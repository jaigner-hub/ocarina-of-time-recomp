// From metroid-prime-4-recomp runtime/main.c @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied with mp4 renamed to oot; the crash dump shows the memory behind each register.
// From fe3h-pc-recomp runtime/main.c @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied with fe3h renamed to mp4 only; adds MP4_BREAK_AT, a log at chosen lifted function entries.
// From totk-pc-recomp runtime/main.c @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), copied with totk renamed to fe3h only.
// oot.exe entry: reserves the guest address space, parses the command line and runs one mode.
//
//   oot.exe [--exefs DIR] [--romfs DIR] [--profile DIR] <mode>
//     --selftest        layout, dispatch table and host-function registry; needs no game files
//     --probe-loader    load and relocate the modules, check them against the measured 1.2.1 values
//     --probe-kernel, --probe-ipc, --probe-services, --probe-fs    host probes
//     --probe-dpi       the process DPI awareness (per-monitor v2) and the primary monitor's size; no game files
//     --probe-gfx-abi   NVN object layouts against nvn.h's reserved sizes; needs no game files
//     --probe-nvn, --probe-nvn-tex, --probe-window, --probe-shader, --probe-vkmem, --probe-vkexec    graphics probes
//     --probe-astc      the compute ASTC decoder against astc_decode_ldr; needs no game files
//     --probe-movie     host VP9 movie::Decoder on one RomFS movie; skips without game data
//     --boot            run the game to the first NVN call
//     --seconds S       run the game (implies --boot's start-up) until the world clock (game time since the loaded
//                       save's first frame) passed S seconds, then exit 0 at that present
//     --shot PNG        with --seconds: save that frame; env OOT_GPU=null|vulkan selects the back end
//     (no mode)         run the game
//
// Also here: the runtime services the generated code calls besides rt_dispatch and rt_svc (rt_brk, rt_udf,
// rt_ticks, the function profiler hook), logging, SHA-256, and the crash reporter (a vectored exception handler that
// names the lifted function, the guest registers and the guest frame-pointer chain).
#include <limits.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>   // the crash reporter, the 1 ms timer and the process memory counters
#include <psapi.h>
#include <timeapi.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "platform/platform.h"

#define MAIN_PATH_MAX 4096

#define MAX_MODULES 16

// ---- logging, tracing, hashing
unsigned rt_trace;

// Log file. A game run whose stderr is not a file already (started from Explorer, a shortcut or a console) also
// writes its log to logs\oot.log next to oot.exe, so a player has one to attach to a report; run_game.py and
// `2> file` keep their own file. Each start moves the earlier logs to oot.1.log .. oot.4.log. A file stops at 64 MiB
// (OOT_TRACE can write that much). Lines from before log_open wait in a static buffer: mem_init runs first so that
// nothing else lands in the guest address space, and an earlier open would put the CRT's heap there.
#define LOG_KEEP 5
#define LOG_CAP (64ull << 20)
enum { LOG_EARLY, LOG_FILE, LOG_OFF };
static int g_log_state;
static PlatAppend *g_log_file;
static char g_log_path[MAIN_PATH_MAX * 3];
static char g_log_early[32768];
static size_t g_log_early_n;
static uint64_t g_log_bytes;

void rt_log_raw(const char *p, size_t n) {
    fwrite(p, 1, n, stderr);
    int state = __atomic_load_n(&g_log_state, __ATOMIC_ACQUIRE);
    if (state == LOG_EARLY) {
        size_t at = __atomic_fetch_add(&g_log_early_n, n, __ATOMIC_RELAXED);
        if (at + n <= sizeof g_log_early) memcpy(g_log_early + at, p, n);
    } else if (state == LOG_FILE) {
        uint64_t at = __atomic_fetch_add(&g_log_bytes, n, __ATOMIC_RELAXED);
        if (at + n <= LOG_CAP) {
            plat_append_write(g_log_file, p, n);
        } else if (at <= LOG_CAP) {   // the first write past the cap says so, once
            static const char cut[] = "[host] log: the file reached 64 MiB, the rest of this run goes to stderr only\n";
            plat_append_write(g_log_file, cut, sizeof cut - 1);
        }
    }
}

void rt_vlogf(const char *fmt, va_list ap) {
    char buf[2048];
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    if (n > 0) rt_log_raw(buf, n < (int)sizeof buf ? (size_t)n : sizeof buf - 1);
}

void rt_logf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    rt_vlogf(fmt, ap);
    va_end(ap);
}

void rt_log(const char *fmt, ...) {
    char buf[2048];
    int n = snprintf(buf, sizeof buf, "[host] ");
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(buf + n, sizeof buf - n - 1, fmt, ap);
    va_end(ap);
    n = k < 0 ? n : n + k < (int)sizeof buf - 1 ? n + k : (int)sizeof buf - 2;
    buf[n++] = '\n';
    rt_log_raw(buf, n);
}

const char *rt_log_path(void) {
    return __atomic_load_n(&g_log_state, __ATOMIC_ACQUIRE) == LOG_FILE ? g_log_path : NULL;
}

// Opens the log file for a game run (game = 1) and writes the buffered lines to it; anything else drops them. Runs on
// the main thread before the boot starts other threads.
static void log_open(int game) {
    char dir[MAIN_PATH_MAX * 3], from[MAIN_PATH_MAX * 3 + 16], to[MAIN_PATH_MAX * 3 + 16];
    char *cut = NULL;
    if (game && !plat_stderr_path(dir, sizeof dir) && plat_exe_path(dir, sizeof dir)) {
        cut = strrchr(dir, '\\');
        char *f = strrchr(dir, '/');
        if (f > cut) cut = f;
    }
    if (cut && (size_t)(cut - dir) + sizeof "\\logs" <= sizeof dir) {
        strcpy(cut, "\\logs");
        plat_file_mkdir_p(dir);
        for (int i = LOG_KEEP - 1; i > 0; i--) {   // oot.3.log -> oot.4.log, ..., oot.log -> oot.1.log
            if (i > 1) snprintf(from, sizeof from, "%s\\oot.%d.log", dir, i - 1);
            else snprintf(from, sizeof from, "%s\\oot.log", dir);
            snprintf(to, sizeof to, "%s\\oot.%d.log", dir, i);
            if (plat_file_exists(from)) plat_file_replace(from, to);
        }
        snprintf(g_log_path, sizeof g_log_path, "%s\\oot.log", dir);
        g_log_file = plat_append_open(g_log_path);
    }
    if (!g_log_file) {
        __atomic_store_n(&g_log_state, LOG_OFF, __ATOMIC_RELEASE);
        return;
    }
    size_t n = __atomic_load_n(&g_log_early_n, __ATOMIC_RELAXED);
    if (n > sizeof g_log_early) n = sizeof g_log_early;
    plat_append_write(g_log_file, g_log_early, n);
    g_log_bytes = n;
    __atomic_store_n(&g_log_state, LOG_FILE, __ATOMIC_RELEASE);
    rt_log("log: %s", g_log_path);
}

static void break_init(void);   // below, with the hook

static void parse_trace(void) {
    break_init();
    const char *e = cfg_get("OOT_TRACE");   // no default: nothing is traced when unset
    if (!e) return;
    static const struct { const char *name; unsigned bit; } names[] = {
        {"svc", RT_TRACE_SVC}, {"host", RT_TRACE_HOSTCALL}, {"ipc", RT_TRACE_IPC}, {"fs", RT_TRACE_FS},
        {"gfx", RT_TRACE_GFX}, {"perf", RT_TRACE_PERF}, {"sched", RT_TRACE_SCHED}, {"audio", RT_TRACE_AUDIO},
        {"present", RT_TRACE_PRESENT}, {"all", ~0u}};
    for (const char *p = e; *p;) {
        size_t n = strcspn(p, ",");
        int known = 0;
        for (unsigned i = 0; i < sizeof names / sizeof *names; i++)
            if (strlen(names[i].name) == n && !strncmp(p, names[i].name, n)) rt_trace |= names[i].bit, known = 1;
        if (!known && n) rt_log("OOT_TRACE: unknown item '%.*s' (svc, host, ipc, fs, gfx, perf, sched, audio, present, all)", (int)n, p);
        p += n + (p[n] == ',');
    }
}

void rt_sha256(const void *p, size_t n, uint8_t out[32]) {
    if (!plat_sha256(p, n, out)) rt_fatal("SHA-256 failed");
}

// ---- configuration
static RtConfig cfg;
const RtConfig *rt_config(void) { return &cfg; }

// ---- runtime services the generated code calls (cpu.h)
volatile int g_fnprof = 0;

// OOT_BREAK_AT=<module>+0x<offset>[,...] (at most 8): a log line at every entry of those lifted functions with the
// caller, the argument registers and the first bytes behind x0 and x1 when they point into committed memory, for
// chasing a bad argument without a debugger. The generated code calls fnprof_hit at every function entry once
// g_fnprof is set, so this costs a compare per call while it is on. The addresses resolve at the first hit, after
// the modules are loaded.
// OOT_BREAK_WATCH=<module>+0x<offset>[,...]: global pointer slots (a GOT entry, a static) whose pointee's first
// bytes are dumped at every break as well, for an object the broken function reaches through a global.
#define BREAK_MAX 8
static struct { char module[16]; uint64_t offset, addr; } g_breaks[BREAK_MAX], g_watches[BREAK_MAX];
static int g_break_count, g_watch_count;
static volatile int32_t g_break_hits;

static void break_init(void) {
    const char *e = cfg_get("OOT_BREAK_AT");
    if (!e || !*e) return;
    for (const char *p = e; *p && g_break_count < BREAK_MAX;) {
        size_t n = strcspn(p, ",");
        const char *plus = memchr(p, '+', n);
        if (!plus || plus - p >= 16) rt_fatal("OOT_BREAK_AT: '%.*s' is not <module>+0x<offset>", (int)n, p);
        memcpy(g_breaks[g_break_count].module, p, (size_t)(plus - p));
        g_breaks[g_break_count].offset = strtoull(plus + 1, NULL, 16);
        g_break_count++;
        p += n + (p[n] == ',');
    }
    const char *w = cfg_get("OOT_BREAK_WATCH");
    for (const char *p = w ? w : ""; *p && g_watch_count < BREAK_MAX;) {
        size_t n = strcspn(p, ",");
        const char *plus = memchr(p, '+', n);
        if (!plus || plus - p >= 16) rt_fatal("OOT_BREAK_WATCH: '%.*s' is not <module>+0x<offset>", (int)n, p);
        memcpy(g_watches[g_watch_count].module, p, (size_t)(plus - p));
        g_watches[g_watch_count].offset = strtoull(plus + 1, NULL, 16);
        g_watch_count++;
        p += n + (p[n] == ',');
    }
    g_fnprof = 1;
}

// Guest memory is reserved by the platform layer (memory.c), so plat_mem_state answers for every address the guest uses.
static int guest_readable(uint64_t a, size_t n) {
    if (!a) return 0;
    uint64_t end = 0;
    int st = plat_mem_state(a, a + n, &end);
    if (st == PLAT_PROT_NONE || st == PLAT_MEM_RESERVED || st == PLAT_MEM_FREE) return 0;
    return end >= a + n;
}

static void break_dump(const char *what, uint64_t a, size_t n) {
    if (!guest_readable(a, n)) return;
    char line[3 * 64 + 1];
    for (size_t off = 0; off < n; off += 32) {
        size_t k = 0;
        for (size_t i = 0; i < 32 && off + i < n; i++)
            k += (size_t)snprintf(line + k, sizeof line - k, "%02x ", *(const uint8_t *)(uintptr_t)(a + off + i));
        rt_log("break:   %s+%#zx: %s", what, off, line);
    }
}

// OOT_BREAK_FOLLOW=[x<n>:]<off>[,<off>...][@<off>]: from x0 (or x<n>), load the pointer at each offset in turn and
// dump 64 bytes at the end of the chain, from "@<off>" into the final object (an object reached through a list entry,
// say: its first word is its vtable).
static void break_follow(const Cpu *c) {
    const char *follow = cfg_value("OOT_BREAK_FOLLOW");
    if (follow && *follow) {
        int reg = 0;
        const char *s = follow;
        if (s[0] == 'x' && s[1] >= '0' && s[1] <= '7' && s[2] == ':') reg = s[1] - '0', s += 3;
        uint64_t q = c->x[reg], at = 0;
        char chain[96];
        snprintf(chain, sizeof chain, "x%d", reg);
        for (; *s && q;) {
            char *end;
            if (*s == '@') { at = strtoull(s + 1, &end, 16); s = end; break; }
            uint64_t off = strtoull(s, &end, 16);
            if (end == s) break;
            q = guest_readable(q + off, 8) ? *(const uint64_t *)(uintptr_t)(q + off) : 0;
            snprintf(chain + strlen(chain), sizeof chain - strlen(chain), "->[+%#llx]", (unsigned long long)off);
            s = *end == ',' ? end + 1 : end;
        }
        if (q) {
            snprintf(chain + strlen(chain), sizeof chain - strlen(chain), "=%#llx+%#llx", (unsigned long long)q, (unsigned long long)at);
            break_dump(chain, q + at, 64);
        }
    }
}

void fnprof_hit(uint64_t addr, void *cpu) {
    const Cpu *c = cpu;
    static volatile int32_t resolved;
    if (!resolved) {
        for (int k = 0; k < g_break_count; k++)
            for (int i = 0; i < ldr_count(); i++)
                if (!strcmp(ldr_module(i)->name, g_breaks[k].module))
                    g_breaks[k].addr = ldr_module(i)->base + g_breaks[k].offset;
        for (int k = 0; k < g_watch_count; k++)
            for (int i = 0; i < ldr_count(); i++)
                if (!strcmp(ldr_module(i)->name, g_watches[k].module))
                    g_watches[k].addr = ldr_module(i)->base + g_watches[k].offset;
        __atomic_exchange_n(&resolved, 1, __ATOMIC_SEQ_CST);
    }
    for (int k = 0; k < g_break_count; k++) {
        if (addr != g_breaks[k].addr) continue;
        // OOT_BREAK_BRIEF=1: one line per hit with x0..x3 and 32 bytes at x0+0x30, up to 400,000 hits, for a value's
        // trajectory over a run; otherwise the full dump, 4,000 hits.
        // OOT_BREAK_COUNT=1: no line per hit; hits are counted per caller, and every 10 s the callers with the most
        // hits since the last summary are logged (for finding which call sites of a common function run per frame).
        static int count = -1;
        if (count < 0) { const char *e = cfg_get("OOT_BREAK_COUNT"); count = e && !strcmp(e, "1"); }
        if (count) {
            enum { NC = 1024 };
            static struct { uint64_t lr; volatile int32_t n; } cs[NC];
            static PlatMutex cl = PLAT_MUTEX_INIT;
            static uint64_t t_last;
            uint64_t lr = c->x[30] - 4;
            plat_mutex_lock(&cl);
            uint32_t h = (uint32_t)((lr >> 2) * 0x9e3779b1u) & (NC - 1);
            while (cs[h].lr && cs[h].lr != lr) h = (h + 1) & (NC - 1);
            cs[h].lr = lr, cs[h].n++;
            uint64_t now = plat_tick_ms();
            if (!t_last) t_last = now;
            if (now - t_last >= 10000) {
                t_last = now;
                rt_log("break count: frame %llu, callers of %s:", (unsigned long long)gfx_stats().frames,
                       ldr_describe(addr, (char[128]){0}, 128));
                for (int r = 0; r < 60; r++) {
                    int best = -1;
                    for (int i = 0; i < NC; i++)
                        if (cs[i].lr && cs[i].n > 0 && (best < 0 || cs[i].n > cs[best].n)) best = i;
                    if (best < 0) break;
                    char d[128];
                    rt_log("  %8ld from %s", (long)cs[best].n, ldr_describe(cs[best].lr, d, sizeof d));
                    cs[best].n = -cs[best].n;   // listed; restored below
                }
                for (int i = 0; i < NC; i++) cs[i].n = 0;
            }
            plat_mutex_unlock(&cl);
            continue;
        }
        static int brief = -1;
        if (brief < 0) brief = !strcmp(cfg_value("OOT_BREAK_BRIEF"), "1");
        int32_t hits = __atomic_add_fetch(&g_break_hits, 1, __ATOMIC_SEQ_CST);
        if (hits > (brief ? 400000 : 4000)) return;
        char at[128], lr[128];
        if (brief) {
            char peek[3 * 32 + 1] = "";
            if (guest_readable(c->x[0] + 0x30, 32))
                for (int i = 0; i < 32; i++) snprintf(peek + 3 * i, 4, "%02x ", *(const uint8_t *)(uintptr_t)(c->x[0] + 0x30 + i));
            float s0, s1;   // the first float arguments
            memcpy(&s0, &c->v[0], 4), memcpy(&s1, &c->v[1], 4);
            rt_log("break: %s from %s: x0 %#llx x1 %#llx x2 %#llx x3 %#llx s0 %g s1 %g; x0+0x30: %s", ldr_describe(addr, at, sizeof at),
                   ldr_describe(c->x[30] - 4, lr, sizeof lr), (unsigned long long)c->x[0], (unsigned long long)c->x[1],
                   (unsigned long long)c->x[2], (unsigned long long)c->x[3], s0, s1, peek);
            break_follow(c);
            continue;
        }
        rt_log("break: %s from %s: x0 %#llx x1 %#llx x2 %#llx x3 %#llx x4 %#llx x8 %#llx sp %#llx", ldr_describe(addr, at, sizeof at),
               ldr_describe(c->x[30] - 4, lr, sizeof lr), (unsigned long long)c->x[0], (unsigned long long)c->x[1],
               (unsigned long long)c->x[2], (unsigned long long)c->x[3], (unsigned long long)c->x[4],
               (unsigned long long)c->x[8], (unsigned long long)c->sp);
        break_dump("x0", c->x[0], 192);
        break_dump("x1", c->x[1], 64);
        // OOT_BREAK_FOLLOW=<off>[,<off>...]: from x0, load the pointer at each offset in turn and dump 64 bytes at
        // the end of the chain (an object reached through a list entry, say: its first word is its vtable)
        break_follow(c);
        for (int w = 0; w < g_watch_count; w++) {
            if (!guest_readable(g_watches[w].addr, 8)) continue;
            uint64_t obj = *(const uint64_t *)(uintptr_t)g_watches[w].addr;
            char name[40];
            snprintf(name, sizeof name, "*%s+%#llx=%#llx", g_watches[w].module, (unsigned long long)g_watches[w].offset,
                     (unsigned long long)obj);
            break_dump(name, obj, 128);
        }
    }
}

void rt_brk(Cpu *c, uint32_t imm, uint64_t pc) {
    char at[128], lr[128];
    rt_fatal("brk #%#x at %s (lr %s)", imm, ldr_describe(pc, at, sizeof at), ldr_describe(c->x[30], lr, sizeof lr));
}

void rt_udf(Cpu *c, uint64_t pc) {
    char at[128], lr[128];
    rt_fatal("undefined instruction at %s (lr %s)", ldr_describe(pc, at, sizeof at),
             ldr_describe(c->x[30], lr, sizeof lr));
}

// CNTPCT_EL0: the Switch counter runs at 19.2 MHz. The scale is set before main, so no thread reads it half written.
static PlatScale tick_scale;

static uint64_t g_start_ns;
double rt_uptime_s(void) { return (double)(plat_now_ns() - g_start_ns) / 1e9; }
__attribute__((constructor)) static void uptime_init(void) { g_start_ns = plat_now_ns(); }
__attribute__((constructor)) static void tick_scale_init(void) { tick_scale = plat_scale(19200000u, 1000000000u); }

uint64_t rt_ticks(void) { return plat_scaled(tick_scale, plat_now_ns()); }

#ifdef _WIN32
typedef struct { uintptr_t host; uint64_t guest; } FnRev;
static FnRev *fn_rev;
static uint64_t fn_rev_n;

static int by_host(const void *a, const void *b) {
    uintptr_t x = ((const FnRev *)a)->host, y = ((const FnRev *)b)->host;
    return x < y ? -1 : x > y;
}

// The lifted function that contains a host code address, by the nearest function start below it.
static uint64_t lifted_function(uintptr_t rip) {
    if (!fn_rev) {
        static RtModule mods[MAX_MODULES];
        int n = rt_linked_modules(mods, MAX_MODULES);
        uint64_t total = 0;
        for (int i = 0; i < n; i++) total += mods[i].count;
        fn_rev = malloc((total ? total : 1) * sizeof *fn_rev);
        if (!fn_rev) return 0;
        for (int i = 0; i < n; i++)
            for (uint64_t k = 0; k < mods[i].count; k++)
                fn_rev[fn_rev_n++] = (FnRev){(uintptr_t)mods[i].fns[k], mods[i].addrs[k]};
        qsort(fn_rev, fn_rev_n, sizeof *fn_rev, by_host);
    }
    uint64_t lo = 0, hi = fn_rev_n;
    while (lo < hi) {
        uint64_t mid = (lo + hi) / 2;
        if (fn_rev[mid].host <= rip) lo = mid + 1;
        else hi = mid;
    }
    return lo ? fn_rev[lo - 1].guest : 0;
}

static int readable(uint64_t a, uint64_t n) {
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery((void *)(uintptr_t)a, &mbi, sizeof mbi) || mbi.State != MEM_COMMIT) return 0;
    if (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) return 0;
    return a + n <= (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize;
}

static LONG CALLBACK crash_handler(EXCEPTION_POINTERS *ep) {
    static volatile int32_t busy;
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_ILLEGAL_INSTRUCTION &&
        code != EXCEPTION_INT_DIVIDE_BY_ZERO && code != EXCEPTION_PRIV_INSTRUCTION && code != EXCEPTION_STACK_OVERFLOW &&
        code != EXCEPTION_IN_PAGE_ERROR)
        return EXCEPTION_CONTINUE_SEARCH;
    uintptr_t rip = (uintptr_t)ep->ContextRecord->Rip;
    HMODULE self = GetModuleHandleW(NULL);
    MODULEINFO mi;
    if (!GetModuleInformation(GetCurrentProcess(), self, &mi, sizeof mi) || rip - (uintptr_t)mi.lpBaseOfDll >= mi.SizeOfImage) {
        // Not our code: left to the handler that expects it, but said once, as a process that dies in a driver or
        // a DLL otherwise leaves no trace (2026-11-05: exit code 40 after frame 1, no FATAL line).
        static volatile int32_t foreign;
        if (__atomic_add_fetch(&foreign, 1, __ATOMIC_SEQ_CST) <= 4) {
            HMODULE m = NULL;
            wchar_t name[MAIN_PATH_MAX] = L"?";
            if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   (LPCWSTR)rip, &m) && m)
                GetModuleFileNameW(m, name, MAIN_PATH_MAX);
            rt_logf("exception %#lx at host %p (%ls+%#llx, thread %lu), left to its own handler\n", code, (void *)rip, name,
                    (unsigned long long)(m ? rip - (uintptr_t)m : 0), GetCurrentThreadId());
            // Where it came from, for an address in no module (a jump through a bad pointer): the registers, the
            // faulting address, and the stack's first words that point into the exe (tools/resolve_map.py names them).
            const CONTEXT *cx = ep->ContextRecord;
            uint64_t fault = ep->ExceptionRecord->NumberParameters >= 2 ? ep->ExceptionRecord->ExceptionInformation[1] : 0;
            rt_logf("  rsp %#llx rbp %#llx rax %#llx rcx %#llx rdx %#llx r8 %#llx r9 %#llx %s %#llx\n",
                    (unsigned long long)cx->Rsp, (unsigned long long)cx->Rbp, (unsigned long long)cx->Rax,
                    (unsigned long long)cx->Rcx, (unsigned long long)cx->Rdx, (unsigned long long)cx->R8,
                    (unsigned long long)cx->R9, ep->ExceptionRecord->ExceptionInformation[0] == 8 ? "executing" : "address",
                    (unsigned long long)fault);
            MEMORY_BASIC_INFORMATION mb;
            if (VirtualQuery((const void *)(uintptr_t)rip, &mb, sizeof mb))
                rt_logf("  rip's region: base %p size %#llx state %#lx protect %#lx type %#lx\n", mb.BaseAddress,
                        (unsigned long long)mb.RegionSize, mb.State, mb.Protect, mb.Type);
            const uint64_t *st = (const uint64_t *)(uintptr_t)cx->Rsp;
            int shown = 0;
            for (int i = 0; i < 256 && shown < 12; i++) {
                if (VirtualQuery(st + i, &mb, sizeof mb) == 0 || mb.State != MEM_COMMIT || (mb.Protect & (PAGE_GUARD | PAGE_NOACCESS))) break;
                uint64_t v = st[i];
                if (v - (uintptr_t)mi.lpBaseOfDll < mi.SizeOfImage) {
                    rt_logf("  stack[%d] = exe+%#llx\n", i, (unsigned long long)(v - (uintptr_t)mi.lpBaseOfDll));
                    shown++;
                }
            }
            fflush(stderr);
        }
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if (__atomic_exchange_n(&busy, 1, __ATOMIC_SEQ_CST)) ExitProcess(71);
    char d[160];
    rt_logf("FATAL: exception %#lx at host %p (oot.exe+%#llx)", code, (void *)rip,
            (unsigned long long)(rip - (uintptr_t)mi.lpBaseOfDll));
    if (code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2)
        rt_logf(", %s address %#llx",
                ep->ExceptionRecord->ExceptionInformation[0] == 1 ? "writing" : ep->ExceptionRecord->ExceptionInformation[0] == 8 ? "executing" : "reading",
                (unsigned long long)ep->ExceptionRecord->ExceptionInformation[1]);
    rt_log_raw("\n", 1);
    uint64_t f = lifted_function(rip);
    if (f) rt_logf("  in lifted function f_%llx = %s\n", (unsigned long long)f, ldr_describe(f, d, sizeof d));
    Cpu *c = kern_cpu();
    if (c) {
        for (int i = 0; i < 31; i++)
            rt_logf("  x%-2d %016llx%s", i, (unsigned long long)c->x[i], i % 4 == 3 ? "\n" : "");
        rt_logf("  sp  %016llx\n  pc  %016llx  lr %s\n", (unsigned long long)c->sp, (unsigned long long)c->pc,
                ldr_describe(c->x[30], d, sizeof d));
        uint64_t fp = c->x[29];
        for (int n = 0; n < 32 && fp && !(fp & 7) && readable(fp, 16); n++) {
            uint64_t next = LD64(fp), lr = LD64(fp + 8);
            rt_logf("  guest frame %2d: %s\n", n, ldr_describe(lr, d, sizeof d));
            if (next <= fp || next - fp > (64u << 20)) break;
            fp = next;
        }
        // The first 160 bytes behind each register that points at readable memory, as hex and text: the object or
        // string the faulting code was working on. The registers are as of the last sync point (the latest call).
        // The thread's TLS slots: SDK 22 keeps them in the thread object behind TLR+0x1F8, at +0x80 (nn::os::GetTlsValue).
        if (c->tpidrro && readable(c->tpidrro + 0x1F8, 8)) {
            uint64_t thr = LD64(c->tpidrro + 0x1F8);
            rt_logf("  TLR %#llx thread object %#llx\n", (unsigned long long)c->tpidrro, (unsigned long long)thr);
            if (thr && readable(thr + 0x80, 0x80)) {
                for (int k = 0; k < 16; k += 4)
                    rt_logf("  tls[%2d..%2d] %016llx %016llx %016llx %016llx\n", k, k + 3, (unsigned long long)LD64(thr + 0x80 + 8 * k),
                            (unsigned long long)LD64(thr + 0x88 + 8 * k), (unsigned long long)LD64(thr + 0x90 + 8 * k),
                            (unsigned long long)LD64(thr + 0x98 + 8 * k));
                // sead keeps a per-thread block behind one slot; its current heap sits at +0x88 (main+0x3ea70)
                for (int k = 0; k < 16; k++) {
                    uint64_t v = LD64(thr + 0x80 + 8 * k);
                    if (v > 0x10000 && !(v >> 48) && readable(v + 0x80, 0x20))
                        rt_logf("  tls[%2d] -> [+0x80] %016llx [+0x88] %016llx [+0x90] %016llx\n", k, (unsigned long long)LD64(v + 0x80),
                                (unsigned long long)LD64(v + 0x88), (unsigned long long)LD64(v + 0x90));
                }
            }
        }
        // OOT_CRASH_PEEK=<bytes> widens the peek (up to 16 KiB) to see a whole object.
        const char *pk = cfg_get("OOT_CRASH_PEEK");
        size_t peek = pk ? (size_t)strtoul(pk, NULL, 0) : 160;
        if (peek < 16) peek = 16;
        if (peek > 16384) peek = 16384;
        for (int i = 0; i < 29; i++) {
            uint64_t a = c->x[i];
            if (a < 0x10000 || (a & 0xffff000000000000ull) || !readable(a, peek)) continue;
            const uint8_t *m = (const uint8_t *)(uintptr_t)a;
            rt_logf("  [x%-2d] at %#llx:\n", i, (unsigned long long)a);
            for (size_t row = 0; row < peek; row += 32) {
                char hex[32 * 3 + 1], txt[33];
                size_t n = peek - row < 32 ? peek - row : 32;
                for (size_t k = 0; k < n; k++) {
                    snprintf(hex + k * 3, 4, "%02x ", m[row + k]);
                    txt[k] = m[row + k] >= 0x20 && m[row + k] < 0x7f ? (char)m[row + k] : '.';
                }
                txt[n] = 0;
                rt_logf("    +%04zx %s |%s|\n", row, hex, txt);
            }
        }
    } else {
        rt_logf("  not on a guest thread\n");
    }
    fflush(stderr);
    ExitProcess(71);
}
#endif

// ---- self-test: layout, dispatch table, host-function registry, without game files

// A block aligned to 64 bytes for the probe's Cpu: _aligned_malloc on Windows, aligned_alloc elsewhere.
static void *aligned_block(size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n, 64);
#else
    return aligned_alloc(64, (n + 63) & ~(size_t)63);
#endif
}

static void aligned_block_free(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("  FAIL: " __VA_ARGS__); putchar('\n'); } } while (0)

static double now_s(void) { return (double)plat_now_ns() / 1e9; }

static volatile int host_probe_hits;
static void host_probe(Cpu *c) { host_probe_hits++; c->x[0] = 42; }

static void check_cpu(void) {
    __builtin_cpu_init();
    int ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("bmi2");
    printf("cpu: x86-64-v3 (AVX2, FMA, BMI2) %s\n", ok ? "present" : "MISSING");
    CHECK(ok, "the lifted code is built for x86-64-v3");
}

// Layout: ascending, no overlap, below 2^39, and inside the window mem_init reserved.
static void check_layout(void) {
    const RtLayoutEntry *lay = rt_layout;
    int n = rt_layout_count;
    printf("layout: %d modules\n", n);
    for (int i = 0; i < n; i++) {
        printf("  %-8s base %#llx  size %#10llx  end %#llx\n", lay[i].name, (unsigned long long)lay[i].base,
               (unsigned long long)lay[i].size, (unsigned long long)(lay[i].base + lay[i].size));
        CHECK(lay[i].size > 0, "%s has size 0", lay[i].name);
        CHECK(lay[i].base % 0x10000 == 0, "%s base is not 64 KiB aligned", lay[i].name);
        CHECK(lay[i].base + lay[i].size <= RT_KERNEL_BASE, "%s reaches into the kernel region", lay[i].name);
        if (i) {
            CHECK(lay[i].base > lay[i - 1].base, "%s base is not above %s", lay[i].name, lay[i - 1].name);
            CHECK(lay[i - 1].base + lay[i - 1].size <= lay[i].base, "%s overlaps %s", lay[i - 1].name, lay[i].name);
        }
        uint64_t end = 0;
        int ours = plat_mem_state(lay[i].base, lay[i].base + 1, &end) == PLAT_MEM_RESERVED;
        CHECK(ours, "%s is not inside the reserved module window", lay[i].name);
    }
    const RtGuestLayout *g = mem_layout();
    printf("guest memory: aslr %#llx+%#llx stack %#llx heap %#llx alias %#llx+%#llx kernel %#llx host functions %#llx\n",
           (unsigned long long)g->aslr.base, (unsigned long long)g->aslr.size, (unsigned long long)g->stack.base,
           (unsigned long long)g->heap.base, (unsigned long long)g->alias.base, (unsigned long long)g->alias.size,
           (unsigned long long)g->kernel.base, (unsigned long long)g->hostfn.base);
    CHECK(g->alias.base + g->alias.size <= lay[0].base && g->heap.base + g->heap.size <= g->alias.base,
          "guest regions overlap");
}

// Every function start dispatches to its own function. The word after a start (unless that is a start too), a
// misaligned address and the word below the module dispatch to nothing.
static void check_modules(const RtModule *mods, int n) {
    uint64_t total = 0;
    for (int i = 0; i < n; i++) {
        const RtModule *m = &mods[i];
        const RtLayoutEntry *l = NULL;
        for (int k = 0; k < rt_layout_count; k++)
            if (!strcmp(rt_layout[k].name, m->name)) l = &rt_layout[k];
        CHECK(l && l->base == m->base && l->size == m->size, "%s: linked base/size differ from the layout", m->name);

        uint64_t bad = 0, bad_neg = 0;
        for (uint64_t k = 0; k < m->count; k++) {
            uint64_t a = m->addrs[k];
            if (rt_dispatch_lookup(a) != m->fns[k] && bad++ < 5)
                printf("  FAIL: %s_fn_addrs[%llu] %#llx does not dispatch to its function\n", m->name,
                       (unsigned long long)k, (unsigned long long)a);
            int next_is_start = k + 1 < m->count && m->addrs[k + 1] == a + 4;
            if ((!next_is_start && rt_dispatch_lookup(a + 4)) || rt_dispatch_lookup(a + 2)) bad_neg++;
        }
        if (rt_dispatch_lookup(m->base - 4)) bad_neg++;
        failures += bad > 0;
        CHECK(!bad_neg, "%s: %llu addresses that are not function starts dispatch to a function", m->name,
              (unsigned long long)bad_neg);
        uint64_t lo = m->count ? m->addrs[0] : 0, hi = m->count ? m->addrs[m->count - 1] : 0;
        printf("  %-8s %8llu functions, first %#llx, last %#llx: %s\n", m->name, (unsigned long long)m->count,
               (unsigned long long)lo, (unsigned long long)hi, bad || bad_neg ? "FAIL" : "ok");
        total += m->count;
    }
    printf("  total    %8llu functions\n", (unsigned long long)total);
}

static void bench(const RtModule *mods, int n) {
    uint64_t total = 0, sink = 0;
    double t0 = now_s();
    for (int i = 0; i < n; i++) {
        for (uint64_t k = 0; k < mods[i].count; k++) sink += (uintptr_t)rt_dispatch_lookup(mods[i].addrs[k]);
        total += mods[i].count;
    }
    double dt = now_s() - t0;
    __asm__ volatile("" ::"r"(sink));
    if (total) printf("lookup: %.1f ns per function start (address order)\n", dt * 1e9 / (double)total);
}

// Memory SVCs without modules: heap, physical memory in the alias region, a stack move, QueryMemory, GetInfo.
static void check_memory(void) {
    const RtGuestLayout *g = mem_layout();
    RtMemoryInfo q;
    uint64_t heap = 0, v = 0;
    int before = failures;
    CHECK(mem_set_heap_size(0x400000, &heap) == 0 && heap == g->heap.base, "SetHeapSize 4 MiB");
    memset((void *)(uintptr_t)heap, 0xAB, 0x400000);
    mem_query(heap + 0x1000, &q);
    CHECK(q.base == heap && q.size == 0x400000 && q.state == 5 && q.perm == 3, "QueryMemory on the heap");
    CHECK(mem_set_heap_size(0x300000, &heap) != 0, "SetHeapSize accepts a size that is not a 2 MiB multiple");

    uint64_t a = g->alias.base + 0x100000;
    CHECK(mem_map_physical(a, 0x10000) == 0, "MapPhysicalMemory");
    ST64(a + 0x8000, 0x1234);
    CHECK(mem_map_physical(a - 0x4000, 0x20000) == 0 && LD64(a + 0x8000) == 0x1234,
          "MapPhysicalMemory over a mapped range lost its contents");
    mem_query(a, &q);
    CHECK(q.base == a - 0x4000 && q.size == 0x20000 && q.state == 5, "QueryMemory on mapped alias pages");
    mem_info(7, &v);
    CHECK(v == 0x400000 + 0x20000, "GetInfo 7 (used memory) %#llx", (unsigned long long)v);

    uint64_t stack = g->stack.base + 0x100000;
    ST64(a + 8, 0x99);
    CHECK(mem_map(stack, a, 0x4000) == 0 && LD64(stack + 8) == 0x99, "MapMemory alias -> stack did not copy");
    ST64(stack + 0x10, 0x5678);
    mem_query(stack, &q);
    CHECK(q.base == stack && q.size == 0x4000 && q.state == 0xB, "QueryMemory on a moved stack");
    mem_query(a, &q);
    CHECK(q.base == a && q.state == 5 && q.perm == 0, "QueryMemory on a moved stack's source");
    CHECK(mem_unmap_physical(a, 0x4000) != 0, "UnmapPhysicalMemory under a moved stack");
    CHECK(mem_unmap(stack, a, 0x4000) == 0 && LD64(a + 0x10) == 0x5678, "UnmapMemory did not copy the stack back");

    CHECK(mem_unmap_physical(a - 0x4000, 0x20000) == 0, "UnmapPhysicalMemory");
    mem_query(a, &q);
    CHECK(q.state == 0 && q.base == g->alias.base && q.size == g->alias.size, "QueryMemory on the free alias region");
    CHECK(mem_map_physical(a, 0x1000) == 0 && LD64(a + 0x10) == 0 && mem_unmap_physical(a, 0x1000) == 0,
          "remapped physical memory is not zeroed");
    CHECK(mem_map_physical(a, 0x100000000ull) == 0x10801, "MapPhysicalMemory beyond the pool is not LimitReached");
    mem_query(RT_HOSTFN_BASE + 0x40, &q);
    CHECK(q.state == 0x10 && q.base == RT_HOSTFN_BASE, "QueryMemory on the host-function window");
    mem_query(g->alias.base + g->alias.size + 0x1000, &q);
    CHECK(q.state == 3 && q.base == g->alias.base + g->alias.size, "QueryMemory on host memory");
    CHECK(mem_set_heap_size(0, &heap) == 0, "SetHeapSize 0");
    printf("memory: heap, physical memory, stack move, QueryMemory, GetInfo: %s\n", failures > before ? "FAIL" : "ok");
}

static int selftest(void) {
    static RtModule mods[MAX_MODULES];
    int n = rt_linked_modules(mods, MAX_MODULES);

    check_cpu();
    check_layout();
    uint64_t probe = hostfn_register("<selftest probe>", host_probe);

    double t0 = now_s();
    hostfn_seal();
    double dt = now_s() - t0;
    RtDispatchStats s = rt_dispatch_stats();
    printf("dispatch: %d modules linked, table covers %#llx-%#llx: %llu slots, %.1f MiB reserved, built in %.0f ms\n",
           n, (unsigned long long)s.lo, (unsigned long long)s.hi, (unsigned long long)s.slots,
           (double)s.table_bytes / (1 << 20), dt * 1e3);
    check_modules(mods, n);
    CHECK(s.functions > 0, "no functions linked");
    Cpu *c = aligned_block(sizeof(Cpu));
    memset(c, 0, sizeof *c);
    uint64_t r = rt_call(c, probe, 1, (uint64_t)7);
    CHECK(rt_dispatch_lookup(probe) && r == 42 && host_probe_hits == 1 && c->x[30] == 0,
          "a registered host function does not dispatch through rt_call");
    CHECK(!rt_dispatch_lookup(probe + 16), "an unregistered host-function address dispatches");
    CHECK(rt_dispatch_lookup(RT_RET_HOST), "the rt_call return sentinel is not registered");
    aligned_block_free(c);
    bench(mods, n);

    void *k1 = mem_kernel_alloc(0x200, 16), *k2 = mem_kernel_alloc(0x2000, 0x1000);
    CHECK((uintptr_t)k1 - RT_KERNEL_BASE < RT_KERNEL_SIZE && !((uintptr_t)k2 & 0xFFF) && k1 != k2,
          "kernel-region allocation");
    mem_kernel_free(k1, 0x200);
    CHECK(mem_kernel_alloc(0x200, 16) == k1, "a freed kernel-region block is not reused");
    check_memory();

#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc = {.cb = sizeof pmc};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc))
        printf("process: peak working set %.1f MiB\n", (double)pmc.PeakWorkingSetSize / (1 << 20));
#else
    float used_mb, total_mb, proc_mb;
    plat_mem_info(&used_mb, &total_mb, &proc_mb);
    printf("process: working set %.1f MiB\n", (double)proc_mb);
#endif
    printf("selftest: %s (%d failures)\n", failures ? "FAILED" : "passed", failures);
    return failures ? 1 : 0;
}

static int usage(const char *argv0) {
    fprintf(stderr, "usage: %s [--exefs DIR] [--romfs DIR] [--profile DIR] [--selftest | --probe-loader | "
                    "--probe-cfg | --probe-dpi | --probe-kernel | --probe-ipc | --probe-services | --probe-fs | --probe-gfx-abi | "
                    "--probe-nvn | --probe-nvn-tex | --probe-window | --probe-shader | --probe-vkmem | "
                    "--probe-vkexec | --probe-tensor | --probe-astc | --probe-movie | --prepare-shaders | --warm-shaders | --boot |--seconds S [--shot PNG]]\n", argv0);
    return 2;
}

// Paths for a copied folder (tools/package_build.py). Per path: OOT_EXEFS, OOT_ROMFS, OOT_PROFILE_DIR, else the
// folder next to oot.exe (exefs, romfs, profile) if it exists, else the path baked in at build time. A relative
// value is relative to oot.exe's folder.
static char pp_exefs[MAIN_PATH_MAX * 2], pp_romfs[MAIN_PATH_MAX * 2], pp_profile[MAIN_PATH_MAX * 2];

static const char *portable_path(const char *dir, const char *var, const char *sub, const char *must, char *out, size_t cap,
                                 const char *fallback) {
    const char *e = cfg_get(var);
    if (e && *e) {
        int abs_path = e[0] == '\\' || e[0] == '/' || e[1] == ':';
        if (abs_path) snprintf(out, cap, "%s", e);
        else snprintf(out, cap, "%s/%s", dir, e);
        return out;
    }
    char probe[MAIN_PATH_MAX * 3];
    snprintf(out, cap, "%s/%s", dir, sub);
    snprintf(probe, sizeof probe, "%s/%s", out, must);
    struct stat st;
    return stat(probe, &st) == 0 ? out : fallback;
}

static void portable_paths(RtConfig *c) {
    char exe[MAIN_PATH_MAX * 2];
    int ok = plat_exe_path(exe, sizeof exe);
    char *cut = ok ? strrchr(exe, '/') : NULL, *bcut = ok ? strrchr(exe, '\\') : NULL;
    if (bcut && (!cut || bcut > cut)) cut = bcut;
    if (!cut) return;
    *cut = 0;
    c->exefs = portable_path(exe, "OOT_EXEFS", "exefs", "main.npdm", pp_exefs, sizeof pp_exefs, c->exefs);
    c->romfs = portable_path(exe, "OOT_ROMFS", "romfs", "romfs.bin", pp_romfs, sizeof pp_romfs, c->romfs);
    c->profile = portable_path(exe, "OOT_PROFILE_DIR", "profile", ".", pp_profile, sizeof pp_profile, c->profile);
}

int main(int argc, char **argv) {
    mem_init();   // first: claim the guest address space before anything else can land there; reads oot.ini
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
#ifdef _WIN32
    AddVectoredExceptionHandler(0, crash_handler);
#endif
    parse_trace();
    prof_start();
    // Timed guest waits (kernel.c block_inner) sleep in whole milliseconds; at the default 15.6 ms tick a 1 ms wait
    // costs a full tick. OOT_TIMER_1MS=0 keeps the default for comparison.
#ifdef _WIN32
    if (cfg_bool("OOT_TIMER_1MS")) timeBeginPeriod(1);
#endif
    cfg = (RtConfig){.exefs = rt_default_exefs, .romfs = rt_default_romfs, .profile = rt_default_profile,
                     .gpu = cfg_get("OOT_GPU")};
    portable_paths(&cfg);   // command-line paths below still win
    if (cfg.gpu && strcmp(cfg.gpu, "null") && strcmp(cfg.gpu, "vulkan")) {
        rt_logf("OOT_GPU must be null or vulkan, not '%s'\n", cfg.gpu);
        return 2;
    }
    const char *mode = NULL, *cs_dir = "probe-cs";
    for (int i = 1; i < argc; i++) {
        const char **path = !strcmp(argv[i], "--exefs") ? &cfg.exefs : !strcmp(argv[i], "--romfs") ? &cfg.romfs
                          : !strcmp(argv[i], "--profile") ? &cfg.profile : NULL;
        if (path) {
            if (++i == argc) return usage(argv[0]);
            *path = argv[i];
        } else if (!strcmp(argv[i], "--seconds")) {
            char *end;
            if (++i == argc) return usage(argv[0]);
            double s = strtod(argv[i], &end);
            if (*end || !(s > 0) || s > 1e6) return usage(argv[0]);
            cfg.seconds = s;
        } else if (!strcmp(argv[i], "--shot")) {
            if (++i == argc) return usage(argv[0]);
            cfg.shot = argv[i];
        } else if (!mode && !strcmp(argv[i], "--probe-coopmat")) {   // --probe-coopmat [shader.spv]: coopmat_probe.c
            mode = argv[i];
            if (i + 1 < argc && strncmp(argv[i + 1], "--", 2)) cs_dir = argv[++i];
            else cs_dir = NULL;
        } else if (!mode && !strcmp(argv[i], "--probe-cs")) {   // --probe-cs <dir>: vk_exec_probe.c probe_cs
            mode = argv[i];
            if (++i == argc) return usage(argv[0]);
            cs_dir = argv[i];
        } else if (!mode && !strncmp(argv[i], "--", 2)) {
            mode = argv[i];
        } else {
            return usage(argv[0]);
        }
    }
    if (cfg.shot && !cfg.seconds) return usage(argv[0]);
    if (cfg.seconds && mode) return usage(argv[0]);
    log_open(!mode);
    if (!mode) {
        cfg.boot = cfg.seconds > 0;   // --seconds starts the game like --boot, with its checkpoints
        rt_boot();
        return 0;
    }
    if (!strcmp(mode, "--selftest")) return selftest();
    if (!strcmp(mode, "--probe-loader")) return probe_loader();
    if (!strcmp(mode, "--probe-cfg")) return probe_cfg();
    if (!strcmp(mode, "--probe-dpi")) return probe_dpi();
    if (!strcmp(mode, "--probe-kernel")) return probe_kernel();
    if (!strcmp(mode, "--probe-ipc")) return probe_ipc();
    if (!strcmp(mode, "--probe-services")) return probe_services();
    if (!strcmp(mode, "--probe-fs")) return probe_fs();
    if (!strcmp(mode, "--probe-gfx-abi")) return probe_gfx_abi();
    if (!strcmp(mode, "--probe-nvn")) return probe_nvn();
    if (!strcmp(mode, "--probe-nvn-tex")) return probe_nvn_tex();
    if (!strcmp(mode, "--probe-window")) return probe_window();
    if (!strcmp(mode, "--probe-shader")) return probe_shader();
    if (!strcmp(mode, "--probe-vkmem")) return probe_vkmem();
    if (!strcmp(mode, "--probe-vkexec")) return probe_vkexec();
    if (!strcmp(mode, "--probe-tensor")) return probe_tensor();
    if (!strcmp(mode, "--probe-cs")) return probe_cs(cs_dir);
    if (!strcmp(mode, "--probe-coopmat")) return probe_coopmat(cs_dir);
    if (!strcmp(mode, "--probe-astc")) return probe_astc();
    if (!strcmp(mode, "--probe-vkdevice")) return probe_vkdevice();
    if (!strcmp(mode, "--probe-movie")) return probe_movie();
    if (!strcmp(mode, "--prepare-shaders")) return vk_pipeline_prepare_shaders();
    if (!strcmp(mode, "--warm-shaders")) return vk_pipeline_warm_shaders();
    if (!strcmp(mode, "--boot")) { cfg.boot = 1; rt_boot(); return 0; }
    return usage(argv[0]);
}
