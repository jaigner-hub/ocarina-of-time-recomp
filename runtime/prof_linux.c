// Linux port of runtime/prof_win.c, with the same interface (host.h): frame timing (OOT_TRACE=perf), the render thread's
// waits, the GPU and memory line of the 60-frame window, the frame-time summary, the performance overlay's counts and
// the A/B switches rt_ab and rt_knob. prof_win.c is copied from metroid-prime-4-recomp @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0
// (MIT, Copyright (c) 2026 Jeff Aigner), fe3h-pc-recomp @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026
// Jeff Aigner) and totk-pc-recomp @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh).
// Linux: not implemented: the sampling profiler (OOT_PROFILE is logged and no profile is written: prof_win.c suspends
// threads with SuspendThread and reads their stacks through .pdata), OOT_ONE_PER_CORE (prof_win.c sets the process
// affinity with Win32 calls; logged once when set). Thread ids come from plat_thread_id, times from kern_now_ns (which
// is plat_now_ns, CLOCK_MONOTONIC), atomics are the GCC/Clang __atomic builtins in place of the Interlocked calls.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "flight.h"
#include "gfx.h"
#include "host.h"
#include "perfmon.h"
#include "platform/platform.h"

void (*perf_gpu_stats)(PerfGpu *out);

// ---- performance counters: enabled by OOT_TRACE=perf, OOT_FLIGHT=1 or the full performance overlay. Wait times are
// always taken (frame-time summary). perf_base: the part fixed at start.
static int perf_on, perf_base;
static int overlay;   // performance overlay mode (perf_set_overlay): 0 off, 1 fps, 2 full
static volatile uint32_t render_tid;   // the thread that presents (perf_present)
static volatile uint32_t replay_tid;   // asynchronous replay (vk_backend.c): its waits hold up the frame too
static volatile uint32_t main_tid;     // the guest main thread (perf_main_thread)

void perf_set_overlay(int mode) {
    __atomic_store_n(&overlay, mode, __ATOMIC_RELAXED);
    __atomic_store_n(&perf_on, perf_base || mode == 2, __ATOMIC_RELAXED);
}

int perf_gpu_timing(void) { return (rt_trace & RT_TRACE_PERF) || __atomic_load_n(&overlay, __ATOMIC_RELAXED) == 2; }

static uint64_t gpu_busy_total;   // summed over every thread's timed submission parts

void perf_gpu_busy(uint64_t ns) { __atomic_fetch_add(&gpu_busy_total, ns, __ATOMIC_RELAXED); }

uint64_t perf_gpu_busy_total(void) { return __atomic_load_n(&gpu_busy_total, __ATOMIC_RELAXED); }

void prof_start(void) {
    flight_init();
    atexit(perf_summary_flush);   // the exit summary; prof_write does the same first when boot.c or the window closes
    perf_base = (rt_trace & RT_TRACE_PERF) || flight_on;
    perf_on = perf_base || overlay == 2;
    const char *p = cfg_get("OOT_PROFILE");
    if (p && *p) rt_log("prof: OOT_PROFILE: no sampling profiler on Linux yet, no profile is written");
}

void prof_write(void) { perf_summary_flush(); }

// ---- OOT_TRACE=perf: frame interval and the time the render thread spends waiting, logged every 60 presents. The
// flight recorder (OOT_FLIGHT=1, flight.c) takes the same numbers per frame.
static uint64_t wait_ns[PERF_N], frame_ns[PERF_N], worst_ns[PERF_N];   // window totals, this frame, the slowest frame

uint64_t perf_begin(void) { return kern_now_ns(); }

void perf_replay_thread(void) { __atomic_store_n(&replay_tid, plat_thread_id(), __ATOMIC_RELAXED); }
void perf_main_thread(void) { __atomic_store_n(&main_tid, plat_thread_id(), __ATOMIC_RELAXED); }

void perf_end(int kind, uint64_t t0) {
    if (!t0) return;
    uint32_t tid = plat_thread_id();
    if (tid != render_tid && tid != replay_tid) return;
    uint64_t d = kern_now_ns() - t0;
    if (tid == render_tid && !replay_tid) {   // one thread adds: no atomics on the synchronous path
        wait_ns[kind] += d;
        frame_ns[kind] += d;
        return;
    }
    __atomic_fetch_add(&wait_ns[kind], d, __ATOMIC_RELAXED);
    __atomic_fetch_add(&frame_ns[kind], d, __ATOMIC_RELAXED);
}

static uint64_t other_wait_ns;   // perf_gpu_wait on threads other than the render thread, over the window
static uint64_t render_wait_ns;  // the same on the render thread

void perf_gpu_wait(uint64_t t0) {
    if (!t0 || !__atomic_load_n(&perf_on, __ATOMIC_RELAXED)) return;
    uint64_t d = kern_now_ns() - t0;
    __atomic_fetch_add(plat_thread_id() == __atomic_load_n(&render_tid, __ATOMIC_RELAXED) ? &render_wait_ns : &other_wait_ns,
                       d, __ATOMIC_RELAXED);
}

static uint64_t replay_ns;         // perf_replay: summed over the window, any thread
static uint64_t frame_replay_ns;   // the same since the previous present (frame-time summary)

void perf_replay(uint64_t t0) {
    if (!t0) return;
    uint64_t d = kern_now_ns() - t0;
    __atomic_fetch_add(&frame_replay_ns, d, __ATOMIC_RELAXED);
    if (__atomic_load_n(&perf_on, __ATOMIC_RELAXED)) __atomic_fetch_add(&replay_ns, d, __ATOMIC_RELAXED);
}

static int64_t counts[PERF_CNT_N];   // running totals: the perf log and the flight recorder take differences

void perf_count(int kind) {
    if (__atomic_load_n(&perf_on, __ATOMIC_RELAXED)) __atomic_fetch_add(&counts[kind], 1, __ATOMIC_RELAXED);
}

// ---- Frame-time summary, always on: every 60 s of play and at exit one line of frame count, mean fps, 1% low and the
// frames over 25/50/100 ms, then the worst frames of the period with the render thread's waits in them. Per frame: a
// histogram increment, the compare against the worst list, a memset of the wait array. No allocation.
#define FT_BINS 1024              // 0.25 ms bins up to 256 ms; slower frames land in the last bin
#define FT_WORST 5
#define FT_PERIOD_NS 60000000000ull
typedef struct { uint64_t frame, ns, wait[PERF_N], replay; } FtWorst;
static struct {
    uint64_t t0, frames, sum, over25, over50, over100;
    uint32_t hist[FT_BINS];
    FtWorst worst[FT_WORST];   // slowest first
    int nworst;
} ft;
static uint64_t summary_t0;   // start of the current period (0: not started)

static void ft_report(uint64_t now, const char *tag) {
    if (ft.frames && ft.sum) {
        uint64_t cut = (ft.frames + 99) / 100, acc = 0, low = 0;   // the slowest 1% of frames
        for (int b = FT_BINS - 1; b >= 0; b--) {
            acc += ft.hist[b];
            if (acc >= cut) { low = (uint64_t)(b + 1) * 250000ull; break; }   // upper edge of the bin
        }
        rt_log("frametime: %s: %llu frames in %.1f s, mean %.1f fps, 1%% low %.1f fps, over 25 ms %llu, 50 ms %llu, 100 ms %llu",
               tag, (unsigned long long)ft.frames, (now - ft.t0) / 1e9, ft.frames * 1e9 / ft.sum, low ? 1e9 / low : 0.0,
               (unsigned long long)ft.over25, (unsigned long long)ft.over50, (unsigned long long)ft.over100);
        for (int i = 0; i < ft.nworst; i++) {
            const FtWorst *w = &ft.worst[i];
            rt_log("frametime:   worst frame %llu: %.1f ms; shader translation %.1f, pipeline build %.1f, texture upload %.1f (astc %.1f), replay %.1f, fence/present wait %.1f, pacing %.1f",
                   (unsigned long long)w->frame, w->ns / 1e6, w->wait[PERF_XLATE] / 1e6, w->wait[PERF_PIPE] / 1e6,
                   w->wait[PERF_TEX] / 1e6, w->wait[PERF_ASTC] / 1e6, w->replay / 1e6,
                   (w->wait[PERF_FENCE] + w->wait[PERF_DRAIN]) / 1e6, w->wait[PERF_PACE] / 1e6);
        }
    }
    memset(&ft, 0, sizeof ft);
    ft.t0 = now;
}

void perf_summary_flush(void) {   // at exit (prof_write)
    if (summary_t0) {
        ft_report(kern_now_ns(), "exit");
        gfx_exit_log();
    }
    summary_t0 = 0;
}

// Returns the replay time since the previous present.
static uint64_t ft_frame(uint64_t frame, uint64_t now, uint64_t d) {
    uint64_t rep = __atomic_exchange_n(&frame_replay_ns, 0, __ATOMIC_RELAXED);
    if (!summary_t0) {   // the first present starts the period; there is no interval yet
        summary_t0 = now;
        ft.t0 = now;
        return rep;
    }
    ft.frames++;
    ft.sum += d;
    uint64_t b = d / 250000ull;
    ft.hist[b < FT_BINS ? b : FT_BINS - 1]++;
    if (d > 25000000ull) ft.over25++;
    if (d > 50000000ull) ft.over50++;
    if (d > 100000000ull) ft.over100++;
    if (d > 25000000ull && (ft.nworst < FT_WORST || d > ft.worst[FT_WORST - 1].ns)) {
        int i = ft.nworst < FT_WORST ? ft.nworst++ : FT_WORST - 1;
        while (i > 0 && ft.worst[i - 1].ns < d) ft.worst[i] = ft.worst[i - 1], i--;
        ft.worst[i].frame = frame;
        ft.worst[i].ns = d;
        memcpy(ft.worst[i].wait, frame_ns, sizeof frame_ns);
        ft.worst[i].replay = rep;
    }
    if (now - summary_t0 >= FT_PERIOD_NS) {
        ft_report(now, "60 s");
        summary_t0 = now;
    }
    return rep;
}

// "n/a" for a number the system does not give (a negative value), else the value with the format.
static void mb_text(char *out, size_t cap, const char *fmt, double v) {
    if (v < 0) snprintf(out, cap, "n/a");
    else snprintf(out, cap, fmt, v);
}

// The second line of the 60-frame window: GPU busy time (timestamps), the time threads waited for GPU fences, VRAM and
// process memory, and which side holds the frame back. pace: the frame waited for pacing (vsync, the frame limiter) at
// least 15% of the window; gpu: the GPU was busy for 80% of the frame interval or more (without timestamps: threads
// waited for fences for half the interval or more); cpu otherwise.
static void perf_gpu_line(uint64_t frame, double interval_ms, double pace_share, uint64_t n) {
    PerfGpu g;
    memset(&g, 0, sizeof g);
    if (perf_gpu_stats) perf_gpu_stats(&g);
    double rw = __atomic_exchange_n(&render_wait_ns, 0, __ATOMIC_RELAXED) / 1e6 / n;
    double ow = __atomic_exchange_n(&other_wait_ns, 0, __ATOMIC_RELAXED) / 1e6 / n;
    double gpu_ms = g.have_gpu ? g.gpu_ns / 1e6 / n : 0;
    char gs[96], vs[96];
    if (g.have_gpu) snprintf(gs, sizeof gs, "gpu avg %.2f ms (%llu parts timed, %llu not)", gpu_ms,
                             (unsigned long long)g.gpu_parts, (unsigned long long)g.gpu_missed);
    else snprintf(gs, sizeof gs, "gpu avg n/a");
    if (g.have_vram) snprintf(vs, sizeof vs, "vram %.0f of %.0f MiB%s", g.vram_used / 1048576.0, g.vram_budget / 1048576.0,
                              g.vram_exact ? "" : " (own allocations of the heap size)");
    else snprintf(vs, sizeof vs, "vram n/a");
    // plat_mem_info: the process working set; plat_commit_info: its private bytes (both -1 when unknown).
    float used_mb, total_mb, ws_mb;
    double priv_mb, commit_used_mb, commit_limit_mb;
    plat_mem_info(&used_mb, &total_mb, &ws_mb);
    plat_commit_info(&priv_mb, &commit_used_mb, &commit_limit_mb);
    char wss[32], privs[32];
    mb_text(wss, sizeof wss, "%.0f MiB", ws_mb);
    mb_text(privs, sizeof privs, "%.0f MiB", priv_mb);
    const char *bound = pace_share >= 0.15 ? "pace"
                      : g.have_gpu ? (gpu_ms >= 0.8 * interval_ms ? "gpu" : "cpu")
                      : (rw + ow >= 0.5 * interval_ms ? "gpu" : "cpu");
    rt_log("perf: frame %llu: %s; fence wait per frame: render thread %.2f ms, other threads %.2f ms; %s; ram working set "
           "%s, private %s; bound: %s",
           (unsigned long long)frame, gs, rw, ow, vs, wss, privs, bound);
}

// ---- A/B switches (host.h): OOT_AB=<seconds> splits the world clock into blocks; rt_ab is 1 in the B blocks.
int rt_ab(void) {
    static int64_t period = -1;   // us of world clock; 0: off
    if (period < 0) {
        const char *e = cfg_get("OOT_AB");
        period = e && *e ? (int64_t)(atof(e) * 1e6) : 0;
    }
    int64_t w = period ? frame_step_world_us() : -1;
    if (w < 0) return 0;   // before the world clock: A
    int64_t k = w / period % 4;
    return k == 1 || k == 2;
}

// OOT_<name>: 0 off, "ab" 2 (on in the B blocks only, see RT_KNOB_ON), anything else 1; def when unset or empty.
int rt_knob(const char *name, const char *def) {
    char key[64];
    snprintf(key, sizeof key, "OOT_%s", name);
    const char *e = cfg_get(key);
    if (!e || !*e) e = def;
    return !strcmp(e, "ab") ? 2 : strcmp(e, "0") != 0;
}

// OOT_ONE_PER_CORE is read once; prof_win.c then keeps the first logical processor of each core. Linux: not implemented.
static void cpu_placement(void) {
    static int checked;
    if (checked) return;
    checked = 1;
    if (rt_knob("ONE_PER_CORE", "0")) rt_log("cpu: OOT_ONE_PER_CORE is not implemented on Linux; the threads keep the default placement");
}

void perf_present(uint64_t frame) {
    static uint64_t last, sum, max, n, win_start, ab_b;
    static int64_t win_prev[PERF_CNT_N];
    __atomic_store_n(&render_tid, plat_thread_id(), __ATOMIC_RELAXED);
    if (rt_trace & RT_TRACE_SCHED) kern_sched_report(frame);
    uint64_t now = kern_now_ns(), d = last ? now - last : 0;
    uint64_t rep = ft_frame(frame, now, d);   // d is 0 on the first present: it only starts the period
    ab_b += rt_ab();
    cpu_placement();
    if (__atomic_load_n(&overlay, __ATOMIC_RELAXED))   // the overlay's only cost while it is off: this branch
        perfmon_frame(now, d, frame_ns, rep, __atomic_load_n(&counts[PERF_CNT_TEX_UP], __ATOMIC_RELAXED));
    kern_core_diag_frame(frame, d);
    if (flight_on) flight_frame(frame, now, d, frame_ns, counts);
    if (!(rt_trace & RT_TRACE_PERF)) {
        memset(frame_ns, 0, sizeof frame_ns);
        last = now;
        return;
    }
    static uint64_t slow_ns;   // OOT_PERF_SLOW_MS (default 50): frames at least this long get a line of their own
    if (!slow_ns) {
        const char *e = cfg_get("OOT_PERF_SLOW_MS");
        slow_ns = (uint64_t)(e && atoi(e) > 0 ? atoi(e) : 50) * 1000000ull;
    }
    if (last) {
        sum += d;
        n++;
        if (d > max) max = d, memcpy(worst_ns, frame_ns, sizeof worst_ns);
        if (d >= slow_ns)
            rt_log("perf: slow frame %llu: %.1f ms; blocked pipeline %.1f xlate %.1f texture %.1f astc %.1f fence %.1f pacing %.1f",
                   (unsigned long long)frame, d / 1e6, frame_ns[PERF_PIPE] / 1e6, frame_ns[PERF_XLATE] / 1e6,
                   frame_ns[PERF_TEX] / 1e6, frame_ns[PERF_ASTC] / 1e6, frame_ns[PERF_FENCE] / 1e6,
                   frame_ns[PERF_PACE] / 1e6);
    }
    memset(frame_ns, 0, sizeof frame_ns);
    last = now;
    if (!win_start) win_start = now;
    if (frame % 60) return;
    uint64_t span = now - win_start;
    int64_t win[PERF_CNT_N];
    for (int i = 0; i < PERF_CNT_N; i++) {
        int64_t v = __atomic_load_n(&counts[i], __ATOMIC_RELAXED);
        win[i] = v - win_prev[i];
        win_prev[i] = v;
    }
    int64_t async = win[PERF_CNT_PIPE_ASYNC], skips = win[PERF_CNT_PIPE_SKIP], up = win[PERF_CNT_TEX_UP];
    int64_t reup = win[PERF_CNT_TEX_REUP], sskip = win[PERF_CNT_SHADER_SKIP], waited = win[PERF_CNT_PIPE_WAIT];
    int64_t oneshot = win[PERF_CNT_ONESHOT_WAIT];
    uint64_t shader_keys, shader_xlated;
    gfx_shader_counts(&shader_keys, &shader_xlated);
    if (n && span)
        rt_log("perf: frame %llu: interval avg %.2f ms max %.2f ms (%.1f fps); render thread blocked: pacing %.0f%% fence %.0f%% drain %.0f%% pipeline build %.0f%% shader xlate %.0f%% texture %.0f%% (astc %.0f%%); pipelines built in background %lld, draws skipped pending %lld, waited %lld, waiting for a shader %lld, one-shot waited %lld; textures uploaded %lld re-uploaded %lld; slowest frame: xlate %.1f ms texture %.1f ms (astc %.1f) pipeline %.1f ms fence %.1f ms; shaders: %llu keys, %llu translated; render busy avg %.2f ms; replay avg %.2f ms; ab %llu/%llu; world %.1f s",
               (unsigned long long)frame, sum / 1e6 / n, max / 1e6, n * 1e9 / sum, 100.0 * wait_ns[PERF_PACE] / span,
               100.0 * wait_ns[PERF_FENCE] / span, 100.0 * wait_ns[PERF_DRAIN] / span,
               100.0 * wait_ns[PERF_PIPE] / span, 100.0 * wait_ns[PERF_XLATE] / span,
               100.0 * wait_ns[PERF_TEX] / span, 100.0 * wait_ns[PERF_ASTC] / span, (long long)async, (long long)skips, (long long)waited, (long long)sskip, (long long)oneshot,
               (long long)up, (long long)reup, worst_ns[PERF_XLATE] / 1e6, worst_ns[PERF_TEX] / 1e6,
               worst_ns[PERF_ASTC] / 1e6, worst_ns[PERF_PIPE] / 1e6, worst_ns[PERF_FENCE] / 1e6,
               (unsigned long long)shader_keys, (unsigned long long)shader_xlated,
               (span > wait_ns[PERF_PACE] ? span - wait_ns[PERF_PACE] : 0) / 1e6 / n,
               __atomic_exchange_n(&replay_ns, 0, __ATOMIC_RELAXED) / 1e6 / n, (unsigned long long)ab_b, (unsigned long long)n, frame_step_world_us() / 1e6);
    if (n && span) perf_gpu_line(frame, sum / 1e6 / n, (double)wait_ns[PERF_PACE] / span, n);
    sum = max = n = ab_b = 0;
    memset(wait_ns, 0, sizeof wait_ns);
    win_start = now;
}
