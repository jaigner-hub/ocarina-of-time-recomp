// From totk-pc-recomp runtime/platform/linux.c @ 0ae47a578285cde080642c9e6c40316c99171ef1 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), adapted to this project's platform.h, which is smaller than TotK's: the functions it declares only. Copied from TotK, with the reader-writer lock (plat_mutex_lock_shared, and the exclusive lock now shares its word with it), the virtual memory functions with the write watch (userfaultfd write protection and PAGEMAP_SCAN, and the page-state table) and the write-fault hook (here a SIGSEGV handler that only runs the hook; TotK's crash hooks are left out), and the write-watch ABI block. Added later for runtime/kernel.c, from the same commit: thread stacks of a given size, thread interruption, fibers, plat_wait_ns, aligned memory and module lookup. Left out: the crash hook, IPC, restart and the TotK-only helpers. plat_thread_detach frees its PlatThread (TotK keeps it), and plat_lib_open leaves the name's search to dlopen.
// Linux implementation of platform.h: pthreads, CLOCK_MONOTONIC, futex locks and condition variables, POSIX files,
// /proc and sysfs for the load counters, posix_spawn for child processes.
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/futex.h>
#include <linux/userfaultfd.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <spawn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/random.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

// The write-watch ABI of Linux 6.7 (uapi linux/fs.h and linux/userfaultfd.h), for headers older than it. The kernel refuses
// what it lacks at run time.
#ifndef PAGEMAP_SCAN
struct page_region {
    __u64 start;
    __u64 end;
    __u64 categories;
};
struct pm_scan_arg {
    __u64 size;
    __u64 flags;
    __u64 start;
    __u64 end;
    __u64 walk_end;
    __u64 vec;
    __u64 vec_len;
    __u64 max_pages;
    __u64 category_inverted;
    __u64 category_mask;
    __u64 category_anyof_mask;
    __u64 return_mask;
};
#define PAGEMAP_SCAN _IOWR('f', 16, struct pm_scan_arg)
#define PAGE_IS_WRITTEN (1 << 1)
#define PAGE_IS_PRESENT (1 << 3)
#define PAGE_IS_SWAPPED (1 << 4)
#define PAGE_IS_PFNZERO (1 << 5)
#define PM_SCAN_WP_MATCHING (1 << 0)
#define PM_SCAN_CHECK_WPASYNC (1 << 1)
#endif
#ifndef UFFD_FEATURE_WP_ASYNC
#define UFFD_FEATURE_WP_UNPOPULATED (1 << 13)
#define UFFD_FEATURE_WP_ASYNC (1 << 15)
#endif
#ifndef UFFD_USER_MODE_ONLY
#define UFFD_USER_MODE_ONLY 1
#endif

#include "platform.h"

// SDL3's clipboard call (SDL3/SDL_clipboard.h, the same declaration). Declared here so this file builds before the
// SDL layer's headers are on the include path; the link resolves it once SDL is in the build.
bool SDL_SetClipboardText(const char *text);

// Spin-wait hint for the lock loops. x86-64 only, like fp_x86.h.
static inline void spin_pause(void) { __builtin_ia32_pause(); }

// ---- threads

// A PlatThread holds only the pthread handle, so detaching can free it while the thread runs. The thread's own
// start block (fn, arg, name) is allocated separately and freed by the thread once it has copied them.
struct PlatThread {
    pthread_t t;
};

struct ThreadStart {
    PlatThreadFn fn;
    void *arg;
    char name[16];   // the kernel keeps 15 characters
};

static void *thread_main(void *p) {
    struct ThreadStart *s = p;
    PlatThreadFn fn = s->fn;
    void *arg = s->arg;
    if (s->name[0]) pthread_setname_np(pthread_self(), s->name);
    free(s);
    fn(arg);
    return NULL;
}

static PlatThread *thread_create(PlatThreadFn fn, void *arg, const char *name, size_t stack) {
    struct ThreadStart *s = calloc(1, sizeof *s);
    PlatThread *t = calloc(1, sizeof *t);
    if (!s || !t) {
        free(s);
        free(t);
        return NULL;
    }
    s->fn = fn;
    s->arg = arg;
    if (name) snprintf(s->name, sizeof s->name, "%s", name);
    pthread_attr_t a;
    pthread_attr_init(&a);
    // The stack is mapped lazily, as Windows reserves it: only the pages a thread touches cost memory.
    if (stack) pthread_attr_setstacksize(&a, stack);
    int e = pthread_create(&t->t, &a, thread_main, s);
    pthread_attr_destroy(&a);
    if (e) {
        free(s);
        free(t);
        return NULL;
    }
    return t;
}

PlatThread *plat_thread_new(PlatThreadFn fn, void *arg, const char *name) { return thread_create(fn, arg, name, 0); }

// Adapted from totk-pc-recomp runtime/platform/linux.c @ 0ae47a578285cde080642c9e6c40316c99171ef1 (plat_thread_new_ex,
// plat_thread_spawn).
PlatThread *plat_thread_new_ex(PlatThreadFn fn, void *arg, const char *name, size_t stack) {
    return thread_create(fn, arg, name, stack ? stack : PLAT_STACK_DEFAULT);
}

int plat_thread_spawn(PlatThreadFn fn, void *arg, const char *name, size_t stack) {
    PlatThread *t = plat_thread_new_ex(fn, arg, name, stack);
    if (t) plat_thread_detach(t);
    return t != NULL;
}

void plat_thread_join(PlatThread *t) {
    pthread_join(t->t, NULL);
    free(t);
}

void plat_thread_detach(PlatThread *t) {
    pthread_detach(t->t);
    free(t);
}

// ---- auto-reset events, on a lock and a condition variable: set is sticky until a wait takes it.
struct PlatEvent {
    PlatMutex m;
    PlatCond c;
    int set;
};

PlatEvent *plat_event_new(void) {
    PlatEvent *e = calloc(1, sizeof *e);
    return e;
}

void plat_event_free(PlatEvent *e) { free(e); }

void plat_event_set(PlatEvent *e) {
    if (!e) return;   // SetEvent(NULL) did nothing before the events became PlatEvent; callers signal early (vk_headroom.c)
    plat_mutex_lock(&e->m);
    e->set = 1;
    plat_mutex_unlock(&e->m);
    plat_cond_signal(&e->c);
}

int plat_event_wait_ns(PlatEvent *e, uint64_t ns) {
    uint64_t end = ns == UINT64_MAX ? UINT64_MAX : plat_now_ns() + ns;
    plat_mutex_lock(&e->m);
    while (!e->set) {
        if (end == UINT64_MAX) {
            plat_cond_wait(&e->c, &e->m);
            continue;
        }
        uint64_t now = plat_now_ns();
        if (now >= end) break;
        plat_cond_wait_ns(&e->c, &e->m, end - now);   // a spurious wake-up just loops
    }
    int got = e->set;
    e->set = 0;
    plat_mutex_unlock(&e->m);
    return got;
}

void plat_yield(void) { sched_yield(); }

// Thread priority is not changed on Linux (see platform.h): an unprivileged thread cannot raise its priority back.
void plat_thread_priority(PlatThread *t, int level) {
    (void)t;
    (void)level;
}

// Linux has no process-wide affinity: the mask goes to the thread (or the caller when t is NULL).
uint64_t plat_thread_affinity(PlatThread *t, uint64_t mask) {
    pthread_t th = t ? t->t : pthread_self();
    cpu_set_t old, set;
    if (pthread_getaffinity_np(th, sizeof old, &old)) return 0;
    CPU_ZERO(&set);
    for (int i = 0; i < 64; i++)
        if (mask >> i & 1) CPU_SET(i, &set);
    if (pthread_setaffinity_np(th, sizeof set, &set)) return 0;
    uint64_t prev = 0;
    for (int i = 0; i < 64; i++)
        if (CPU_ISSET(i, &old)) prev |= 1ull << i;
    return prev;
}

uint32_t plat_cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (uint32_t)n : 1;
}

// Cached per thread: gettid is a system call, and callers ask for the id often.
uint32_t plat_thread_id(void) {
    static __thread uint32_t tid;
    if (!tid) tid = (uint32_t)syscall(SYS_gettid);
    return tid;
}

// The first field of /proc/self/task/<tid>/schedstat is the time the thread has run, in ns. From totk-pc-recomp
// runtime/platform/linux.c @ 0ae47a578285cde080642c9e6c40316c99171ef1 (plat_thread_cpu_ns).
uint64_t plat_thread_cpu_ns(uint32_t tid) {
    char path[64];
    snprintf(path, sizeof path, "/proc/self/task/%u/schedstat", tid);
    FILE *f = fopen(path, "r");
    unsigned long long ns = 0;
    if (f) {
        if (fscanf(f, "%llu", &ns) != 1) ns = 0;
        fclose(f);
    }
    return ns;
}

// No unwinder here: kernel.c's thread dump falls back to the instruction pointer plat_intr_stop reads.
int plat_stack_walk(uint32_t tid, uint64_t *frames, int max) {
    (void)tid, (void)frames, (void)max;
    return 0;
}

// ---- thread interruption: a signal handshake in place of Windows' SuspendThread / Get/SetThreadContext. From
// totk-pc-recomp runtime/platform/linux.c @ 0ae47a578285cde080642c9e6c40316c99171ef1 (PlatIntr and plat_intr_*),
// adapted: the handler stores the interrupted rsp too, and the signal is real-time signal SIGRTMIN + 4.
// The caller moves the handle from IDLE to REQ and signals the thread. The thread's handler stores where it was and
// moves REQ to STOPPED, then spins until the caller moves it on to GO (with the redirect in to), changes its own
// return context, and goes back to IDLE. A caller that sees no STOPPED within INTR_WAIT_NS (the thread is blocked in
// the kernel and runs no handler) takes REQ back to IDLE; a handler that then finds IDLE returns at once.
enum { INTR_IDLE, INTR_REQ, INTR_STOPPED, INTR_GO };
#define INTR_WAIT_NS 200000u

struct PlatIntr {
    pthread_t pt;
    volatile int32_t state;
    uint64_t rip, rsp, to;
};

static __thread PlatIntr *intr_me;
static int intr_sig;

static void intr_handler(int sig, siginfo_t *si, void *uc) {
    (void)sig, (void)si;
    int saved_errno = errno;
    PlatIntr *t = intr_me;
    ucontext_t *u = uc;
    if (t) {
        t->rip = (uint64_t)u->uc_mcontext.gregs[REG_RIP];
        t->rsp = (uint64_t)u->uc_mcontext.gregs[REG_RSP];
        int32_t req = INTR_REQ;
        if (__atomic_compare_exchange_n(&t->state, &req, INTR_STOPPED, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            while (__atomic_load_n(&t->state, __ATOMIC_ACQUIRE) != INTR_GO) spin_pause();
            if (t->to) u->uc_mcontext.gregs[REG_RIP] = (greg_t)t->to;
            __atomic_store_n(&t->state, INTR_IDLE, __ATOMIC_RELEASE);
        }
    }
    errno = saved_errno;
}

PlatIntr *plat_intr_self(void) {
    static PlatMutex install = PLAT_MUTEX_INIT;
    if (!__atomic_load_n(&intr_sig, __ATOMIC_ACQUIRE)) {
        plat_mutex_lock(&install);
        if (!intr_sig) {
            int sig = SIGRTMIN + 4;   // glibc keeps the first real-time signals for itself; SIGRTMIN skips them
            struct sigaction sa = {.sa_sigaction = intr_handler, .sa_flags = SA_SIGINFO | SA_RESTART};
            sigemptyset(&sa.sa_mask);
            if (!sigaction(sig, &sa, NULL)) __atomic_store_n(&intr_sig, sig, __ATOMIC_RELEASE);
        }
        plat_mutex_unlock(&install);
    }
    if (!intr_sig) return NULL;
    PlatIntr *t = calloc(1, sizeof *t);
    if (!t) return NULL;
    t->pt = pthread_self();
    intr_me = t;
    return t;
}

// Called by the thread itself as it ends, or by another once the thread no longer runs.
void plat_intr_free(PlatIntr *t) {
    if (!t) return;
    if (intr_me == t) intr_me = NULL;
    free(t);
}

int plat_intr_stop(PlatIntr *t, uint64_t *rip, uint64_t *rsp) {
    int32_t idle = INTR_IDLE;
    if (!__atomic_compare_exchange_n(&t->state, &idle, INTR_REQ, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return 0;
    if (pthread_kill(t->pt, intr_sig)) {
        __atomic_store_n(&t->state, INTR_IDLE, __ATOMIC_RELEASE);
        return 0;
    }
    uint64_t end = plat_now_ns() + INTR_WAIT_NS;
    while (__atomic_load_n(&t->state, __ATOMIC_ACQUIRE) != INTR_STOPPED) {
        if (plat_now_ns() < end) {
            spin_pause();
            continue;
        }
        int32_t req = INTR_REQ;
        if (__atomic_compare_exchange_n(&t->state, &req, INTR_IDLE, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return 0;
    }
    *rip = t->rip;
    *rsp = t->rsp;
    return 1;
}

int plat_intr_resume(PlatIntr *t, uint64_t to) {
    t->to = to;
    __atomic_store_n(&t->state, INTR_GO, __ATOMIC_RELEASE);
    return 1;
}

// ---- fibers: a stack switch of our own. From totk-pc-recomp runtime/platform/linux.c @
// 0ae47a578285cde080642c9e6c40316c99171ef1 (plat_fiber_*), copied. glibc's swapcontext saves and restores the signal
// mask with a system call on every switch, and the game switches fibers often. fiber_swap saves the callee-saved
// registers, the MXCSR and the x87 control word on the current stack, stores the stack pointer in *save and loads the
// other one. A new fiber's stack is laid out as fiber_swap leaves one, returning into fiber_entry, which realigns the
// stack and calls fn(arg) (r12 and r13).
struct PlatFiber {
    void *sp;
    void *stack;   // mapped stack (NULL for a thread's own context)
    size_t size;
};

void plat_fiber_swap(void **save, void *load);
void plat_fiber_entry(void);
__asm__(".text\n"
        ".globl plat_fiber_swap\n"
        ".type plat_fiber_swap, @function\n"
        ".p2align 4\n"
        "plat_fiber_swap:\n"
        "    .cfi_startproc\n"
        "    pushq %rbp\n"
        "    pushq %rbx\n"
        "    pushq %r12\n"
        "    pushq %r13\n"
        "    pushq %r14\n"
        "    pushq %r15\n"
        "    subq $16, %rsp\n"
        "    stmxcsr 8(%rsp)\n"
        "    fnstcw (%rsp)\n"
        "    movq %rsp, (%rdi)\n"
        "    movq %rsi, %rsp\n"
        "    fldcw (%rsp)\n"
        "    ldmxcsr 8(%rsp)\n"
        "    addq $16, %rsp\n"
        "    popq %r15\n"
        "    popq %r14\n"
        "    popq %r13\n"
        "    popq %r12\n"
        "    popq %rbx\n"
        "    popq %rbp\n"
        "    ret\n"
        "    .cfi_endproc\n"
        ".size plat_fiber_swap, .-plat_fiber_swap\n"
        ".globl plat_fiber_entry\n"
        ".type plat_fiber_entry, @function\n"
        ".p2align 4\n"
        "plat_fiber_entry:\n"
        // fiber_swap's ret lands here with rsp = top - 8, as at the entry of a called function. The call below must
        // be made with rsp 16-byte aligned so that fn starts with rsp = 8 mod 16, as the System V ABI demands.
        "    andq $-16, %rsp\n"
        "    movq %r13, %rdi\n"
        "    callq *%r12\n"
        "    ud2\n"
        ".size plat_fiber_entry, .-plat_fiber_entry\n");

static __thread PlatFiber *fiber_cur;
static __thread PlatFiber fiber_own;

PlatFiber *plat_fiber_self(void) {
    fiber_own = (PlatFiber){0};
    fiber_cur = &fiber_own;
    return &fiber_own;
}

void plat_fiber_unself(void) { fiber_cur = NULL; }

PlatFiber *plat_fiber_new(size_t stack, void (*fn)(void *arg), void *arg) {
    PlatFiber *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    // Reserved, not committed; the lowest page stays inaccessible, so an overflow faults instead of corrupting.
    void *m = mmap(NULL, stack, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_STACK, -1, 0);
    if (m == MAP_FAILED) {
        free(f);
        return NULL;
    }
    mprotect(m, 4096, PROT_NONE);
    f->stack = m, f->size = stack;
    uint64_t *top = (uint64_t *)(((uintptr_t)m + stack) & ~(uintptr_t)15);
    uint64_t *sp = top - 10;   // fpu state (2 words), r15, r14, r13, r12, rbx, rbp, return address, padding
    sp[0] = 0x037f;            // x87 control word: the default
    sp[1] = 0x1f80;            // MXCSR: the default
    sp[2] = sp[3] = 0;         // r15, r14
    sp[4] = (uint64_t)(uintptr_t)arg;   // r13
    sp[5] = (uint64_t)(uintptr_t)fn;    // r12
    sp[6] = sp[7] = 0;         // rbx, rbp
    sp[8] = (uint64_t)(uintptr_t)plat_fiber_entry;
    sp[9] = 0;
    f->sp = sp;
    return f;
}

void plat_fiber_switch(PlatFiber *to) {
    PlatFiber *from = fiber_cur;
    fiber_cur = to;   // on this thread; the switch back into from sets it on whichever thread that is
    plat_fiber_swap(&from->sp, to->sp);
}

void plat_fiber_free(PlatFiber *f) {
    if (!f) return;
    if (f->stack) munmap(f->stack, f->size);
    free(f);
}

// ---- time

uint64_t plat_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

// clock_nanosleep wakes within the timer slack (50 us by default) of its deadline; the last PLAT_SPIN_NS are spent
// yielding so the deadline is met closely, as on Windows.
#define PLAT_SPIN_NS 100000u

void plat_sleep_ns(uint64_t ns) {
    uint64_t end = plat_now_ns() + ns;
    if (ns > PLAT_SPIN_NS) {
        uint64_t at = end - PLAT_SPIN_NS;
        struct timespec ts = {(time_t)(at / 1000000000u), (long)(at % 1000000000u)};
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
    }
    while (plat_now_ns() < end) sched_yield();
}

// From totk-pc-recomp runtime/platform/linux.c @ 0ae47a578285cde080642c9e6c40316c99171ef1 (plat_wait_ns).
void plat_wait_ns(uint64_t ns) {
    struct timespec ts = {(time_t)(ns / 1000000000u), (long)(ns % 1000000000u)};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}

// ---- locks: a reader-writer lock in one 32-bit word, zero when free. Adapted from totk-pc-recomp runtime/platform/linux.c
// (see the header).
// bit 31: a writer holds it; bit 30: some thread sleeps on the word; bits 0-29: readers holding it.
// A thread that cannot take the lock sets the sleeper bit and waits on the futex; every release that finds the bit
// clears it and wakes all sleepers, which retry (and set it again if they still cannot get in). Readers do not enter
// while the sleeper bit is set, so a waiting writer is not starved by a stream of readers.

#define LK_W 0x80000000u
#define LK_S 0x40000000u
#define LK_R 0x3fffffffu
#define LK_SPINS 64

static long futex(volatile uint32_t *w, int op, uint32_t val, const struct timespec *ts) {
    return syscall(SYS_futex, w, op | FUTEX_PRIVATE_FLAG, val, ts, NULL, 0);
}

static inline volatile uint32_t *lk(PlatMutex *m) { return (volatile uint32_t *)&m->p; }

// Sets the sleeper bit on s and sleeps while the word still equals the result.
static void lk_sleep(volatile uint32_t *w, uint32_t s) {
    if (!(s & LK_S) && !__atomic_compare_exchange_n(w, &s, s | LK_S, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return;
    futex(w, FUTEX_WAIT, s | LK_S, NULL);
}

static void lk_wake(volatile uint32_t *w) {
    __atomic_fetch_and(w, ~LK_S, __ATOMIC_RELAXED);
    futex(w, FUTEX_WAKE, INT_MAX, NULL);
}

void plat_mutex_init(PlatMutex *m) { m->p = 0; }

int plat_mutex_trylock(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
    while (!(s & (LK_W | LK_R)))
        if (__atomic_compare_exchange_n(w, &s, s | LK_W, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 1;
    return 0;
}

void plat_mutex_lock(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    for (int spin = 0;; spin++) {
        uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
        if (!(s & (LK_W | LK_R))) {
            if (__atomic_compare_exchange_n(w, &s, s | LK_W, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return;
            continue;
        }
        if (spin < LK_SPINS) spin_pause();
        else lk_sleep(w, s);
    }
}

void plat_mutex_unlock(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_fetch_and(w, ~LK_W, __ATOMIC_RELEASE);
    if (s & LK_S) lk_wake(w);
}

void plat_mutex_lock_shared(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    for (int spin = 0;; spin++) {
        uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
        if (!(s & (LK_W | LK_S))) {
            if (__atomic_compare_exchange_n(w, &s, s + 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return;
            continue;
        }
        if (spin < LK_SPINS) spin_pause();
        else lk_sleep(w, s);
    }
}

void plat_mutex_unlock_shared(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_sub_fetch(w, 1, __ATOMIC_RELEASE);
    if (!(s & LK_R) && (s & LK_S)) lk_wake(w);
}

// ---- condition variables: a sequence number in the first 32 bits. A waiter reads it before releasing the lock and
// sleeps only while it is unchanged, so a signal between the release and the sleep is not lost.

static inline volatile uint32_t *cv(PlatCond *c) { return (volatile uint32_t *)&c->p; }

void plat_cond_init(PlatCond *c) { c->p = 0; }

void plat_cond_wait(PlatCond *c, PlatMutex *m) {
    uint32_t seq = __atomic_load_n(cv(c), __ATOMIC_RELAXED);
    plat_mutex_unlock(m);
    futex(cv(c), FUTEX_WAIT, seq, NULL);
    plat_mutex_lock(m);
}

int plat_cond_wait_ns(PlatCond *c, PlatMutex *m, uint64_t ns) {
    uint32_t seq = __atomic_load_n(cv(c), __ATOMIC_RELAXED);
    struct timespec ts = {(time_t)(ns / 1000000000u), (long)(ns % 1000000000u)};
    plat_mutex_unlock(m);
    long r = futex(cv(c), FUTEX_WAIT, seq, &ts);
    int timed_out = r == -1 && errno == ETIMEDOUT;
    plat_mutex_lock(m);
    return !timed_out;
}

void plat_cond_signal(PlatCond *c) {
    __atomic_add_fetch(cv(c), 1, __ATOMIC_RELEASE);
    futex(cv(c), FUTEX_WAKE, 1, NULL);
}

void plat_cond_broadcast(PlatCond *c) {
    __atomic_add_fetch(cv(c), 1, __ATOMIC_RELEASE);
    futex(cv(c), FUTEX_WAKE, INT_MAX, NULL);
}

// ---- virtual memory. Reservations are PROT_NONE MAP_NORESERVE mappings; commit is mprotect read-write, decommit
// MADV_DONTNEED and PROT_NONE. Linux keeps no committed/reserved distinction, so a byte per guest page records the state
// (pstate, mapped lazily: only the pages touched cost memory). Adapted from totk-pc-recomp runtime/platform/linux.c.
//
// Write watch: a userfaultfd in asynchronous write-protect mode (Linux 6.7) over the watched reservations. The kernel
// clears a page's protection on the first write without a round trip to us, and PAGEMAP_SCAN lists the pages whose
// protection was cleared and protects them again. Without it (an older kernel), plat_mem_written reports the ranges as
// untracked and the back end uploads them in full.

#define PG 4096ull
#define PS_LIMIT (1ull << 39)
enum { PS_FREE, PS_RESERVED, PS_NONE, PS_READ, PS_RW };   // PS_NONE.. are committed: PLAT_PROT_* + PS_NONE
static uint8_t *pstate;
static int ww_fd = -1, pm_fd = -1;   // userfaultfd and /proc/self/pagemap; -1 when write watch is unavailable
static PlatMutex vm_lock = PLAT_MUTEX_INIT;

static void ps_set(uint64_t addr, uint64_t n, uint8_t v) {
    if (addr < PS_LIMIT) memset(pstate + addr / PG, v, (size_t)((n + PG - 1) / PG));
}

static void ww_open(void) {
    int fd = (int)syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
    struct uffdio_api api = {.api = UFFD_API,
                             .features = UFFD_FEATURE_WP_ASYNC | UFFD_FEATURE_WP_UNPOPULATED};
    if (fd < 0 || ioctl(fd, UFFDIO_API, &api)) {
        fprintf(stderr, "platform: no asynchronous userfaultfd write protection (%s): written memory is not tracked\n",
                strerror(errno));
        if (fd >= 0) close(fd);
        return;
    }
    pm_fd = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    if (pm_fd < 0) {
        close(fd);
        return;
    }
    ww_fd = fd;
}

int plat_mem_reserve(uint64_t addr, uint64_t n, int watch) {
    plat_mutex_lock(&vm_lock);
    if (!pstate) {
        void *p = mmap(NULL, PS_LIMIT / PG, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        pstate = p == MAP_FAILED ? NULL : p;
    }
    if (watch && ww_fd < 0 && pm_fd < 0) ww_open();
    void *p = pstate && addr + n <= PS_LIMIT
                  ? mmap((void *)(uintptr_t)addr, n, PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0)
                  : MAP_FAILED;
    int ok = p == (void *)(uintptr_t)addr;
    if (p != MAP_FAILED && !ok) munmap(p, n);   // a kernel without MAP_FIXED_NOREPLACE put it elsewhere
    if (ok) {
        ps_set(addr, n, PS_RESERVED);
        // Transparent huge pages would populate (and count as written) 2 MiB at the first write of one page; writes are
        // tracked per 4 KiB page, as on Windows.
        if (watch) madvise(p, n, MADV_NOHUGEPAGE);
        struct uffdio_register reg = {.range = {addr, n}, .mode = UFFDIO_REGISTER_MODE_WP};
        if (watch && ww_fd >= 0 && ioctl(ww_fd, UFFDIO_REGISTER, &reg))
            fprintf(stderr, "platform: write watch over %#llx+%#llx failed (%s)\n", (unsigned long long)addr,
                    (unsigned long long)n, strerror(errno));
    }
    plat_mutex_unlock(&vm_lock);
    return ok;
}

void *plat_mem_alloc(size_t n) {
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

int plat_mem_commit(uint64_t addr, size_t n) {
    if (mprotect((void *)(uintptr_t)addr, n, PROT_READ | PROT_WRITE)) return 0;
    ps_set(addr, n, PS_RW);
    return 1;
}

int plat_mem_decommit(uint64_t addr, uint64_t n) {
    if (madvise((void *)(uintptr_t)addr, n, MADV_DONTNEED) || mprotect((void *)(uintptr_t)addr, n, PROT_NONE)) return 0;
    ps_set(addr, n, PS_RESERVED);
    return 1;
}

int plat_mem_protect(uint64_t addr, uint64_t n, int prot) {
    int p = prot == PLAT_PROT_RW ? PROT_READ | PROT_WRITE : prot == PLAT_PROT_READ ? PROT_READ : PROT_NONE;
    if (mprotect((void *)(uintptr_t)addr, n, p)) return 0;
    ps_set(addr, n, (uint8_t)(PS_NONE + prot));
    return 1;
}

void plat_mem_release(uint64_t addr, uint64_t n) {
    munmap((void *)(uintptr_t)addr, n);
    ps_set(addr, n, PS_FREE);
}

int plat_mem_state(uint64_t addr, uint64_t limit, uint64_t *end) {
    if (!pstate || addr >= PS_LIMIT) {
        *end = limit;
        return PLAT_MEM_FREE;
    }
    uint64_t lim = limit < PS_LIMIT ? limit : PS_LIMIT, i = addr / PG;
    uint8_t v = pstate[i];
    while ((i + 1) * PG < lim && pstate[i + 1] == v) i++;
    *end = (i + 1) * PG < lim ? (i + 1) * PG : lim;
    return v == PS_FREE ? PLAT_MEM_FREE : v == PS_RESERVED ? PLAT_MEM_RESERVED : (int)v - PS_NONE;
}

uint64_t plat_mem_written(uint64_t addr, uint64_t n, uint64_t *pages, uint64_t max, int reset) {
    if (!max || !n) return 0;
    if (ww_fd < 0) return UINT64_MAX;
    struct page_region r[256];
    uint64_t got = 0;
    for (uint64_t at = addr, end = addr + n; at < end && got < max;) {
        // Written means: write-protection cleared, on a populated page (present or swapped out) that is not the shared
        // zero page. A page never touched since its commit has no protection marker either and counts as written to
        // the kernel; a read maps the zero page there. Neither is a write, as on Windows.
        struct pm_scan_arg arg = {.size = sizeof arg,
                                  .flags = reset ? PM_SCAN_WP_MATCHING | PM_SCAN_CHECK_WPASYNC : 0,
                                  .start = at, .end = end, .vec = (uint64_t)(uintptr_t)r, .vec_len = 256,
                                  .max_pages = max - got,
                                  .category_inverted = PAGE_IS_PFNZERO,
                                  .category_mask = PAGE_IS_WRITTEN | PAGE_IS_PFNZERO,
                                  .category_anyof_mask = PAGE_IS_PRESENT | PAGE_IS_SWAPPED,
                                  .return_mask = PAGE_IS_WRITTEN};
        long k = ioctl(pm_fd, PAGEMAP_SCAN, &arg);
        if (k < 0) return UINT64_MAX;
        for (long i = 0; i < k; i++)
            for (uint64_t p = r[i].start; p < r[i].end && got < max; p += PG) pages[got++] = p;
        if (arg.walk_end <= at) break;
        at = arg.walk_end;
    }
    return got;
}

// A write fault (SIGSEGV with the write bit of the page-fault error code) goes to the hook first: memory.c makes a frozen
// region writable there. Any other fault, and a write the hook refuses, gets the default action.
static int (*fault_hook)(uint64_t addr);

static void fault_handler(int sig, siginfo_t *si, void *uc) {
    ucontext_t *u = uc;
    long err = u->uc_mcontext.gregs[REG_ERR];
    if (sig == SIGSEGV && fault_hook && (err & 2) && fault_hook((uint64_t)(uintptr_t)si->si_addr)) return;
    signal(sig, SIG_DFL);   // returns into the faulting instruction, which faults again with the default action
}

void plat_mem_fault_hook(int (*fn)(uint64_t addr)) {
    fault_hook = fn;
    struct sigaction sa = {.sa_sigaction = fault_handler, .sa_flags = SA_SIGINFO | SA_NODEFER};
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, NULL);
}

int plat_mem_read(uint64_t addr, void *out, size_t n) {
    struct iovec local = {out, n}, remote = {(void *)(uintptr_t)addr, n};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == (ssize_t)n;
}

// ---- time, errors, environment, re-running

uint64_t plat_tick_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

uint32_t plat_last_error(void) { return (uint32_t)errno; }

int plat_setenv(const char *name, const char *value) {
    return value ? setenv(name, value, 1) == 0 : unsetenv(name) == 0;
}

// Replaces this process with the same program and arguments (the environment goes on as it is now).
int plat_reexec(uint32_t *err) {
    static char args[16384];
    char *argv[256];
    int argc = 0;
    FILE *f = fopen("/proc/self/cmdline", "rb");
    size_t n = f ? fread(args, 1, sizeof args - 1, f) : 0;
    if (f) fclose(f);
    args[n] = 0;
    for (size_t i = 0; i < n && argc < 255; i += strlen(args + i) + 1) argv[argc++] = args + i;
    argv[argc] = NULL;
    fflush(stdout);
    fflush(stderr);
    execv("/proc/self/exe", argv);
    if (err) *err = (uint32_t)errno;
    return 0;
}

// ---- files

int plat_file_exists(const char *path) {
    struct stat st;
    return !stat(path, &st) && !S_ISDIR(st.st_mode);
}

void plat_file_mkdir_p(const char *path) {
    char tmp[PATH_MAX];
    if (snprintf(tmp, sizeof tmp, "%s", path) >= (int)sizeof tmp) return;
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char c = *p;
            *p = 0;
            mkdir(tmp, 0755);
            *p = c;
        }
    }
    mkdir(tmp, 0755);
}

static int write_all(int fd, const void *data, size_t n) {
    const uint8_t *p = data;
    while (n) {
        ssize_t k = write(fd, p, n);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return 0;
        p += k, n -= (size_t)k;
    }
    return 1;
}

uint8_t *plat_file_read(const char *path, uint64_t *size) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return NULL;
    struct stat st;
    uint8_t *buf = NULL;
    if (!fstat(fd, &st) && S_ISREG(st.st_mode) && st.st_size > 0 && st.st_size < (1ll << 30)) {
        buf = malloc((size_t)st.st_size + 1);
        size_t got = 0;
        while (buf && got < (size_t)st.st_size) {
            ssize_t k = read(fd, buf + got, (size_t)st.st_size - got);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) break;
            got += (size_t)k;
        }
        if (buf && got != (size_t)st.st_size) {
            free(buf);
            buf = NULL;
        }
        if (buf) {
            buf[st.st_size] = 0;
            *size = (uint64_t)st.st_size;
        }
    }
    close(fd);
    return buf;
}

int plat_file_write(const char *path, const void *data, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return 0;
    int ok = write_all(fd, data, n);
    return close(fd) == 0 && ok;
}

// A PlatAppend is the descriptor + 1, so descriptor 0 is not NULL. O_APPEND makes each write land whole at the end of
// the file, from any thread.
struct PlatAppend;
PlatAppend *plat_append_open(const char *path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0644);
    return fd < 0 ? NULL : (PlatAppend *)(intptr_t)(fd + 1);
}

int plat_append_write(PlatAppend *f, const void *data, size_t n) {
    int fd = (int)(intptr_t)f - 1;
    ssize_t k;
    do k = write(fd, data, n);
    while (k < 0 && errno == EINTR);
    return k == (ssize_t)n;
}

// Flushes the directory holding path, so a rename into it survives a crash.
static void sync_dir_of(const char *path) {
    char dir[PATH_MAX];
    if (snprintf(dir, sizeof dir, "%s", path) >= (int)sizeof dir) return;
    char *slash = strrchr(dir, '/');
    if (slash == dir) slash[1] = 0;
    else if (slash) *slash = 0;
    else strcpy(dir, ".");
    int fd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
}

int plat_file_replace(const char *from, const char *to) {
    if (rename(from, to)) return 0;
    sync_dir_of(to);
    return 1;
}

int plat_file_write_atomic(const char *path, const void *data, size_t n) {
    // A rename replaces a read-only file here (only the folder's permission counts) but not on Windows: a file the
    // player made read-only stays as it is on both.
    if (access(path, W_OK) && errno != ENOENT) return 0;
    char tmp[PATH_MAX + 48];
    int len = snprintf(tmp, sizeof tmp, "%s.%u.%u.tmp", path, (unsigned)getpid(), plat_thread_id());
    if (len < 0 || len >= (int)sizeof tmp) return 0;
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return 0;
    int ok = write_all(fd, data, n) && fsync(fd) == 0;
    ok = close(fd) == 0 && ok;
    if (ok && !rename(tmp, path)) {
        sync_dir_of(path);
        return 1;
    }
    unlink(tmp);
    return 0;
}

int plat_file_flush(const char *path) {
    int fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    int ok = fsync(fd) == 0;
    close(fd);
    return ok;
}

void plat_file_delete(const char *path) { unlink(path); }

int plat_file_truncate(const char *path, uint64_t size) { return truncate(path, (off_t)size) == 0; }

int64_t plat_file_size(const char *path) {
    struct stat st;
    return stat(path, &st) ? -1 : (int64_t)st.st_size;
}

// The owner's write permission stands for Windows' read-only attribute.
int plat_file_set_readonly(const char *path, int on) {
    struct stat st;
    if (stat(path, &st)) return 0;
    mode_t m = on ? st.st_mode & ~(mode_t)0222 : st.st_mode | S_IWUSR;
    return chmod(path, m & 07777) == 0;
}

int plat_file_list(const char *dir, const char *ext, int (*cb)(const char *name, void *ctx), void *ctx) {
    DIR *d = opendir(dir);
    if (!d) return 0;
    size_t el = strlen(ext);
    for (struct dirent *e; (e = readdir(d));) {
        size_t n = strlen(e->d_name);
        if (n < el || strcmp(e->d_name + n - el, ext) || e->d_name[0] == '.') continue;
        if (!cb(e->d_name, ctx)) break;
    }
    closedir(d);
    return 1;
}

// From totk-pc-recomp runtime/platform/linux.c @ 0ae47a578285cde080642c9e6c40316c99171ef1 (plat_aligned_alloc,
// plat_aligned_free, plat_exe_base, plat_module_of).
void *plat_aligned_alloc(size_t size, size_t align) {
    void *p = NULL;
    return posix_memalign(&p, align < sizeof(void *) ? sizeof(void *) : align, size) ? NULL : p;
}

void plat_aligned_free(void *p) { free(p); }

uintptr_t plat_exe_base(void) {
    Dl_info d;
    return dladdr((void *)plat_exe_base, &d) ? (uintptr_t)d.dli_fbase : 0;
}

int plat_module_of(uintptr_t addr, char *name, size_t cap, uintptr_t *base) {
    Dl_info d;
    if (!dladdr((void *)addr, &d) || !d.dli_fname) return 0;
    const char *n = strrchr(d.dli_fname, '/');
    *base = (uintptr_t)d.dli_fbase;
    return snprintf(name, cap, "%s", n ? n + 1 : d.dli_fname) < (int)cap;
}

int plat_exe_path(char *out, size_t cap) {
    ssize_t n = readlink("/proc/self/exe", out, cap - 1);
    if (n <= 0 || (size_t)n >= cap - 1) return 0;
    out[n] = 0;
    return 1;
}

size_t plat_getenv(const char *name, char *out, size_t cap) {
    const char *v = getenv(name);
    size_t n = v ? strlen(v) : 0;
    if (!n || n >= cap) return 0;
    memcpy(out, v, n + 1);
    return n;
}

// The first set variable of names ("ru_RU.UTF-8") as a BCP-47 tag ("ru-RU") into tag; "" for C and POSIX.
static void locale_tag(const char *const *names, int n, char *tag, size_t cap) {
    const char *v = NULL;
    for (int i = 0; i < n && !(v && *v); i++) v = getenv(names[i]);
    size_t k = 0;
    if (v && strcmp(v, "C") && strncmp(v, "C.", 2) && strcmp(v, "POSIX"))
        for (; v[k] && v[k] != '.' && v[k] != '@' && k < cap - 1; k++) tag[k] = v[k] == '_' ? '-' : v[k];
    tag[k] = 0;
}

const char *plat_ui_language(void) {
    static char tag[32];
    static const char *const names[] = {"LC_ALL", "LC_MESSAGES", "LANG"};
    if (!tag[0]) locale_tag(names, 3, tag, sizeof tag);
    return tag;
}

// Noto CJK as the common distributions and SteamOS install it; one collection covers every CJK language.
int plat_font_path(const char *lang, char *out, size_t cap) {
    static const char *files[] = {
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    };
    if (strcmp(lang, "ja") && strcmp(lang, "ko") && strcmp(lang, "zh-Hans") && strcmp(lang, "zh-Hant")) return 0;
    for (size_t i = 0; i < sizeof files / sizeof *files; i++)
        if (plat_file_exists(files[i])) return snprintf(out, cap, "%s", files[i]) < (int)cap;
    return 0;
}

// PRETTY_NAME of /etc/os-release and the kernel release: "Fedora Linux 43 (Linux 7.2.11-100.fc43.x86_64)".
int plat_os_version(char *out, size_t cap) {
    char name[128] = "Linux";
    FILE *f = fopen("/etc/os-release", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f))
            if (!strncmp(line, "PRETTY_NAME=", 12)) {
                char *v = line + 12, *e;
                if (*v == '"') v++;
                if ((e = strpbrk(v, "\"\n"))) *e = 0;
                snprintf(name, sizeof name, "%s", v);
                break;
            }
        fclose(f);
    }
    struct utsname u;
    if (uname(&u)) return snprintf(out, cap, "%s", name) < (int)cap;
    return snprintf(out, cap, "%s (Linux %s)", name, u.release) < (int)cap;
}

int plat_stderr_path(char *out, size_t cap) {
    struct stat st;
    if (fstat(2, &st) || !S_ISREG(st.st_mode)) return 0;
    ssize_t n = readlink("/proc/self/fd/2", out, cap - 1);
    if (n <= 0 || (size_t)n >= cap - 1) return 0;
    out[n] = 0;
    return 1;
}

// ---- desktop

int plat_clipboard_set(const char *utf8) { return SDL_SetClipboardText(utf8) ? 1 : 0; }

static void reap_main(void *arg) {
    waitpid((pid_t)(intptr_t)arg, NULL, 0);
}

// xdg-open picks the desktop's file manager; it runs detached, so a slow answer does not hold up the caller, and a
// thread of its own reaps it so it does not stay a zombie.
int plat_open_folder(const char *path) {
    struct stat st;
    if (stat(path, &st) || !S_ISDIR(st.st_mode)) return 0;
    posix_spawnattr_t a;
    posix_spawnattr_init(&a);
    posix_spawnattr_setflags(&a, POSIX_SPAWN_SETSID);
    char *argv[] = {"xdg-open", (char *)path, NULL};
    pid_t pid;
    int e = posix_spawnp(&pid, "xdg-open", NULL, &a, argv, environ);
    posix_spawnattr_destroy(&a);
    if (e) return 0;
    PlatThread *r = plat_thread_new(reap_main, (void *)(intptr_t)pid, "reap");
    if (r) plat_thread_detach(r);
    return 1;
}

// ---- dynamic libraries

void *plat_lib_open(const char *name) { return dlopen(name, RTLD_NOW | RTLD_LOCAL); }
void *plat_lib_sym(void *lib, const char *name) { return lib ? dlsym(lib, name) : NULL; }

// ---- machine load: /proc/stat and /proc/self/stat for the CPU, hwmon for the temperature, the amdgpu sysfs files of
// the busiest DRM card for the GPU.

struct PlatLoad {
    uint64_t busy, total, proc;   // the previous call's jiffies; total 0 before the first
    uint64_t proc_ns;
    char gpu[256];                // /sys/class/drm/cardN/device of the GPU in use; empty when unknown
    char temp[256];               // the CPU's hwmon temp1_input; empty when unknown
};

static uint64_t process_cpu_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t)) return 0;
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

static int read_u64_file(const char *path, uint64_t *v) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    unsigned long long x;
    int ok = fscanf(f, "%llu", &x) == 1;
    fclose(f);
    if (ok) *v = x;
    return ok;
}

// The hwmon device of the CPU package: k10temp (AMD), coretemp (Intel), or the first that has a temp1_input.
static void find_cpu_temp(char *out, size_t cap) {
    out[0] = 0;
    DIR *d = opendir("/sys/class/hwmon");
    if (!d) return;
    for (struct dirent *e; (e = readdir(d));) {
        if (e->d_name[0] == '.') continue;
        char path[256], name[64] = "";
        snprintf(path, sizeof path, "/sys/class/hwmon/%s/name", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        if (!fgets(name, sizeof name, f)) name[0] = 0;
        fclose(f);
        name[strcspn(name, "\n")] = 0;
        int cpu = !strcmp(name, "k10temp") || !strcmp(name, "coretemp") || !strcmp(name, "zenpower");
        snprintf(path, sizeof path, "/sys/class/hwmon/%s/temp1_input", e->d_name);
        if ((cpu || !out[0]) && plat_file_exists(path)) snprintf(out, cap, "%s", path);
        if (cpu && out[0]) break;
    }
    closedir(d);
}

// The DRM card with gpu_busy_percent (amdgpu) that has the most VRAM in use. None when no card has that file (NVIDIA):
// the caller then reads its load from another source, not from an idle integrated GPU.
static void find_gpu(char *out, size_t cap) {
    out[0] = 0;
    uint64_t best = 0;
    DIR *d = opendir("/sys/class/drm");
    if (!d) return;
    for (struct dirent *e; (e = readdir(d));) {
        if (strncmp(e->d_name, "card", 4) || strchr(e->d_name, '-')) continue;
        char dev[256], path[300];
        snprintf(dev, sizeof dev, "/sys/class/drm/%s/device", e->d_name);
        snprintf(path, sizeof path, "%s/gpu_busy_percent", dev);
        uint64_t v, used = 0;
        if (!read_u64_file(path, &v)) continue;
        snprintf(path, sizeof path, "%s/mem_info_vram_used", dev);
        read_u64_file(path, &used);
        if (!out[0] || used > best) snprintf(out, cap, "%s", dev), best = used;
    }
    closedir(d);
}

// Linux has no adapter LUIDs: the busiest amdgpu card is used whatever the caller passes as gpu_luid.
PlatLoad *plat_load_open(const uint8_t *gpu_luid) {
    (void)gpu_luid;
    PlatLoad *l = calloc(1, sizeof *l);
    if (!l) return NULL;
    find_cpu_temp(l->temp, sizeof l->temp);
    find_gpu(l->gpu, sizeof l->gpu);
    return l;
}

void plat_load_close(PlatLoad *l) { free(l); }

void plat_cpu_load(PlatLoad *l, float *total_pct, float *proc_pct, float *temp_c) {
    *total_pct = *proc_pct = *temp_c = -1;
    if (!l) return;
    FILE *f = fopen("/proc/stat", "r");
    unsigned long long v[10] = {0};
    int n = f ? fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4],
                       &v[5], &v[6], &v[7], &v[8], &v[9])
              : 0;
    if (f) fclose(f);
    uint64_t pns = process_cpu_ns();
    if (n >= 8) {
        uint64_t idle = v[3] + v[4], total = 0;
        for (int i = 0; i < 8; i++) total += v[i];   // guest time is already inside user time
        uint64_t busy = total - idle;
        if (l->total && total > l->total) {
            uint64_t dt = total - l->total;
            *total_pct = 100.0f * (float)(busy - l->busy) / (float)dt;
            // the process time is in ns, the machine's in jiffies summed over every CPU
            double dt_ns = (double)dt * 1e9 / (double)sysconf(_SC_CLK_TCK);
            *proc_pct = (float)(100.0 * (double)(pns - l->proc_ns) / dt_ns);
        }
        l->busy = busy, l->total = total, l->proc_ns = pns;
    }
    uint64_t mc;
    if (l->temp[0] && read_u64_file(l->temp, &mc)) *temp_c = (float)mc / 1000.0f;
}

void plat_gpu_load(PlatLoad *l, float *load_pct, float *vram_used_mb) {
    *load_pct = *vram_used_mb = -1;
    if (!l || !l->gpu[0]) return;
    char path[300];
    uint64_t v;
    snprintf(path, sizeof path, "%s/gpu_busy_percent", l->gpu);
    if (read_u64_file(path, &v)) *load_pct = (float)v;
    snprintf(path, sizeof path, "%s/mem_info_vram_used", l->gpu);
    if (read_u64_file(path, &v)) *vram_used_mb = (float)(v >> 20);
}

void plat_mem_info(float *used_mb, float *total_mb, float *proc_mb) {
    *used_mb = *total_mb = *proc_mb = -1;
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[128];
        unsigned long long total = 0, avail = 0, k;
        while (fgets(line, sizeof line, f)) {
            if (sscanf(line, "MemTotal: %llu kB", &k) == 1) total = k;
            else if (sscanf(line, "MemAvailable: %llu kB", &k) == 1) avail = k;
        }
        fclose(f);
        if (total) *total_mb = (float)(total >> 10), *used_mb = (float)((total - avail) >> 10);
    }
    f = fopen("/proc/self/statm", "r");
    unsigned long long size, resident;
    if (f && fscanf(f, "%llu %llu", &size, &resident) == 2)
        *proc_mb = (float)(resident * (uint64_t)sysconf(_SC_PAGESIZE) >> 20);
    if (f) fclose(f);
}

static void process_memory(uint64_t *resident, uint64_t *private_bytes) {
    *resident = *private_bytes = 0;
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[128];
    unsigned long long k;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "VmRSS: %llu kB", &k) == 1) *resident = (uint64_t)k << 10;
        else if (sscanf(line, "RssAnon: %llu kB", &k) == 1) *private_bytes = (uint64_t)k << 10;
    }
    fclose(f);
}

// Linux overcommits by default, so CommitLimit binds only with vm.overcommit_memory = 2; Committed_AS is what the
// system has promised. The process's share is its resident anonymous memory, the nearest to Windows' private bytes.
void plat_commit_info(double *proc_mb, double *used_mb, double *limit_mb) {
    *proc_mb = *used_mb = *limit_mb = -1;
    char line[128];
    unsigned long long k;
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        while (fgets(line, sizeof line, f)) {
            if (sscanf(line, "CommitLimit: %llu kB", &k) == 1) *limit_mb = (double)(k >> 10);
            else if (sscanf(line, "Committed_AS: %llu kB", &k) == 1) *used_mb = (double)(k >> 10);
        }
        fclose(f);
    }
    uint64_t resident, private_bytes;
    process_memory(&resident, &private_bytes);
    if (private_bytes) *proc_mb = (double)(private_bytes >> 20);
}

// ---- child processes

struct PlatChild {
    pid_t pid;
    int in, out;   // in: our write end, out: our read end
};

// Children start one at a time: a pipe made by one thread must not be inherited by a child another thread starts.
static PlatMutex child_lock = PLAT_MUTEX_INIT;

PlatChild *plat_child_start(const char *exe, const char *arg, uint32_t *err) {
    if (err) *err = 0;
    int in[2], out[2];
    plat_mutex_lock(&child_lock);
    if (pipe2(in, O_CLOEXEC)) {
        plat_mutex_unlock(&child_lock);
        return NULL;
    }
    if (pipe2(out, O_CLOEXEC)) {
        close(in[0]), close(in[1]);
        plat_mutex_unlock(&child_lock);
        return NULL;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in[0], 0);
    posix_spawn_file_actions_adddup2(&fa, out[1], 1);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    char *argv[] = {(char *)exe, (char *)arg, NULL};
    pid_t pid;
    int e = posix_spawn(&pid, exe, &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(in[0]);
    close(out[1]);
    plat_mutex_unlock(&child_lock);
    if (e) {
        close(in[1]), close(out[0]);
        if (err) *err = (uint32_t)e;
        return NULL;
    }
    PlatChild *c = malloc(sizeof *c);
    if (!c) {
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        close(in[1]), close(out[0]);
        return NULL;
    }
    c->pid = pid, c->in = in[1], c->out = out[0];
    return c;
}

void plat_child_stop(PlatChild *c) {
    if (!c) return;
    close(c->in);
    close(c->out);
    kill(c->pid, SIGKILL);
    waitpid(c->pid, NULL, 0);
    free(c);
}

// A child that died makes the write fail with EPIPE; SIGPIPE is ignored for the process (main.c), as Windows has none.
int plat_child_write(PlatChild *c, const void *data, size_t n) { return write_all(c->in, data, n); }

long plat_child_read(PlatChild *c, void *buf, size_t cap) {
    ssize_t k;
    do k = read(c->out, buf, cap);
    while (k < 0 && errno == EINTR);
    return (long)k;
}

// getrandom(2) fills up to 256 bytes per call without blocking once the kernel has its entropy.
int plat_random(void *out, size_t n) {
    uint8_t *p = out;
    while (n) {
        ssize_t k = getrandom(p, n, 0);
        if (k < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        p += k, n -= (size_t)k;
    }
    return 1;
}

// SHA-256 in portable C: the block function, padding and output of totk-pc-recomp runtime/sha256.c @ c944f9f94520874f2a669add30dd08d29ee9f7b5
// (MIT, Copyright (c) 2026 Valiantsin Dzerakh), without its SHA-NI path.
static const uint32_t sha_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static uint32_t sha_ror(uint32_t x, int n) { return x >> n | x << (32 - n); }

// n 64-byte blocks at p into the state h.
static void sha_blocks(uint32_t h[8], const uint8_t *p, size_t n) {
    for (; n; n--, p += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = sha_ror(w[i - 15], 7) ^ sha_ror(w[i - 15], 18) ^ w[i - 15] >> 3;
            uint32_t s1 = sha_ror(w[i - 2], 17) ^ sha_ror(w[i - 2], 19) ^ w[i - 2] >> 10;
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = k + (sha_ror(e, 6) ^ sha_ror(e, 11) ^ sha_ror(e, 25)) + ((e & f) ^ (~e & g)) + sha_k[i] + w[i];
            uint32_t t2 = (sha_ror(a, 2) ^ sha_ror(a, 13) ^ sha_ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            k = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += k;
    }
}

int plat_sha256(const void *data, size_t n, uint8_t out[32]) {
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t h[8];
    memcpy(h, iv, sizeof h);
    const uint8_t *p = data;
    size_t full = n / 64, rest = n % 64;
    sha_blocks(h, p, full);
    // The last block (one, or two when the length does not fit after the 0x80 byte) holds the rest, 0x80, zeros and the
    // message length in bits, big-endian in its last eight bytes.
    uint8_t tail[128] = {0};
    if (rest) memcpy(tail, p + full * 64, rest);
    tail[rest] = 0x80;
    size_t len = rest < 56 ? 64 : 128;
    uint64_t bits = (uint64_t)n * 8;
    for (int i = 0; i < 8; i++) tail[len - 1 - i] = (uint8_t)(bits >> (8 * i));
    sha_blocks(h, tail, len / 64);
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (uint8_t)(h[i] >> 24), out[4 * i + 1] = (uint8_t)(h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(h[i] >> 8), out[4 * i + 3] = (uint8_t)h[i];
    }
    return 1;
}

// The streamed form: whole 64-byte blocks go through sha_blocks as they fill, the rest waits in buf for the next piece
// or for plat_sha256_end, which pads it the way plat_sha256 does.
struct PlatSha256 {
    uint32_t h[8];
    uint8_t buf[64];
    size_t fill;
    uint64_t total;
};

PlatSha256 *plat_sha256_begin(void) {
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    PlatSha256 *s = calloc(1, sizeof *s);
    if (s) memcpy(s->h, iv, sizeof iv);
    return s;
}

int plat_sha256_update(PlatSha256 *s, const void *data, size_t n) {
    const uint8_t *p = data;
    s->total += n;
    if (s->fill) {
        size_t k = 64 - s->fill < n ? 64 - s->fill : n;
        memcpy(s->buf + s->fill, p, k);
        s->fill += k, p += k, n -= k;
        if (s->fill < 64) return 1;
        sha_blocks(s->h, s->buf, 1);
        s->fill = 0;
    }
    size_t full = n / 64;
    sha_blocks(s->h, p, full);
    p += full * 64, n -= full * 64;
    memcpy(s->buf, p, n);
    s->fill = n;
    return 1;
}

int plat_sha256_end(PlatSha256 *s, uint8_t out[32]) {
    uint8_t tail[128] = {0};
    memcpy(tail, s->buf, s->fill);
    tail[s->fill] = 0x80;
    size_t len = s->fill < 56 ? 64 : 128;
    uint64_t bits = s->total * 8;
    for (int i = 0; i < 8; i++) tail[len - 1 - i] = (uint8_t)(bits >> (8 * i));
    sha_blocks(s->h, tail, len / 64);
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (uint8_t)(s->h[i] >> 24), out[4 * i + 1] = (uint8_t)(s->h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(s->h[i] >> 8), out[4 * i + 3] = (uint8_t)s->h[i];
    }
    free(s);
    return 1;
}

// ---- files and the host file system under the save data (fs_host.c; declared in platform.h). Adapted from
// totk-pc-recomp runtime/platform/linux.c @ 0ae47a578285cde080642c9e6c40316c99171ef1 (MIT, Copyright (c) 2026 Valiantsin
// Dzerakh): a PlatFile is the descriptor + 1, so descriptor 0 is not NULL.
static int fd_of(PlatFile *f) { return (int)(intptr_t)f - 1; }
static PlatFile *file_of_fd(int fd) { return fd < 0 ? NULL : (PlatFile *)(intptr_t)(fd + 1); }

PlatFile *plat_file_open(const char *path) { return file_of_fd(open(path, O_RDONLY | O_CLOEXEC)); }

void plat_file_close(PlatFile *f) {
    if (f) close(fd_of(f));
}

int64_t plat_file_pread(PlatFile *f, void *buf, uint32_t n, uint64_t off) {
    ssize_t k;
    do k = pread(fd_of(f), buf, n, (off_t)off);
    while (k < 0 && errno == EINTR);
    return k >= 0 ? k : errno == EFAULT ? -2 : -1;
}

int64_t plat_file_pwrite(PlatFile *f, const void *buf, uint32_t n, uint64_t off) {
    ssize_t k;
    do k = pwrite(fd_of(f), buf, n, (off_t)off);
    while (k < 0 && errno == EINTR);
    return k;
}

int plat_file_id(PlatFile *f, PlatFileId *out) {
    struct stat st;
    if (fstat(fd_of(f), &st)) return 0;
    *out = (PlatFileId){(uint64_t)st.st_dev, (uint64_t)st.st_ino, (uint64_t)st.st_size,
                        (uint64_t)st.st_mtim.tv_sec * 1000000000u + (uint64_t)st.st_mtim.tv_nsec,
                        (uint64_t)st.st_ctim.tv_sec * 1000000000u + (uint64_t)st.st_ctim.tv_nsec};
    return 1;
}

static int fs_err(int e);

int plat_file_set_size(PlatFile *f, uint64_t size) { return ftruncate(fd_of(f), (off_t)size) ? fs_err(errno) : 0; }

int plat_file_fsync(PlatFile *f) { return fsync(fd_of(f)) == 0; }

int plat_fs_os_error(void) { return errno; }

// ENOENT and ENOTDIR both mean the path is not there; EBUSY is the "try again" of the save commit.
static int fs_err(int e) {
    switch (e) {
    case ENOENT: case ENOTDIR: return PLAT_FS_NOT_FOUND;
    case EEXIST: return PLAT_FS_EXISTS;
    case ENOTEMPTY: return PLAT_FS_NOT_EMPTY;
    case EBUSY: case ETXTBSY: return PLAT_FS_BUSY;
    default: return PLAT_FS_ERROR;
    }
}

int plat_fs_kind(const char *path) {
    struct stat st;
    if (lstat(path, &st)) return PLAT_FS_MISSING;
    return S_ISLNK(st.st_mode) ? PLAT_FS_LINK : S_ISDIR(st.st_mode) ? PLAT_FS_DIR : PLAT_FS_FILE;
}

int plat_fs_mkdir(const char *path) { return mkdir(path, 0755) ? fs_err(errno) : PLAT_FS_OK; }

// A symbolic link goes as a file does (Windows: a directory link is removed with RemoveDirectory).
int plat_fs_rmdir(const char *path) {
    struct stat st;
    if (!lstat(path, &st) && S_ISLNK(st.st_mode)) return unlink(path) ? fs_err(errno) : PLAT_FS_OK;
    return rmdir(path) ? fs_err(errno) : PLAT_FS_OK;
}

int plat_fs_unlink(const char *path) { return unlink(path) ? fs_err(errno) : PLAT_FS_OK; }

// RENAME_NOREPLACE keeps Windows' rule that a rename never replaces; a file system without it (EINVAL) gets a check
// first, which another process could race but the save directories are ours.
int plat_fs_rename(const char *from, const char *to, int durable) {
    if (syscall(SYS_renameat2, AT_FDCWD, from, AT_FDCWD, to, RENAME_NOREPLACE)) {
        if (errno != EINVAL && errno != ENOSYS) return fs_err(errno);
        if (plat_fs_kind(to) != PLAT_FS_MISSING) return PLAT_FS_EXISTS;
        if (rename(from, to)) return fs_err(errno);
    }
    if (durable) {
        sync_dir_of(from);
        sync_dir_of(to);
    }
    return PLAT_FS_OK;
}

int plat_fs_link(const char *existing, const char *path) { return link(existing, path) ? fs_err(errno) : PLAT_FS_OK; }

int plat_fs_links(const char *path) {
    struct stat st;
    return stat(path, &st) ? 0 : (int)st.st_nlink;
}

void plat_fs_sync_dir(const char *path) {
    int fd = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
}

int plat_fs_list(const char *dir, int (*cb)(const char *name, int kind, uint64_t size, void *ctx), void *ctx) {
    DIR *d = opendir(dir);
    if (!d) return fs_err(errno);
    for (struct dirent *e; (e = readdir(d));) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        struct stat st;
        if (fstatat(dirfd(d), e->d_name, &st, AT_SYMLINK_NOFOLLOW)) continue;
        int kind = S_ISLNK(st.st_mode) ? PLAT_FS_LINK : S_ISDIR(st.st_mode) ? PLAT_FS_DIR : PLAT_FS_FILE;
        if (!cb(e->d_name, kind, kind == PLAT_FS_FILE ? (uint64_t)st.st_size : 0, ctx)) break;
    }
    closedir(d);
    return PLAT_FS_OK;
}

PlatFile *plat_fs_open(const char *path, int write, int *err) {
    struct stat st;
    int fd = open(path, (write ? O_RDWR : O_RDONLY) | O_CLOEXEC);
    if (fd >= 0 && (fstat(fd, &st) || !S_ISREG(st.st_mode))) {
        close(fd);
        fd = -1, errno = ENOENT;
    }
    if (fd < 0 && err) *err = fs_err(errno);
    return file_of_fd(fd);
}

PlatFile *plat_fs_create(const char *path, int *err) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0 && err) *err = fs_err(errno);
    return file_of_fd(fd);
}

int plat_temp_dir(char *out, size_t cap) {
    const char *t = getenv("TMPDIR");
    if (!t || !*t) t = "/tmp";
    size_t n = strlen(t);
    while (n > 1 && t[n - 1] == '/') n--;
    return snprintf(out, cap, "%.*s", (int)n, t) < (int)cap;
}

void plat_utc_now(struct tm *out, unsigned *ms) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    time_t s = ts.tv_sec;
    gmtime_r(&s, out);
    *ms = (unsigned)(ts.tv_nsec / 1000000);
}
