// From metroid-prime-4-recomp runtime/vk_sync.c @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied with mp4 renamed to oot.
// From fe3h-pc-recomp runtime/vk_sync.c @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied with fe3h renamed to mp4 only.
// From totk-pc-recomp runtime/vk_sync.c @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), copied with totk renamed to fe3h only.
// Submission, fences and the completion thread of the Vulkan back end.
// One timeline semaphore (VkCtx.timeline) orders everything: each submission records its command handles into a
// primary command buffer (vk_exec_record) and signals the next timeline value, which is the NVN fence value
// the front end hands to syncs. The completion thread waits on the timeline and, in submit order, runs the CPU side
// effects queued for each value (event writes, counter reports, sync state), then frees the memory and texture objects whose users
// completed. A value counts as done only after its side effects ran, so a guest woken by fence_wait sees them.
//
// The submission shape (one timeline value per submission, signalled at ALL_COMMANDS, vkWaitSemaphores for waits, a
// failed wait treated as a lost device) follows LAS Recomp runtime/nvn/vulkan/exec.c:39-110, 355-397
// (https://github.com/peiikai/LASRecomp @ da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai).
// Side effects written after GPU completion, as MacNativeGames does: games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c:7-9
// (https://github.com/ly05010419/MacNativeGames @ b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026
// jhaizhou; origin d2r-mobile).
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include "gfx.h"
#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"

#define WEAK __attribute__((weak))

typedef struct { uint64_t fence; void (*fn)(void *); void *arg; } Pending;
typedef struct { VkCommandBuffer cb; uint64_t fence; } CmdSlot;   // reusable once the timeline reached fence

static SRWLOCK start_lock = SRWLOCK_INIT;
static int started;
static HANDLE completion_thread;

// Submissions: one at a time, so timeline values are signalled in the order they are handed out.
static SRWLOCK sub_lock = SRWLOCK_INIT;
static VkCommandPool cmd_pool;
static CmdSlot *slots;
static int nslots;
static int cur_slot;          // the slot vk_sync_submit records into now (an index: take_slot may move slots)
static uint64_t cur_fence;    // the value of the submission being recorded

// Shared with the completion thread (st_lock).
static SRWLOCK st_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE cv_work = CONDITION_VARIABLE_INIT;   // a submission, a callback or stop
static CONDITION_VARIABLE cv_done = CONDITION_VARIABLE_INIT;   // done advanced
static uint64_t issued, done;     // newest submitted value; newest value whose side effects ran
static uint64_t reserved;         // asynchronous replay: newest value handed out before its submission (>= issued)

static int stop;                  // set by vk_sync_shutdown, checked by the threads

// ---- OOT_FENCE_TRACE=1 (diagnostic): the completion thread's stage, so a stall says who is blocked.
// ft_stage: 0 idle, 1 inside a completion callback, 2 inside vk_mem_collect, 3 inside the timeline wait.
static int ftrace = -1;
static volatile LONG64 ft_stage, ft_last_cb, ft_waits, ft_cur_fence;
static void ft_init(void) {
    if (ftrace >= 0) return;
    const char *e = cfg_get("OOT_FENCE_TRACE");
    ftrace = e && *e && *e != '0';
    if (ftrace) {
        extern int vk_exec_trace_on;   // vk_exec.c
        vk_exec_trace_on = 1;
        rt_log("fence: trace on (OOT_FENCE_TRACE)");
    }
}
static void ft_set(int stage) { if (ftrace > 0) __atomic_store_n(&ft_stage, stage, __ATOMIC_RELEASE); }
static volatile LONG64 ft_cstage;   // the completion thread's own stage: it polls the timeline every 100 ms and must not
                                    // overwrite the submit path's stage, which is what a stall needs to show
static void ft_cset(int stage) { if (ftrace > 0) __atomic_store_n(&ft_cstage, stage, __ATOMIC_RELEASE); }

// Says, once a second, what the completion thread is doing while `done` stops advancing behind `issued`.
// Also prints a heartbeat every 5 s, so a stall shows the stage the submit path is stuck in.
static DWORD WINAPI ft_watch_main(void *arg) {
    VkCtx *c = arg;
    uint64_t last = 0;
    int still = 0, beat = 0;
    for (;;) {
        Sleep(1000);
        if (__atomic_load_n(&stop, __ATOMIC_ACQUIRE)) return 0;
        uint64_t d = __atomic_load_n(&done, __ATOMIC_ACQUIRE), i = __atomic_load_n(&issued, __ATOMIC_ACQUIRE);
        if (d == last && i > d) {
            if (++still == 2) {
                uint64_t tl = 0;
                vkGetSemaphoreCounterValue(c->device, c->timeline, &tl);
                rt_log("fence: STALL done=%llu issued=%llu timeline=%llu stage=%lld last_cb=%lld waits=%lld",
                       (unsigned long long)d, (unsigned long long)i, (unsigned long long)tl,
                       (long long)__atomic_load_n(&ft_stage, __ATOMIC_ACQUIRE),
                       (long long)__atomic_load_n(&ft_last_cb, __ATOMIC_ACQUIRE),
                       (long long)__atomic_load_n(&ft_waits, __ATOMIC_ACQUIRE));
            }
        } else {
            still = 0;
        }
        if (++beat >= 5) {   // heartbeat: stage 4+ means a submit holds sub_lock
            beat = 0;
            uint64_t tl = 0;
            vkGetSemaphoreCounterValue(c->device, c->timeline, &tl);
            extern volatile LONG64 vk_exec_cur_op;   // vk_exec.c
            long long op = __atomic_load_n(&vk_exec_cur_op, __ATOMIC_RELAXED);
            rt_log("fence: hb done=%llu issued=%llu timeline=%llu stage=%lld cstage=%lld cur=%lld op=%s(%lld) waits=%lld",
                   (unsigned long long)d, (unsigned long long)i, (unsigned long long)tl,
                   (long long)__atomic_load_n(&ft_stage, __ATOMIC_ACQUIRE),
                   (long long)__atomic_load_n(&ft_cstage, __ATOMIC_ACQUIRE),
                   (long long)__atomic_load_n(&ft_cur_fence, __ATOMIC_ACQUIRE),
                   nvn_op_name(op < 0 ? 0xffffffffu : (uint32_t)op), op,
                   (long long)__atomic_load_n(&ft_waits, __ATOMIC_ACQUIRE));
        }
        last = d;
    }
}
static Pending *queue;            // side effects in the order they were added
static int qhead, qn, qcap;

// ---- GPU time (OOT_TRACE=perf): each submission part (vk_sync_split) writes a timestamp when it starts and when it
// ends, into a ring of query pairs. The completion thread reads a pair once the part's value is reached and adds the
// part's interval to the busy time, counting overlap with the previous part once. Off (one log line) when the queue
// family has no timestamp bits or the period is 0. Upload batches (vk_memory.c) are not timed.
#define TS_PAIRS 1024
static VkQueryPool ts_pool;
static int ts_state;                       // 0 not set up, 1 on, -1 off
static double ts_period;                   // ns per tick
static uint64_t ts_mask;                   // valid bits of a timestamp
static uint32_t ts_next;                   // submitting thread (sub_lock): next pair
static volatile uint8_t ts_busy[TS_PAIRS]; // written until the completion thread read the pair
static int ts_cur = -1;                    // the pair of the part being recorded, -1 untimed
static uint64_t ts_last_end;               // completion thread: end of the latest part read (ticks)
static volatile LONG64 ts_busy_ns, ts_parts, ts_missed;

// On the first submission that wants timing (perf_gpu_timing: OOT_TRACE=perf or the full performance overlay).
static void ts_setup(VkCtx *c) {
    ts_state = -1;
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    if (!qp) return;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, qp);
    uint32_t bits = c->gfx_family < nq ? qp[c->gfx_family].timestampValidBits : 0;
    free(qp);
    float period = c->props.limits.timestampPeriod;
    if (!bits || period <= 0) {
        rt_log("vk: no GPU timestamps on this queue (valid bits %u, period %.3f ns): the perf line shows no GPU time", bits, period);
        return;
    }
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                .queryCount = 2 * TS_PAIRS};
    VkResult r = vkCreateQueryPool(c->device, &qi, NULL, &ts_pool);
    if (r != VK_SUCCESS) {
        rt_log("vk: timestamp query pool failed (%d): the perf line shows no GPU time", (int)r);
        return;
    }
    ts_period = period;
    ts_mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
    ts_state = 1;
}

typedef struct { uint32_t pair; } TsRead;

static void ts_read_cb(void *arg) {   // completion thread, in submission order
    TsRead *t = arg;
    VkCtx *c = vk_ctx();
    uint64_t v[2];
    if (vkGetQueryPoolResults(c->device, ts_pool, 2 * t->pair, 2, sizeof v, v, sizeof v[0], VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        uint64_t s = v[0] & ts_mask, e = v[1] & ts_mask;
        if (e >= s) {
            uint64_t from = s > ts_last_end ? s : ts_last_end;
            if (e > from) {
                LONG64 ns = (LONG64)((double)(e - from) * ts_period);
                InterlockedAdd64(&ts_busy_ns, ns);
                perf_gpu_busy((uint64_t)ns);
            }
            if (e > ts_last_end) ts_last_end = e;
            InterlockedIncrement64(&ts_parts);
        } else InterlockedIncrement64(&ts_missed);   // the counter wrapped
    } else InterlockedIncrement64(&ts_missed);
    __atomic_store_n(&ts_busy[t->pair], 0, __ATOMIC_RELEASE);
    free(t);
}

// At the start of a command buffer (sub_lock held): the pair it times, -1 untimed.
static int ts_open(VkCommandBuffer cb) {
    if (ts_state != 1 || !perf_gpu_timing()) {
        if (ts_state || !perf_gpu_timing()) return -1;
        ts_setup(vk_ctx());   // the overlay turned full: timestamps from now on
        if (ts_state != 1) return -1;
    }
    uint32_t p = ts_next % TS_PAIRS;
    if (__atomic_load_n(&ts_busy[p], __ATOMIC_ACQUIRE)) { InterlockedIncrement64(&ts_missed); return -1; }
    ts_next++;
    ts_busy[p] = 1;
    vkCmdResetQueryPool(cb, ts_pool, 2 * p, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ts_pool, 2 * p);
    return (int)p;
}

// At its end, before vkEndCommandBuffer; fence: the value whose completion covers it.
static void ts_close(VkCommandBuffer cb, int pair, uint64_t fence) {
    if (pair < 0) return;
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool, 2 * (uint32_t)pair + 1);
    TsRead *t = malloc(sizeof *t);
    if (!t) rt_fatal("vk: out of memory");
    t->pair = (uint32_t)pair;
    vk_completion_add(fence, ts_read_cb, t);
}

// The pair of the part being recorded.
static void ts_begin(VkCommandBuffer cb) { ts_cur = ts_open(cb); }

static void ts_end(VkCommandBuffer cb, uint64_t fence) {
    ts_close(cb, ts_cur, fence);
    ts_cur = -1;
}

// An empty submission records no command buffer: its pair is never written.
static void ts_drop(void) {
    if (ts_cur < 0) return;
    __atomic_store_n(&ts_busy[ts_cur], 0, __ATOMIC_RELEASE);
    ts_cur = -1;
}

static void gpu_stats(PerfGpu *g) {
    g->have_gpu = ts_state == 1;
    g->gpu_ns = (uint64_t)InterlockedExchange64(&ts_busy_ns, 0);
    g->gpu_parts = (uint64_t)InterlockedExchange64(&ts_parts, 0);
    g->gpu_missed = (uint64_t)InterlockedExchange64(&ts_missed, 0);
    g->have_vram = vk_mem_vram(&g->vram_used, &g->vram_budget, &g->vram_exact);
}

// Weak stand-in for vk_exec.c's replay: only empty recordings ({OP_BEGIN, OP_END}) can be submitted.
WEAK int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence) {
    (void)cb, (void)fence;
    for (int i = 0; i < n; i++) {
        const op_hdr *h = handles[i] ? nvn_op_first(handles[i]) : NULL;
        if (!h || h->op != OP_BEGIN || h->size < sizeof *h || nvn_op_next(h)->op != OP_END)
            rt_fatal("vk: op replay is not linked");
    }
    return 0;
}

static DWORD WINAPI completion_main(void *arg) {
    VkCtx *c = arg;
    AcquireSRWLockExclusive(&st_lock);
    for (;;) {
        while (!stop && done == issued && !(qn && queue[qhead].fence <= done))
            SleepConditionVariableSRW(&cv_work, &st_lock, INFINITE, 0);
        if (stop) break;
        uint64_t reached = done;
        if (issued > done) {
            uint64_t want = done + 1;
            ReleaseSRWLockExclusive(&st_lock);
            ft_cset(3);   // OOT_FENCE_TRACE: waiting on the timeline
            VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
                                      .pSemaphores = &c->timeline, .pValues = &want};
            VkResult r = vkWaitSemaphores(c->device, &wi, 100000000ull);   // 100 ms: stop is checked between waits
            ft_cset(0);
            if (r != VK_SUCCESS && r != VK_TIMEOUT) {
                vk_device_fault_report(r);
                rt_fatal("vk: vkWaitSemaphores failed (%d; device lost?)", (int)r);
            }
            vkGetSemaphoreCounterValue(c->device, c->timeline, &reached);
            AcquireSRWLockExclusive(&st_lock);
            if (reached > issued) reached = issued;
        }
        while (qn && queue[qhead].fence <= reached) {
            Pending p = queue[qhead++];
            qn--;
            ReleaseSRWLockExclusive(&st_lock);
            if (ftrace > 0) {
                __atomic_store_n(&ft_last_cb, (long long)p.fence, __ATOMIC_RELEASE);
                ft_cset(1);
            }
            p.fn(p.arg);
            ft_cset(0);
            AcquireSRWLockExclusive(&st_lock);
        }
        if (reached > done) {
            ReleaseSRWLockExclusive(&st_lock);
            ft_cset(2);
            vk_mem_collect();
            ft_cset(0);
            AcquireSRWLockExclusive(&st_lock);
            __atomic_store_n(&done, reached, __ATOMIC_RELEASE);
            WakeAllConditionVariable(&cv_done);
        }
    }
    ReleaseSRWLockExclusive(&st_lock);
    return 0;
}

void vk_sync_start(void) {
    if (__atomic_load_n(&started, __ATOMIC_ACQUIRE)) return;
    AcquireSRWLockExclusive(&start_lock);
    if (!started) {
        if (!vk_init()) rt_fatal("vk: no usable Vulkan device (see the vk: lines above; OOT_GPU=null runs without a GPU)");
        VkCtx *c = vk_ctx();
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                      .queueFamilyIndex = c->gfx_family};
        if (vkCreateCommandPool(c->device, &pi, NULL, &cmd_pool) != VK_SUCCESS) rt_fatal("vk: submission command pool");
        if (!ts_state && perf_gpu_timing()) ts_setup(c);
        perf_gpu_stats = gpu_stats;
        uint64_t now = 0;
        vkGetSemaphoreCounterValue(c->device, c->timeline, &now);
        if (issued > now) {   // a new device after vk_shutdown: values keep increasing for the front end
            VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = c->timeline,
                                        .value = issued};
            if (vkSignalSemaphore(c->device, &si) != VK_SUCCESS) rt_fatal("vk: timeline restart at %llu", (unsigned long long)issued);
            now = issued;
        }
        AcquireSRWLockExclusive(&st_lock);
        issued = done = now;
        if (reserved < now) reserved = now;
        stop = 0;
        ReleaseSRWLockExclusive(&st_lock);
        vk_mem_note_submit(now);
        completion_thread = CreateThread(NULL, 0, completion_main, c, 0, NULL);
        if (!completion_thread) rt_fatal("vk: cannot start the completion thread (error %lu)", GetLastError());
        ft_init();   // OOT_FENCE_TRACE: after `c` exists, before the first submit
        if (ftrace > 0) CreateThread(NULL, 0, ft_watch_main, c, 0, NULL);
        __atomic_store_n(&started, 1, __ATOMIC_RELEASE);
        vk_astc_mode();   // decides OOT_ASTC and builds the ASTC decoder now, not at the first ASTC upload
    }
    ReleaseSRWLockExclusive(&start_lock);
}

// The index of a command buffer the GPU no longer uses, claimed for value `fence`. Called with sub_lock held; may move
// slots[].
static int claim_slot(VkCtx *c, uint64_t fence) {
    uint64_t gpu = 0;
    vkGetSemaphoreCounterValue(c->device, c->timeline, &gpu);
    for (int i = 0; i < nslots; i++)
        if (slots[i].fence <= gpu) {
            slots[i].fence = fence;
            return i;
        }
    CmdSlot *n = realloc(slots, sizeof *n * (size_t)(nslots + 1));
    if (!n) rt_fatal("vk: out of memory (command buffers)");
    slots = n;
    CmdSlot *s = &slots[nslots];
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cmd_pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    if (vkAllocateCommandBuffers(c->device, &ai, &s->cb) != VK_SUCCESS) rt_fatal("vk: command buffer allocation");
    s->fence = fence;
    return nslots++;
}

static CmdSlot *take_slot(VkCtx *c, uint64_t fence) {
    cur_slot = claim_slot(c, fence);
    return &slots[cur_slot];
}

// The copies the replay's first accesses to VRAM mirrors queued (vk_memory.c), in a command buffer of their own that is
// submitted right before the part that needs them and timed like a part; VK_NULL_HANDLE when there are none. After the
// part was recorded, sub_lock held; may move slots[].
static VkCommandBuffer prologue(VkCtx *c, uint64_t fence) {
    if (!vk_mem_mirror_pending()) return VK_NULL_HANDLE;
    int i = claim_slot(c, fence);   // before reading slots: the claim may move it
    VkCommandBuffer cb = slots[i].cb;
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &bi);
    int pair = ts_open(cb);
    vk_mem_mirror_prologue(cb);
    ts_close(cb, pair, fence);
    vkEndCommandBuffer(cb);
    return cb;
}

VkCommandBuffer vk_sync_split(void) {
    VkCtx *c = vk_ctx();
    VkCommandBuffer cb = slots[cur_slot].cb;
    ts_end(cb, cur_fence);
    if (!rec_end_cb(cb, 0, 0)) vkEndCommandBuffer(cb);   // every command of the part is in cb, ended by its recorder
    vk_upload_flush(0);   // the uploads this part asked for precede it in queue order
    VkCommandBuffer pro = prologue(c, cur_fence);
    VkCommandBufferSubmitInfo cbi[2] = {{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = pro},
                                        {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb}};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = pro ? 2 : 1,
                        .pCommandBufferInfos = pro ? cbi : cbi + 1};
    vk_queue_lock();
    VkResult r = vkQueueSubmit2(c->gfx_queue, 1, &si, VK_NULL_HANDLE);
    vk_queue_unlock();
    if (r != VK_SUCCESS) {
        vk_device_fault_report(r);
        rt_fatal("vk: vkQueueSubmit2 of a part of value %llu failed (%d)", (unsigned long long)cur_fence, (int)r);
    }
    vk_mem_note_submit(cur_fence);   // objects destroyed from here on may be in use by the part just submitted
    CmdSlot *s = take_slot(c, cur_fence);   // the submitted part keeps its slot until the value is reached
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(s->cb, &bi);
    vk_crumb(s->cb, VK_CRUMB_SUBMIT, cur_fence);
    ts_begin(s->cb);
    rec_begin(s->cb);
    return s->cb;
}

// Asynchronous replay (vk_backend.c): the value the next queued submission or present will signal. Values are handed
// out in queue order and the replay thread submits them in that order, so each vk_sync_submit gets exactly its value.
uint64_t vk_sync_reserve(void) {
    vk_sync_start();
    AcquireSRWLockExclusive(&st_lock);
    if (reserved < issued) reserved = issued;
    uint64_t v = ++reserved;
    ReleaseSRWLockExclusive(&st_lock);
    return v;
}

// Held by pool finalization and virtual remapping (vk_backend.c): a submission being recorded on another thread names
// the pool's buffers until it is submitted, and the vkQueueWaitIdle of the finalization covers only submitted work.
// FE3H's loader threads free a pool per resource while its render thread submits; in battle the GPU then drew into
// freed memory (an invalid write, the device lost).
void vk_sync_exclusive(int on) {
    vk_sync_start();
    if (on) AcquireSRWLockExclusive(&sub_lock);
    else ReleaseSRWLockExclusive(&sub_lock);
}

// OOT_GPU_DELAY=1 (an experiment): a submission is not handed to the queue at once but held until the CPU waits for a
// fence at or past it (nvnSyncWait, the window acquire, the present) or until vk_sync_finish; the next submission flushes
// the held ones first, so queue order holds. The console's GPU runs behind the CPU by about a frame, and a game that
// finishes writing a buffer after submitting the draw that reads it gets away with it there; here the GPU starts at
// once. Held submissions are visible to nothing but the queue: issued, the completion thread and the fences see them
// as submitted, so a wait for one flushes it.
static SRWLOCK pend_lock = SRWLOCK_INIT;
static struct { VkCommandBufferSubmitInfo cbi[2]; VkSemaphoreSubmitInfo sig; VkSubmitInfo2 si; uint64_t fence; } pend[64];
static int npend, gpu_delay = -1;
static void pend_flush_upto(uint64_t value) {   // pend_lock held
    int k = 0;
    while (k < npend && pend[k].fence <= value) k++;
    if (!k) return;
    VkCtx *c = vk_ctx();
    vk_queue_lock();
    for (int i = 0; i < k; i++) {
        pend[i].si.pCommandBufferInfos = pend[i].si.pCommandBufferInfos == pend[i].cbi ? pend[i].cbi : pend[i].cbi + 1;
        pend[i].si.pSignalSemaphoreInfos = &pend[i].sig;
        VkResult r = vkQueueSubmit2(c->gfx_queue, 1, &pend[i].si, VK_NULL_HANDLE);
        if (r != VK_SUCCESS) vk_device_fault_report(r), rt_fatal("vk: vkQueueSubmit2 of held value %llu failed (%d)", (unsigned long long)pend[i].fence, (int)r);
    }
    vk_queue_unlock();
    memmove(pend, pend + k, (size_t)(npend - k) * sizeof *pend);
    npend -= k;
}
static void pend_flush(uint64_t value) {
    if (gpu_delay <= 0) return;
    AcquireSRWLockExclusive(&pend_lock);
    pend_flush_upto(value);
    ReleaseSRWLockExclusive(&pend_lock);
}

uint64_t vk_sync_submit(const uint64_t *handles, int n) {
    vk_sync_start();
    VkCtx *c = vk_ctx();
    AcquireSRWLockExclusive(&sub_lock);
    ft_set(4);   // OOT_FENCE_TRACE: sub_lock held; a stall at 4 blocks every other submit
    vk_mem_dirty_submit();   // OOT_TRACE=perf: what a mirror would upload before this submit
    uint64_t fence = __atomic_load_n(&issued, __ATOMIC_ACQUIRE) + 1;
    cur_fence = fence;
    if (ftrace > 0) __atomic_store_n(&ft_cur_fence, (long long)fence, __ATOMIC_RELEASE);
    // Noted before the recording, not after the submit: an object another thread releases while this submission is
    // recorded (a texture replaced at its address, a sweep of released textures) may be named by it, and deferred
    // destruction (vk_memory.c collect_locked) must then wait for this fence, not the previous one. FE3H's battles
    // recreate transient textures while the render thread records; the GPU then wrote into destroyed images.
    vk_mem_note_submit(fence);
    CmdSlot *s = take_slot(c, fence);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    uint64_t t0 = perf_begin();
    vkBeginCommandBuffer(s->cb, &bi);
    vk_crumb(s->cb, VK_CRUMB_SUBMIT, fence);
    ts_begin(s->cb);
    rec_begin(s->cb);
    ft_set(5);   // recording
    int recorded = vk_exec_record(s->cb, handles, n, fence);
    ft_set(4);
    s = &slots[cur_slot];   // the last part (vk_sync_split)
    if (recorded) ts_end(s->cb, fence);
    else ts_drop();
    // The recorder is done with the part: the end-of-submission crumb, then the end, by the recorder when it records.
    if (!rec_end_cb(s->cb, VK_CRUMB_SUBMIT_END, fence)) vkEndCommandBuffer(s->cb);
    perf_replay(t0);
    // After the recording: the uploads it asked for (and every earlier one) precede it in queue order.
    ft_set(6);   // uploading
    vk_upload_flush(0);
    ft_set(7);   // prologue
    VkCommandBuffer pro = prologue(c, fence);
    ft_set(4);
    s = &slots[cur_slot];
    VkCommandBufferSubmitInfo cbi[2] = {{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = pro},
                                        {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->cb}};
    VkSemaphoreSubmitInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = c->timeline,
                                 .value = fence, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                        .commandBufferInfoCount = (pro ? 1u : 0u) + (recorded ? 1u : 0u),
                        .pCommandBufferInfos = pro ? cbi : cbi + 1, .signalSemaphoreInfoCount = 1,
                        .pSignalSemaphoreInfos = &sig};
    if (gpu_delay < 0) {
        const char *e = cfg_get("OOT_GPU_DELAY");
        gpu_delay = e && *e && *e != '0';
        if (gpu_delay) rt_log("vk: submissions are held until a wait needs them (OOT_GPU_DELAY)");
    }
    if (gpu_delay > 0) {
        AcquireSRWLockExclusive(&pend_lock);
        if (npend == 64) pend_flush_upto(pend[0].fence);
        typeof(pend[0]) *e = &pend[npend++];
        memcpy(e->cbi, cbi, sizeof e->cbi);
        e->sig = sig;
        e->si = si;
        e->si.pCommandBufferInfos = pro ? e->cbi : e->cbi + 1;
        e->si.pSignalSemaphoreInfos = &e->sig;
        e->fence = fence;
        ReleaseSRWLockExclusive(&pend_lock);
    } else {
        ft_set(8);   // vkQueueSubmit2
        vk_queue_lock();
        VkResult r = vkQueueSubmit2(c->gfx_queue, 1, &si, VK_NULL_HANDLE);
        vk_queue_unlock();
        ft_set(4);
        if (r != VK_SUCCESS) {
            vk_device_fault_report(r);
            rt_fatal("vk: vkQueueSubmit2 of value %llu failed (%d)", (unsigned long long)fence, (int)r);
        }
    }
    s->fence = recorded ? fence : 0;
    vk_mem_note_submit(fence);
    AcquireSRWLockExclusive(&st_lock);
    __atomic_store_n(&issued, fence, __ATOMIC_RELEASE);
    WakeAllConditionVariable(&cv_work);
    ReleaseSRWLockExclusive(&st_lock);
    static unsigned subs;
    if (++subs % 4096 == 0) rec_stats_log();
    ft_set(0);
    ReleaseSRWLockExclusive(&sub_lock);
    return fence;
}

void vk_completion_add(uint64_t fence, void (*fn)(void *), void *arg) {
    AcquireSRWLockExclusive(&st_lock);
    if (qhead + qn == qcap) {
        if (qhead) {
            memmove(queue, queue + qhead, sizeof *queue * (size_t)qn);
            qhead = 0;
        } else {
            int cap = qcap ? qcap * 2 : 256;
            Pending *q = realloc(queue, sizeof *q * (size_t)cap);
            if (!q) rt_fatal("vk: out of memory (completion queue)");
            queue = q, qcap = cap;
        }
    }
    queue[qhead + qn++] = (Pending){fence, fn, arg};
    // The completion thread sleeps only while nothing is issued beyond done: a callback for a later value (the replay's,
    // for the submission it records: counter reports, freed resources) becomes due through the submit, which wakes it.
    if (fence <= done) WakeAllConditionVariable(&cv_work);
    ReleaseSRWLockExclusive(&st_lock);
}

static int fence_wait_impl(uint64_t value, uint64_t timeout_ns);

int vk_sync_fence_wait(uint64_t value, uint64_t timeout_ns) {
    uint64_t p0 = perf_begin();
    int ok = fence_wait_impl(value, timeout_ns);
    perf_end(PERF_FENCE, p0);
    perf_gpu_wait(p0);
    return ok;
}

static int fence_wait_impl(uint64_t value, uint64_t timeout_ns) {
    vk_sync_start();
    pend_flush(value);   // OOT_GPU_DELAY: the held submissions this wait needs
    AcquireSRWLockExclusive(&st_lock);
    int ok;
    if (value > issued && value > reserved) {
        ok = 0;   // never submitted nor queued: never reached (as in the null back end)
    } else {
        uint64_t t0 = kern_now_ns();
        if (ftrace > 0 && value > done) {
            InterlockedIncrement64(&ft_waits);
            rt_log("fence: wait %llu issued=%llu done=%llu", (unsigned long long)value,
                   (unsigned long long)issued, (unsigned long long)done);
        }
        while (done < value) {
            DWORD ms = INFINITE;
            if (timeout_ns < (1ull << 62)) {
                uint64_t el = kern_now_ns() - t0;
                if (el >= timeout_ns) break;
                uint64_t left = (timeout_ns - el + 999999) / 1000000;
                ms = left >= INFINITE ? INFINITE - 1 : (DWORD)left;
            }
            SleepConditionVariableSRW(&cv_done, &st_lock, ms, 0);
        }
        ok = done >= value;
    }
    ReleaseSRWLockExclusive(&st_lock);
    return ok;
}

uint64_t vk_sync_fence_done(void) { return __atomic_load_n(&done, __ATOMIC_ACQUIRE); }

uint64_t vk_sync_issued(void) { return __atomic_load_n(&issued, __ATOMIC_ACQUIRE); }

void vk_sync_finish(void) {
    vk_sync_start();
    pend_flush(UINT64_MAX);
    vk_sync_fence_wait(__atomic_load_n(&issued, __ATOMIC_ACQUIRE), UINT64_MAX);
    vk_upload_flush(1);
}

void vk_sync_shutdown(void) {
    AcquireSRWLockExclusive(&start_lock);
    if (started) {
        vk_sync_finish();
        AcquireSRWLockExclusive(&st_lock);
        stop = 1;
        WakeAllConditionVariable(&cv_work);
        ReleaseSRWLockExclusive(&st_lock);
        WaitForSingleObject(completion_thread, INFINITE);
        CloseHandle(completion_thread);
        completion_thread = NULL;
        AcquireSRWLockExclusive(&st_lock);
        qhead = qn = 0;   // callbacks of values never submitted
        ReleaseSRWLockExclusive(&st_lock);
        AcquireSRWLockExclusive(&sub_lock);
        VkCtx *c = vk_ctx();
        if (c && cmd_pool) vkDestroyCommandPool(c->device, cmd_pool, NULL);
        perf_gpu_stats = NULL;
        if (c && ts_pool) vkDestroyQueryPool(c->device, ts_pool, NULL);
        ts_pool = VK_NULL_HANDLE;
        ts_state = 0;
        ts_last_end = 0;
        memset((void *)ts_busy, 0, sizeof ts_busy);
        cmd_pool = VK_NULL_HANDLE;
        free(slots);
        slots = NULL;
        nslots = 0;
        ReleaseSRWLockExclusive(&sub_lock);
        __atomic_store_n(&started, 0, __ATOMIC_RELEASE);
    }
    ReleaseSRWLockExclusive(&start_lock);
}
