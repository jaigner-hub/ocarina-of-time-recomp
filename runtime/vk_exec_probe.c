// From metroid-prime-4-recomp runtime/vk_exec_probe.c @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied with mp4 renamed to oot.
// From fe3h-pc-recomp runtime/vk_exec_probe.c @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied with fe3h renamed to mp4 only.
// From totk-pc-recomp runtime/vk_exec_probe.c @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), copied with totk renamed to fe3h only.
// oot.exe --probe-vkexec: the Vulkan replay, through gfx_vulkan_backend. Sync part: empty
// submissions get consecutive fence values; fence_wait and fence_done agree; completion callbacks run in submit order on
// the completion thread; a guest-memory word a callback writes is visible once fence_wait returns; finish drains every
// submission; a present of a cleared vk_texture.c image (layout GENERAL) reads back exactly through the swapchain
// capture. Clear part: a hand-built op stream (SET_RT to an RGBA8 and a D32 texture, whole-target clears,
// then scissored clears) through submit reads back exactly with read_texels.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "sm86_lower.h"
#include "platform/platform.h"

#define MAX_PATH 260   // the buffers keep the Windows sizes (MAX_PATH * 2) on both systems

// The last path separator of s, either kind (Windows executables have backslashes): NULL when there is none.
static char *path_sep(const char *s) {
    char *a = strrchr(s, '/'), *b = strrchr(s, '\\');
    return !a ? b : !b ? a : a > b ? a : b;
}

#define SUBMITS 100
#define TAIL 10
#define TEX_W 256
#define TEX_H 144
#define CHECK(cond, ...)                                                                                                \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            printf("probe-vkexec: FAILED: ");                                                                          \
            printf(__VA_ARGS__);                                                                                       \
            printf("\n");                                                                                              \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

static uint32_t main_tid;
static uint64_t ran[SUBMITS + TAIL + 8];
static long nran, on_main;

static void note(void *arg) {   // completion thread
    if (nran < (long)(sizeof ran / sizeof ran[0])) ran[nran] = (uint64_t)(uintptr_t)arg;
    nran++;
    if (plat_thread_id() == main_tid) on_main++;
}

typedef struct { volatile uint32_t *word; uint32_t value; } WordWrite;
static void write_word(void *arg) {
    const WordWrite *w = arg;
    *w->word = w->value;
}

static long ran_now(void) { return __atomic_load_n(&nran, __ATOMIC_ACQUIRE); }

static int test_sync(const GfxBackend *b, uint8_t *guest, uint64_t *first, uint64_t *last) {
    // An empty recording as the front end hands it over: {OP_BEGIN, OP_END}.
    static op_hdr empty[2] = {{OP_BEGIN, sizeof(op_hdr)}, {OP_END, sizeof(op_hdr)}};
    uint64_t handles[2] = {(uint64_t)(uintptr_t)empty, (uint64_t)(uintptr_t)empty};
    uint64_t prev = b->fence_done();
    *first = prev + 1;
    for (int i = 0; i < SUBMITS; i++) {
        vk_completion_add(prev + 1, note, (void *)(uintptr_t)(prev + 1));   // queued before the value exists, as vk_exec.c does
        uint64_t f = b->submit(i % 3 ? handles : NULL, i % 3);
        CHECK(f == prev + 1, "submission %d got fence %llu after %llu", i, (unsigned long long)f, (unsigned long long)prev);
        CHECK(b->fence_done() <= f, "fence_done %llu ahead of the newest submission %llu",
              (unsigned long long)b->fence_done(), (unsigned long long)f);
        CHECK(!b->fence_wait(f + 1, 0), "fence_wait of a value never submitted returned 1");
        if (i % 10 == 9) {
            CHECK(b->fence_wait(f, 5000000000ull), "fence_wait(%llu) timed out", (unsigned long long)f);
            CHECK(b->fence_done() >= f, "fence_done %llu after fence_wait(%llu)", (unsigned long long)b->fence_done(),
                  (unsigned long long)f);
            CHECK(ran_now() == i + 1, "%ld callbacks ran after fence_wait of submission %d", ran_now(), i);
        }
        prev = f;
    }
    // A guest-memory word written by a completion callback is visible to the waiter.
    volatile uint32_t *word = (volatile uint32_t *)(guest + 0x1000);
    *word = 0;
    static WordWrite ww;
    ww = (WordWrite){word, 0x5EC0DE01u};
    vk_completion_add(prev + 1, write_word, &ww);
    uint64_t f = b->submit(NULL, 0);
    CHECK(f == prev + 1, "word submission got fence %llu", (unsigned long long)f);
    CHECK(b->fence_wait(f, 5000000000ull), "fence_wait of the word submission timed out");
    CHECK(*word == 0x5EC0DE01u, "guest word is %#x after fence_wait", *word);
    prev = f;
    // finish drains everything.
    for (int i = 0; i < TAIL; i++) {
        vk_completion_add(prev + 1, note, (void *)(uintptr_t)(prev + 1));
        prev = b->submit(handles, 2);
    }
    b->finish();
    CHECK(b->fence_done() == prev, "fence_done %llu after finish, newest %llu", (unsigned long long)b->fence_done(),
          (unsigned long long)prev);
    CHECK(ran_now() == SUBMITS + TAIL, "%ld of %d callbacks ran after finish", ran_now(), SUBMITS + TAIL);
    for (int i = 0; i < SUBMITS + TAIL; i++) {
        uint64_t want = *first + (uint64_t)i + (i >= SUBMITS);   // the word submission sits between the two runs
        CHECK(ran[i] == want, "callback %d ran for fence %llu, expected %llu", i, (unsigned long long)ran[i],
              (unsigned long long)want);
    }
    CHECK(!on_main, "%ld callbacks ran on the submitting thread", on_main);
    *last = prev;
    return 0;
}

static uint8_t to_byte(float f) { return (uint8_t)(f * 255.0f + 0.5f); }

static int test_present(const GfxBackend *b, uint8_t *guest, int *w_out, int *h_out) {
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x100000, .size = 0x100000,
                   .game_mem = guest + 0x100000};
    CHECK(b->pool_init(&pool), "pool_init of a CPU-visible pool");
    tex_t t;
    memset(&t, 0, sizeof t);
    t.magic = NVN_TEX_MAGIC;
    t.d.target = NVN_TEXTURE_TARGET_2D, t.d.format = NVN_FORMAT_RGBA8, t.d.width = TEX_W, t.d.height = TEX_H;
    t.d.depth = 1, t.d.levels = 1;
    nvn_tex_layout(&t.d, &t.L);
    CHECK(t.L.size <= pool.size, "texture storage %#llx exceeds the pool", (unsigned long long)t.L.size);
    t.pool = &pool, t.off = 0, t.mem = pool.mem, t.serial = 1;
    const VkTexImage *img = vk_texture_get(&t);
    CHECK(img && img->format == VK_FORMAT_R8G8B8A8_UNORM, "vk_texture_get of an RGBA8 texture");
    vk_texture_gpu_written(&t);   // the clear below is its content, not the storage
    const float rgba[4] = {0.2f, 0.4f, 0.6f, 1.0f};
    VkClearColorValue cv;
    memcpy(cv.float32, rgba, sizeof cv.float32);
    VkImageSubresourceRange rng = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkCommandBuffer cb = vk_upload_begin();
    vkCmdClearColorImage(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &rng);
    vk_upload_end();

    uint64_t frames = gfx_stats().frames;
    // the exact colour check needs the bilinear blit: FSR's approximate reciprocals move a flat colour by 1/255
    plat_setenv("OOT_UPSCALER", "bilinear");
    vk_present_capture(1);
    const int crop[4] = {0, 0, TEX_W, TEX_H};
    b->present(&t, crop, 0);
    b->finish();         // the GPU work so far; a queued present is not part of it (vk_backend.c vk_be_finish)
    vk_present_drain();  // the present: submitted and, with capture on, read back
    const uint8_t *px;
    int w, h;
    VkFormat fmt;
    CHECK(vk_present_captured(&px, &w, &h, &fmt), "the present was not captured");
    CHECK(gfx_stats().frames == frames + 1, "frame counter %llu after one present from %llu",
          (unsigned long long)gfx_stats().frames, (unsigned long long)frames);
    uint8_t want[4] = {0, 0, 0, 255};
    if (fmt == VK_FORMAT_B8G8R8A8_UNORM) want[0] = to_byte(rgba[2]), want[1] = to_byte(rgba[1]), want[2] = to_byte(rgba[0]);
    else if (fmt == VK_FORMAT_R8G8B8A8_UNORM) want[0] = to_byte(rgba[0]), want[1] = to_byte(rgba[1]), want[2] = to_byte(rgba[2]);
    else CHECK(0, "swapchain format %d is not UNORM RGBA/BGRA", (int)fmt);
    uint32_t wv;
    memcpy(&wv, want, 4);
    const uint32_t *p = (const uint32_t *)px;
    for (long k = 0, n = (long)w * h; k < n; k++)
        CHECK(p[k] == wv, "presented pixel %ld is %08x, expected %08x", k, p[k], wv);
    vk_present_capture(0);
    b->texture_release(&t);
    b->pool_fini(&pool);
    *w_out = w, *h_out = h;
    return 0;
}

// ---- clear part: a hand-built op stream through submit, read back with read_texels.

static uint8_t *put(uint8_t *p, uint32_t op, const void *payload, uint32_t n) {
    op_hdr h = {op, (uint32_t)((sizeof h + n + 7) & ~7u)};
    memset(p, 0, h.size);
    memcpy(p, &h, sizeof h);
    if (n) memcpy(p + sizeof h, payload, n);
    return p + h.size;
}

static void init_tex(tex_t *t, pool_t *pool, int64_t off, int format) {
    memset(t, 0, sizeof *t);
    t->magic = NVN_TEX_MAGIC;
    t->d.target = NVN_TEXTURE_TARGET_2D, t->d.format = format, t->d.width = TEX_W, t->d.height = TEX_H;
    t->d.depth = 1, t->d.levels = 1;
    nvn_tex_layout(&t->d, &t->L);
    t->pool = pool, t->off = off, t->mem = pool->mem + off, t->serial = 100 + (uint64_t)off;
}

// Pixel of the RGBA8 target after the stream: the scissored second clear inside the rectangle, the first elsewhere.
#define SC_X 40
#define SC_Y 24
#define SC_W 64
#define SC_H 32

static int test_clear(const GfxBackend *b, uint8_t *guest, int *n_inside) {
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x400000, .size = 0x200000,
                   .game_mem = guest + 0x400000};
    CHECK(b->pool_init(&pool), "pool_init of a CPU-visible pool for the clear part");
    static tex_t color, depth;   // payload texture pointers live as long as the submission
    init_tex(&color, &pool, 0, NVN_FORMAT_RGBA8);
    init_tex(&depth, &pool, 0x100000, NVN_FORMAT_DEPTH32F);
    CHECK(color.L.size <= 0x100000 && depth.L.size <= 0x100000, "clear-part textures exceed their pool halves");

    static _Alignas(8) uint8_t stream[4096];
    uint8_t *p = stream;
    p = put(p, OP_BEGIN, NULL, 0);
    p_set_rt rt;
    memset(&rt, 0, sizeof rt);
    rt.n = 1, rt.color[0] = &color, rt.depth = &depth;
    p = put(p, OP_SET_RT, &rt, sizeof rt);
    const p_clear_color c0 = {0, 0xf, {0.2f, 0.4f, 0.6f, 1.0f}};
    p = put(p, OP_CLEAR_COLOR, &c0, sizeof c0);   // whole target outside a pass: vkCmdClearColorImage
    const p_clear_ds d0 = {0.5f, 1, 0, 0xff};
    p = put(p, OP_CLEAR_DS, &d0, sizeof d0);      // vkCmdClearDepthStencilImage
    const p_scissor sc = {SC_X, SC_Y, SC_W, SC_H};
    p = put(p, OP_SCISSOR, &sc, sizeof sc);
    const p_clear_color c1 = {0, 0xf, {1.0f, 0.0f, 0.0f, 1.0f}};
    p = put(p, OP_CLEAR_COLOR, &c1, sizeof c1);   // scissored: a pass and vkCmdClearAttachments
    const p_clear_ds d1 = {0.25f, 1, 0, 0xff};
    p = put(p, OP_CLEAR_DS, &d1, sizeof d1);
    const p_scissor full = {0, 0, 1 << 20, 1 << 20};
    p = put(p, OP_SCISSOR, &full, sizeof full);   // leaves the replay's scissor unbounded for later submissions
    p = put(p, OP_END, NULL, 0);
    CHECK(p <= stream + sizeof stream, "op stream overflow");

    uint64_t handle = (uint64_t)(uintptr_t)stream;
    uint64_t f = b->submit(&handle, 1);
    CHECK(b->fence_wait(f, 5000000000ull), "fence_wait of the clear submission timed out");

    static uint32_t px[TEX_W * TEX_H];
    static float dpx[TEX_W * TEX_H];
    NVNcopyRegion r = {0, 0, 0, TEX_W, TEX_H, 1};
    b->read_texels(&color, NULL, &r, (uint8_t *)px, TEX_W * 4, TEX_W * TEX_H * 4);
    b->read_texels(&depth, NULL, &r, (uint8_t *)dpx, TEX_W * 4, TEX_W * TEX_H * 4);
    const uint8_t out_b[4] = {to_byte(0.2f), to_byte(0.4f), to_byte(0.6f), 255}, in_b[4] = {255, 0, 0, 255};
    uint32_t out_v, in_v;
    memcpy(&out_v, out_b, 4), memcpy(&in_v, in_b, 4);
    int inside = 0;
    for (int y = 0; y < TEX_H; y++)
        for (int x = 0; x < TEX_W; x++) {
            int in = x >= SC_X && x < SC_X + SC_W && y >= SC_Y && y < SC_Y + SC_H;
            inside += in;
            uint32_t want = in ? in_v : out_v;
            float dwant = in ? 0.25f : 0.5f;
            CHECK(px[y * TEX_W + x] == want, "colour texel (%d,%d) is %08x, expected %08x", x, y, px[y * TEX_W + x], want);
            CHECK(dpx[y * TEX_W + x] == dwant, "depth texel (%d,%d) is %g, expected %g", x, y, dpx[y * TEX_W + x], dwant);
        }
    b->texture_release(&color);
    b->texture_release(&depth);
    b->pool_fini(&pool);
    *n_inside = inside;
    return 0;
}

// ---- draw part: the pipeline self-test, then a hand-built stream that clears an RGBA8 target, binds a census
// vertex + fragment pair and draws one triangle; a census compute shader is dispatched. The census (build/v121/shaders,
// tools/shader_census.py) is read the way vk_pipeline_selftest reads it: a pair is a fragment shader and a vertex
// shader of the same source file.

void vk_exec_draw_shutdown(void);   // vk_exec_draw.c: the stand-in resources; the device must be idle
void vk_exec_xfer_shutdown(void);   // vk_exec_xfer.c: the occlusion query pool; the device must be idle

typedef struct { const char *key, *c1, *src; size_t c1_len, src_len; int stage; uint32_t local[3]; } DrawEnt;

static const char *dr_field(const char *from, const char *to, const char *name, size_t *len) {
    char pat[32];
    snprintf(pat, sizeof pat, "\"%s\": \"", name);
    const char *p = strstr(from, pat);
    if (!p || p >= to) return NULL;
    p += strlen(pat);
    const char *e = memchr(p, '"', (size_t)(to - p));
    if (!e) return NULL;
    *len = (size_t)(e - p);
    return p;
}

static int dr_cmp(const void *x, const void *y) {
    const DrawEnt *a = x, *b = y;
    size_t n = a->src_len < b->src_len ? a->src_len : b->src_len;
    int c = memcmp(a->src, b->src, n);
    if (c) return c;
    if (a->src_len != b->src_len) return a->src_len < b->src_len ? -1 : 1;
    return a->stage - b->stage;
}

static int dr_parse(const char *jobs, DrawEnt *out, int max) {
    int n = 0;
    for (const char *p = jobs; n < max && (p = strstr(p, "\"job\": {")); p += 8) {
        const char *end = strstr(p, "\"origin\": ");
        if (!end) break;
        size_t kl, sl, cl, rl;
        const char *k = dr_field(p, end, "key", &kl), *s = dr_field(p, end, "stage", &sl), *c = dr_field(p, end, "c1", &cl),
                   *r = dr_field(p, end, "source", &rl);
        if (!k || !s || !c || !r || kl != 64) continue;
        DrawEnt *e = &out[n++];
        e->key = k, e->c1 = c, e->c1_len = cl / 2, e->src = r, e->src_len = rl;
        e->stage = !strncmp(s, "vs", sl) ? GFX_STAGE_VS : !strncmp(s, "ps", sl) ? GFX_STAGE_FS : GFX_STAGE_CS;
        e->local[0] = e->local[1] = e->local[2] = 1;
        const char *l = strstr(p, "\"local\": [");
        if (l && l < end) {
            l += 10;
            for (int i = 0; i < 3; i++) {
                while (*l == ' ' || *l == '\n' || *l == '\r' || *l == ',') l++;
                e->local[i] = (uint32_t)strtoul(l, (char **)&l, 10);
            }
        }
    }
    return n;
}

static char *dr_read(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) free(b), b = NULL;
    fclose(f);
    if (b) b[n] = 0, *size = n;
    return b;
}

static int dr_root(char *root, size_t n) {
    char exe[MAX_PATH * 2];
    if (!plat_exe_path(exe, sizeof exe)) return 0;
    for (char *cut = path_sep(exe); cut; cut = path_sep(exe)) {
        *cut = 0;
        char p[MAX_PATH * 2 + 40];
        snprintf(p, sizeof p, "%s/build/v121/shaders/jobs.json", exe);
        if (plat_file_exists(p)) return snprintf(root, n, "%s", exe) < (int)n;
    }
    return 0;
}

static int dr_hex(char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; }

// A program of one census shader as the game hands it to NVN: control section, code with its 0x30-byte header.
static prog_t *dr_prog(const char *root, const DrawEnt *e[], int n) {
    static uint64_t serial = 0x7000;
    prog_t *p = calloc(1, sizeof *p);
    p->magic = NVN_PROG_MAGIC, p->serial = ++serial;
    for (int i = 0; i < n; i++) {
        char path[MAX_PATH * 2 + 128];
        snprintf(path, sizeof path, "%s/build/v121/shaders/code/%.64s.code", root, e[i]->key);
        long body;
        char *raw = dr_read(path, &body);
        if (!raw) return free(p), NULL;
        size_t skip = e[i]->stage == GFX_STAGE_CS ? 0 : 0x30, total = skip + (size_t)body;
        uint8_t *code = calloc(1, total), *ctl = calloc(1, 0x880);
        memcpy(code + skip, raw, (size_t)body);
        free(raw);
        uint32_t c1_off = (uint32_t)(total - e[i]->c1_len), length = (uint32_t)total;
        uint32_t nvn_stage = e[i]->stage == GFX_STAGE_VS ? 0 : e[i]->stage == GFX_STAGE_FS ? 1 : 5;
        memcpy(ctl + 0x700, &c1_off, 4), memcpy(ctl + 0x704, &length, 4), memcpy(ctl + 0x714, &nvn_stage, 4);
        if (nvn_stage == 5) memcpy(ctl + 0x748, e[i]->local, 12);
        p->sd[i].data = (uint64_t)(uintptr_t)code, p->sd[i].control = ctl;
        p->nstage++;
    }
    return p;
}

#define DR_ATTRS 8
#define DR_STRIDE (DR_ATTRS * 16)

static int layouts_equal;   // texels compared equal between the two layouts (0: not compared)
static int ind_draw_texels, ind_dispatches;   // indirect draws' texels compared, indirect dispatches run

// The draw part's stream up to the draw: target cleared, viewport, program, three RGBA32F vertices at 0x80000 of the
// pool, identity uniforms at 0x90000.
static uint8_t *draw_prefix(uint8_t *q, prog_t *prog, const pool_t *pool, tex_t *color) {
    q = put(q, OP_BEGIN, NULL, 0);
    p_set_rt rt;
    memset(&rt, 0, sizeof rt);
    rt.n = 1, rt.color[0] = color;
    q = put(q, OP_SET_RT, &rt, sizeof rt);
    const p_clear_color c0 = {0, 0xf, {0.2f, 0.4f, 0.6f, 1.0f}};
    q = put(q, OP_CLEAR_COLOR, &c0, sizeof c0);
    p_viewports vp;
    memset(&vp, 0, sizeof vp);
    vp.n = 1, vp.v[0][2] = TEX_W, vp.v[0][3] = TEX_H, vp.v[0][5] = 1.0f;
    q = put(q, OP_VIEWPORTS, &vp, sizeof vp);
    const p_bind_program bp = {prog, 3};
    q = put(q, OP_BIND_PROGRAM, &bp, sizeof bp);
    p_bind_vattr va;
    memset(&va, 0, sizeof va);
    va.n = DR_ATTRS;
    for (int a = 0; a < DR_ATTRS; a++) {
        va_st s = {NVN_FORMAT_RGBA32F, (uint32_t)(16 * a), 0};
        memcpy(&va.st[a], &s, sizeof s);
    }
    q = put(q, OP_BIND_VATTR, &va, sizeof va);
    p_bind_vstream vs;
    memset(&vs, 0, sizeof vs);
    vs.n = 1, vs.st[0].stride = DR_STRIDE;
    q = put(q, OP_BIND_VSTREAM, &vs, sizeof vs);
    p_bind_vbuf vb;
    memset(&vb, 0, sizeof vb);
    vb.n = 1, vb.r[0].address = (uint64_t)(uintptr_t)pool->mem + 0x80000, vb.r[0].size = 3 * DR_STRIDE;
    q = put(q, OP_BIND_VBUF, &vb, sizeof vb);
    for (int stage = 0; stage < 2; stage++)
        for (int i = 0; i < 4; i++) {
            p_bind_buf bu = {stage, i, (uint64_t)(uintptr_t)pool->mem + 0x90000, 4096};
            q = put(q, OP_BIND_UBO, &bu, sizeof bu);
        }
    return q;
}

static int test_draw(const GfxBackend *b, uint8_t *guest, int *n_drawn, int *n_tries, int *n_cs) {
    char root[MAX_PATH * 2], path[MAX_PATH * 2 + 64];
    if (!dr_root(root, sizeof root)) {   // TotK's shader census (tools/shader_census.py), which FE3H does not carry
        printf("probe-vkexec: draw part skipped: build/v121/shaders/jobs.json (the TotK census) is not present\n");
        return 0;
    }
    snprintf(path, sizeof path, "%s/build/v121/shaders/jobs.json", root);
    long size;
    char *jobs = dr_read(path, &size);
    DrawEnt *e = malloc(sizeof *e * 20000);
    int n = jobs && e ? dr_parse(jobs, e, 20000) : 0;
    CHECK(n > 0, "no census entries");
    qsort(e, (size_t)n, sizeof *e, dr_cmp);
    int (*pairs)[2] = malloc(sizeof *pairs * (size_t)n), np = 0, *cs = malloc(sizeof *cs * (size_t)n), nc = 0;
    for (int i = 0; i < n;) {
        int j = i, v0 = -1, nv = 0, nf = 0;
        for (; j < n && e[j].src_len == e[i].src_len && !memcmp(e[j].src, e[i].src, e[i].src_len); j++) {
            if (e[j].stage == GFX_STAGE_VS && v0 < 0) v0 = j;
            nv += e[j].stage == GFX_STAGE_VS;
            if (e[j].stage == GFX_STAGE_CS) cs[nc++] = j;
        }
        for (int k = i; k < j && nv; k++)
            if (e[k].stage == GFX_STAGE_FS) pairs[np][0] = v0 + nf++ % nv, pairs[np][1] = k, np++;
        i = j;
    }
    CHECK(np > 0 && nc > 0, "census has %d pairs and %d compute shaders", np, nc);

    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x700000, .size = 0xC0000,
                   .game_mem = guest + 0x700000};
    CHECK(b->pool_init(&pool), "pool_init of a CPU-visible pool for the draw part");
    static tex_t color;
    init_tex(&color, &pool, 0, NVN_FORMAT_RGBA8);
    CHECK(color.L.size <= 0x80000, "draw-part target exceeds its pool slot");
    // Vertex data at 0x80000: three vertices of DR_ATTRS RGBA32F attributes, attribute 0 the position. Uniform data at
    // 0x90000: identity matrices back to back, so that a vertex shader multiplying the position by one passes it through.
    float *vtx = (float *)(pool.mem + 0x80000);
    static const float pos[3][4] = {{-0.8f, -0.8f, 0.0f, 1.0f}, {0.8f, -0.8f, 0.0f, 1.0f}, {0.0f, 0.8f, 0.0f, 1.0f}};
    for (int v = 0; v < 3; v++)
        for (int a = 0; a < DR_ATTRS; a++)
            for (int c = 0; c < 4; c++) vtx[(v * DR_ATTRS + a) * 4 + c] = a == 0 ? pos[v][c] : c == 3 ? 1.0f : 0.5f;
    float *ub = (float *)(pool.mem + 0x90000);
    for (int m = 0; m < 256; m++)
        for (int i = 0; i < 16; i++) ub[m * 16 + i] = i % 5 == 0 ? 1.0f : 0.0f;

    const uint8_t clear_b[4] = {to_byte(0.2f), to_byte(0.4f), to_byte(0.6f), 255};
    uint32_t clear_v;
    memcpy(&clear_v, clear_b, 4);
    static uint32_t px[TEX_W * TEX_H];
    static _Alignas(8) uint8_t stream[8192];
    int tries = 0, drawn = 0;
    prog_t *hit = NULL;
    for (int t = 0; t < 150 && !drawn; t++) {
        const int *pr = pairs[(long long)t * np / (np < 150 ? np : 150)];
        if (t && pr == pairs[(long long)(t - 1) * np / (np < 150 ? np : 150)]) continue;
        const DrawEnt *both[2] = {&e[pr[0]], &e[pr[1]]};
        prog_t *prog = dr_prog(root, both, 2);
        CHECK(prog, "census code file of %.16s is missing", both[0]->key);
        tries++;
        uint8_t *q = draw_prefix(stream, prog, &pool, &color);
        const p_draw d = {NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3, 0, 0};
        q = put(q, OP_DRAW, &d, sizeof d);
        q = put(q, OP_END, NULL, 0);
        CHECK(q <= stream + sizeof stream, "draw op stream overflow");
        uint64_t handle = (uint64_t)(uintptr_t)stream;
        uint64_t f = b->submit(&handle, 1);
        CHECK(b->fence_wait(f, 20000000000ull), "fence_wait of the draw submission timed out");
        const NVNcopyRegion r = {0, 0, 0, TEX_W, TEX_H, 1};
        b->read_texels(&color, NULL, &r, (uint8_t *)px, TEX_W * 4, TEX_W * TEX_H * 4);
        int diff = 0;
        for (int k = 0; k < TEX_W * TEX_H; k++) diff += px[k] != clear_v;
        if (diff) drawn = diff, hit = prog;
        if (diff && vk_caps.gpl) {   // the same draw through the shared set 0 layout: texel-equal
            static uint32_t px2[TEX_W * TEX_H];
            int was = vk_pipeline_set_indep(0);
            f = b->submit(&handle, 1);
            CHECK(b->fence_wait(f, 20000000000ull), "fence_wait of the shared-layout draw timed out");
            b->read_texels(&color, NULL, &r, (uint8_t *)px2, TEX_W * 4, TEX_W * TEX_H * 4);
            vk_pipeline_set_indep(was);
            int ne = 0;
            for (int k = 0; k < TEX_W * TEX_H; k++) ne += px[k] != px2[k];
            CHECK(!ne, "%d of %d texels differ between independent and shared layouts", ne, TEX_W * TEX_H);
            layouts_equal = TEX_W * TEX_H;
        }
    }
    CHECK(drawn, "none of %d census pairs changed a pixel of the cleared target", tries);

    // DrawElementsIndirect (OP_DRAW_IDX_IND) with the pair that drew: indices {0, 1, 2} at 0xA0000; the arguments are
    // copied by the GPU (COPY_B2B, barrier) to 0xA0200 right before the draw, so the replay must read them on the GPU.
    // {3, 1, 0, 0, 0} draws the same triangle (texel-equal to the direct draw); count 0 draws nothing.
    uint32_t *ix = (uint32_t *)(pool.mem + 0xA0000), *args = (uint32_t *)(pool.mem + 0xA0100);
    ix[0] = 0, ix[1] = 1, ix[2] = 2;
    for (int pass = 0; pass < 2; pass++) {
        const uint32_t want[5] = {pass ? 0u : 3u, 1, 0, 0, 0};
        memcpy(args, want, sizeof want);
        memset(pool.mem + 0xA0200, 0, 20);
        uint8_t *q = draw_prefix(stream, hit, &pool, &color);
        const p_copy_b2b cp = {(uint64_t)(uintptr_t)args, (uint64_t)(uintptr_t)(pool.mem + 0xA0200), 20};
        q = put(q, OP_COPY_B2B, &cp, sizeof cp);
        const p_barrier bar = {0, 0};
        q = put(q, OP_BARRIER, &bar, sizeof bar);
        const p_draw_idx_ind di = {NVN_DRAW_PRIMITIVE_TRIANGLES, NVN_INDEX_TYPE_UNSIGNED_INT, (uint64_t)(uintptr_t)ix,
                                   (uint64_t)(uintptr_t)(pool.mem + 0xA0200)};
        q = put(q, OP_DRAW_IDX_IND, &di, sizeof di);
        q = put(q, OP_END, NULL, 0);
        uint64_t handle = (uint64_t)(uintptr_t)stream;
        uint64_t f = b->submit(&handle, 1);
        CHECK(b->fence_wait(f, 20000000000ull), "fence_wait of the indirect draw timed out");
        static uint32_t pxi[TEX_W * TEX_H];
        const NVNcopyRegion r = {0, 0, 0, TEX_W, TEX_H, 1};
        b->read_texels(&color, NULL, &r, (uint8_t *)pxi, TEX_W * 4, TEX_W * TEX_H * 4);
        int ne = 0;
        for (int k = 0; k < TEX_W * TEX_H; k++) ne += pxi[k] != (pass ? clear_v : px[k]);
        CHECK(!ne, "indirect draw with count %u: %d of %d texels differ from the %s", want[0], ne, TEX_W * TEX_H,
              pass ? "cleared target" : "direct draw");
        ind_draw_texels += TEX_W * TEX_H;
    }

    // Compute: a few census compute shaders, one dispatch each, resources unbound (zero buffers and stand-in images).
    int dispatched = 0;
    for (int t = 0; t < 4 && t < nc; t++) {
        const DrawEnt *one[1] = {&e[cs[(long long)t * nc / (nc < 4 ? nc : 4)]]};
        prog_t *prog = dr_prog(root, one, 1);
        CHECK(prog, "census code file of %.16s is missing", one[0]->key);
        uint8_t *q = stream;
        q = put(q, OP_BEGIN, NULL, 0);
        const p_bind_program bp = {prog, 1 << 5};
        q = put(q, OP_BIND_PROGRAM, &bp, sizeof bp);
        const p_dispatch dp = {1, 1, 1, 0};
        q = put(q, OP_DISPATCH, &dp, sizeof dp);
        // DispatchComputeIndirect: group counts {1, 1, 1} copied by the GPU to 0xA0500 first
        const uint32_t groups[3] = {1, 1, 1};
        memcpy(pool.mem + 0xA0400, groups, sizeof groups);
        memset(pool.mem + 0xA0500, 0, sizeof groups);
        const p_copy_b2b cp = {(uint64_t)(uintptr_t)(pool.mem + 0xA0400), (uint64_t)(uintptr_t)(pool.mem + 0xA0500), 12};
        q = put(q, OP_COPY_B2B, &cp, sizeof cp);
        const p_barrier bar = {0, 0};
        q = put(q, OP_BARRIER, &bar, sizeof bar);
        const p_dispatch_ind dpi = {(uint64_t)(uintptr_t)(pool.mem + 0xA0500), 0};
        q = put(q, OP_DISPATCH_IND, &dpi, sizeof dpi);
        q = put(q, OP_END, NULL, 0);
        uint64_t handle = (uint64_t)(uintptr_t)stream;
        uint64_t f = b->submit(&handle, 1);
        CHECK(b->fence_wait(f, 20000000000ull), "fence_wait of the dispatch submission timed out");
        dispatched++;
        ind_dispatches++;
    }
    b->texture_release(&color);
    b->pool_fini(&pool);
    free(pairs), free(cs), free(e), free(jobs);
    *n_drawn = drawn, *n_tries = tries, *n_cs = dispatched;
    return 0;
}

// ---- xfer part: copies, texture clear, event and counter through submit.

#define XF_W 64
#define XF_H 64
#define XF_BYTES (XF_W * XF_H * 4)

// Texture in the xfer pool at `off`.
static int xf_tex(tex_t *t, pool_t *pool, int64_t off, uint64_t serial) {
    init_tex(t, pool, off, NVN_FORMAT_RGBA8);
    t->d.width = XF_W, t->d.height = XF_H;
    nvn_tex_layout(&t->d, &t->L);
    t->serial = serial;
    return t->L.size <= 0x20000;
}

static int xf_submit(const GfxBackend *b, uint8_t *stream, uint8_t *end) {
    end = put(end, OP_END, NULL, 0);
    (void)stream;
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    uint64_t f = b->submit(&handle, 1);
    return b->fence_wait(f, 5000000000ull);
}

static int test_xfer(const GfxBackend *b, uint8_t *guest, int *bytes) {
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x800000, .size = 0x100000,
                   .game_mem = guest + 0x800000};
    CHECK(b->pool_init(&pool), "pool_init of a CPU-visible pool for the xfer part");
    uint8_t *m = pool.mem;
    static tex_t ta, tb, tc;
    CHECK(xf_tex(&ta, &pool, 0x40000, 201) && xf_tex(&tb, &pool, 0x60000, 202) && xf_tex(&tc, &pool, 0x80000, 203),
          "xfer textures exceed their pool slots");
    // Linear buffers: B2B source 0x0, target 0x1000 (4 KiB); texture source 0x2000, read-backs 0x6000 and 0xA000.
    for (int i = 0; i < 4096; i++) m[i] = (uint8_t)(i * 7 + 3), m[0x1000 + i] = 0;
    uint32_t *img = (uint32_t *)(m + 0x2000);
    for (int i = 0; i < XF_W * XF_H; i++) img[i] = 0x01010101u * (uint32_t)(i & 0xff) + (uint32_t)i * 0x00010000u;
    memset(m + 0x6000, 0, XF_BYTES), memset(m + 0xA000, 0, XF_BYTES);
    volatile uint32_t *event = (volatile uint32_t *)(m + 0xE000);
    NVNcounterData *ctr = (NVNcounterData *)(m + 0xE100), *sp = (NVNcounterData *)(m + 0xE200);
    *event = 0, ctr->counter = 99, ctr->timestamp = 0;
    sp->counter = 99, sp->timestamp = 0;

    static _Alignas(8) uint8_t stream[4096];
    uint8_t *p = stream;
    p = put(p, OP_BEGIN, NULL, 0);
    const p_copy_b2b bb = {(uint64_t)(uintptr_t)m, (uint64_t)(uintptr_t)(m + 0x1000), 4096};
    p = put(p, OP_COPY_B2B, &bb, sizeof bb);
    p_copy_b2t up;
    memset(&up, 0, sizeof up);
    up.src = (uint64_t)(uintptr_t)(m + 0x2000), up.tex = &ta, up.r = (NVNcopyRegion){0, 0, 0, XF_W, XF_H, 1};
    p = put(p, OP_COPY_B2T, &up, sizeof up);
    p_copy_b2t down = up;
    down.src = (uint64_t)(uintptr_t)(m + 0x6000);
    p = put(p, OP_COPY_T2B, &down, sizeof down);
    p_copy_t2t tt;
    memset(&tt, 0, sizeof tt);
    tt.src = &ta, tt.dst = &tb, tt.sr = tt.dr = up.r;
    p = put(p, OP_COPY_T2T, &tt, sizeof tt);
    down.tex = &tb, down.src = (uint64_t)(uintptr_t)(m + 0xA000);
    p = put(p, OP_COPY_T2B, &down, sizeof down);
    p_clear_tex ct;
    memset(&ct, 0, sizeof ct);
    ct.tex = &tc, ct.r = (NVNcopyRegion){0, 0, 0, XF_W, XF_H, 1}, ct.mask = 0xf;
    ct.c.f[0] = 0.2f, ct.c.f[1] = 0.4f, ct.c.f[2] = 0.6f, ct.c.f[3] = 1.0f;
    p = put(p, OP_CLEAR_TEX, &ct, sizeof ct);
    const p_signal_event ev1 = {event, NVN_EVENT_SIGNAL_MODE_WRITE, 7}, ev2 = {event, NVN_EVENT_SIGNAL_MODE_ADD, 5};
    p = put(p, OP_SIGNAL_EVENT, &ev1, sizeof ev1);
    p = put(p, OP_SIGNAL_EVENT, &ev2, sizeof ev2);
    const p_report_counter rc = {NVN_COUNTER_TYPE_TIMESTAMP, (uint64_t)(uintptr_t)ctr};
    p = put(p, OP_REPORT_COUNTER, &rc, sizeof rc);
    // Samples passed: a reset, no draw, a report: {0, time} through the occlusion-query path.
    const p_reset_counter rs = {NVN_COUNTER_TYPE_SAMPLES_PASSED};
    p = put(p, OP_RESET_COUNTER, &rs, sizeof rs);
    const p_report_counter rsp = {NVN_COUNTER_TYPE_SAMPLES_PASSED, (uint64_t)(uintptr_t)sp};
    p = put(p, OP_REPORT_COUNTER, &rsp, sizeof rsp);
    CHECK(p <= stream + sizeof stream - sizeof(op_hdr), "xfer op stream overflow");
    CHECK(xf_submit(b, stream, p), "fence_wait of the xfer submission timed out");

    for (int i = 0; i < 4096; i++) CHECK(m[0x1000 + i] == m[i], "B2B byte %d is %02x, expected %02x", i, m[0x1000 + i], m[i]);
    for (int i = 0; i < XF_W * XF_H; i++) {
        CHECK(((uint32_t *)(m + 0x6000))[i] == img[i], "B2T then T2B texel %d is %08x, expected %08x", i,
              ((uint32_t *)(m + 0x6000))[i], img[i]);
        CHECK(((uint32_t *)(m + 0xA000))[i] == img[i], "T2T then T2B texel %d is %08x, expected %08x", i,
              ((uint32_t *)(m + 0xA000))[i], img[i]);
    }
    static uint32_t px[XF_W * XF_H];
    const NVNcopyRegion r = {0, 0, 0, XF_W, XF_H, 1};
    b->read_texels(&ta, NULL, &r, (uint8_t *)px, XF_W * 4, XF_BYTES);
    for (int i = 0; i < XF_W * XF_H; i++) CHECK(px[i] == img[i], "read_texels of the copied texture, texel %d is %08x", i, px[i]);
    b->read_texels(&tc, NULL, &r, (uint8_t *)px, XF_W * 4, XF_BYTES);
    const uint8_t cl[4] = {to_byte(0.2f), to_byte(0.4f), to_byte(0.6f), 255};
    uint32_t clv;
    memcpy(&clv, cl, 4);
    for (int i = 0; i < XF_W * XF_H; i++) CHECK(px[i] == clv, "CLEAR_TEX texel %d is %08x, expected %08x", i, px[i], clv);
    CHECK(*event == 12, "event word is %u after WRITE 7 and ADD 5", *event);
    CHECK(ctr->counter == 0 && ctr->timestamp != 0, "counter report is {%llu, %llu}", (unsigned long long)ctr->counter,
          (unsigned long long)ctr->timestamp);
    CHECK(sp->counter == 0 && sp->timestamp != 0, "samples-passed report after a reset and no draw is {%llu, %llu}",
          (unsigned long long)sp->counter, (unsigned long long)sp->timestamp);
    b->texture_release(&ta);
    b->texture_release(&tb);
    b->texture_release(&tc);
    b->pool_fini(&pool);
    *bytes = 4096 + 2 * XF_BYTES;
    return 0;
}

// c0 as vk_c0_fill builds it (CPU only): compute storage and uniform tables, dispatch size, unbound slots zero; the
// vertex stage's point size clamp and base instance.
static int test_c0(void) {
    static p_bind_buf ssbo[16], ubo[16];
    static uint8_t bank[VK_C0_BYTES];
    ssbo[1] = (p_bind_buf){5, 1, 0x12340000, 0x400};
    ubo[4] = (p_bind_buf){5, 4, 0x55660000, 0x100};
    const uint32_t dim[3] = {3, 2, 1};
    uint32_t u[4];
    uint64_t a;
    vk_c0_fill(bank, 5, ssbo, ubo, dim, 7);
    memcpy(u, bank + 0x200, 12);
    CHECK(u[0] == 3 && u[1] == 2 && u[2] == 1, "c0 dispatch size is %u %u %u", u[0], u[1], u[2]);
    memcpy(&a, bank + 0x320, 8), memcpy(u, bank + 0x328, 4);
    CHECK(a == 0x12340000 && u[0] == 0x400, "c0 0x320 (storage slot 1) is {%#llx, %#x}", (unsigned long long)a, u[0]);
    memcpy(&a, bank + 0x250, 8), memcpy(u, bank + 0x258, 4);
    CHECK(a == 0x55660000 && u[0] == 0x100, "c0 0x250 (uniform slot 4) is {%#llx, %#x}", (unsigned long long)a, u[0]);
    for (int i = 0; i < 16; i++) CHECK(!bank[0x310 + i], "c0 0x310 (unbound storage slot 0) byte %d is %02x", i, bank[0x310 + i]);
    CHECK(!bank[0x644] && !bank[0x610], "c0 of a compute stage has graphics words");
    vk_c0_fill(bank, 0, ssbo, ubo, dim, 61);
    float ps;
    memcpy(&ps, bank + 0x610, 4), memcpy(u, bank + 0x644, 4);
    CHECK(ps == 2047.0f && u[0] == 61, "vertex c0: point size %g at 0x610, base instance %u at 0x644", ps, u[0]);
    memcpy(&a, bank + 0x120, 8);
    CHECK(a == 0x12340000 && !bank[0x200], "vertex c0: storage slot 1 at 0x120 is %#llx", (unsigned long long)a);
    return 0;
}

// A texture in a physical pool's memory that nothing wrote reads zero, as Switch pool memory starts zeroed
// (OOT_FIX_ZERO; the driver may return zeros without it, then this is a guard).
static int test_zero(const GfxBackend *b, uint8_t *guest, int *bytes) {
    pool_t phys = {.magic = NVN_POOL_MAGIC, .flags = 0x189, .mem = (uint8_t *)(uintptr_t)RT_GPU_VA_BASE, .size = 0x100000,
                   .game_mem = guest + 0xA00000};
    pool_t cpu = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x800000, .size = 0x100000, .game_mem = guest + 0x800000};
    CHECK(b->pool_init(&phys), "pool_init of a physical pool");
    CHECK(b->pool_init(&cpu), "pool_init of a CPU-visible pool for the zero part");
    static tex_t t;
    init_tex(&t, &phys, 0, NVN_FORMAT_RGBA8);
    t.d.width = 64, t.d.height = 64;
    nvn_tex_layout(&t.d, &t.L);
    t.serial = 301;
    memset(cpu.mem, 0xab, 64 * 64 * 4);
    static _Alignas(8) uint8_t stream[512];
    uint8_t *p = put(stream, OP_BEGIN, NULL, 0);
    p_copy_b2t down;
    memset(&down, 0, sizeof down);
    down.src = (uint64_t)(uintptr_t)cpu.mem, down.tex = &t, down.r = (NVNcopyRegion){0, 0, 0, 64, 64, 1};
    p = put(p, OP_COPY_T2B, &down, sizeof down);
    CHECK(xf_submit(b, stream, p), "fence_wait of the zero submission timed out");
    for (int i = 0; i < 64 * 64 * 4; i++) CHECK(!cpu.mem[i], "unwritten texel byte %d of a physical-pool texture is %02x", i, cpu.mem[i]);
    b->texture_release(&t);
    b->pool_fini(&cpu);
    b->pool_fini(&phys);
    *bytes = 64 * 64 * 4;
    return 0;
}

// ClearBuffer / ReportValue (OP_CLEAR_BUF), Downsample (a linear OP_COPY_T2T to half size) and the stencil view
// aspect. Fills: 64 bytes in a CPU-visible pool, a 4-byte report over the first word in the same submission (GPU order:
// the report wins), and 64 bytes in a CPU-invisible physical pool copied out with COPY_B2B. Downsample: a 64x64 RGBA8
// texture to 32x32, each texel the 2x2 average within 1. Stencil: views of a D24S8 texture pick the stencil aspect in
// STENCIL mode and depth otherwise, and a stencil-aspect sampled view of the image is created.
static int test_nvnx(const GfxBackend *b, uint8_t *guest, int *bytes) {
    pool_t cpu = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x800000, .size = 0x100000, .game_mem = guest + 0x800000};
    pool_t phys = {.magic = NVN_POOL_MAGIC, .flags = 0x189, .mem = (uint8_t *)(uintptr_t)RT_GPU_VA_BASE, .size = 0x100000,
                   .game_mem = guest + 0xA00000};
    CHECK(b->pool_init(&cpu) && b->pool_init(&phys), "pool_init of the nvnx pools");
    uint8_t *m = cpu.mem;
    memset(m, 0x11, 0x400);
    static tex_t src, dst;
    init_tex(&src, &cpu, 0x40000, NVN_FORMAT_RGBA8);
    src.d.width = 64, src.d.height = 64, src.serial = 401;
    nvn_tex_layout(&src.d, &src.L);
    init_tex(&dst, &cpu, 0x60000, NVN_FORMAT_RGBA8);
    dst.d.width = 32, dst.d.height = 32, dst.serial = 402;
    nvn_tex_layout(&dst.d, &dst.L);
    CHECK(src.L.size <= 0x20000 && dst.L.size <= 0x20000, "downsample textures exceed their pool slots");
    uint8_t *img = m + 0x1000, *out = m + 0x8000;
    for (int i = 0; i < 64 * 64 * 4; i++) img[i] = (uint8_t)((i * 37 + (i >> 8) * 11) & 0xff);
    memset(out, 0, 32 * 32 * 4);

    static _Alignas(8) uint8_t stream[2048];
    uint8_t *p = put(stream, OP_BEGIN, NULL, 0);
    const p_clear_buf f1 = {(uint64_t)(uintptr_t)m, 64, 0xdeadbeefu, 0}, rv = {(uint64_t)(uintptr_t)m, 4, 0x1234u, 0};
    p = put(p, OP_CLEAR_BUF, &f1, sizeof f1);
    p = put(p, OP_CLEAR_BUF, &rv, sizeof rv);
    const p_clear_buf f2 = {RT_GPU_VA_BASE + 0x100, 64, 0xa5a5a5a5u, 0};
    p = put(p, OP_CLEAR_BUF, &f2, sizeof f2);
    const p_copy_b2b back = {RT_GPU_VA_BASE + 0x100, (uint64_t)(uintptr_t)(m + 0x200), 64};
    p = put(p, OP_COPY_B2B, &back, sizeof back);
    p_copy_b2t up;
    memset(&up, 0, sizeof up);
    up.src = (uint64_t)(uintptr_t)img, up.tex = &src, up.r = (NVNcopyRegion){0, 0, 0, 64, 64, 1};
    p = put(p, OP_COPY_B2T, &up, sizeof up);
    p_copy_t2t ds;   // as nvnCommandBufferDownsample records it
    memset(&ds, 0, sizeof ds);
    ds.src = &src, ds.dst = &dst, ds.sr = (NVNcopyRegion){0, 0, 0, 64, 64, 1}, ds.dr = (NVNcopyRegion){0, 0, 0, 32, 32, 1};
    ds.flags = NVN_COPY_FLAGS_LINEAR_FILTER_BIT;
    p = put(p, OP_COPY_T2T, &ds, sizeof ds);
    p_copy_b2t down;
    memset(&down, 0, sizeof down);
    down.src = (uint64_t)(uintptr_t)out, down.tex = &dst, down.r = (NVNcopyRegion){0, 0, 0, 32, 32, 1};
    p = put(p, OP_COPY_T2B, &down, sizeof down);
    CHECK(xf_submit(b, stream, p), "fence_wait of the nvnx submission timed out");

    uint32_t w0;
    memcpy(&w0, m, 4);
    CHECK(w0 == 0x1234u, "ReportValue word after the fill is %#x", w0);
    for (int i = 1; i < 16; i++) {
        uint32_t w;
        memcpy(&w, m + 4 * i, 4);
        CHECK(w == 0xdeadbeefu, "ClearBuffer word %d is %#x", i, w);
    }
    CHECK(m[64] == 0x11, "ClearBuffer wrote past its 64 bytes (%02x)", m[64]);
    for (int i = 0; i < 64; i++) CHECK(m[0x200 + i] == 0xa5, "physical-pool fill byte %d is %02x", i, m[0x200 + i]);
    int worst = 0;
    for (int y = 0; y < 32; y++)
        for (int x = 0; x < 32; x++)
            for (int ch = 0; ch < 4; ch++) {
                int sum = 0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) sum += img[((2 * y + dy) * 64 + 2 * x + dx) * 4 + ch];
                int d = abs(out[(y * 32 + x) * 4 + ch] - (sum + 2) / 4);
                if (d > worst) worst = d;
            }
    CHECK(worst <= 1, "downsampled texel off the 2x2 average by %d", worst);

    static tex_t dsx;
    init_tex(&dsx, &cpu, 0x80000, NVN_FORMAT_DEPTH24_STENCIL8);
    dsx.d.width = 64, dsx.d.height = 64, dsx.serial = 403;
    nvn_tex_layout(&dsx.d, &dsx.L);
    const VkTexImage *di = vk_texture_get(&dsx);
    CHECK(di && (di->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) && (di->aspect & VK_IMAGE_ASPECT_DEPTH_BIT), "D24S8 image with both aspects");
    view_t sv, dv;
    memset(&sv, 0, sizeof sv), memset(&dv, 0, sizeof dv);
    sv.set_dsmode = 1, sv.dsmode = NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL;
    dv.set_dsmode = 1, dv.dsmode = NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH;
    CHECK(vk_exec_view_aspect(di, &sv, 1) == VK_IMAGE_ASPECT_STENCIL_BIT && vk_exec_view_aspect(di, &dv, 1) == VK_IMAGE_ASPECT_DEPTH_BIT &&
              vk_exec_view_aspect(di, &sv, 0) == VK_IMAGE_ASPECT_DEPTH_BIT, "view aspects of a D24S8 texture");
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = di->image,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = di->format,
                                .subresourceRange = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1}};
    VkImageView view = VK_NULL_HANDLE;
    CHECK(vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) == VK_SUCCESS, "stencil-aspect view of the D24S8 image");
    vkDestroyImageView(vk_ctx()->device, view, NULL);
    b->texture_release(&src);
    b->texture_release(&dst);
    b->texture_release(&dsx);
    b->pool_fini(&cpu);
    b->pool_fini(&phys);
    *bytes = 64 + 64 + 32 * 32 * 4;
    return 0;
}

// VRAM mirror (vk_memory.c; skipped in import mode): a pool given its mirror as on a first buffer use, after which
// buffer copies and fills read and write the mirror. The first use copies the whole pool; CPU writes between
// submissions reach the mirror through write watch; GPU writes reach guest memory by the fence, and only the bytes
// written: CPU bytes beside a GPU fill keep their values.
static int test_mirror(const GfxBackend *b, uint8_t *guest, int *bytes) {
    *bytes = 0;
    if (!vk_mirror_on) return 0;
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0xC00000, .size = 0x100000,
                   .game_mem = guest + 0xC00000};
    CHECK(b->pool_init(&pool), "pool_init of a CPU-visible pool for the mirror part");
    uint8_t *m = pool.mem;
    for (int i = 0; i < 4096; i++) m[i] = (uint8_t)(i * 13 + 1);
    memset(m + 0x1000, 0xEE, 0x5000);
    CHECK(vk_mem_mirror_probe((uint64_t)(uintptr_t)m), "the pool got no VRAM mirror");
    static _Alignas(8) uint8_t stream[1024];
    uint64_t a = (uint64_t)(uintptr_t)m;
    // 1: B2B in the mirror after its first-use copy, and a fill; both written back.
    uint8_t *p = put(stream, OP_BEGIN, NULL, 0);
    const p_copy_b2b c1 = {a, a + 0x1000, 4096};
    const p_clear_buf f1 = {a + 0x2000, 0x800, 0x5a5a5a5au, 0};
    p = put(p, OP_COPY_B2B, &c1, sizeof c1);
    p = put(p, OP_CLEAR_BUF, &f1, sizeof f1);
    CHECK(xf_submit(b, stream, p), "fence_wait of the first mirror submission timed out");
    for (int i = 0; i < 4096; i++) CHECK(m[0x1000 + i] == m[i], "mirror B2B byte %d is %02x, expected %02x", i, m[0x1000 + i], m[i]);
    for (int i = 0; i < 0x800; i++) CHECK(m[0x2000 + i] == 0x5a, "mirror fill byte %d is %02x", i, m[0x2000 + i]);
    for (int i = 0x800; i < 0x1000; i++) CHECK(m[0x2000 + i] == 0xEE, "byte %#x past the mirror fill is %02x", i, m[0x2000 + i]);
    // 2: the CPU rewrites the source; the next submission copies the new bytes.
    for (int i = 0; i < 4096; i++) m[i] = (uint8_t)(i * 7 + 5);
    p = put(stream, OP_BEGIN, NULL, 0);
    const p_copy_b2b c2 = {a, a + 0x3000, 4096};
    p = put(p, OP_COPY_B2B, &c2, sizeof c2);
    CHECK(xf_submit(b, stream, p), "fence_wait of the second mirror submission timed out");
    for (int i = 0; i < 4096; i++)
        CHECK(m[0x3000 + i] == m[i], "B2B byte %d after a CPU write is %02x, expected %02x (write not mirrored)", i,
              m[0x3000 + i], m[i]);
    // 3: CPU bytes in the page of a GPU fill, written before the submission, survive the write-back.
    memset(m + 0x4000, 0x11, 0x800);
    p = put(stream, OP_BEGIN, NULL, 0);
    const p_clear_buf f3 = {a + 0x4800, 0x800, 0x77777777u, 0};
    p = put(p, OP_CLEAR_BUF, &f3, sizeof f3);
    CHECK(xf_submit(b, stream, p), "fence_wait of the third mirror submission timed out");
    for (int i = 0; i < 0x800; i++) CHECK(m[0x4000 + i] == 0x11, "CPU byte %#x beside a fill is %02x", i, m[0x4000 + i]);
    for (int i = 0; i < 0x800; i++) CHECK(m[0x4800 + i] == 0x77, "fill byte %#x is %02x", i, m[0x4800 + i]);
    b->pool_fini(&pool);
    *bytes = 4096 + 0x1000 + 4096 + 0x1000;
    return 0;
}

int probe_vkexec(void) {
    layouts_equal = 0;
    main_tid = plat_thread_id();
    const GfxBackend *b = &gfx_vulkan_backend;
    uint64_t heap = 0;
    CHECK(mem_set_heap_size(16ull << 20, &heap) == 0 && heap, "SetHeapSize 16 MiB");
    uint8_t *guest = (uint8_t *)(uintptr_t)heap;
    uint64_t first = 0, last = 0;
    int w = 0, h = 0, inside = 0, xfer_bytes = 0;
    int bad = test_sync(b, guest, &first, &last) || test_present(b, guest, &w, &h);
    int sync_bad = bad;
    if (!bad) bad = test_clear(b, guest, &inside);
    int draw_bad = 0, drawn = 0, tries = 0, dispatched = 0;
    if (!bad) bad = draw_bad = vk_pipeline_selftest() || test_draw(b, guest, &drawn, &tries, &dispatched);
    if (!bad) bad = test_xfer(b, guest, &xfer_bytes);
    int zero_bytes = 0;
    if (!bad) bad = test_c0() || test_zero(b, guest, &zero_bytes);
    int nvnx_bytes = 0, mirror_bytes = 0;
    if (!bad) bad = test_nvnx(b, guest, &nvnx_bytes);
    if (!bad) bad = test_mirror(b, guest, &mirror_bytes);
    b->finish();
    vk_pipeline_shutdown();   // the pipelines and the draw part's stand-ins go before the device
    vk_exec_draw_shutdown();
    vk_exec_xfer_shutdown();
    vk_backend_shutdown();
    long errors = vk_validation_errors_total();
    if (bad || errors) {
        printf("probe-vkexec: FAILED (%s, %ld validation errors)\n",
               sync_bad ? "a sync check failed" : draw_bad ? "a draw check failed" : bad ? "a clear or xfer check failed" : "checks passed", errors);
        return 1;
    }
    printf("probe-vkexec: sync part passed (%d submissions, fences %llu..%llu, %d completion callbacks in submit order "
           "on the completion thread, guest word visible after fence_wait, finish drained, present %dx%d exact, "
           "0 validation errors)\n",
           SUBMITS + TAIL + 1, (unsigned long long)first, (unsigned long long)last, SUBMITS + TAIL, w, h);
    printf("probe-vkexec: clear part passed (RGBA8 + D32 %dx%d: image clears outside a pass, scissored attachment "
           "clears in a pass on %d texels, read back exact, 0 validation errors)\n",
           TEX_W, TEX_H, inside);
    printf("probe-vkexec: draw part passed (pipeline self-test; one triangle of census pair %d changed %d of %d texels of the "
           "cleared RGBA8 target; %d census compute shaders dispatched; 0 validation errors)\n",
           tries, drawn, TEX_W * TEX_H, dispatched);
    if (layouts_equal)
        printf("probe-vkexec: layout part passed (the same draw through independent sets 0 and 1 and the shared set 0: "
               "%d of %d texels equal)\n", layouts_equal, TEX_W * TEX_H);
    else
        printf("probe-vkexec: layout part skipped (no graphics pipeline library: shared set 0 only)\n");
    printf("probe-vkexec: xfer part passed (B2B between CPU-visible pools, B2T + T2B round trip and T2T of a %dx%d RGBA8 "
           "texture exact, CLEAR_TEX exact, event WRITE + ADD, timestamp and samples-passed reports in guest memory after "
           "fence_wait, %d bytes compared, 0 validation errors)\n",
           XF_W, XF_H, xfer_bytes);
    printf("probe-vkexec: indirect part passed (DrawElementsIndirect with GPU-copied arguments: count 3 texel-equal to the "
           "direct draw, count 0 draws nothing, %d texels compared; %d census compute shaders dispatched indirectly)\n",
           ind_draw_texels, ind_dispatches);
    printf("probe-vkexec: nvnx part passed (ClearBuffer then ReportValue in GPU order, a CPU-invisible pool fill copied out, "
           "Downsample 64x64 -> 32x32 within 1 of the 2x2 average, stencil view aspect of D24S8, %d bytes compared)\n",
           nvnx_bytes);
    printf("probe-vkexec: c0 and zero parts passed (c0 tables, dispatch size, point size and base instance at their offsets, "
           "unbound slots zero; %d bytes of an unwritten physical-pool texture read zero, OOT_FIX_ZERO=%s)\n",
           zero_bytes, cfg_value("OOT_FIX_ZERO"));
    if (vk_mirror_on)
        printf("probe-vkexec: mirror part passed (B2B and fill in a VRAM mirror written back by the fence, a CPU write "
               "between submissions mirrored, CPU bytes beside a fill kept, %d bytes compared)\n", mirror_bytes);
    else
        printf("probe-vkexec: mirror part skipped (memory mode import)\n");
    return 0;
}

// ---- tensor part (oot.exe --probe-tensor): sm86_lower.c's emulation of the SM86 tensor-core instructions (LDSM,
// HMMA) run on the host GPU against a CPU reference. A hand-assembled kernel of one 32-lane workgroup: every lane
// loads its 16 bytes of A (16x16 f16, row-major) and 8 bytes of B (16x8 f16, row-major) from storage buffer slot 0,
// stores them to shared memory (A row r at r*32, B row k at 512 + k*16), LDSM.x4 loads the A fragment from the row
// addresses the CPU tabulated at +768 (lane L: (L&15)*32 + (L&16)), LDSM.x2.trans the B fragment from +896
// (512 + (L&15)*16), HMMA.16816.F16 multiplies them into zero accumulators, a second HMMA adds the product to that in place, and the fragments go back to the
// buffer: D at +1024 (8 bytes per lane), A at +1280 (16 per lane), B at +1792 (8 per lane). The SM86 encodings
// are the ones sm86_lower.c decodes (NAK's); word 3 holds scheduling bits the lowering ignores.

typedef struct { uint32_t w[4]; } TnInsn;

static void tn_emit(TnInsn **p, uint32_t w0, uint32_t w1, uint32_t w2) {
    (*p)->w[0] = w0 | 0x7000u, (*p)->w[1] = w1, (*p)->w[2] = w2, (*p)->w[3] = 0x000fe200u;   // predicate PT
    (*p)++;
}

static uint16_t tn_f2h(float f) {   // the test's values are small integers: exact
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int exp = (int)((u >> 23) & 0xff) - 127 + 15;
    if (f == 0 || exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    uint32_t mant = u & 0x7fffff, h = sign | ((uint32_t)exp << 10) | (mant >> 13), rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return (uint16_t)h;
}

static float tn_h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff, u;
    if (exp == 0) u = sign | (mant ? (uint32_t)(((uint32_t)(mant) << 13) | 0x33800000u) : 0);   // subnormals: not exact, unused
    else if (exp == 31) u = sign | 0x7f800000u | (mant << 13);
    else u = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    memcpy(&f, &u, 4);
    return f;
}

#define TN_BUF 4096

static int probe_tensor_run(const GfxBackend *b, uint8_t *guest, int *n_bad) {
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x400000, .size = 0x100000,
                   .game_mem = guest + 0x400000};
    CHECK(b->pool_init(&pool), "pool_init of a CPU-visible pool for the tensor part");
    uint8_t *buf = pool.mem;
    memset(buf, 0, TN_BUF);
    // A[i][k] in -3..3, B[k][j] in -2..2: every partial sum is an integer below 2048, exact in f16 whatever the order.
    float A[16][16], B[16][8], D[16][8];
    for (int i = 0; i < 16; i++)
        for (int k = 0; k < 16; k++) A[i][k] = (float)((i * 3 + k * 5) % 7 - 3);
    for (int k = 0; k < 16; k++)
        for (int j = 0; j < 8; j++) B[k][j] = (float)((k * 2 + j * 3) % 5 - 2);
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 8; j++) {
            float s = 0;
            for (int k = 0; k < 16; k++) s += A[i][k] * B[k][j];
            D[i][j] = s;
        }
    uint16_t *ha = (uint16_t *)buf, *hb = (uint16_t *)(buf + 512);
    for (int i = 0; i < 16; i++)
        for (int k = 0; k < 16; k++) ha[i * 16 + k] = tn_f2h(A[i][k]);
    for (int k = 0; k < 16; k++)
        for (int j = 0; j < 8; j++) hb[k * 8 + j] = tn_f2h(B[k][j]);
    uint32_t *addr_a = (uint32_t *)(buf + 768), *addr_b = (uint32_t *)(buf + 896);
    for (uint32_t L = 0; L < 32; L++) addr_a[L] = (L & 15) * 32 + (L & 16), addr_b[L] = 512 + (L & 15) * 16;

    // The kernel. Registers: R0 lane, R1 = 16L, R2 = 8L, R3 = 4L, R4-7 A row piece, R8-9 B row piece, R10/R11 the
    // LDSM row addresses, R12-15 the A fragment, R16-17 the B fragment, R20-21 D. UR4:5 = the storage buffer's address
    // (bank 0 + 0x10, the NVN2 layout with no pushed constants).
    static _Alignas(256) TnInsn code[64];
    memset(code, 0, sizeof code);
    TnInsn *p = code;
    tn_emit(&p, 0x0919 | 0 << 16, 0, 0);                                       // S2R R0, SR_LANEID
    tn_emit(&p, 0x0824 | 1 << 16 | 0 << 24, 16, 0x078e00ffu);                  // IMAD R1, R0, 16, RZ
    tn_emit(&p, 0x0824 | 2 << 16 | 0 << 24, 8, 0x078e00ffu);                   // IMAD R2, R0, 8, RZ
    tn_emit(&p, 0x0824 | 3 << 16 | 0 << 24, 4, 0x078e00ffu);                   // IMAD R3, R0, 4, RZ
    tn_emit(&p, 0x0ab9 | 4 << 16 | 0xff << 24, 0x10 << 6, 5 << 9);             // ULDC.64 UR4, c[0][0x10]
    tn_emit(&p, 0x0981 | 4 << 16 | 1 << 24, 4, 0x081e8100u | 6 << 9);          // LDG.E.128 R4, [R1.U32 + UR4]
    tn_emit(&p, 0x0981 | 8 << 16 | 2 << 24, 4 | 512 << 8, 0x081e8100u | 5 << 9);    // LDG.E.64 R8, [R2 + UR4 + 512]
    tn_emit(&p, 0x0981 | 10 << 16 | 3 << 24, 4 | 768 << 8, 0x081e8100u | 4 << 9);   // LDG.E R10, [R3 + UR4 + 768]
    tn_emit(&p, 0x0981 | 11 << 16 | 3 << 24, 4 | 896 << 8, 0x081e8100u | 4 << 9);   // LDG.E R11, [R3 + UR4 + 896]
    tn_emit(&p, 0x0388 | 1 << 24, 4, 6 << 9);                                  // STS.128 [R1], R4
    tn_emit(&p, 0x0388 | 2 << 24, 8 | 512 << 8, 5 << 9);                       // STS.64 [R2 + 512], R8
    tn_emit(&p, 0x0b1d, 0, 0);                                                 // BAR.SYNC 0
    tn_emit(&p, 0x083b | 12 << 16 | 10 << 24, 0, 2 << 8);                      // LDSM.16.M88.4 R12, [R10]
    tn_emit(&p, 0x083b | 16 << 16 | 11 << 24, 0, 1 << 8 | 1 << 14);            // LDSM.16.MT88.2 R16, [R11]
    tn_emit(&p, 0x023c | 20 << 16 | 12 << 24, 16, 0xff | 8 << 8);              // HMMA.16816.F16 R20, R12, R16, RZ
    tn_emit(&p, 0x023c | 20 << 16 | 12 << 24, 16, 20 | 8 << 8);                // HMMA.16816.F16 R20, R12, R16, R20: 2AB
    tn_emit(&p, 0x0986 | 2 << 24, 20 | 1024 << 8, 0x0810a100u | 4 | 5 << 9);   // STG.E.64 [R2 + UR4 + 1024], R20
    tn_emit(&p, 0x0986 | 1 << 24, 12 | 1280 << 8, 0x0810a100u | 4 | 6 << 9);   // STG.E.128 [R1 + UR4 + 1280], R12
    tn_emit(&p, 0x0986 | 2 << 24, 16 | 1792 << 8, 0x0810a100u | 4 | 5 << 9);   // STG.E.64 [R2 + UR4 + 1792], R16
    // Diagnostics: the loaded A piece, the plain shared-memory round trips, the LDSM row addresses.
    tn_emit(&p, 0x0984 | 24 << 16 | 1 << 24, 0, 6 << 9);                       // LDS.128 R24, [R1]
    tn_emit(&p, 0x0984 | 28 << 16 | 2 << 24, 512 << 8, 5 << 9);                // LDS.64 R28, [R2 + 512]
    tn_emit(&p, 0x0986 | 1 << 24, 4 | 2048 << 8, 0x0810a100u | 4 | 6 << 9);    // STG.E.128 [R1 + UR4 + 2048], R4
    tn_emit(&p, 0x0986 | 1 << 24, 24 | 2560 << 8, 0x0810a100u | 4 | 6 << 9);   // STG.E.128 [R1 + UR4 + 2560], R24
    tn_emit(&p, 0x0986 | 2 << 24, 28 | 3072 << 8, 0x0810a100u | 4 | 5 << 9);   // STG.E.64 [R2 + UR4 + 3072], R28
    tn_emit(&p, 0x0986 | 2 << 24, 10 | 3328 << 8, 0x0810a100u | 4 | 5 << 9);   // STG.E.64 [R2 + UR4 + 3328], R10
    tn_emit(&p, 0x094d, 0, 0x03800000u);                                       // EXIT
    uint32_t code_end = (uint32_t)((p - code) * 16), code_size = 512;
    static uint8_t ctl[0x100];
    memset(ctl, 0, sizeof ctl);
    const uint32_t magic = SM86_CONTROL_MAGIC, ctl_size = SM86_CONTROL_MIN, local[3] = {32, 1, 1};
    memcpy(ctl, &magic, 4), memcpy(ctl + 0x30, &ctl_size, 4), memcpy(ctl + 0xd4, &ctl_size, 4);
    memcpy(ctl + 0x38, &code_end, 4), memcpy(ctl + 0x40, &code_size, 4), memcpy(ctl + 0x44, &code_size, 4);
    ctl[0x56] = 5, ctl[0xb2] = 1;                                              // compute; one storage buffer slot
    memcpy(ctl + 0x58, local, 12);
    static prog_t prog;
    memset(&prog, 0, sizeof prog);
    prog.magic = NVN_PROG_MAGIC, prog.serial = 0x7100, prog.nstage = 1;
    prog.sd[0].data = (uint64_t)(uintptr_t)code, prog.sd[0].control = ctl;

    static _Alignas(8) uint8_t stream[1024];
    uint8_t *q = stream;
    q = put(q, OP_BEGIN, NULL, 0);
    const p_bind_program bp = {&prog, 1 << 5};
    q = put(q, OP_BIND_PROGRAM, &bp, sizeof bp);
    const p_bind_buf sb = {5, 0, (uint64_t)(uintptr_t)buf, TN_BUF};
    q = put(q, OP_BIND_SSBO, &sb, sizeof sb);
    const p_dispatch dp = {1, 1, 1, 0};
    q = put(q, OP_DISPATCH, &dp, sizeof dp);
    const p_barrier bar = {0, 0};
    q = put(q, OP_BARRIER, &bar, sizeof bar);
    q = put(q, OP_END, NULL, 0);
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    uint64_t f = b->submit(&handle, 1);
    CHECK(b->fence_wait(f, 20000000000ull), "fence_wait of the tensor dispatch timed out");

    // Lane L (g = L >> 2, t = L & 3): A fragment R12 = (A[g][2t], A[g][2t+1]), R13 = A[g+8][2t..], R14 = A[g][2t+8..],
    // R15 = A[g+8][2t+8..]; B fragment R16 = (B[2t][g], B[2t+1][g]), R17 = (B[2t+8][g], B[2t+9][g]); D R20 =
    // (D[g][2t], D[g][2t+1]), R21 = (D[g+8][2t], D[g+8][2t+1]).
    int bad = 0, bad_a = 0, bad_b = 0, bad_d = 0, shown = 0, bad_ldg = 0, bad_lds = 0, bad_addr = 0;
    for (uint32_t L = 0; L < 32; L++) {
        const uint32_t *ldg = (const uint32_t *)(buf + 2048 + L * 16), *lds = (const uint32_t *)(buf + 2560 + L * 16);
        const uint32_t *ldsb = (const uint32_t *)(buf + 3072 + L * 8), *ad = (const uint32_t *)(buf + 3328 + L * 8);
        const uint32_t *fa = (const uint32_t *)(buf + 1280 + L * 16), *fb = (const uint32_t *)(buf + 1792 + L * 8);
        const uint32_t *fd = (const uint32_t *)(buf + 1024 + L * 8);
        bad_ldg += memcmp(ldg, buf + L * 16, 16) != 0;
        bad_lds += memcmp(lds, buf + L * 16, 16) != 0 || memcmp(ldsb, buf + 512 + L * 8, 8) != 0;
        bad_addr += ad[0] != addr_a[L] || ad[1] != addr_b[L];
        printf("probe-tensor: lane %2u ldg %08x %08x %08x %08x lds %08x %08x %08x %08x / %08x %08x addr %u %u | A %08x %08x %08x %08x | B %08x %08x | D %08x %08x\n",
               L, ldg[0], ldg[1], ldg[2], ldg[3], lds[0], lds[1], lds[2], lds[3], ldsb[0], ldsb[1], ad[0], ad[1],
               fa[0], fa[1], fa[2], fa[3], fb[0], fb[1], fd[0], fd[1]);
    }
    printf("probe-tensor: lanes with a wrong LDG piece %d, wrong shared round trip %d, wrong LDSM addresses %d\n", bad_ldg, bad_lds, bad_addr);
    for (uint32_t L = 0; L < 32; L++) {
        uint32_t g = L >> 2, t = L & 3;
        const uint16_t *fa = (const uint16_t *)(buf + 1280 + L * 16), *fb = (const uint16_t *)(buf + 1792 + L * 8);
        const uint16_t *fd = (const uint16_t *)(buf + 1024 + L * 8);
        float wa[8] = {A[g][2 * t], A[g][2 * t + 1], A[g + 8][2 * t], A[g + 8][2 * t + 1],
                       A[g][2 * t + 8], A[g][2 * t + 9], A[g + 8][2 * t + 8], A[g + 8][2 * t + 9]};
        float wb[4] = {B[2 * t][g], B[2 * t + 1][g], B[2 * t + 8][g], B[2 * t + 9][g]};
        float wd[4] = {2 * D[g][2 * t], 2 * D[g][2 * t + 1], 2 * D[g + 8][2 * t], 2 * D[g + 8][2 * t + 1]};   // twice
        for (int i = 0; i < 8; i++)
            if (tn_h2f(fa[i]) != wa[i]) {
                bad_a++;
                if (shown++ < 12) printf("probe-tensor: lane %u A half %d is %g (%04x), expected %g\n", L, i, tn_h2f(fa[i]), fa[i], wa[i]);
            }
        for (int i = 0; i < 4; i++)
            if (tn_h2f(fb[i]) != wb[i]) {
                bad_b++;
                if (shown++ < 12) printf("probe-tensor: lane %u B half %d is %g (%04x), expected %g\n", L, i, tn_h2f(fb[i]), fb[i], wb[i]);
            }
        for (int i = 0; i < 4; i++)
            if (tn_h2f(fd[i]) != wd[i]) {
                bad_d++;
                if (shown++ < 12) printf("probe-tensor: lane %u D half %d is %g (%04x), expected %g\n", L, i, tn_h2f(fd[i]), fd[i], wd[i]);
            }
    }
    printf("probe-tensor: A fragment %d of 256 halves wrong, B fragment %d of 128, D %d of 128\n", bad_a, bad_b, bad_d);
    bad = bad_a + bad_b + bad_d;
    b->pool_fini(&pool);
    *n_bad = bad;
    return 0;
}

int probe_tensor(void) {
    main_tid = plat_thread_id();
    const GfxBackend *b = &gfx_vulkan_backend;
    uint64_t heap = 0;
    CHECK(mem_set_heap_size(16ull << 20, &heap) == 0 && heap, "SetHeapSize 16 MiB");
    uint8_t *guest = (uint8_t *)(uintptr_t)heap;
    int n_bad = -1;
    int bad = probe_tensor_run(b, guest, &n_bad);
    b->finish();
    vk_pipeline_shutdown();
    vk_exec_draw_shutdown();
    vk_exec_xfer_shutdown();
    vk_backend_shutdown();
    long errors = vk_validation_errors_total();
    if (bad || n_bad || errors) {
        printf("probe-tensor: FAILED (%d fragment halves wrong, %ld validation errors)\n", n_bad, errors);
        return 1;
    }
    printf("probe-tensor: passed (LDSM.x4, LDSM.x2.trans and HMMA.16816.F16 fragments of a 16x16 by 16x8 product exact, "
           "0 validation errors)\n");
    return 0;
}

// ---- kernel part (oot.exe --probe-cs <dir>): one NVN2 compute kernel of the game dispatched in isolation with
// stand-in resources, to bisect a device loss or a wrong result outside the game (2026-11-06: the cinematic's first
// neural-network layer faulted the GPU). <dir> holds the control section (ctl), the code block (code) and bindings.txt:
//   ssbo <slot> <size> [<file>]      storage buffer slot: zeros, or the file's bytes (the game's, from OOT_DUMP_BUF)
//   tex <slot> <w> <h>               a texture slot: an RGBA8 2D texture of zeros (unbound slots get the dummy)
//   dispatch <x> <y> <z>
//   trunc <n>                        lower only the first n instructions (OOT_SM86_TRUNC) to bisect inside the kernel
// After the fence the storage buffers are written back as <dir>/out_ssbo<slot>.bin.

static uint8_t *cs_read(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = calloc(1, (size_t)*size + 16);
    if (b && fread(b, 1, (size_t)*size, f) != (size_t)*size) free(b), b = NULL;
    fclose(f);
    return b;
}

int probe_cs(const char *dir) {
    main_tid = plat_thread_id();
    const GfxBackend *b = &gfx_vulkan_backend;
    char path[MAX_PATH * 2];
    long ctl_size = 0, code_size = 0;
    snprintf(path, sizeof path, "%s/ctl", dir);
    uint8_t *ctl = cs_read(path, &ctl_size);
    snprintf(path, sizeof path, "%s/code", dir);
    uint8_t *code = cs_read(path, &code_size);
    CHECK(ctl && code, "cannot read %s/ctl and %s/code", dir, dir);
    snprintf(path, sizeof path, "%s/bindings.txt", dir);
    FILE *bf = fopen(path, "r");
    CHECK(bf, "cannot read %s", path);
    uint64_t heap = 0;
    CHECK(mem_set_heap_size(64ull << 20, &heap) == 0 && heap, "SetHeapSize 64 MiB");
    uint8_t *guest = (uint8_t *)(uintptr_t)heap;
    pool_t pool = {.magic = NVN_POOL_MAGIC, .flags = 0x22, .mem = guest + 0x400000, .size = 0x3c00000,
                   .game_mem = guest + 0x400000};
    CHECK(b->pool_init(&pool), "pool_init of a CPU-visible pool for the kernel part");
    static _Alignas(256) uint8_t code_buf[1 << 20];
    CHECK(code_size <= (long)sizeof code_buf, "code block too large");
    memcpy(code_buf, code, (size_t)code_size);
    static prog_t prog;
    memset(&prog, 0, sizeof prog);
    prog.magic = NVN_PROG_MAGIC, prog.serial = 0x7200, prog.nstage = 1;
    prog.sd[0].data = (uint64_t)(uintptr_t)code_buf, prog.sd[0].control = ctl;

    static _Alignas(8) uint8_t stream[4096];
    uint8_t *q = stream;
    q = put(q, OP_BEGIN, NULL, 0);
    const p_bind_program bp = {&prog, 1 << 5};
    q = put(q, OP_BIND_PROGRAM, &bp, sizeof bp);
    uint64_t off = 0x10000;
    struct { int slot; uint64_t off, size; } ssbo[16];
    int nssbo = 0;
    p_dispatch dp = {1, 1, 1, 0};
    char line[600];
    while (fgets(line, sizeof line, bf)) {
        int slot, x, y, z;
        unsigned long long size;
        char file[400];
        if (sscanf(line, "ssbo %d %llx %399s", &slot, &size, file) >= 2) {
            if (sscanf(line, "ssbo %d %llx %399s", &slot, &size, file) == 2) file[0] = 0;
            CHECK(nssbo < 16 && off + size <= pool.size, "too many or too large storage buffers");
            uint8_t *mem = pool.mem + off;
            memset(mem, 0, (size_t)size);
            if (file[0]) {
                long n = 0;
                snprintf(path, sizeof path, "%s/%s", dir, file);
                uint8_t *d = cs_read(path, &n);
                CHECK(d, "cannot read %s", path);
                memcpy(mem, d, (size_t)(n < (long)size ? n : (long)size));
                free(d);
                printf("probe-cs: ssbo slot %d: %ld bytes from %s\n", slot, n, file);
            }
            const p_bind_buf sb = {5, slot, (uint64_t)(uintptr_t)mem, size};
            q = put(q, OP_BIND_SSBO, &sb, sizeof sb);
            ssbo[nssbo].slot = slot, ssbo[nssbo].off = off, ssbo[nssbo].size = size, nssbo++;
            off = (off + size + 0xffff) & ~0xffffull;
        } else if (sscanf(line, "dispatch %d %d %d", &x, &y, &z) == 3) {
            dp.x = x, dp.y = y, dp.z = z;
        } else if (sscanf(line, "trunc %d", &x) == 1) {
            char v[32];
            snprintf(v, sizeof v, "%d", x);
            plat_setenv("OOT_SM86_TRUNC", v);
            printf("probe-cs: lowering truncated after %d instructions\n", x);
        }
    }
    fclose(bf);
    q = put(q, OP_DISPATCH, &dp, sizeof dp);
    const p_barrier bar = {0, 0};
    q = put(q, OP_BARRIER, &bar, sizeof bar);
    q = put(q, OP_END, NULL, 0);
    printf("probe-cs: dispatch %dx%dx%d of a %ld-byte kernel with %d storage buffers\n", dp.x, dp.y, dp.z, code_size, nssbo);
    fflush(stdout);
    uint64_t handle = (uint64_t)(uintptr_t)stream;
    uint64_t f = b->submit(&handle, 1);
    uint64_t q0 = plat_now_ns();
    int done = b->fence_wait(f, 60000000000ull);
    uint64_t q1 = plat_now_ns();
    printf("probe-cs: fence %s after %.1f ms (the pipeline build is before the submit)\n", done ? "reached" : "TIMED OUT",
           (double)(q1 - q0) / 1e6);
    for (int i = 0; i < nssbo && done; i++) {
        snprintf(path, sizeof path, "%s/out_ssbo%d.bin", dir, ssbo[i].slot);
        FILE *o = fopen(path, "wb");
        if (o) { fwrite(pool.mem + ssbo[i].off, 1, (size_t)ssbo[i].size, o); fclose(o); }
        uint64_t nz = 0;
        for (uint64_t k = 0; k < ssbo[i].size; k++) nz += pool.mem[ssbo[i].off + k] != 0;
        printf("probe-cs: ssbo slot %d: %llu of %llu bytes non-zero after the dispatch -> %s\n", ssbo[i].slot,
               (unsigned long long)nz, (unsigned long long)ssbo[i].size, path);
    }
    b->finish();
    vk_pipeline_shutdown();
    vk_exec_draw_shutdown();
    vk_exec_xfer_shutdown();
    vk_backend_shutdown();
    long errors = vk_validation_errors_total();
    printf("probe-cs: %s (%ld validation errors)\n", done ? "done" : "FAILED", errors);
    return done ? 0 : 1;
}
