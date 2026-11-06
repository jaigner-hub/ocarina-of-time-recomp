// From metroid-prime-4-recomp runtime/gfx_shader_probe.c @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied with mp4 renamed to oot.
// From fe3h-pc-recomp runtime/gfx_shader_probe.c @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied with fe3h renamed to mp4 only.
// From totk-pc-recomp runtime/gfx_shader_probe.c @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), copied with totk renamed to fe3h only.
// --probe-shader: the shader service client (gfx_shader.c) against 32 census shaders. Needs build/xlate/oot_xlate.exe and
// build/v131072/shaders/ (tools/shader_census.py); uses a throwaway cache directory.
//   1. Translate 32 shaders (mixed stages) through gfx_shader_get; each SPIR-V starts with the magic 0x07230203.
//   2. Drop the memory cache and ask again: 32 hits from disk, 0 service requests. A third pass returns the same pointers.
//   3. Send a malformed line straight to the service: it answers with an error line and still serves a valid request.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gfx.h"
#include "platform/platform.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define MAX_PATH 260   // the buffers keep the Windows sizes (MAX_PATH * 2) on both systems

// The last path separator of s, either kind (Windows executables have backslashes): NULL when there is none.
static char *path_sep(const char *s) {
    char *a = strrchr(s, '/'), *b = strrchr(s, '\\');
    return !a ? b : !b ? a : a > b ? a : b;
}

int gfx_shader_raw_request(const char *line, char *reply, size_t reply_size);
uint64_t gfx_shader_requests(void);
uint64_t gfx_shader_cache_hits(void);
void gfx_shader_drop_memory(void);

#define WANT 32
#define CANDIDATES 64                                      // census entries tried: a program the lowering cannot take yet is skipped
#define CHECK(cond, ...) do { if (!(cond)) { printf("probe-shader: FAILED: " __VA_ARGS__); printf("\n"); return 1; } } while (0)

typedef struct { char key[65]; int stage; uint32_t local[3]; } Entry;   // OoT: a tools/shader_census.py program

static char *slurp(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) free(b), b = NULL;
    fclose(f);
    if (b) b[n] = 0, *size = (size_t)n;
    return b;
}

// Value of `"name": "` inside [from, to), NULL when absent; *len is the value length.
static const char *str_field(const char *from, const char *to, const char *name, size_t *len) {
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

// The repository root: the nearest parent of oot.exe holding build/v131072/shaders/programs.json.
static int find_root(char *root, size_t n) {
    char exe[MAX_PATH * 2];
    if (!plat_exe_path(exe, sizeof exe)) return 0;
    for (char *cut = path_sep(exe); cut; cut = path_sep(exe)) {
        *cut = 0;
        char p[MAX_PATH * 2 + 40];
        snprintf(p, sizeof p, "%s/build/v131072/shaders/programs.json", exe);
        if (plat_file_exists(p)) return snprintf(root, n, "%s", exe) < (int)n;
    }
    return 0;
}

// programs.json: `"<key>": {` then the fields, one per line (json.dumps with indent 0); "stage" names the stage
// (vs, ps, cs, task, mesh...) and "local" is the compute workgroup size as a three-line list.
static int pick_entries(const char *jobs, Entry *out) {
    enum { MAXN = 20000 };
    Entry *all = malloc(sizeof(Entry) * MAXN);
    int n = 0, per_stage[3] = {0, 0, 0};
    for (const char *p = jobs; n < MAXN && (p = strstr(p, "\": {")); p += 4) {
        const char *end = strstr(p, "\n}");
        if (!end || p - jobs < 65 || p[-65] != '"') continue;
        size_t sl;
        const char *s = str_field(p, end, "stage", &sl);
        if (!s || (sl != 2 && !(sl == 2))) continue;
        if (strncmp(s, "vs", sl) && strncmp(s, "ps", sl) && strncmp(s, "cs", sl)) continue;
        Entry *e = &all[n];
        memcpy(e->key, p - 64, 64), e->key[64] = 0;
        e->stage = !strncmp(s, "vs", sl) ? GFX_STAGE_VS : !strncmp(s, "ps", sl) ? GFX_STAGE_FS : GFX_STAGE_CS;
        e->local[0] = e->local[1] = e->local[2] = 1;
        const char *l = strstr(p, "\"local\": [");
        if (l && l < end) {
            l += 10;
            for (int i = 0; i < 3; i++) {
                while (*l == ' ' || *l == '\n' || *l == '\r' || *l == ',') l++;
                e->local[i] = (uint32_t)strtoul(l, (char **)&l, 10);
                if (!e->local[i]) e->local[i] = 1;
            }
        }
        per_stage[e->stage == GFX_STAGE_VS ? 0 : e->stage == GFX_STAGE_FS ? 1 : 2]++;
        n++;
    }
    // 22 vertex, 22 fragment, 20 compute shaders, spread evenly over each stage's list; the first WANT that lower
    // and translate are the probe's
    int want[3] = {22, 22, 20}, taken[3] = {0, 0, 0}, seen[3] = {0, 0, 0}, got = 0;
    for (int i = 0; i < n; i++) {
        int s = all[i].stage == GFX_STAGE_VS ? 0 : all[i].stage == GFX_STAGE_FS ? 1 : 2;
        if (per_stage[s] < want[s]) want[s] = per_stage[s];
        if (taken[s] < want[s] && seen[s] == (int)((long long)taken[s] * per_stage[s] / want[s])) out[got++] = all[i], taken[s]++;
        seen[s]++;
    }
    free(all);
    return got;
}

static char root[MAX_PATH * 2];

// The program the game hands NVN for a census entry (its control section and code block as the archive holds
// them), translated.
static const GfxShader *translate(const Entry *e, int *err) {
    char path[MAX_PATH * 2 + 64];
    uint8_t control[0x400] = {0};
    snprintf(path, sizeof path, "%s/build/v131072/shaders/ctl/%s.ctl", root, e->key);
    size_t ctl_size, code_size;
    char *ctl = slurp(path, &ctl_size);
    if (!ctl || ctl_size > sizeof control) return free(ctl), *err = 1, NULL;
    memcpy(control, ctl, ctl_size);
    free(ctl);
    snprintf(path, sizeof path, "%s/build/v131072/shaders/code/%s.code", root, e->key);
    char *code = slurp(path, &code_size);
    if (!code) return *err = 1, NULL;
    GfxShaderAnswers a = {0};
    if (e->stage == GFX_STAGE_CS) for (int k = 0; k < 3; k++) a.local[k] = (uint16_t)e->local[k];
    const GfxShader *s = gfx_shader_get(e->stage, control, (const uint8_t *)code, code_size, NULL, &a);
    free(code);
    return s;
}

static int remove_one(const char *name, void *ctx) {
    char f[MAX_PATH * 2];
    snprintf(f, sizeof f, "%s/%s", (const char *)ctx, name);
    plat_file_delete(f);   // files only: the probe's folder has no subfolders
    return 1;
}

static void remove_dir(const char *dir) {
    plat_file_list(dir, "", remove_one, (void *)dir);
#ifdef _WIN32
    RemoveDirectoryA(dir);
#else
    rmdir(dir);
#endif
}

#define FAIL_IF(cond, ...) do { if (cond) { printf("probe-shader: FAILED: " __VA_ARGS__); printf("\n"); remove_dir(dir); return 1; } } while (0)

int probe_shader(void) {
    char path[MAX_PATH * 2 + 64];
    if (!find_root(root, sizeof root)) {   // the shader census (tools/shader_census.py)
        printf("probe-shader: skipped: build/v131072/shaders/programs.json (the shader census) is not present\n");
        return 0;
    }
    snprintf(path, sizeof path, "%s/build/v131072/shaders/programs.json", root);
    size_t jobs_size;
    char *jobs = slurp(path, &jobs_size);
    CHECK(jobs, "cannot read %s", path);
    static Entry candidates[CANDIDATES], entries[WANT];
    int count = pick_entries(jobs, candidates);
    CHECK(count >= WANT, "only %d census shaders usable", count);

    char dir[MAX_PATH];
#ifdef _WIN32
    snprintf(dir, sizeof dir, "%s/oot-shader-probe-%lu", getenv("TEMP") ? getenv("TEMP") : ".", GetCurrentProcessId());
#else
    snprintf(dir, sizeof dir, "%s/oot-shader-probe-%lu", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (unsigned long)getpid());
#endif
    CHECK(plat_setenv("OOT_SHADER_CACHE", dir), "cannot set OOT_SHADER_CACHE");

    // pass 1: translate through the service; a candidate the lowering cannot take yet (docs/phase3.md) is skipped
    const GfxShader *first[WANT];
    int n = 0, skipped = 0;
    uint64_t req0 = gfx_shader_requests();
    for (int i = 0; i < count && n < WANT; i++) {
        int err = 0;
        const GfxShader *s = translate(&candidates[i], &err);
        if (!s) { skipped++; continue; }
        uint64_t words;
        const uint32_t *w = gfx_shader_spirv(s, &words);
        FAIL_IF(words < 5 || w[0] != 0x07230203, "shader %s: bad SPIR-V magic", candidates[i].key);
        entries[n] = candidates[i], first[n++] = s;
    }
    FAIL_IF(n < WANT, "only %d of %d candidates translated", n, count);
    FAIL_IF(gfx_shader_requests() - req0 != WANT, "pass 1 sent %llu requests, expected %d",   // a skipped one never reaches the service
            (unsigned long long)(gfx_shader_requests() - req0), WANT);

    // pass 2: memory cache dropped, so every answer comes from the disk cache
    gfx_shader_drop_memory();
    req0 = gfx_shader_requests();
    uint64_t hits0 = gfx_shader_cache_hits();
    const GfxShader *second[WANT];
    for (int i = 0; i < WANT; i++) {
        int err = 0;
        second[i] = translate(&entries[i], &err);
        FAIL_IF(!second[i], "shader %s missing from the disk cache", entries[i].key);
        uint64_t words;
        const uint32_t *w = gfx_shader_spirv(second[i], &words);
        FAIL_IF(words < 5 || w[0] != 0x07230203, "shader %s: bad SPIR-V magic from disk", entries[i].key);
    }
    int hits = (int)(gfx_shader_cache_hits() - hits0);
    FAIL_IF(gfx_shader_requests() != req0, "pass 2 contacted the service");
    FAIL_IF(hits != WANT, "pass 2 had %d cache hits", hits);

    // pass 3: memory hits return the same pointer
    for (int i = 0; i < WANT; i++) {
        int err = 0;
        FAIL_IF(translate(&entries[i], &err) != second[i], "repeat call for %s returned another pointer", entries[i].key);
    }
    FAIL_IF(gfx_shader_requests() != req0, "pass 3 contacted the service");
    FAIL_IF(gfx_shader_failures() != (uint64_t)skipped, "%llu translation failures, %d skipped candidates",
            (unsigned long long)gfx_shader_failures(), skipped);

    // the service survives a malformed line and still translates a valid one
    char reply[4096];
    FAIL_IF(!gfx_shader_raw_request("this is not json", reply, sizeof reply), "service gone after the malformed line");
    FAIL_IF(strncmp(reply, "fail ", 5), "malformed line answered with \"%s\"", reply);
    FAIL_IF(!gfx_shader_raw_request("{\"job\":{\"stage\":\"ps\"},\"out\":\"x\"}", reply, sizeof reply) || strncmp(reply, "fail ", 5),
            "job without key not rejected");
    char dir_json[MAX_PATH * 2], line[MAX_PATH * 4 + 1024], *o = dir_json;
    for (const char *s = dir; *s; s++) { if (*s == '\\') *o++ = '\\'; *o++ = *s; }
    *o = 0;
    char esc_root[MAX_PATH * 2];
    o = esc_root;
    for (const char *s = root; *s; s++) { if (*s == '\\') *o++ = '\\'; *o++ = *s; }
    *o = 0;
    // A valid request: the first entry's lowered program (tools/shader_census.py --lower writes lowered/<key>.code
    // and .c1), the translator's input form.
    const Entry *e = &entries[0];
    char c1_path[MAX_PATH * 2 + 128], c1_hex[16384] = "";
    snprintf(c1_path, sizeof c1_path, "%s/build/v131072/shaders/lowered/%.64s.c1", root, e->key);
    size_t c1_size = 0;
    char *c1 = slurp(c1_path, &c1_size);
    FAIL_IF(!c1 || c1_size * 2 >= sizeof c1_hex, "%s is missing (tools/shader_census.py --lower writes it)", c1_path);
    for (size_t b = 0; b < c1_size; b++) sprintf(c1_hex + 2 * b, "%02x", (unsigned char)c1[b]);
    free(c1);
    snprintf(line, sizeof line,
             "{\"job\":{\"key\":\"%.64sa\",\"stage\":\"%s\",\"debug\":false,\"c1\":\"%s\",\"code\":\"%s\\\\build\\\\v131072\\\\shaders\\\\lowered\\\\%.64s.code\"},\"out\":\"%s\"}",
             e->key, e->stage == GFX_STAGE_VS ? "vs" : e->stage == GFX_STAGE_FS ? "ps" : "cs", c1_hex, esc_root, e->key, dir_json);
    FAIL_IF(!gfx_shader_raw_request(line, reply, sizeof reply), "service did not answer a valid request after the malformed one");
    FAIL_IF(strncmp(reply, "ok ", 3), "valid request after the malformed one answered \"%s\"", reply);

    remove_dir(dir);
    printf("probe-shader: passed (%d shaders, %d candidates the lowering cannot take yet, %d cache hits, service survived "
           "malformed request)\n", WANT, skipped, hits);
    return 0;
}
