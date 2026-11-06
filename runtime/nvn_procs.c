// From metroid-prime-4-recomp runtime/nvn_procs.c @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied; up to 256 distinct unknown proc names are logged; NVN2 procs without an implementation get logging stubs (nvn_procs_stub.inc).
// From fe3h-pc-recomp runtime/nvn_procs.c @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied unchanged.
// From totk-pc-recomp runtime/nvn_procs.c @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), copied unchanged.
// NVN proc table. Every name the game's proc loader asks for gets a synthetic host
// address through hostfn_register before hostfn_seal; nvnBootstrapLoader and nvnDeviceGetProcAddress hand them out.
// nvnp_<name> is the implementation of one proc: the weak definitions here stop with the proc's name, nvn_api_*.c
// and nvn_ext.c define the strong ones. The name list is runtime/nvn_procs.inc (tools/gen_nvn_procs.py).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gfx.h"
#include "host.h"
#include "platform/platform.h"

// The strong nvnp_nvnDeviceGetProcAddress below replaces the stand-in generated for the same row.
#define nvnp_nvnDeviceGetProcAddress nvnp_stub_nvnDeviceGetProcAddress

#define NVN_PROC(n) void nvnp_##n(Cpu *c);
#include "nvn_procs.inc"
#undef NVN_PROC

#define NVN_PROC(n)                                                                                                  \
    __attribute__((weak)) void nvnp_##n(Cpu *c) {                                                                     \
        (void)c;                                                                                                     \
        rt_fatal("NVN " #n " is not implemented");                                                                   \
    }
#include "nvn_procs.inc"
#undef NVN_PROC
#undef nvnp_nvnDeviceGetProcAddress
void nvnp_nvnDeviceGetProcAddress(Cpu *c);

// NVN2 names the game loads that have no implementation (nvn_procs_stub.inc): a stub logs its first three calls with
// the caller and the first arguments and returns 0, so the game runs on and the log shows which procs it uses.
#define NVN_STUB(n)                                                                                                  \
    static void nvnp_stub_##n(Cpu *c) {                                                                               \
        static volatile int32_t seen;                                                                                   \
        int32_t k = __atomic_add_fetch(&seen, 1, __ATOMIC_SEQ_CST);                                                                         \
        char at[128];                                                                                                \
        if (k <= 3) rt_log("nvn2 stub " #n " from %s: x0 %#llx x1 %#llx x2 %#llx x3 %#llx (returns 0)",                \
                           ldr_describe(c->x[30] - 4, at, sizeof at), (unsigned long long)c->x[0],                      \
                           (unsigned long long)c->x[1], (unsigned long long)c->x[2], (unsigned long long)c->x[3]);      \
        ret_x(c, 0);                                                                                                 \
    }
#include "nvn_procs_stub.inc"
#undef NVN_STUB
typedef struct { const char *name; uint64_t addr; } Proc;

#define NVN_PROC(n) +1
#define NVN_STUB(n) +1
enum { PROC_COUNT = 0
#include "nvn_procs.inc"
#include "nvn_procs_stub.inc"
};
#undef NVN_PROC
#undef NVN_STUB

static Proc procs[PROC_COUNT];
static int nprocs;

static int by_name(const void *a, const void *b) { return strcmp(((const Proc *)a)->name, ((const Proc *)b)->name); }

void nvn_procs_register(void) {
    if (nprocs) return;
#define NVN_PROC(n) procs[nprocs++] = (Proc){#n, hostfn_register("nvn:" #n, nvnp_##n)};
#include "nvn_procs.inc"
#undef NVN_PROC
#define NVN_STUB(n) procs[nprocs++] = (Proc){#n, hostfn_register("nvn:" #n, nvnp_stub_##n)};
#include "nvn_procs_stub.inc"
#undef NVN_STUB
    qsort(procs, (size_t)nprocs, sizeof *procs, by_name);
}

uint64_t nvn_proc_address(const char *name) {
    if (!name) return 0;
    const Proc *p = bsearch(&(Proc){name, 0}, procs, (size_t)nprocs, sizeof *procs, by_name);
    if (p) return p->addr;
    // Unknown names get NULL, as on hardware; each distinct one is logged once.
    static PlatMutex lock = PLAT_MUTEX_INIT;
    static char seen[256][96];
    static int nseen;
    plat_mutex_lock(&lock);
    int fresh = 1;
    for (int i = 0; i < nseen; i++)
        if (!strncmp(seen[i], name, sizeof seen[i] - 1)) fresh = 0;
    if (fresh && nseen < 256) {
        strncpy(seen[nseen], name, sizeof seen[0] - 1);
        nseen++;
    } else {
        fresh = 0;
    }
    plat_mutex_unlock(&lock);
    if (fresh) rt_log("nvn: no such proc \"%s\", returning NULL", name);
    return 0;
}

// PFNNVNGENERICFUNC nvnDeviceGetProcAddress(const NVNdevice *device, const char *name)
void nvnp_nvnDeviceGetProcAddress(Cpu *c) {
    const char *name = arg_p(c, 1);
    ret_x(c, nvn_proc_address(name));
}
