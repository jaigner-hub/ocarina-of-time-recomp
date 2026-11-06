// RenderDoc in-app capture on Linux: the stub for rdoc_win.c (the Win32 loader of renderdoc.dll), with the same interface
// declared in vk.h. rdoc_win.c is copied from metroid-prime-4-recomp @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT,
// Copyright (c) 2026 Jeff Aigner), fe3h-pc-recomp @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff
// Aigner) and totk-pc-recomp @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh).
// Linux: not implemented. No library is loaded and no frame is captured; rdoc_init logs that OOT_RDOC was ignored so the
// setting does not fail silently.
#include <stdint.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"

void rdoc_init(void) {
    const char *e = cfg_get("OOT_RDOC");
    if (e && *e) rt_log("rdoc: OOT_RDOC is not implemented on Linux; no frame is captured");
}

int rdoc_active(void) { return 0; }

void rdoc_after_present(uint64_t frame) { (void)frame; }
