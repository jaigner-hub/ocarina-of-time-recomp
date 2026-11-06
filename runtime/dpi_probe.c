// From metroid-prime-4-recomp runtime/dpi_probe.c @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied with mp4 renamed to oot.
// From fe3h-pc-recomp runtime/dpi_probe.c @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied with fe3h renamed to mp4 only.
// From totk-pc-recomp runtime/dpi_probe.c @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), copied with totk renamed to fe3h only.
// --probe-dpi: the process DPI awareness that oot.manifest declares, and what the primary monitor reports under it.
// A DPI-unaware process gets every window and monitor size scaled down by the display scale and its output stretched
// back up by Windows; the probe fails unless the process is per-monitor aware (v2). Needs no game files and no GPU.
#include <stdio.h>

#include "host.h"

#if defined(_WIN32)
#include <windows.h>

static const char *awareness_name(DPI_AWARENESS_CONTEXT ctx) {
    if (AreDpiAwarenessContextsEqual(ctx, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) return "per-monitor aware v2";
    switch (GetAwarenessFromDpiAwarenessContext(ctx)) {
    case DPI_AWARENESS_UNAWARE: return "unaware";
    case DPI_AWARENESS_SYSTEM_AWARE: return "system aware";
    case DPI_AWARENESS_PER_MONITOR_AWARE: return "per-monitor aware v1";
    default: return "invalid";
    }
}

int probe_dpi(void) {
    DPI_AWARENESS_CONTEXT ctx = GetThreadDpiAwarenessContext();
    MONITORINFO mi = {sizeof mi};
    HMONITOR mon = MonitorFromPoint((POINT){0, 0}, MONITOR_DEFAULTTOPRIMARY);
    int have = GetMonitorInfoW(mon, &mi);
    printf("dpi: awareness %s\n", awareness_name(ctx));
    printf("dpi: system dpi %u (scale %u%%)\n", GetDpiForSystem(), GetDpiForSystem() * 100 / 96);
    if (have)
        printf("dpi: primary monitor %ldx%ld at %ld,%ld; screen metrics %dx%d\n", mi.rcMonitor.right - mi.rcMonitor.left,
               mi.rcMonitor.bottom - mi.rcMonitor.top, mi.rcMonitor.left, mi.rcMonitor.top, GetSystemMetrics(SM_CXSCREEN),
               GetSystemMetrics(SM_CYSCREEN));
    int ok = AreDpiAwarenessContextsEqual(ctx, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) && have;
    printf("probe-dpi: %s\n", ok ? "PASS" : "FAIL (the process is not per-monitor aware v2)");
    return ok ? 0 : 1;
}

#else
// The DPI awareness of a window is a Windows setting: elsewhere the probe has nothing to check and fails.
int probe_dpi(void) {
    printf("probe-dpi: Windows only\n");
    return 1;
}
#endif
