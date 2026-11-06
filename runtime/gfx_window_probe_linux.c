// --probe-window on Linux: the Linux counterpart of gfx_window_probe.c, which is Win32 only (its checks read back the
// presented swapchain image, count vsync ticks and send WM_ACTIVATEAPP and cursor messages). Here only the window
// system is checked: an SDL3 window with Vulkan support is created hidden and destroyed again. The present readback,
// focus, cursor and vsync checks are not run on Linux. Copied in its shape from gfx_window_probe.c of this tree
// (metroid-prime-4-recomp @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0, MIT, Copyright (c) 2026 Jeff Aigner; fe3h-pc-recomp
// @ 47ef99126793dbd13e4a475feb910a13ef6639b8, MIT, Copyright (c) 2026 Jeff Aigner; totk-pc-recomp @
// c944f9f94520874f2a669add30dd08d29ee9f7b5, MIT, Copyright (c) 2026 Valiantsin Dzerakh); the checks are not copied.
#include <stdio.h>

#include <SDL3/SDL.h>

int probe_window(void) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        printf("probe-window: FAILED (SDL video init: %s)\n", SDL_GetError());
        return 1;
    }
    SDL_Window *w = SDL_CreateWindow("probe-window", 640, 360, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    if (!w) {
        printf("probe-window: FAILED (SDL window: %s)\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }
    SDL_DestroyWindow(w);
    SDL_Quit();
    printf("probe-window: hidden SDL3 Vulkan window created and destroyed; the present, focus, cursor and vsync checks "
           "of the Win32 probe are not run on Linux\n");
    return 0;
}
