// The event seam of runtime/input_sdl.c: on hosts without Win32, the window thread gives the keyboard, mouse, focus and
// cursor state to the input code here, where input_win.c takes the Win32 window messages (input_window_msg). The window
// (gfx_window_sdl.c) calls input_sdl_window once the window exists, input_sdl_event for every event (before the menu's
// own handler), and input_sdl_idle when no event has come for a while. Window thread, except where noted.
#pragma once
#include <SDL3/SDL.h>

#ifdef __cplusplus
extern "C" {
#endif

void input_sdl_window(SDL_Window *w);      // the game window was created (or recreated): its focus and cursor state
void input_sdl_event(const SDL_Event *e);  // every event of the window, in order
void input_sdl_idle(void);                 // no event for a while: applies a cursor change asked for from another thread

#ifdef __cplusplus
}
#endif
