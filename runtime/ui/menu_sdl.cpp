// SDL adapter of the in-game menu: the SDL3 events of the window (gfx_window_sdl.c) to menu events (menu.h). It is the
// counterpart of menu_win.cpp, with the same keys, buttons and focus rules: F1 opens and closes the menu, Esc opens it
// (with the menu open, Esc goes to the menu, which closes itself), F10 cycles the performance overlay, and while the
// menu is open its keys, the mouse and the focus go to it. Window thread.
#include <SDL3/SDL.h>

#include "imgui.h"

extern "C" {
#include "ui/menu.h"
}

// Keys the menu uses, by SDL keycode (the layout's letters, as menu_win.cpp's virtual keys are).
static ImGuiKey key_of(SDL_Keycode k) {
    switch (k) {
    case SDLK_UP: return ImGuiKey_UpArrow;
    case SDLK_DOWN: return ImGuiKey_DownArrow;
    case SDLK_LEFT: return ImGuiKey_LeftArrow;
    case SDLK_RIGHT: return ImGuiKey_RightArrow;
    case SDLK_RETURN: return ImGuiKey_Enter;
    case SDLK_SPACE: return ImGuiKey_Space;
    case SDLK_ESCAPE: return ImGuiKey_Escape;
    case SDLK_TAB: return ImGuiKey_Tab;
    case SDLK_BACKSPACE: return ImGuiKey_Backspace;
    case SDLK_DELETE: return ImGuiKey_Delete;   // clears a cell of the remapping rows
    case SDLK_HOME: return ImGuiKey_Home;
    case SDLK_END: return ImGuiKey_End;
    case SDLK_PAGEUP: return ImGuiKey_PageUp;
    case SDLK_PAGEDOWN: return ImGuiKey_PageDown;
    case SDLK_LSHIFT: case SDLK_RSHIFT: return ImGuiKey_LeftShift;
    case SDLK_LCTRL: case SDLK_RCTRL: return ImGuiKey_LeftCtrl;
    case SDLK_Q: return ImGuiKey_Q;   // previous tab
    case SDLK_E: return ImGuiKey_E;   // next tab
    default: return ImGuiKey_None;
    }
}

static void menu_key(const SDL_KeyboardEvent *k) {
    bool down = k->type == SDL_EVENT_KEY_DOWN;
    if (down && !k->repeat && (k->key == SDLK_F1 || (k->key == SDLK_ESCAPE && !menu_is_open()))) {
        menu_toggle();   // Esc with the menu open is not a toggle: it goes to the menu, which closes itself
        return;
    }
    if (down && !k->repeat && k->key == SDLK_F10) {   // also keeps F10 from opening anything else
        menu_perf_cycle();
        return;
    }
    if (!menu_is_open()) return;
    if (ImGuiKey key = key_of(k->key)) menu_event_key((int)key, down);
}

extern "C" void menu_sdl_event(const SDL_Event *e) {
    switch (e->type) {
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
        menu_key(&e->key);
        return;
    case SDL_EVENT_MOUSE_MOTION:
        if (menu_is_open()) menu_event_mouse_pos(e->motion.x, e->motion.y);
        return;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        if (!menu_is_open()) return;
        // SDL numbers: 1 left, 2 middle, 3 right; the menu's: 0 left, 1 right, 2 middle
        int b = e->button.button == SDL_BUTTON_LEFT ? 0 : e->button.button == SDL_BUTTON_RIGHT ? 1 : 2;
        menu_event_mouse_pos(e->button.x, e->button.y);
        menu_event_mouse_button(b, e->type == SDL_EVENT_MOUSE_BUTTON_DOWN);
        return;
    }
    case SDL_EVENT_MOUSE_WHEEL:
        if (menu_is_open()) menu_event_wheel(e->wheel.x, e->wheel.y);
        return;
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
        menu_event_focus(1);
        return;
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        menu_event_focus(0);
        return;
    default:
        return;
    }
}
