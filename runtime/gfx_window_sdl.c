// From totk-pc-recomp runtime/gfx_window_sdl.c @ 0ae47a5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), the SDL3 window and its
// event loop, adapted to OoT's gfx.h and vk.h (OOT_ settings, OoT's present and menu interface).
// From totk-pc-recomp runtime/gfx_present.c @ 0ae47a5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), the window-system-free
// half (swapchain, present blit, capture, flight thumbnails, vsync, focus), ported here from this tree's gfx_window.c.
// From metroid-prime-4-recomp runtime/gfx_window.c @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied with mp4 renamed to oot.
// From fe3h-pc-recomp runtime/gfx_window.c @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied with fe3h renamed to mp4; the window class and title name this port; the present blit turns the frame over for the lower-left window origin.
// From totk-pc-recomp runtime/gfx_window.c @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), copied with totk renamed to fe3h only.
// Swapchain set-up ported from LAS Recomp, runtime/nvn/vulkan/device.c:254-330 (https://github.com/peiikai/LASRecomp @
// da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai). Focus and vsync behaviour from
// MacNativeGames, games/totk/runtime/frontend.m:88-97, 175-176, 350-352 (https://github.com/ly05010419/MacNativeGames @
// b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026 MacNativeGames contributors).
//
// Linux (SDL3) window back end for gfx.h and vk.h; gfx_window.c is the Win32 one, and build_game.py compiles this file
// in its place on Linux (SDL_TWIN). Keyboard and mouse go to input_sdl.c, the in-game menu's events to ui/menu_sdl.cpp.
//
// Threads: the window thread runs SDL's event loop and every SDL video call (SDL wants them on one thread). The others
// hand it work through run_on_window (a user event and a wait) or post a user event. The vsync thread signals the bound
// system events. Presents come from any host thread and take g_pl; every queue submission takes the recursive queue
// lock (vk_queue_lock). The swapchain is rebuilt lazily: a resize only marks it dirty, the next present (or
// gfx_window_resize) rebuilds it. The window thread never runs guest code.
//
// Settings: OOT_WINDOW_SIZE=<w>x<h> is the client area in pixels (default from cfg.c; on a scaled display the window asks
// for its pixels in the display's units, SDL_GetWindowDisplayScale), OOT_FULLSCREEN=1 the desktop's mode (F11 toggles it).
// OOT_BACKGROUND_WINDOW has no effect here: SDL windows take focus where the desktop puts them. The window and the menu
// keep the screen awake (SDL_HINT_VIDEO_ALLOW_SCREENSAVER), as the Win32 window does through SetThreadExecutionState.
//
// OOT_ASSUME_FOCUS is applied inside oe_set_focus (ov_system.c), so a focus event here needs no check of its own.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include "cfg.h"
#include "flight.h"
#include "fs.h"
#include "gfx.h"
#include "input.h"
#include "nvn_api.h"
#include "platform/platform.h"
#include "ui/menu.h"
#include "vk.h"

// The seams of input_sdl.c and ui/menu_sdl.cpp (TotK's input_sdl.h has the same declarations; the files are not in this
// tree yet). input_cursor_suspend is declared in input.h and defined by input_sdl.c.
void input_sdl_window(SDL_Window *w);      // the game window was created
void input_sdl_event(const SDL_Event *e);  // every event, first
void input_sdl_idle(void);                 // no event for a while: a cursor change asked for from another thread
void menu_sdl_event(const SDL_Event *e);   // every event, after input_sdl_event

#define FRAMES 2
#define MAX_IMAGES 16
#define VSYNC_NS 1000000000ull
#define VSYNC_HZ 60

// ---- locks. platform.h has a plain mutex; the queue lock and g_pl are taken re-entrantly (the menu's overlay submits
// while the present holds g_pl), so the owner and its depth are kept here. A thread's own id in owner means it holds
// the lock: only the holder stores its id, and it clears it before it unlocks.
typedef struct { PlatMutex m; uint32_t owner; unsigned depth; } RMutex;
#define RMUTEX_INIT {PLAT_MUTEX_INIT, 0, 0}

static void rmutex_lock(RMutex *r) {
    uint32_t me = plat_thread_id();
    if (__atomic_load_n(&r->owner, __ATOMIC_RELAXED) == me) {
        r->depth++;
        return;
    }
    plat_mutex_lock(&r->m);
    __atomic_store_n(&r->owner, me, __ATOMIC_RELAXED);
    r->depth = 1;
}

static void rmutex_unlock(RMutex *r) {
    if (--r->depth == 0) {
        __atomic_store_n(&r->owner, 0, __ATOMIC_RELAXED);
        plat_mutex_unlock(&r->m);
    }
}

// One-shot event: waited for once, set once (video up, a run_on_window request done).
typedef struct { PlatMutex m; PlatCond c; int set; } Latch;

static void latch_init(Latch *l) {
    plat_mutex_init(&l->m);
    plat_cond_init(&l->c);
    l->set = 0;
}

static void latch_set(Latch *l) {
    plat_mutex_lock(&l->m);
    l->set = 1;
    plat_cond_broadcast(&l->c);
    plat_mutex_unlock(&l->m);
}

static void latch_wait(Latch *l) {
    plat_mutex_lock(&l->m);
    while (!l->set) plat_cond_wait(&l->c, &l->m);
    plat_mutex_unlock(&l->m);
}

static RMutex g_ql = RMUTEX_INIT;          // the queue lock
static RMutex g_pl = RMUTEX_INIT;          // swapchain and present state
static RMutex g_start_lock = RMUTEX_INIT;  // gfx_window_start and the vsync start

void vk_queue_lock(void) { rmutex_lock(&g_ql); }
void vk_queue_unlock(void) { rmutex_unlock(&g_ql); }

// ---- window state
static SDL_Window *g_win;                  // set on the window thread; read from any thread
static uint32_t g_ev_base;                 // SDL_RegisterEvents: the user events below
enum { EV_RUN, EV_SETTINGS, EV_CLOSE, EV_COUNT };
static Latch g_video_ready;                // set once SDL video is up (or failed)
static int g_video_ok;
static PlatMutex g_video_lock = PLAT_MUTEX_INIT;
static int g_video_started;
static uint32_t g_window_tid;              // the window thread, which runs the video calls
static int g_open_ok;
static int32_t g_density_milli = 1000;     // pixels per window unit, x1000
static int32_t g_dirty;                    // the window was resized; the swapchain is stale
static int32_t g_focused = 1;
static int32_t g_focus_msgs;
static int g_started;
static int32_t g_rate_mhz;                 // the monitor's refresh rate in millihertz, 0 unknown
static char g_applied_size[32];            // the size and fullscreen state last taken from the settings
static int g_applied_fs;

// ---- work on the window thread
typedef struct {
    void (*fn)(void *arg);
    void *arg;
    Latch done;
} RunReq;

static void run_on_window(void (*fn)(void *arg), void *arg) {
    if (plat_thread_id() == g_window_tid) {
        fn(arg);
        return;
    }
    RunReq r = {.fn = fn, .arg = arg};
    latch_init(&r.done);
    SDL_Event e;
    memset(&e, 0, sizeof e);
    e.user.type = g_ev_base + EV_RUN;
    e.user.data1 = &r;
    if (!SDL_PushEvent(&e)) rt_fatal("gfx: cannot reach the window thread (%s)", SDL_GetError());
    latch_wait(&r.done);
}

static void post(int ev) {
    SDL_Event e;
    memset(&e, 0, sizeof e);
    e.user.type = g_ev_base + (uint32_t)ev;
    SDL_PushEvent(&e);
}

// ---- seams for input_sdl.c and ui/menu_sdl.cpp
SDL_Window *gfx_window_sdl(void) { return g_win; }
double gfx_window_sdl_density(void) { return __atomic_load_n(&g_density_milli, __ATOMIC_RELAXED) / 1000.0; }
void gfx_window_sdl_run(void (*fn)(void *arg), void *arg) { run_on_window(fn, arg); }

// ---- Win32 handle: none on this host. Callers that need the window use gfx_window_sdl().
void *gfx_window_handle(void) { return NULL; }

// ---- settings
static int gfx_cfg_fullscreen(void) { return cfg_bool("OOT_FULLSCREEN"); }

static void cfg_window_size(int *w, int *h) {
    int x = 0, y = 0;
    if (sscanf(cfg_value("OOT_WINDOW_SIZE"), "%dx%d", &x, &y) != 2 || x < 640 || y < 360 || x > 7680 || y > 4320)
        sscanf(cfg_meta("WINDOW_SIZE")->def, "%dx%d", &x, &y);   // unparsable or out of range: the default
    *w = x;
    *h = y;
}

static void note_applied(void) {
    snprintf(g_applied_size, sizeof g_applied_size, "%s", cfg_value("OOT_WINDOW_SIZE"));
    g_applied_fs = gfx_cfg_fullscreen();
}

static int is_fullscreen(void) { return g_win && (SDL_GetWindowFlags(g_win) & SDL_WINDOW_FULLSCREEN) != 0; }

// The window's units for w x h pixels: the settings are in pixels, SDL sizes the window in its own units.
static void set_pixel_size(int w, int h) {
    float scale = SDL_GetWindowDisplayScale(g_win);
    if (!(scale > 0)) scale = 1;
    SDL_SetWindowSize(g_win, (int)(w / scale + 0.5f), (int)(h / scale + 0.5f));
}

// F11 and the settings: borderless over the display in the desktop's mode, and back. The cursor is released meanwhile.
static void set_fullscreen(int on) {
    input_cursor_suspend(1);
    SDL_SetWindowFullscreen(g_win, on);
    SDL_SyncWindow(g_win);
    input_cursor_suspend(0);
}

// Window thread: the window takes the size and fullscreen state of the settings when they changed. A window resized by
// hand keeps its size when another setting changes. Fullscreen has to end for a new size to take, and comes back after.
static void apply_settings(void) {
    int size_changed = strcmp(cfg_value("OOT_WINDOW_SIZE"), g_applied_size) != 0;
    int fs = gfx_cfg_fullscreen(), fs_changed = fs != g_applied_fs;
    note_applied();
    if (size_changed) {
        int w, h, was = is_fullscreen();
        cfg_window_size(&w, &h);
        if (was) {
            SDL_SetWindowFullscreen(g_win, false);
            SDL_SyncWindow(g_win);
        }
        set_pixel_size(w, h);
        if (was) {
            SDL_SetWindowFullscreen(g_win, true);
            SDL_SyncWindow(g_win);
        }
    }
    if (fs_changed && fs != is_fullscreen()) set_fullscreen(fs);
}

void gfx_window_apply_settings(void) {
    if (g_win) post(EV_SETTINGS);
}

void gfx_window_request_close(void) {
    if (g_win) post(EV_CLOSE);
}

// The DPI of the window, 96 at 100% display scale; 96 before the window exists.
unsigned gfx_window_dpi(void) {
    float scale = g_win ? SDL_GetWindowDisplayScale(g_win) : 0;
    return scale > 0 ? (unsigned)(scale * 96 + 0.5f) : 96;
}

// ---- display: refresh rate, size, pixel density
// Window thread. The exact rational rate when SDL has it (59.94, 143.998), else the mode's float rate.
static void refresh_rate_update(void) {
    if (!g_win) return;
    const SDL_DisplayMode *m = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(g_win));
    double hz = 0;
    if (m && m->refresh_rate_denominator) hz = (double)m->refresh_rate_numerator / m->refresh_rate_denominator;
    else if (m) hz = m->refresh_rate;
    int32_t mhz = (int32_t)(hz * 1000 + 0.5);
    if (__atomic_exchange_n(&g_rate_mhz, mhz, __ATOMIC_RELAXED) != mhz) rt_log("gfx: monitor refresh rate %.3f Hz", hz);
}

double gfx_window_refresh_hz(void) { return __atomic_load_n(&g_rate_mhz, __ATOMIC_RELAXED) / 1000.0; }

int gfx_window_monitor_size(int *w, int *h) {
    if (!g_win) return 0;
    const SDL_DisplayMode *m = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(g_win));
    if (!m) return 0;
    *w = (int)(m->w * m->pixel_density + 0.5f);
    *h = (int)(m->h * m->pixel_density + 0.5f);
    return 1;
}

// Window thread.
static void density_update(void) {
    float d = g_win ? SDL_GetWindowPixelDensity(g_win) : 1;
    __atomic_store_n(&g_density_milli, (int32_t)((d > 0 ? d : 1) * 1000 + 0.5f), __ATOMIC_RELAXED);
}

// Any thread: the swapchain is rebuilt at the next present.
static void mark_dirty(void) { __atomic_store_n(&g_dirty, 1, __ATOMIC_RELAXED); }

// ---- focus and close
// The window gained or lost the focus: the game hears it as nn::oe's focus state and a notification message.
static void focus_changed(int in) {
    __atomic_store_n(&g_focused, in, __ATOMIC_RELAXED);
    oe_set_focus(in);   // pinned to 1 by OOT_ASSUME_FOCUS inside
    oe_push_message(OE_MSG_FOCUS_STATE_CHANGED);
    __atomic_fetch_add(&g_focus_msgs, 1, __ATOMIC_RELAXED);
    time_t now = time(NULL);
    struct tm t;
    localtime_r(&now, &t);
    rt_log("gfx: window %s the focus at %04d-%02d-%02d %02d:%02d:%02d local time", in ? "gained" : "lost",
           t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
}

int gfx_window_focused(void) { return __atomic_load_n(&g_focused, __ATOMIC_RELAXED); }
unsigned gfx_window_focus_messages(void) { return (unsigned)__atomic_load_n(&g_focus_msgs, __ATOMIC_RELAXED); }

// The window was closed: the process ends with it, as in MNG, once a save Commit still running in the background is
// done. _exit, as on Windows and boot.c: no atexit handlers run while guest threads may still be going.
static void window_closed(void) {
    rt_log("gfx: window closed; exiting");
    fs_save_flush();
    vk_pipeline_save();
    prof_write();
    _exit(0);
}

// ---- vsync (60 Hz). The schedule starts when the first vsync user binds; a stall of more than 100 ms restarts it.
static PlatMutex g_vs_lock = PLAT_MUTEX_INIT;
static uint64_t g_vs_ev[16];
static int g_vs_nev;
static volatile long *g_vs_ctr[4];
static int g_vs_nctr;
static int64_t g_ticks;
static int64_t g_vs_start_ns;
static int g_vs_started;

static void vsync_main(void *arg) {
    (void)arg;
    uint64_t n = 0;
    for (;;) {
        uint64_t start = (uint64_t)__atomic_load_n(&g_vs_start_ns, __ATOMIC_RELAXED);
        uint64_t due = start + (n + 1) * VSYNC_NS / VSYNC_HZ;
        uint64_t now = kern_now_ns();
        if (now > due + 100000000ull) {   // the process stalled: restart the schedule, no catch-up burst
            __atomic_store_n(&g_vs_start_ns, (int64_t)(now - (n + 1) * VSYNC_NS / VSYNC_HZ), __ATOMIC_RELAXED);
        } else if (due > now) {
            plat_sleep_ns(due - now);
        }
        n++;
        __atomic_store_n(&g_ticks, (int64_t)n, __ATOMIC_RELAXED);
        uint64_t ev[16];
        volatile long *ctr[4];
        int ne, nc;
        plat_mutex_lock(&g_vs_lock);
        ne = g_vs_nev;
        nc = g_vs_nctr;
        memcpy(ev, g_vs_ev, sizeof ev);
        memcpy(ctr, g_vs_ctr, sizeof ctr);
        plat_mutex_unlock(&g_vs_lock);
        for (int i = 0; i < nc; i++) __atomic_fetch_add(ctr[i], 1, __ATOMIC_RELAXED);
        for (int i = 0; i < ne; i++) kern_signal_system_event(ev[i]);
    }
}

static void vsync_ensure(void) {
    rmutex_lock(&g_start_lock);
    if (!g_vs_started) {
        g_vs_started = 1;
        __atomic_store_n(&g_vs_start_ns, (int64_t)kern_now_ns(), __ATOMIC_RELAXED);
        PlatThread *t = plat_thread_new(vsync_main, NULL, "vsync");
        if (!t) rt_fatal("gfx: cannot start the vsync thread");
        plat_thread_detach(t);
    }
    rmutex_unlock(&g_start_lock);
}

void gfx_vsync_bind(uint64_t system_event) {
    plat_mutex_lock(&g_vs_lock);
    int have = 0;
    for (int i = 0; i < g_vs_nev; i++) have |= g_vs_ev[i] == system_event;
    if (!have) {
        if (g_vs_nev == 16) rt_fatal("gfx: more than 16 vsync events");
        g_vs_ev[g_vs_nev++] = system_event;
    }
    plat_mutex_unlock(&g_vs_lock);
    vsync_ensure();
}

void gfx_vsync_bind_counter(volatile long *counter) {
    plat_mutex_lock(&g_vs_lock);
    if (g_vs_nctr == 4) rt_fatal("gfx: more than 4 vsync counters");
    g_vs_ctr[g_vs_nctr++] = counter;
    plat_mutex_unlock(&g_vs_lock);
    vsync_ensure();
}

uint64_t gfx_vsync_ticks(void) { return (uint64_t)__atomic_load_n(&g_ticks, __ATOMIC_RELAXED); }
uint64_t gfx_vsync_time_ns(uint64_t tick) {
    return (uint64_t)__atomic_load_n(&g_vs_start_ns, __ATOMIC_RELAXED) + tick * VSYNC_NS / VSYNC_HZ;
}

// ---- window thread: events
static void handle(const SDL_Event *e);

// The F11 and P keys; the menu and input see the event first (input_sdl_event, menu_sdl_event).
static void handle_key(const SDL_KeyboardEvent *k) {
    if (k->type != SDL_EVENT_KEY_DOWN || k->repeat) return;
    if (k->scancode == SDL_SCANCODE_F11) {
        int on = !is_fullscreen();
        set_fullscreen(on);
        cfg_set("OOT_FULLSCREEN", on ? "1" : "0");   // the menu shows it; the next start keeps it
        cfg_save();
        note_applied();   // what cfg_get reads now (the environment may still override it)
    }
    if (k->scancode == SDL_SCANCODE_P) {
        rt_shot_ring_dump();   // OOT_SHOT_RING: the last presents to PNGs
        vk_dump_arm(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 2);   // OOT_DUMP=when: the frame after
    }
}

static void handle(const SDL_Event *e) {
    if (e->type == g_ev_base + EV_RUN) {
        RunReq *r = e->user.data1;
        r->fn(r->arg);
        latch_set(&r->done);
        return;
    }
    if (e->type == g_ev_base + EV_SETTINGS) {
        if (g_win) apply_settings();
        return;
    }
    if (e->type == g_ev_base + EV_CLOSE) window_closed();
    if (!g_win) return;
    input_sdl_event(e);
    menu_sdl_event(e);
    switch (e->type) {
    case SDL_EVENT_KEY_DOWN:
        handle_key(&e->key);
        break;
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
    case SDL_EVENT_WINDOW_RESIZED:
    case SDL_EVENT_WINDOW_RESTORED:
        density_update();
        mark_dirty();
        break;
    case SDL_EVENT_WINDOW_DISPLAY_CHANGED:
    case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
    case SDL_EVENT_DISPLAY_CURRENT_MODE_CHANGED:
        density_update();
        refresh_rate_update();
        mark_dirty();
        break;
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        focus_changed(e->type == SDL_EVENT_WINDOW_FOCUS_GAINED);
        break;
    case SDL_EVENT_QUIT:
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
        window_closed();
        break;
    }
}

static void window_main(void *arg) {
    (void)arg;
    g_window_tid = plat_thread_id();
    SDL_SetHint(SDL_HINT_APP_ID, "oot-recomp");   // Wayland's app id: the desktop file and the taskbar match it
    SDL_SetHint(SDL_HINT_VIDEO_ALLOW_SCREENSAVER, "0");
    g_video_ok = SDL_Init(SDL_INIT_VIDEO);
    if (!g_video_ok) rt_log("gfx: SDL video init failed (%s)", SDL_GetError());
    // SDL's Vulkan queries need its Vulkan loader, which a window would load at creation: the instance and the device
    // are made before the window.
    else if (!SDL_Vulkan_LoadLibrary(NULL)) {
        rt_log("gfx: SDL cannot load Vulkan (%s)", SDL_GetError());
        g_video_ok = 0;
    } else {
        g_ev_base = SDL_RegisterEvents(EV_COUNT);
        if (!g_ev_base) {
            rt_log("gfx: SDL cannot register its user events (%s)", SDL_GetError());
            g_video_ok = 0;
        } else {
            rt_log("gfx: SDL %d.%d.%d, video driver %s", SDL_VERSIONNUM_MAJOR(SDL_GetVersion()),
                   SDL_VERSIONNUM_MINOR(SDL_GetVersion()), SDL_VERSIONNUM_MICRO(SDL_GetVersion()), SDL_GetCurrentVideoDriver());
        }
    }
    latch_set(&g_video_ready);
    if (!g_video_ok) return;
    for (;;) {
        SDL_Event e;
        if (SDL_WaitEventTimeout(&e, 100)) handle(&e);
        else input_sdl_idle();   // a cursor change requested from another thread meanwhile
    }
}

// Starts the window thread (video, the event loop) once; any thread. g_video_ok says whether SDL video came up.
static int video_up(void) {
    plat_mutex_lock(&g_video_lock);
    if (!g_video_started) {
        g_video_started = 1;
        latch_init(&g_video_ready);
        PlatThread *t = plat_thread_new(window_main, NULL, "window");
        if (!t) rt_fatal("gfx: cannot start the window thread");
        plat_thread_detach(t);
        latch_wait(&g_video_ready);
    }
    plat_mutex_unlock(&g_video_lock);
    return g_video_ok;
}

// The Vulkan instance extensions SDL needs for its surface (VK_KHR_xcb_surface or VK_KHR_wayland_surface, ...), written to
// out (at most max). Called by vk_device.c before the instance is made. Returns the count, 0 without SDL video.
int gfx_window_vk_extensions(const char **out, int max) {
    if (!video_up()) return 0;
    Uint32 n = 0;
    const char *const *ext = SDL_Vulkan_GetInstanceExtensions(&n);
    if (!ext) {
        rt_log("gfx: SDL_Vulkan_GetInstanceExtensions failed (%s)", SDL_GetError());
        return 0;
    }
    int k = 0;
    for (Uint32 i = 0; i < n && k < max; i++) out[k++] = ext[i];
    return k;
}

// 1 when queue family of pd can present to the window's display system (SDL, before a window exists); vk_device.c.
int gfx_window_vk_presentable(VkInstance inst, VkPhysicalDevice pd, uint32_t family) {
    return video_up() && SDL_Vulkan_GetPresentationSupport(inst, pd, family);
}

// ---- window creation (window thread), surface (window thread)
static void open_window(void *arg) {
    (void)arg;
    int w, h;
    cfg_window_size(&w, &h);
    g_win = SDL_CreateWindow("Ocarina of Time Recomp", w, h, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!g_win) {
        rt_log("gfx: SDL_CreateWindow failed (%s)", SDL_GetError());
        return;
    }
    set_pixel_size(w, h);   // w x h were window units; the setting is in pixels
    density_update();
    note_applied();
    refresh_rate_update();
    const char *bg = cfg_get("OOT_BACKGROUND_WINDOW");   // no row in the settings table: read it as gfx_window.c does
    if (bg && atoi(bg)) rt_log("gfx: OOT_BACKGROUND_WINDOW has no effect with SDL windows");
    if (gfx_cfg_fullscreen()) SDL_SetWindowFullscreen(g_win, true);
    SDL_SyncWindow(g_win);
    input_sdl_window(g_win);
    g_open_ok = 1;
}

// Makes the window once; 1 when it is open. Any thread.
static int window_open(void) {
    if (!video_up()) return 0;
    if (!g_win) run_on_window(open_window, NULL);
    return g_open_ok;
}

typedef struct { VkInstance inst; VkSurfaceKHR surface; } SurfaceReq;

static void make_surface(void *arg) {
    SurfaceReq *r = arg;
    if (!SDL_Vulkan_CreateSurface(g_win, r->inst, NULL, &r->surface)) {
        rt_log("gfx: SDL_Vulkan_CreateSurface failed (%s)", SDL_GetError());
        r->surface = VK_NULL_HANDLE;
    }
}

static VkSurfaceKHR window_surface(VkInstance inst) {
    SurfaceReq r = {inst, VK_NULL_HANDLE};
    run_on_window(make_surface, &r);
    return r.surface;
}

// The drawable area in pixels. Any thread.
static void client_size(int *w, int *h) {
    *w = *h = 0;
    if (g_win) SDL_GetWindowSizeInPixels(g_win, w, h);
}

static int window_minimized(void) {
    return !g_win || (SDL_GetWindowFlags(g_win) & (SDL_WINDOW_MINIMIZED | SDL_WINDOW_HIDDEN)) != 0;
}

typedef struct { int w, h; } ResizeReq;

static void resize_window(void *arg) {
    ResizeReq *r = arg;
    set_pixel_size(r->w, r->h);
    SDL_SyncWindow(g_win);
}

// ---- swapchain and present state (g_pl)
static VkSurfaceKHR g_surface;
static VkSwapchainKHR g_sc;
static VkFormat g_sc_fmt;
static VkExtent2D g_sc_ext;
static int g_scaler;                        // the last present: 0 the game image 1:1, 1 scaled by the blit, 2 by FSR
static uint32_t g_sc_n;
static VkImage g_sc_img[MAX_IMAGES];
static VkImageView g_sc_view[MAX_IMAGES];   // colour attachment views for the menu overlay
static VkSemaphore g_render_done[MAX_IMAGES];
static VkCommandPool g_pool;
static VkCommandBuffer g_cb[FRAMES];
static VkFence g_fence[FRAMES];
static VkSemaphore g_acquire[FRAMES];
static unsigned g_frame;
static int g_capture;
static VkBuffer g_cap_buf;
static VmaAllocation g_cap_alloc;
static uint8_t *g_cap_ptr;
static VkDeviceSize g_cap_size;
static int g_cap_w, g_cap_h, g_cap_valid;
static int g_flip_on;            // vk_present_blit submitted the blit of swapchain image g_flip_idx; vk_present_flip presents it
static uint32_t g_flip_idx;

// ---- OOT_FLIGHT thumbnails (flight.c; g_pl): each present also blits the swapchain image down into slot frame % g_th_n
// of a host-visible ring. A slot's tag names the frame it holds complete; recording a new frame clears the tag first,
// and the tag is set once the fence of that present has signalled.
#define TH_BYTES ((VkDeviceSize)FLIGHT_THUMB_W * FLIGHT_THUMB_H * 4)
static int g_th_n, g_th_failed;
static uint64_t g_th_frame;                // frame of the next vk_present_image call (vk_thumbs_frame)
static uint64_t *g_th_tag;                 // per ring slot
static uint64_t g_th_pending[FRAMES];      // per fence slot: the frame whose thumbnail its submission writes
static VkImage g_th_img;
static VmaAllocation g_th_img_alloc;
static VkFormat g_th_fmt;
static VkBuffer g_th_buf;
static VmaAllocation g_th_alloc;
static uint8_t *g_th_ptr;

// The OOT_VK_PRESENT mode the swapchain got (its name for gfx_window_present_mode).
static const char *g_present_got = "";

static VkPresentModeKHR pick_present_mode(VkPhysicalDevice pd) {
    const char *e = cfg_value("OOT_VK_PRESENT");
    VkPresentModeKHR want = !strcmp(e, "mailbox") ? VK_PRESENT_MODE_MAILBOX_KHR
                          : !strcmp(e, "immediate") ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_FIFO_KHR;
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, g_surface, &n, NULL);
    VkPresentModeKHR m[16];
    if (n > 16) n = 16;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, g_surface, &n, m);
    VkPresentModeKHR got = VK_PRESENT_MODE_FIFO_KHR;
    for (uint32_t i = 0; i < n; i++) if (m[i] == want) got = want;
    const char *name = got == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : got == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate" : "fifo";
    __atomic_store_n(&g_present_got, name, __ATOMIC_RELEASE);
    static int logged = -1;   // the mode last logged: a change from the menu shows in the log
    if ((int)got != logged) {
        logged = (int)got;
        rt_log("gfx: present mode %s (OOT_VK_PRESENT=%s)", name, e);
    }
    return got;
}

const char *gfx_window_present_mode(void) { return __atomic_load_n(&g_present_got, __ATOMIC_ACQUIRE); }

// Queues the present of a blit vk_present_blit left waiting (g_pl held). A swapchain rebuild and the next blit flush it
// first; the device release drops it.
static void flip_locked(void) {
    if (!g_flip_on) return;
    g_flip_on = 0;
    if (!g_sc) return;
    VkPresentInfoKHR pi = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1,
                           .pWaitSemaphores = &g_render_done[g_flip_idx], .swapchainCount = 1, .pSwapchains = &g_sc,
                           .pImageIndices = &g_flip_idx};
    vk_queue_lock();
    VkResult r = vkQueuePresentKHR(vk_ctx()->gfx_queue, &pi);
    vk_queue_unlock();
    vk_device_fault_report(r);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) mark_dirty();
}

static void free_capture(void) {
    VkCtx *c = vk_ctx();
    if (g_cap_buf && c) vmaDestroyBuffer(c->vma, g_cap_buf, g_cap_alloc);
    g_cap_buf = VK_NULL_HANDLE;
    g_cap_ptr = NULL;
    g_cap_size = 0;
    g_cap_valid = 0;
}

// Returns 1 with a usable swapchain, 0 when the window has no area (minimised). Called with g_pl held.
static int create_swapchain(void) {
    flip_locked();   // the acquired image goes out before its swapchain is replaced
    VkCtx *c = vk_ctx();
    VkSurfaceCapabilitiesKHR caps;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c->phys, g_surface, &caps) != VK_SUCCESS) return 0;
    int cw = 0, ch = 0;
    client_size(&cw, &ch);
    VkExtent2D ext = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent
                                                            : (VkExtent2D){(uint32_t)cw, (uint32_t)ch};
    if (!ext.width || !ext.height) return 0;
    uint32_t nf = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, g_surface, &nf, NULL);
    VkSurfaceFormatKHR fmts[64];
    if (nf > 64) nf = 64;
    vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, g_surface, &nf, fmts);
    if (!nf) return 0;
    // UNORM: the blit writes the bytes the game wrote (no sRGB encode).
    VkSurfaceFormatKHR sf = fmts[0];
    int found = 0;
    for (uint32_t i = 0; i < nf && !found; i++)
        if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM && fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { sf = fmts[i]; found = 1; }
    for (uint32_t i = 0; i < nf && !found; i++)
        if (fmts[i].format == VK_FORMAT_R8G8B8A8_UNORM && fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { sf = fmts[i]; found = 1; }
    uint32_t count = caps.minImageCount + 1;
    if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;
    if (count > 8) count = 8;

    vk_queue_lock();
    vkDeviceWaitIdle(c->device);
    VkSwapchainKHR old = g_sc;
    VkSwapchainCreateInfoKHR ci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = g_surface, .minImageCount = count,
        .imageFormat = sf.format, .imageColorSpace = sf.colorSpace, .imageExtent = ext, .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,   // every surface supports it (the menu overlay draws into it)
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, .presentMode = pick_present_mode(c->phys), .clipped = VK_TRUE,
        .oldSwapchain = old};
    VkResult r = vkCreateSwapchainKHR(c->device, &ci, NULL, &g_sc);
    if (r != VK_SUCCESS) {
        rt_log("gfx: vkCreateSwapchainKHR failed (%d)", (int)r);
        g_sc = old;
        vk_queue_unlock();
        return 0;
    }
    for (uint32_t i = 0; i < g_sc_n; i++) {
        vkDestroySemaphore(c->device, g_render_done[i], NULL);
        if (g_sc_view[i]) vkDestroyImageView(c->device, g_sc_view[i], NULL);
        g_sc_view[i] = VK_NULL_HANDLE;
    }
    if (old) vkDestroySwapchainKHR(c->device, old, NULL);
    g_sc_fmt = sf.format;
    g_sc_ext = ext;
    g_sc_n = MAX_IMAGES;
    vkGetSwapchainImagesKHR(c->device, g_sc, &g_sc_n, g_sc_img);
    VkSemaphoreCreateInfo semi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (uint32_t i = 0; i < g_sc_n; i++) {
        vkCreateSemaphore(c->device, &semi, NULL, &g_render_done[i]);
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_sc_img[i],
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = sf.format,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &g_sc_view[i]) != VK_SUCCESS) g_sc_view[i] = VK_NULL_HANDLE;
    }
    free_capture();
    vk_queue_unlock();
    __atomic_store_n(&g_dirty, 0, __ATOMIC_RELAXED);
    return 1;
}

static void sync_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 s0,
                         VkAccessFlags2 a0, VkPipelineStageFlags2 s1, VkAccessFlags2 a1) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = s0, .srcAccessMask = a0,
                               .dstStageMask = s1, .dstAccessMask = a1, .oldLayout = from, .newLayout = to,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &d);
}

static int ensure_capture_buffer(void) {
    VkCtx *c = vk_ctx();
    VkDeviceSize need = (VkDeviceSize)g_sc_ext.width * g_sc_ext.height * 4;
    if (g_cap_buf && g_cap_size == need) return 1;
    free_capture();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = need, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &g_cap_buf, &g_cap_alloc, &info) != VK_SUCCESS) { g_cap_buf = VK_NULL_HANDLE; return 0; }
    g_cap_ptr = info.pMappedData;
    g_cap_size = need;
    return g_cap_ptr != NULL;
}

void vk_thumbs_enable(int n) {
    rmutex_lock(&g_pl);
    if (!g_th_n && n > 0 && (g_th_tag = calloc((size_t)n, sizeof *g_th_tag))) g_th_n = n;
    rmutex_unlock(&g_pl);
}

void vk_thumbs_frame(uint64_t frame) {
    rmutex_lock(&g_pl);
    g_th_frame = frame;
    rmutex_unlock(&g_pl);
}

static void free_thumbs(void) {
    VkCtx *c = vk_ctx();
    if (g_th_img && c) vmaDestroyImage(c->vma, g_th_img, g_th_img_alloc);
    if (g_th_buf && c) vmaDestroyBuffer(c->vma, g_th_buf, g_th_alloc);
    g_th_img = VK_NULL_HANDLE, g_th_buf = VK_NULL_HANDLE, g_th_ptr = NULL;
    if (g_th_tag) memset(g_th_tag, 0, sizeof *g_th_tag * (size_t)g_th_n);
    memset(g_th_pending, 0, sizeof g_th_pending);
}

// The ring and the blit target in the swapchain's format; allocated on the first present, the image again when the
// swapchain format changes. A failure turns the thumbnails off for the session.
static int ensure_thumbs(void) {
    if (!g_th_n || g_th_failed || !g_th_frame) return 0;
    VkCtx *c = vk_ctx();
    if (!g_th_buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = TH_BYTES * (VkDeviceSize)g_th_n,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &g_th_buf, &g_th_alloc, &info) != VK_SUCCESS || !info.pMappedData) {
            if (g_th_buf) vmaDestroyBuffer(c->vma, g_th_buf, g_th_alloc);
            g_th_buf = VK_NULL_HANDLE;
            g_th_failed = 1;
            rt_log("gfx: flight thumbnails off: no host-visible ring of %llu MB", (unsigned long long)(bi.size >> 20));
            return 0;
        }
        g_th_ptr = info.pMappedData;
    }
    if (g_th_img && g_th_fmt != g_sc_fmt) {
        vmaDestroyImage(c->vma, g_th_img, g_th_img_alloc);   // the previous present's fence was waited for: idle
        g_th_img = VK_NULL_HANDLE;
    }
    if (!g_th_img) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = g_sc_fmt,
                                .extent = {FLIGHT_THUMB_W, FLIGHT_THUMB_H, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &g_th_img, &g_th_img_alloc, NULL) != VK_SUCCESS) {
            g_th_img = VK_NULL_HANDLE;
            g_th_failed = 1;
            rt_log("gfx: flight thumbnails off: no %ux%u image", FLIGHT_THUMB_W, FLIGHT_THUMB_H);
            return 0;
        }
        g_th_fmt = g_sc_fmt;
    }
    return 1;
}

// dst (TRANSFER_SRC, the swapchain image just written) -> thumbnail image -> ring slot of g_th_frame.
static void record_thumb(VkCommandBuffer cb, VkImage dst, unsigned fence_slot) {
    uint64_t f = g_th_frame;
    unsigned ring = (unsigned)(f % (uint64_t)g_th_n);
    g_th_tag[ring] = 0;   // under g_pl: vk_thumbs_read no longer hands this slot out
    sync_barrier(cb, g_th_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageBlit region = {
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int)g_sc_ext.width, (int)g_sc_ext.height, 1}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {FLIGHT_THUMB_W, FLIGHT_THUMB_H, 1}}};
    vkCmdBlitImage(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_th_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region,
                   VK_FILTER_LINEAR);
    sync_barrier(cb, g_th_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy bc = {.bufferOffset = TH_BYTES * ring, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                            .imageExtent = {FLIGHT_THUMB_W, FLIGHT_THUMB_H, 1}};
    vkCmdCopyImageToBuffer(cb, g_th_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_th_buf, 1, &bc);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &d);
    g_th_pending[fence_slot] = f;
    g_th_frame = 0;
}

int vk_thumbs_read(uint64_t frame, uint8_t *rgba) {
    int ok = 0;
    rmutex_lock(&g_pl);
    if (g_th_n && g_th_ptr && frame) {
        unsigned ring = (unsigned)(frame % (uint64_t)g_th_n);
        if (g_th_tag[ring] == frame) {
            VkCtx *c = vk_ctx();
            vmaInvalidateAllocation(c->vma, g_th_alloc, TH_BYTES * ring, TH_BYTES);
            memcpy(rgba, g_th_ptr + TH_BYTES * ring, TH_BYTES);
            ok = g_th_fmt == VK_FORMAT_B8G8R8A8_UNORM || g_th_fmt == VK_FORMAT_B8G8R8A8_SRGB ? 2 : 1;
        }
    }
    rmutex_unlock(&g_pl);
    if (ok == 2)
        for (size_t i = 0; i < TH_BYTES; i += 4) {
            uint8_t t = rgba[i];
            rgba[i] = rgba[i + 2], rgba[i + 2] = t;
        }
    return ok != 0;
}

void vk_present_capture(int on) {
    rmutex_lock(&g_pl);
    g_capture = on;
    if (!on) g_cap_valid = 0;
    rmutex_unlock(&g_pl);
}

int vk_present_captured(const uint8_t **px, int *w, int *h, VkFormat *fmt) {
    rmutex_lock(&g_pl);
    int ok = g_cap_valid;
    if (ok) { *px = g_cap_ptr; *w = g_cap_w; *h = g_cap_h; *fmt = g_sc_fmt; }
    rmutex_unlock(&g_pl);
    return ok;
}

// The captured swapchain image as a PNG (the menu's test hook), RGBA with opaque alpha.
static void save_capture(const char *path) {
    size_t n = (size_t)g_cap_w * (size_t)g_cap_h * 4;
    uint8_t *px = malloc(n);
    if (!px) return;
    int bgra = g_sc_fmt == VK_FORMAT_B8G8R8A8_UNORM || g_sc_fmt == VK_FORMAT_B8G8R8A8_SRGB;
    for (size_t i = 0; i < n; i += 4) {
        px[i] = g_cap_ptr[i + (bgra ? 2 : 0)];
        px[i + 1] = g_cap_ptr[i + 1];
        px[i + 2] = g_cap_ptr[i + (bgra ? 0 : 2)];
        px[i + 3] = 255;
    }
    rt_write_png(path, px, (uint32_t)g_cap_w, (uint32_t)g_cap_h);
    free(px);
    rt_log("gfx: menu test shot %s (%dx%d)", path, g_cap_w, g_cap_h);
}

// Where the game's picture (cw x ch) goes on the sw x sh swapchain: OOT_ASPECT=fit (default) keeps its shape and centres
// it, with black bars at the sides (a 16:9 picture in a 21:9 window) or above and below; stretch fills the swapchain.
// Bars under a pixel are not drawn. Returns 1 when the rectangle leaves bars to clear.
static int fit_rect(int cw, int ch, int sw, int sh, int out[4]) {
    out[0] = 0, out[1] = 0, out[2] = sw, out[3] = sh;
    if (cw <= 0 || ch <= 0 || strcmp(cfg_value("OOT_ASPECT"), "fit")) return 0;
    long long a = (long long)cw * sh, b = (long long)ch * sw;
    if (a < b) {   // narrower than the window: bars at the sides
        int fw = (int)((a + ch / 2) / ch);
        if (fw >= sw - 1) return 0;
        out[0] = (sw - fw) / 2, out[2] = fw;
    } else if (a > b) {   // wider: bars above and below
        int fh = (int)(((long long)ch * sw + cw / 2) / cw);
        if (fh >= sh - 1) return 0;
        out[1] = (sh - fh) / 2, out[3] = fh;
    } else {
        return 0;
    }
    return 1;
}

// A settings change (cfg_version) reaches the window thread, which applies a changed size or fullscreen; a changed
// OOT_VK_PRESENT rebuilds the swapchain at this present. g_pl held.
static void window_settings_live(void) {
    static unsigned seen;
    static int started;
    static char present[16];   // the OOT_VK_PRESENT the swapchain was made with
    unsigned v = cfg_version();
    if (started && v == seen) return;
    seen = v;
    const char *e = cfg_value("OOT_VK_PRESENT");
    if (!started || strncmp(e, present, sizeof present - 1)) {
        snprintf(present, sizeof present, "%s", e);
        if (started) mark_dirty();
    }
    if (started) gfx_window_apply_settings();
    started = 1;
}

static void present_image(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4], int defer) {
    int general = layout == VK_IMAGE_LAYOUT_GENERAL;
    gfx_window_start();
    VkCtx *c = vk_ctx();
    rmutex_lock(&g_pl);
    flip_locked();
    if (!g_sc || __atomic_load_n(&g_dirty, __ATOMIC_RELAXED)) {
        if (!window_minimized() && !create_swapchain() && !g_sc) goto out;
    }
    if (window_minimized() || !g_sc) goto out;

    unsigned slot = g_frame % FRAMES;
    if (vkWaitForFences(c->device, 1, &g_fence[slot], VK_TRUE, 5000000000ull) != VK_SUCCESS) {
        rt_log("gfx: present fence wait timed out");
        goto out;
    }
    uint32_t idx = 0;
    VkResult r = vkAcquireNextImageKHR(c->device, g_sc, 1000000000ull, g_acquire[slot], VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        mark_dirty();
        if (!create_swapchain()) goto out;
        r = vkAcquireNextImageKHR(c->device, g_sc, 1000000000ull, g_acquire[slot], VK_NULL_HANDLE, &idx);
    }
    vk_device_fault_report(r);
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) goto out;     // lost, not ready: skip the frame
    if (r == VK_SUBOPTIMAL_KHR) mark_dirty();
    vkResetFences(c->device, 1, &g_fence[slot]);
    if (g_th_pending[slot]) {   // that fence has signalled: the thumbnail its submission wrote is complete
        g_th_tag[g_th_pending[slot] % (uint64_t)g_th_n] = g_th_pending[slot];
        g_th_pending[slot] = 0;
    }

    int cx = 0, cy = 0, cw = w, ch = h;
    if (crop && crop[2] > 0 && crop[3] > 0) { cx = crop[0]; cy = crop[1]; cw = crop[2]; ch = crop[3]; }
    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cx + cw > w) cw = w - cx;
    if (cy + ch > h) ch = h - cy;
    window_settings_live();
    // The menu overlay: laid out before recording (its first frame starts the ImGui back end and uploads the font)
    const char *menu_shot = NULL;
    MenuTarget mt = {c->instance, c->phys, c->device, c->gfx_family, c->gfx_queue, g_sc_fmt, g_sc_ext.width,
                     g_sc_ext.height, cw, ch, g_scaler};
    int overlay = g_sc_view[idx] &&
                  menu_overlay_frame(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), &mt, &menu_shot);
    int cap = (g_capture || menu_shot) && ensure_capture_buffer();
    int thumb = g_th_n && ensure_thumbs();

    VkCommandBuffer cb = g_cb[slot];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &bi);
    uint64_t crumb = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    vk_crumb(cb, VK_CRUMB_PRESENT, crumb);
    VkImage dst = g_sc_img[idx];
    sync_barrier(cb, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    const int rect[4] = {cx, cy, cw, ch};
    int to[4];
    if (fit_rect(cw, ch, (int)g_sc_ext.width, (int)g_sc_ext.height, to)) {
        // black bars: clear the whole image, then the picture's writes below wait for the clear
        VkClearColorValue black = {{0, 0, 0, 1}};
        VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &all);
        sync_barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }
    // FSR (OOT_UPSCALER) when the picture's place on the swapchain is larger than the crop; else the bilinear blit
    int fsr = general && vk_fsr_record(cb, slot, src, fmt, w, h, rect, dst, to);
    g_scaler = fsr ? 2 : cw != to[2] || ch != to[3];
    if (!fsr) {
        if (general)   // a GENERAL source (vk_texture.c images): earlier submissions' writes become visible to the blit here
            sync_barrier(cb, src, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        // NVN's lower-left window origin (nvn_api_core.c): the frame sits bottom-up in memory and the display turns
        // it over, which the blit does by reading the rows in reverse (a negative source extent).
        int y0 = cy, y1 = cy + ch;
        if (nvn_origin_lower_left()) y0 = cy + ch, y1 = cy;
        VkImageBlit region = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{cx, y0, 0}, {cx + cw, y1, 1}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{to[0], to[1], 0}, {to[0] + to[2], to[1] + to[3], 1}}};
        vkCmdBlitImage(cb, src, general ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);
        if (general)   // later submissions that write the source wait for the blit's read
            sync_barrier(cb, src, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
    }
    // last: the image's layout, and the stage and access of its last write, as each step leaves it
    VkImageLayout last = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    VkPipelineStageFlags2 last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    VkAccessFlags2 last_write = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    if (overlay) {
        sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, last_stage, last_write,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = g_sc_view[idx],
                                        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
        VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, g_sc_ext}, .layerCount = 1,
                              .colorAttachmentCount = 1, .pColorAttachments = &ca};
        vkCmdBeginRendering(cb, &ri);
        menu_overlay_draw(cb);
        vkCmdEndRendering(cb);
        last = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        last_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        last_write = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }
    if (cap) {
        sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, last_stage, last_write,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy bc = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                .imageExtent = {g_sc_ext.width, g_sc_ext.height, 1}};
        vkCmdCopyImageToBuffer(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_cap_buf, 1, &bc);
        last = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        last_write = 0;
    }
    if (thumb) {
        if (last != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
            sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, last_stage, last_write,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        last = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        last_write = 0;
        record_thumb(cb, dst, slot);
    }
    sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, last_stage, last_write, VK_PIPELINE_STAGE_2_NONE, 0);
    vk_crumb(cb, VK_CRUMB_PRESENT_END, crumb);
    vkEndCommandBuffer(cb);

    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_acquire[slot],
                                  .stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT};
    VkSemaphoreSubmitInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_render_done[idx],
                                 .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = 1, .pWaitSemaphoreInfos = &wait,
                        .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbi, .signalSemaphoreInfoCount = 1,
                        .pSignalSemaphoreInfos = &sig};
    vk_queue_lock();
    r = vkQueueSubmit2(c->gfx_queue, 1, &si, g_fence[slot]);
    vk_queue_unlock();
    if (r == VK_SUCCESS) {
        g_flip_on = 1;
        g_flip_idx = idx;
        if (!defer) flip_locked();
    } else {
        vk_device_fault_report(r);
        rt_log("gfx: present submit failed (%d)", (int)r);
    }
    g_frame++;
    if (cap) {
        vkWaitForFences(c->device, 1, &g_fence[slot], VK_TRUE, 5000000000ull);
        vmaInvalidateAllocation(c->vma, g_cap_alloc, 0, VK_WHOLE_SIZE);
        g_cap_w = (int)g_sc_ext.width;
        g_cap_h = (int)g_sc_ext.height;
        g_cap_valid = 1;
        if (menu_shot) save_capture(menu_shot);
    }
out:
    rmutex_unlock(&g_pl);
}

void vk_present_image(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4]) {
    present_image(src, layout, fmt, w, h, crop, 0);
}

void vk_present_blit(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4]) {
    present_image(src, layout, fmt, w, h, crop, 1);
}

void vk_present_flip(void) {
    rmutex_lock(&g_pl);
    flip_locked();
    rmutex_unlock(&g_pl);
}

void gfx_window_resize(int cw, int chh) {
    gfx_window_start();
    ResizeReq r = {cw, chh};
    run_on_window(resize_window, &r);
    rmutex_lock(&g_pl);
    create_swapchain();
    rmutex_unlock(&g_pl);
}

// Called by vk_device.c's release before the device goes away; the window and its thread stay.
void gfx_window_release_vk(void) {
    VkCtx *c = vk_ctx();
    rmutex_lock(&g_pl);
    if (c && g_started) {
        vk_queue_lock();
        vkDeviceWaitIdle(c->device);
        vk_queue_unlock();
        g_flip_on = 0;   // its blit is done (device idle); the swapchain goes away unpresented
        menu_vk_release();
        vk_fsr_release();
        free_capture();
        free_thumbs();
        for (int i = 0; i < FRAMES; i++) {
            vkDestroySemaphore(c->device, g_acquire[i], NULL);
            vkDestroyFence(c->device, g_fence[i], NULL);
        }
        for (uint32_t i = 0; i < g_sc_n; i++) {
            vkDestroySemaphore(c->device, g_render_done[i], NULL);
            if (g_sc_view[i]) vkDestroyImageView(c->device, g_sc_view[i], NULL);
            g_sc_view[i] = VK_NULL_HANDLE;
        }
        if (g_pool) vkDestroyCommandPool(c->device, g_pool, NULL);
        if (g_sc) vkDestroySwapchainKHR(c->device, g_sc, NULL);
        if (g_surface) vkDestroySurfaceKHR(c->instance, g_surface, NULL);
        g_sc = VK_NULL_HANDLE; g_surface = VK_NULL_HANDLE; g_pool = VK_NULL_HANDLE; g_sc_n = 0;
        g_started = 0;     // the window and its thread stay; the next gfx_window_start makes a new surface
    }
    rmutex_unlock(&g_pl);
}

void gfx_window_start(void) {
    rmutex_lock(&g_start_lock);
    if (!g_started) {
        if (!vk_init()) rt_fatal("gfx: no usable Vulkan device (see the vk: lines above)");
        VkCtx *c = vk_ctx();
        if (!window_open()) rt_fatal("gfx: window creation failed");
        g_surface = window_surface(c->instance);
        if (!g_surface) rt_fatal("gfx: no Vulkan surface for the window");

        VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                       .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c->gfx_family};
        vkCreateCommandPool(c->device, &pci, NULL, &g_pool);
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = FRAMES};
        vkAllocateCommandBuffers(c->device, &ai, g_cb);
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT};
        VkSemaphoreCreateInfo semi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (int i = 0; i < FRAMES; i++) {
            vkCreateFence(c->device, &fi, NULL, &g_fence[i]);
            vkCreateSemaphore(c->device, &semi, NULL, &g_acquire[i]);
        }
        g_frame = 0;
        g_started = 1;
        rmutex_lock(&g_pl);
        create_swapchain();
        rmutex_unlock(&g_pl);
        int cw = 0, ch = 0;
        client_size(&cw, &ch);
        rt_log("gfx: window %dx%d at DPI %u, swapchain %ux%u format %d", cw, ch, gfx_window_dpi(), g_sc_ext.width,
               g_sc_ext.height, (int)g_sc_fmt);
    }
    rmutex_unlock(&g_start_lock);
    vsync_ensure();
}
