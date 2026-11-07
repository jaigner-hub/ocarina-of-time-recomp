// From metroid-prime-4-recomp runtime/romfs_path.h @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied unchanged.
// From fe3h-pc-recomp runtime/romfs_path.h @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied unchanged.
// From totk-pc-recomp runtime/romfs_path.h @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), copied unchanged.
// From MacNativeGames games/totk/runtime/romfs_path.h @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames
// contributors).
// RomFS path validation shared by the runtime (romfs.c) and its probe (fs_host.c probe_fs).
#pragma once
#include <stddef.h>
#include <string.h>

// A RomFS-relative path: non-empty, no leading '/', no '\', no empty, "." or ".." component.
static inline int romfs_path_valid_relative(const char *path) {
    if (!path || !*path || *path == '/' || strchr(path, '\\')) return 0;
    for (const char *part = path; *part;) {
        const char *end = strchr(part, '/');
        size_t size = end ? (size_t)(end - part) : strlen(part);
        if (!size || (size == 1 && part[0] == '.') || (size == 2 && part[0] == '.' && part[1] == '.')) return 0;
        if (!end) break;
        part = end + 1;
    }
    return 1;
}

// "<mount>:/<path>" -> "<path>", or NULL. Audio stream requests join a mount root and a leading-slash resource.
// The sound streams make "content://Sound/..." (repeated root separators) and the voice streams make
// "content://Voice/Resource//Stream/..." (a repeated separator inside the path): nn::fs accepts both, so every run
// of separators counts as one. Traversal ("." and ".."), backslashes and an empty path are still rejected. A path with
// a repeated interior separator comes back in a per-thread buffer, valid until the thread's next call.
static inline const char *romfs_path_relative(const char *mount, const char *path) {
    if (!mount || !*mount || !path) return NULL;
    size_t size = strlen(mount);
    if (strncmp(path, mount, size) || path[size] != ':' || path[size + 1] != '/') return NULL;
    path += size + 2;
    while (*path == '/') ++path;
    if (!strstr(path, "//")) return romfs_path_valid_relative(path) ? path : NULL;
    static _Thread_local char collapsed[1024];
    size_t n = 0;
    for (const char *s = path; *s; s++) {
        if (*s == '/' && s[1] == '/') continue;
        if (n + 1 >= sizeof collapsed) return NULL;
        collapsed[n++] = *s;
    }
    collapsed[n] = 0;
    return romfs_path_valid_relative(collapsed) ? collapsed : NULL;
}
