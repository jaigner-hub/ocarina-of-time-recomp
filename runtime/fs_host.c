// From metroid-prime-4-recomp runtime/fs_host.c @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied with mp4 renamed to oot.
// From fe3h-pc-recomp runtime/fs_host.c @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied with fe3h renamed to mp4; adds GetSaveDataSize with this game's declared save sizes; the fs trace logs every open and entry-type query; path arguments take '\\' as a separator and drop "." components, as Horizon does and this game writes; file and directory handles are multiples of 8; RomFS directories go to romfs.c.
// From totk-pc-recomp runtime/fs_host.c @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), renamed to fe3h; adds the older SDK calls GetDirectoryEntryCount, CommitSaveData, EnsureSaveData and SetAllocator (FE3H, SDK 6.5.0); logs failed opens with FE3H_TRACE=fs and the backtrace of an invalid close.
// Adapted from MacNativeGames games/totk/runtime/fs_host.c @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames
// contributors); the probe follows MacNativeGames' romfs.c runtime_fs_probe and fs_host.c runtime_save_probe.
// nn::fs host implementation of the functions main imports. RomFS paths and
// handles go to the read-only backend (romfs.c); save, temporary and cache storage are host directories under the
// profile directory (rt_config()->profile, default build/profile):
//
//   <profile>/save/<uid hi><uid lo>/   committed application save data, created on first use as the system launcher
//                                      does for titles that declare save data
//   <profile>/save-tx/                 the save mount's working copy and Commit's staging (section "save
//                                      transactions")
//   <profile>/save-backup/<time>/      the three committed saves before the current one
//   <profile>/temp/                    emptied at start
//   <profile>/cache/<index>/           cache storage
//
// The save mount is journaled as on Horizon: changes go to a working copy and become the committed save atomically at
// Commit; a crash or an Unmount before Commit leaves the previous committed save. Commit returns at once and the copy
// runs on a background thread (section "save transactions"); fs_save_flush lets the exit paths wait for it. Paths are
// UTF-8 and go to the W APIs as UTF-16.
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fs.h"
#include "host.h"
#include "platform/platform.h"
#include "romfs_path.h"

#define OOT_APPLICATION_ID 0x0100f2c0115b6000ull
#define FILE_BASE 0x200000000ull
#define DIR_BASE 0x300000000ull
#define NFILES 1024
#define NDIRS 256
#define NMOUNTS 16
#define PATH_CAP 1100

enum { M_FREE, M_SAVE, M_TEMP, M_CACHE, M_SAVE_RO };   // M_SAVE: the working copy; M_SAVE_RO: the committed save

typedef struct { int kind; char name[32]; char root[1024]; } Mount;
typedef struct { HANDLE h; int mode, save; } File;   // h NULL = free; save: on a save-data mount
typedef struct { HANDLE find; WIN32_FIND_DATAW next; int has_next, mode; wchar_t *pattern; } Dir;   // find NULL = free; pattern: <dir>/* for GetDirectoryEntryCount

static Mount mounts[NMOUNTS];
static File files[NFILES];
static Dir dirs[NDIRS];
static SRWLOCK lock = SRWLOCK_INIT;
static char profile[1024], save_root[1100], work_root[1100];
static volatile LONG storage_ready;

static void need(int c, const char *m) {
    if (!c) rt_fatal("fs: %s", m);
}

wchar_t *fs_wide(const char *s) {
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    if (n <= 0) rt_fatal("fs: path is not valid UTF-8");
    wchar_t *w = malloc((size_t)n * sizeof *w);
    need(w != NULL, "path allocation");
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w, n);
    return w;
}

static DWORD attributes(const char *path) {
    wchar_t *w = fs_wide(path);
    DWORD a = GetFileAttributesW(w);
    free(w);
    return a;
}

// Save in progress: a save-data file open for writing, or save data changed in the last 3 s and not committed yet
// (TotK writes a slot's files back to back and then commits; between the files nothing is open for a moment).
static volatile LONG save_writers, save_dirty;
static volatile LONG64 save_changed_ns;
static void save_wait_commit(void);
static void save_touched(void) {
    save_wait_commit();   // every change to save data comes here first: none lands in a commit's snapshot
    InterlockedExchange64(&save_changed_ns, (LONG64)plat_now_ns());
    InterlockedExchange(&save_dirty, 1);
}
int fs_save_busy(void) {
    if (InterlockedCompareExchange(&save_writers, 0, 0) > 0) return 1;
    return InterlockedCompareExchange(&save_dirty, 0, 0) &&
           plat_now_ns() - (uint64_t)InterlockedCompareExchange64(&save_changed_ns, 0, 0) < 3000000000ull;
}

// ---- host directories
static int make_dir(const char *p) {
    wchar_t *w = fs_wide(p);
    int ok = CreateDirectoryW(w, NULL) || GetLastError() == ERROR_ALREADY_EXISTS;
    free(w);
    return ok;
}

// p and every missing parent.
static void make_dirs(const char *p) {
    char buf[1100];
    need(snprintf(buf, sizeof buf, "%s", p) < (int)sizeof buf, "profile path");
    for (char *q = buf + 1; *q; q++)
        if ((*q == '/' || *q == '\\') && q[-1] != ':') {
            char c = *q;
            *q = 0;
            make_dir(buf);
            *q = c;
        }
    if (!make_dir(buf)) rt_fatal("fs: cannot create directory %s (error %lu)", buf, GetLastError());
}

// Remove a directory tree; a reparse point is removed as a link, never followed. 0 also when nothing is there.
static int remove_tree_w(const wchar_t *path) {
    DWORD a = GetFileAttributesW(path);
    if (a == INVALID_FILE_ATTRIBUTES)
        return GetLastError() == ERROR_FILE_NOT_FOUND || GetLastError() == ERROR_PATH_NOT_FOUND ? 0 : -1;
    if (!(a & FILE_ATTRIBUTE_DIRECTORY)) {
        if (a & FILE_ATTRIBUTE_READONLY) SetFileAttributesW(path, a & ~FILE_ATTRIBUTE_READONLY);
        return DeleteFileW(path) ? 0 : -1;
    }
    int bad = 0;
    if (!(a & FILE_ATTRIBUTE_REPARSE_POINT)) {
        size_t n = wcslen(path);
        wchar_t *pattern = malloc((n + 3) * sizeof *pattern);
        need(pattern != NULL, "path allocation");
        memcpy(pattern, path, n * sizeof *pattern);
        memcpy(pattern + n, L"\\*", 3 * sizeof *pattern);
        WIN32_FIND_DATAW fd;
        HANDLE f = FindFirstFileW(pattern, &fd);
        free(pattern);
        if (f != INVALID_HANDLE_VALUE) {
            do {
                if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
                size_t k = wcslen(fd.cFileName);
                wchar_t *child = malloc((n + k + 2) * sizeof *child);
                need(child != NULL, "path allocation");
                memcpy(child, path, n * sizeof *child);
                child[n] = L'\\';
                memcpy(child + n + 1, fd.cFileName, (k + 1) * sizeof *child);
                bad |= remove_tree_w(child);
                free(child);
            } while (FindNextFileW(f, &fd));
            FindClose(f);
        }
    }
    return bad | (RemoveDirectoryW(path) ? 0 : -1);
}

static int remove_tree(const char *path) {
    wchar_t *w = fs_wide(path);
    int r = remove_tree_w(w);
    free(w);
    return r;
}

// ---- save transactions
// The save mount's root is a working copy, made from the committed save when the first save mount appears. Commit
// copies it to a staging directory and swaps that in for the committed save:
//
//   <profile>/save/<uid>/          committed save: what the next start loads
//   <profile>/save-tx/work/        working copy, the save mount's root; deleted at the last Unmount and at start
//   <profile>/save-tx/new/         Commit's staging copy of the working copy
//   <profile>/save-tx/old/         the committed save a Commit replaced, on its way to save-backup
//
// Commit: (1) copy work to new and flush every file (FlushFileBuffers) and directory, a file equal to the committed
// one as a hard link to it; (2) rename save/<uid> to old;
// (3) rename new to save/<uid>, the commit point; (4) move old to save-backup/<UTC time> and keep the newest three.
// Both renames are MoveFileExW with MOVEFILE_WRITE_THROUGH. The game's Commit returns 0 once steps (1) to (4) are
// queued; a thread runs them (95-135 ms on the game thread before, in the 1.4.3 fight save) while every change to the
// working copy waits in save_touched, and Unmount and the next Commit wait too. A crash in that time loses this Commit
// only, and a failed one is logged instead of returned. save/<uid> is renamed away only after new is complete
// on disk, so the recovery rule at start (save_recover) is:
//   save/<uid> present   it is the committed save (the old one if the crash came before (3), the new one after);
//                        new, if present, is an unfinished copy and is deleted (rollback);
//   save/<uid> absent    the crash came between (2) and (3) and new is complete: rename it to save/<uid> (roll
//                        forward); with no new either, old goes back to save/<uid>.
// Then old, if present, goes to save-backup, and work (changes no Commit took) is deleted.
#define SAVE_BACKUPS 3
#define COPY_CHUNK (1u << 20)

static char tx_root[1100], backup_root[1100];
static wchar_t *w_save, *w_work, *w_new, *w_old, *w_backup;
static SRWLOCK tx_lock = SRWLOCK_INIT;   // taken before lock, never inside it
static int save_mounts;                  // M_SAVE mounts: the working copy exists while this is above 0

// Probe hook: Commit stops at its tx_crash_at-th step (from 1) and leaves the disk as a process killed there would.
static int tx_crash_at, tx_step, tx_dead;
static int crashed(void) {
    if (tx_crash_at && ++tx_step == tx_crash_at) tx_dead = 1;
    return tx_dead;
}

static wchar_t *wjoin(const wchar_t *a, const wchar_t *b) {
    size_t n = wcslen(a), k = wcslen(b);
    wchar_t *w = malloc((n + k + 2) * sizeof *w);
    need(w != NULL, "path allocation");
    memcpy(w, a, n * sizeof *w);
    w[n] = L'\\';
    memcpy(w + n + 1, b, (k + 1) * sizeof *w);
    return w;
}

static int exists_w(const wchar_t *p) { return GetFileAttributesW(p) != INVALID_FILE_ATTRIBUTES; }

// Rename a directory, durably; retried for up to 2 s while another process (a virus scanner, the indexer) holds a
// file in it. 0 on success.
static int move_dir_w(const wchar_t *from, const wchar_t *to) {
    for (int i = 0; i < 40; i++) {
        if (MoveFileExW(from, to, MOVEFILE_WRITE_THROUGH)) return 0;
        DWORD e = GetLastError();
        if (e != ERROR_ACCESS_DENIED && e != ERROR_SHARING_VIOLATION) return -1;
        Sleep(50);
    }
    return -1;
}

// Copy one file; the source may be open for writing (a save file the game has not closed). 0 on success.
static int copy_file_w(const wchar_t *src, const wchar_t *dst, int durable, uint64_t *bytes) {
    HANDLE in = CreateFileW(src, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                            OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (in == INVALID_HANDLE_VALUE) return -1;
    HANDLE out = CreateFileW(dst, GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
    if (out == INVALID_HANDLE_VALUE) {
        CloseHandle(in);
        return -1;
    }
    uint8_t *buf = malloc(COPY_CHUNK);
    need(buf != NULL, "copy buffer");
    int ok = 1;
    for (;;) {
        DWORD got = 0, put = 0;
        if (!ReadFile(in, buf, COPY_CHUNK, &got, NULL)) ok = 0;
        if (!ok || !got) break;
        if (!WriteFile(out, buf, got, &put, NULL) || put != got) ok = 0;
        if (!ok) break;
        *bytes += got;
    }
    if (ok && durable) ok = FlushFileBuffers(out);
    free(buf);
    CloseHandle(in);
    CloseHandle(out);
    return ok ? 0 : -1;
}

// Whether files a and b hold the same bytes; 0 when either cannot be read.
static int same_file_w(const wchar_t *a, const wchar_t *b) {
    HANDLE h[2];
    const wchar_t *p[2] = {a, b};
    for (int i = 0; i < 2; i++)
        h[i] = CreateFileW(p[i], GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                           OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    LARGE_INTEGER na, nb;
    int same = h[0] != INVALID_HANDLE_VALUE && h[1] != INVALID_HANDLE_VALUE && GetFileSizeEx(h[0], &na) &&
               GetFileSizeEx(h[1], &nb) && na.QuadPart == nb.QuadPart;
    uint8_t *buf = same ? malloc(2 * (size_t)COPY_CHUNK) : NULL;
    if (same) need(buf != NULL, "compare buffer");
    while (same) {
        DWORD ga = 0, gb = 0;
        if (!ReadFile(h[0], buf, COPY_CHUNK, &ga, NULL) || !ReadFile(h[1], buf + COPY_CHUNK, COPY_CHUNK, &gb, NULL) ||
            ga != gb || memcmp(buf, buf + COPY_CHUNK, ga))
            same = 0;
        else if (!ga)
            break;
    }
    free(buf);
    for (int i = 0; i < 2; i++)
        if (h[i] != INVALID_HANDLE_VALUE) CloseHandle(h[i]);
    return same;
}

// Copy the tree src to dst, which must not exist; reparse points are skipped. durable: every file is flushed, and
// every directory as far as the file system allows (NTFS journals the entries either way). base (or NULL): the
// committed save; a file whose bytes equal base's file at the same path becomes a hard link to it instead of a copy
// (*nlinked), as committed files are never written in place (the game writes the working copy). A save commit then
// writes the 3-4 files of the slot the game saved instead of all 82 (300-380 ms of flushes on the game thread).
// 0 on success.
static int copy_tree_w(const wchar_t *src, const wchar_t *dst, const wchar_t *base, int durable, int *nfiles,
                       int *nlinked, uint64_t *bytes) {
    if (!CreateDirectoryW(dst, NULL)) return -1;
    wchar_t *pattern = wjoin(src, L"*");
    WIN32_FIND_DATAW fd;
    HANDLE f = FindFirstFileW(pattern, &fd);
    free(pattern);
    if (f == INVALID_HANDLE_VALUE) return -1;
    int bad = 0;
    do {
        if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..") ||
            (fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT))
            continue;
        wchar_t *s = wjoin(src, fd.cFileName), *d = wjoin(dst, fd.cFileName);
        wchar_t *b = base ? wjoin(base, fd.cFileName) : NULL;
        if (b && !exists_w(b)) free(b), b = NULL;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) bad = copy_tree_w(s, d, b, durable, nfiles, nlinked, bytes);
        else {
            if (b && same_file_w(s, b) && CreateHardLinkW(d, b, NULL)) bad = 0, ++*nlinked;
            else bad = copy_file_w(s, d, durable, bytes);
            *nfiles += !bad;
            if (!bad && durable && crashed()) bad = -1;
        }
        free(s);
        free(d);
        free(b);
    } while (!bad && FindNextFileW(f, &fd));
    FindClose(f);
    if (!bad && durable) {
        HANDLE h = CreateFileW(dst, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                               OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
        if (h != INVALID_HANDLE_VALUE) {
            FlushFileBuffers(h);
            CloseHandle(h);
        }
    }
    return bad;
}

static int name_cmp(const void *a, const void *b) { return wcscmp(a, b); }

// Move old to save-backup/<UTC time> and delete all backups but the newest SAVE_BACKUPS.
static void rotate_backup(void) {
    if (!exists_w(w_old)) return;
    SYSTEMTIME t;
    GetSystemTime(&t);
    wchar_t name[48], *dst = NULL;
    for (int k = 0; !dst || exists_w(dst); k++) {
        free(dst);
        int n = swprintf(name, 48, L"%04u%02u%02u-%02u%02u%02u-%03u", (unsigned)t.wYear, (unsigned)t.wMonth,
                         (unsigned)t.wDay, (unsigned)t.wHour, (unsigned)t.wMinute, (unsigned)t.wSecond,
                         (unsigned)t.wMilliseconds);
        if (k) swprintf(name + n, 48 - (size_t)n, L"-%d", k);
        dst = wjoin(w_backup, name);
    }
    if (move_dir_w(w_old, dst)) {
        rt_log("fs: cannot move the replaced save to save-backup (error %lu); deleted", GetLastError());
        remove_tree_w(w_old);
    }
    free(dst);
    static wchar_t names[64][48];
    int n = 0;
    wchar_t *pattern = wjoin(w_backup, L"*");
    WIN32_FIND_DATAW fd;
    HANDLE f = FindFirstFileW(pattern, &fd);
    free(pattern);
    if (f != INVALID_HANDLE_VALUE) {
        do
            if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && fd.cFileName[0] >= L'0' &&
                fd.cFileName[0] <= L'9' && wcslen(fd.cFileName) < 48 && n < 64)
                wcscpy(names[n++], fd.cFileName);
        while (FindNextFileW(f, &fd));
        FindClose(f);
    }
    qsort(names, (size_t)n, sizeof *names, name_cmp);
    for (int i = 0; i + SAVE_BACKUPS < n; i++) {
        wchar_t *p = wjoin(w_backup, names[i]);
        remove_tree_w(p);
        free(p);
    }
}

// Startup: finish or roll back an interrupted Commit and drop uncommitted changes (the rule above).
static void save_recover(void) {
    const char *did = "nothing to recover";
    int has_save = exists_w(w_save);
    if (!has_save && exists_w(w_new)) {
        if (move_dir_w(w_new, w_save)) rt_fatal("fs: cannot complete an interrupted save commit (error %lu)",
                                                GetLastError());
        did = "completed an interrupted commit";
    } else if (!has_save && exists_w(w_old)) {
        if (move_dir_w(w_old, w_save)) rt_fatal("fs: cannot roll back an interrupted save commit (error %lu)",
                                                GetLastError());
        did = "rolled back an interrupted commit";
    } else if (exists_w(w_new)) {
        if (remove_tree_w(w_new)) rt_fatal("fs: cannot delete an unfinished save commit (error %lu)", GetLastError());
        did = "rolled back an unfinished commit";
    } else if (exists_w(w_old))
        did = "moved the save an interrupted commit replaced to save-backup";
    rotate_backup();
    if (exists_w(w_work)) {
        if (remove_tree_w(w_work)) rt_fatal("fs: cannot delete the old save working copy (error %lu)", GetLastError());
        if (!strcmp(did, "nothing to recover"))
            did = "deleted the last run's working copy (changes after its last Commit)";
    }
    rt_log("host save recovery: %s", did);
}

// The first save mount: a fresh working copy of the committed save.
static void work_create(void) {
    int n = 0, linked = 0;
    uint64_t bytes = 0;
    if (remove_tree_w(w_work) || copy_tree_w(w_save, w_work, NULL, 0, &n, &linked, &bytes))
        rt_fatal("fs: cannot make the save working copy (error %lu)", GetLastError());
    rt_log("host save working copy: %d files, %.1f MiB", n, bytes / 1048576.0);
}

// The last save unmount: changes no Commit took are dropped.
static void work_discard(void) {
    if (InterlockedExchange(&save_dirty, 0)) rt_log("fs: save unmounted without Commit; changes discarded");
    if (remove_tree_w(w_work)) rt_log("fs: save working copy not fully deleted (a file still open?)");
}

// Commit of the save mount, tx_lock held. Steps (1) to (4) above; every failure before (3) leaves the committed save.
static uint32_t save_commit_locked(void) {
    uint64_t t0 = plat_now_ns(), bytes = 0;
    int nfiles = 0, linked = 0;
    tx_step = tx_dead = 0;
    rotate_backup();   // an old left by a failed (4)
    const wchar_t *base = exists_w(w_save) ? w_save : NULL;
    if (remove_tree_w(w_new) || copy_tree_w(w_work, w_new, base, 1, &nfiles, &linked, &bytes)) {
        if (tx_dead) return FS_R_INVALID_ARGUMENT;
        rt_log("fs: save commit: copying the working copy failed (error %lu); committed save unchanged",
               GetLastError());
        remove_tree_w(w_new);
        return FS_R_INVALID_ARGUMENT;
    }
    if (crashed()) return FS_R_INVALID_ARGUMENT;
    if (exists_w(w_save) && move_dir_w(w_save, w_old)) {
        rt_log("fs: save commit: cannot move the committed save (error %lu); committed save unchanged",
               GetLastError());
        remove_tree_w(w_new);
        return FS_R_INVALID_ARGUMENT;
    }
    if (crashed()) return FS_R_INVALID_ARGUMENT;
    if (move_dir_w(w_new, w_save)) {
        DWORD e = GetLastError();
        if (move_dir_w(w_old, w_save))
            rt_fatal("fs: save commit: cannot swap in the new save (error %lu); the next start completes it", e);
        rt_log("fs: save commit: cannot swap in the new save (error %lu); committed save unchanged", e);
        remove_tree_w(w_new);
        return FS_R_INVALID_ARGUMENT;
    }
    if (crashed()) return FS_R_INVALID_ARGUMENT;
    rotate_backup();
    InterlockedExchange(&save_dirty, 0);
    rt_log("fs commit save: %d files, %d changed (%.1f MiB written), %d unchanged linked, in %.0f ms", nfiles,
           nfiles - linked, bytes / 1048576.0, linked, (plat_now_ns() - t0) / 1e6);
    return 0;
}

// The background commit: commit_pending from the game's Commit until the thread has run save_commit_locked.
static SRWLOCK pend_lock = SRWLOCK_INIT;
static CONDITION_VARIABLE pend_cv = CONDITION_VARIABLE_INIT;
static volatile LONG commit_pending;

static void save_wait_commit(void) {
    if (!InterlockedCompareExchange(&commit_pending, 0, 0)) return;
    AcquireSRWLockExclusive(&pend_lock);
    while (commit_pending) SleepConditionVariableSRW(&pend_cv, &pend_lock, INFINITE, 0);
    ReleaseSRWLockExclusive(&pend_lock);
}

void fs_save_flush(void) { save_wait_commit(); }

static uint32_t commit_run(void) {
    AcquireSRWLockExclusive(&tx_lock);
    uint32_t r = save_commit_locked();
    InterlockedDecrement(&save_writers);
    ReleaseSRWLockExclusive(&tx_lock);
    AcquireSRWLockExclusive(&pend_lock);
    commit_pending = 0;
    ReleaseSRWLockExclusive(&pend_lock);
    WakeAllConditionVariable(&pend_cv);
    return r;
}

static DWORD WINAPI commit_main(void *arg) {
    (void)arg;
    commit_run();
    return 0;
}

static uint32_t save_commit(void) {
    save_wait_commit();   // the previous Commit
    static int quiet;
    if (!InterlockedCompareExchange(&save_dirty, 0, 0)) {
        if (!quiet++) rt_log("fs commit save: nothing changed (logged once)");
        return 0;
    }
    InterlockedIncrement(&save_writers);   // fs_save_busy until the thread is done
    InterlockedExchange(&commit_pending, 1);
    HANDLE t = CreateThread(NULL, 0, commit_main, NULL, 0, NULL);
    if (t) {
        CloseHandle(t);
        return 0;
    }
    rt_log("fs: cannot start the save commit thread (error %lu); committing on the game thread", GetLastError());
    return commit_run();
}

// ---- storage roots
static BOOL CALLBACK storage_init(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once, (void)param, (void)ctx;
    if (!*profile)
        need(snprintf(profile, sizeof profile, "%s", rt_config()->profile) < (int)sizeof profile, "profile path");
    uint64_t uid[2];
    char p[PATH_CAP];
    make_dirs(profile);
    account_uid(uid);
    snprintf(p, sizeof p, "%s/save", profile);
    make_dirs(p);
    need(snprintf(save_root, sizeof save_root, "%s/save/%016llx%016llx", profile, (unsigned long long)uid[1],
                  (unsigned long long)uid[0]) < (int)sizeof save_root, "save path");
    need(snprintf(tx_root, sizeof tx_root, "%s/save-tx", profile) < (int)sizeof tx_root, "save path");
    snprintf(work_root, sizeof work_root, "%s/work", tx_root);
    snprintf(backup_root, sizeof backup_root, "%s/save-backup", profile);
    make_dirs(tx_root);
    make_dirs(backup_root);
    w_save = fs_wide(save_root);
    w_work = fs_wide(work_root);
    snprintf(p, sizeof p, "%s/new", tx_root);
    w_new = fs_wide(p);
    snprintf(p, sizeof p, "%s/old", tx_root);
    w_old = fs_wide(p);
    w_backup = fs_wide(backup_root);
    save_recover();   // before save_root is created: an absent save_root means a commit to complete
    make_dirs(save_root);
    snprintf(p, sizeof p, "%s/temp", profile);
    remove_tree(p);
    make_dirs(p);
    snprintf(p, sizeof p, "%s/cache", profile);
    make_dirs(p);
    InterlockedExchange(&storage_ready, 1);
    rt_log("host save storage: %s", save_root);
    return TRUE;
}

static void ready(void) {
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, storage_init, NULL, NULL);
}

void fs_host_set_profile(const char *dir) {
    if (InterlockedCompareExchange(&storage_ready, 0, 0)) rt_fatal("fs_host_set_profile after the first storage use");
    need(dir && snprintf(profile, sizeof profile, "%s", dir) < (int)sizeof profile, "profile path");
}

// ---- mounts and paths
static uint32_t mount_add(const char *name, int kind, const char *root) {
    if (!name || !*name || strlen(name) >= 32 || strchr(name, ':') || strchr(name, '/')) return FS_R_INVALID_ARGUMENT;
    char as_path[40];
    snprintf(as_path, sizeof as_path, "%s:", name);
    if (romfs_owns(as_path)) return FS_R_ALREADY_EXISTS;   // the RomFS mount's name
    AcquireSRWLockExclusive(&lock);
    int slot = -1;
    for (int i = 0; i < NMOUNTS; i++) {
        if (mounts[i].kind && !strcmp(mounts[i].name, name)) {
            ReleaseSRWLockExclusive(&lock);
            return FS_R_ALREADY_EXISTS;
        }
        if (!mounts[i].kind && slot < 0) slot = i;
    }
    if (slot < 0) rt_fatal("fs mount table full");
    mounts[slot].kind = kind;
    strcpy(mounts[slot].name, name);
    snprintf(mounts[slot].root, sizeof mounts[slot].root, "%s", root);
    ReleaseSRWLockExclusive(&lock);
    rt_log("fs mount %s: -> %s", name, root);
    return 0;
}

// A path component Windows stores under exactly this name: not empty, ".", "..", no reserved character, no trailing
// dot or space (Windows strips those, which would alias two names).
static int component_ok(const char *c, size_t n) {
    if (!n || (n == 1 && c[0] == '.') || (n == 2 && c[0] == '.' && c[1] == '.')) return 0;
    if (c[n - 1] == '.' || c[n - 1] == ' ') return 0;
    for (size_t i = 0; i < n; i++)
        if ((unsigned char)c[i] < 32 || strchr("\\:*?\"<>|", c[i])) return 0;
    return 1;
}

// "name:/a/b" -> host path; 0 on success; *kind = the mount's kind (M_SAVE ...) when kind is given.
static uint32_t host_path_kind(const char *path, char *out, size_t cap, int *kind) {
    if (!path) return FS_R_INVALID_PATH;
    const char *colon = strchr(path, ':');
    if (!colon || colon[1] != '/') return FS_R_INVALID_PATH;
    size_t n = (size_t)(colon - path);
    char root[1024];
    int k = M_FREE;
    AcquireSRWLockShared(&lock);
    for (int i = 0; i < NMOUNTS; i++)
        if (mounts[i].kind && strlen(mounts[i].name) == n && !strncmp(mounts[i].name, path, n)) {
            strcpy(root, mounts[i].root);
            k = mounts[i].kind;
        }
    ReleaseSRWLockShared(&lock);
    if (k == M_FREE) return FS_R_PATH_NOT_FOUND;
    if (kind) *kind = k;
    const char *rest = colon + 1;
    for (const char *q = rest + 1; *q;) {
        const char *e = strchr(q, '/');
        size_t len = e ? (size_t)(e - q) : strlen(q);
        if (!e && !len) break;
        if (!component_ok(q, len)) return FS_R_INVALID_PATH;
        q = e ? e + 1 : q + len;
    }
    if (snprintf(out, cap, "%s%s", root, rest) >= (int)cap) return FS_R_INVALID_PATH;
    size_t L = strlen(out);
    while (L > strlen(root) && out[L - 1] == '/') out[--L] = 0;
    return 0;
}

static uint32_t host_path(const char *path, char *out, size_t cap) { return host_path_kind(path, out, cap, NULL); }

// A change to save data through a path (create, delete, rename, directories) marks the save as in progress.
static uint32_t save_change_path(const char *path, char *out, size_t cap) {
    int kind = M_FREE;
    uint32_t r = host_path_kind(path, out, cap, &kind);
    if (!r && kind == M_SAVE_RO) return FS_R_UNSUPPORTED_OPEN;
    if (!r && kind == M_SAVE) save_touched();
    return r;
}

static uint32_t win_result(DWORD e) {
    switch (e) {
    case ERROR_FILE_NOT_FOUND: case ERROR_PATH_NOT_FOUND: case ERROR_DIRECTORY: return FS_R_PATH_NOT_FOUND;
    case ERROR_FILE_EXISTS: case ERROR_ALREADY_EXISTS: return FS_R_ALREADY_EXISTS;
    case ERROR_DIR_NOT_EMPTY: return FS_R_NOT_EMPTY;
    default: return FS_R_INVALID_ARGUMENT;
    }
}

// ---- storage mounts
static int my_uid(const uint64_t *uid) {
    uint64_t me[2];
    account_uid(me);
    return uid && uid[0] == me[0] && uid[1] == me[1];
}

static uint32_t mount_save(const char *name, const uint64_t *uid) {
    ready();
    if (!my_uid(uid)) return FS_R_TARGET_NOT_FOUND;
    AcquireSRWLockExclusive(&tx_lock);
    if (!save_mounts) work_create();
    uint32_t r = mount_add(name, M_SAVE, work_root);
    if (!r) save_mounts++;
    else if (!save_mounts) remove_tree_w(w_work);
    ReleaseSRWLockExclusive(&tx_lock);
    return r;
}

static uint8_t save_exists(uint64_t application, const uint64_t *uid) {
    ready();
    return application == OOT_APPLICATION_ID && my_uid(uid);
}

// Other titles' save data (data-link bonuses) is not present on this host.
static uint32_t mount_save_ro(const char *name, uint64_t application, const uint64_t *uid) {
    if (!save_exists(application, uid)) return FS_R_TARGET_NOT_FOUND;
    return mount_add(name, M_SAVE_RO, save_root);
}

static uint32_t mount_temp(const char *name) {
    ready();
    char p[PATH_CAP];
    snprintf(p, sizeof p, "%s/temp", profile);
    return mount_add(name, M_TEMP, p);
}

static void cache_dir(int index, char *out, size_t cap) { snprintf(out, cap, "%s/cache/%d", profile, index); }

static uint32_t create_cache(int index, int64_t size, int64_t journal) {
    ready();
    if (index < 0 || size < 0 || journal < 0) return FS_R_INVALID_ARGUMENT;
    char p[PATH_CAP];
    cache_dir(index, p, sizeof p);
    wchar_t *w = fs_wide(p);
    uint32_t r = CreateDirectoryW(w, NULL) ? 0 : win_result(GetLastError());
    free(w);
    return r;
}

static uint32_t delete_cache(int index) {
    ready();
    char p[PATH_CAP];
    cache_dir(index, p, sizeof p);
    return remove_tree(p) ? FS_R_TARGET_NOT_FOUND : 0;
}

static uint32_t mount_cache(const char *name, int index) {
    ready();
    char p[PATH_CAP];
    cache_dir(index, p, sizeof p);
    DWORD a = attributes(p);
    if (a == INVALID_FILE_ATTRIBUTES || !(a & FILE_ATTRIBUTE_DIRECTORY)) return FS_R_TARGET_NOT_FOUND;
    return mount_add(name, M_CACHE, p);
}

typedef struct { int32_t index; uint8_t reserved[0x1c]; } CacheStorageInfo;
_Static_assert(sizeof(CacheStorageInfo) == 0x20, "nn::fs::CacheStorageInfo");
typedef struct { int list[64]; int count, next; } CacheList;

static uint32_t open_cache_list(uint64_t *out) {
    ready();
    if (!out) return FS_R_INVALID_ARGUMENT;
    CacheList *l = calloc(1, sizeof *l);
    need(l != NULL, "cache list allocation");
    char p[PATH_CAP];
    snprintf(p, sizeof p, "%s/cache/*", profile);
    wchar_t *w = fs_wide(p);
    WIN32_FIND_DATAW fd;
    HANDLE f = FindFirstFileW(w, &fd);
    free(w);
    if (f != INVALID_HANDLE_VALUE) {
        do {
            wchar_t *end;
            long v = wcstol(fd.cFileName, &end, 10);
            if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && *fd.cFileName && !*end && v >= 0 && l->count < 64)
                l->list[l->count++] = (int)v;
        } while (FindNextFileW(f, &fd));
        FindClose(f);
    }
    *out = (uint64_t)(uintptr_t)l;
    return 0;
}

static uint32_t read_cache_list(int *count, CacheStorageInfo *out, uint64_t h, int max) {
    CacheList *l = (CacheList *)(uintptr_t)h;
    if (!l || !count || max < 0 || (!out && max)) return FS_R_INVALID_ARGUMENT;
    int n = 0;
    while (n < max && l->next < l->count) {
        memset(&out[n], 0, sizeof *out);
        out[n++].index = l->list[l->next++];
    }
    *count = n;
    return 0;
}

static void close_cache_list(uint64_t h) { free((void *)(uintptr_t)h); }

static void unmount(const char *name) {
    save_wait_commit();   // else the working copy could go before a Commit queued just now has copied it
    AcquireSRWLockExclusive(&tx_lock);
    AcquireSRWLockExclusive(&lock);
    for (int i = 0; i < NMOUNTS; i++)
        if (name && mounts[i].kind && !strcmp(mounts[i].name, name)) {
            int save = mounts[i].kind == M_SAVE;
            mounts[i].kind = M_FREE;
            ReleaseSRWLockExclusive(&lock);
            if (save && !--save_mounts) work_discard();
            ReleaseSRWLockExclusive(&tx_lock);
            rt_log("fs unmount %s", name);
            return;
        }
    ReleaseSRWLockExclusive(&lock);
    ReleaseSRWLockExclusive(&tx_lock);
    size_t n = name ? strlen(name) : 0;
    char probe[40];
    if (!n || n >= 32 || (snprintf(probe, sizeof probe, "%s:", name), !romfs_owns(probe)))
        rt_fatal("fs: unmount of unknown mount %s", name ? name : "(null)");
}

static uint32_t commit(const char *name) {
    int found = 0, save = 0;
    AcquireSRWLockShared(&lock);
    for (int i = 0; i < NMOUNTS; i++)
        if (name && mounts[i].kind && !strcmp(mounts[i].name, name)) found = 1, save = mounts[i].kind == M_SAVE;
    ReleaseSRWLockShared(&lock);
    if (!found) return FS_R_PATH_NOT_FOUND;
    if (save) return save_commit();   // durability comes from the staging copy, not the working copy's handles
    AcquireSRWLockShared(&lock);
    for (int i = 0; i < NFILES; i++)
        if (files[i].h && (files[i].mode & FS_OPEN_WRITE)) FlushFileBuffers(files[i].h);
    ReleaseSRWLockShared(&lock);
    return 0;
}

// ---- files
static File *file_of(uint64_t h, HANDLE *fh, int *mode) {
    if (h < FILE_BASE || h >= FILE_BASE + NFILES * 8 || ((h - FILE_BASE) & 7)) return NULL;   // multiples of 8, as romfs.c
    AcquireSRWLockShared(&lock);
    File *f = &files[(h - FILE_BASE) / 8];
    HANDLE v = f->h;
    if (fh) *fh = f->h;
    if (mode) *mode = f->mode;
    ReleaseSRWLockShared(&lock);
    return v ? f : NULL;
}

// Positional I/O on a synchronous handle: ReadFile/WriteFile with an OVERLAPPED offset.
static int64_t host_read_at(HANDLE f, void *buf, uint32_t n, uint64_t off) {
    OVERLAPPED ov = {.Offset = (DWORD)off, .OffsetHigh = (DWORD)(off >> 32)};
    DWORD got = 0;
    mem_thaw((uintptr_t)buf, n);   // the kernel's write would fail on a frozen page instead of faulting
    if (!ReadFile(f, buf, n, &got, &ov)) {
        DWORD e = GetLastError();
        if (e == ERROR_NOACCESS)   // a read-only page in the buffer: a frozen region missed its thaw
            rt_log("fs: read of %u bytes into %p refused: the buffer is not writable", n, buf);
        return e == ERROR_HANDLE_EOF ? 0 : -1;
    }
    return got;
}

static uint32_t open_file(uint64_t *out, const char *path, int mode) {
    if (romfs_owns(path)) return romfs_open(out, path, mode);
    fs_trace(mode & FS_OPEN_WRITE ? "write" : "read", path);
    if (!out || !(mode & (FS_OPEN_READ | FS_OPEN_WRITE)) || (mode & ~7)) return FS_R_INVALID_ARGUMENT;
    char p[PATH_CAP];
    int kind = M_FREE;
    uint32_t r = host_path_kind(path, p, sizeof p, &kind);
    if (r) return r;
    DWORD a = attributes(p);
    if (a == INVALID_FILE_ATTRIBUTES || (a & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)))
        return FS_R_PATH_NOT_FOUND;
    if (kind == M_SAVE_RO && (mode & FS_OPEN_WRITE)) return FS_R_UNSUPPORTED_OPEN;
    int save_writer = kind == M_SAVE && (mode & FS_OPEN_WRITE);
    DWORD access = (mode & FS_OPEN_READ ? GENERIC_READ : 0) | (mode & FS_OPEN_WRITE ? GENERIC_WRITE : 0);
    wchar_t *w = fs_wide(p);
    HANDLE f = CreateFileW(w, access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, NULL);
    free(w);
    if (f == INVALID_HANDLE_VALUE) return win_result(GetLastError());
    AcquireSRWLockExclusive(&lock);
    for (int i = 0; i < NFILES; i++)
        if (!files[i].h) {
            files[i] = (File){f, mode, save_writer};
            *out = FILE_BASE + (uint64_t)i * 8;
            ReleaseSRWLockExclusive(&lock);
            if (save_writer) {
                InterlockedIncrement(&save_writers);
                save_touched();
            }
            return 0;
        }
    ReleaseSRWLockExclusive(&lock);
    rt_fatal("fs: host file handle table full (%d)", NFILES);
}

static uint32_t read_file(uint64_t *out, uint64_t h, int64_t offset, void *buffer, uint64_t size) {
    if (romfs_handle(h)) return romfs_read(out, h, offset, buffer, size);
    HANDLE f;
    int mode;
    if (!out || offset < 0 || (!buffer && size) || !file_of(h, &f, &mode)) return FS_R_INVALID_ARGUMENT;
    if (!(mode & FS_OPEN_READ)) return FS_R_OPEN_MODE;
    uint64_t done = 0;
    while (done < size) {
        uint32_t chunk = size - done > 0x40000000 ? 0x40000000 : (uint32_t)(size - done);
        int64_t n = host_read_at(f, (uint8_t *)buffer + done, chunk, (uint64_t)offset + done);
        if (n < 0) rt_fatal("fs: host save read failure (error %lu)", GetLastError());
        if (!n) break;
        done += (uint64_t)n;
    }
    *out = done;
    return 0;
}

static uint32_t read_exact(uint64_t h, int64_t off, void *buf, uint64_t size) {
    uint64_t n = 0;
    uint32_t r = read_file(&n, h, off, buf, size);
    return r ? r : n == size ? 0 : FS_R_INVALID_ARGUMENT;
}

static uint32_t write_file(uint64_t h, int64_t offset, const void *buffer, uint64_t size, const int32_t *option) {
    HANDLE f;
    int mode;
    File *file = offset < 0 || (!buffer && size) ? NULL : file_of(h, &f, &mode);
    if (!file) return FS_R_INVALID_ARGUMENT;
    if (!(mode & FS_OPEN_WRITE)) return FS_R_OPEN_MODE;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz)) rt_fatal("fs: host save size (error %lu)", GetLastError());
    if ((uint64_t)offset + size > (uint64_t)sz.QuadPart && !(mode & FS_OPEN_APPEND)) return FS_R_APPEND_REQUIRED;
    if (file->save) save_touched();
    for (uint64_t done = 0; done < size;) {
        uint32_t chunk = size - done > 0x40000000 ? 0x40000000 : (uint32_t)(size - done);
        uint64_t at = (uint64_t)offset + done;
        OVERLAPPED ov = {.Offset = (DWORD)at, .OffsetHigh = (DWORD)(at >> 32)};
        DWORD put = 0;
        if (!WriteFile(f, (const uint8_t *)buffer + done, chunk, &put, &ov) || !put)
            rt_fatal("fs: host save write failure (error %lu)", GetLastError());
        done += put;
    }
    if (option && (*option & 1)) FlushFileBuffers(f);   // WriteOptionFlag_Flush
    return 0;
}

static uint32_t file_size(int64_t *out, uint64_t h) {
    if (romfs_handle(h)) return romfs_size(out, h);
    HANDLE f;
    LARGE_INTEGER sz;
    if (!out || !file_of(h, &f, NULL) || !GetFileSizeEx(f, &sz)) return FS_R_INVALID_ARGUMENT;
    *out = sz.QuadPart;
    return 0;
}

static uint32_t set_file_size(uint64_t h, int64_t size) {
    HANDLE f;
    int mode;
    File *file = size < 0 ? NULL : file_of(h, &f, &mode);
    if (!file) return FS_R_INVALID_ARGUMENT;
    if (!(mode & FS_OPEN_WRITE)) return FS_R_OPEN_MODE;
    if (file->save) save_touched();
    FILE_END_OF_FILE_INFO eof = {.EndOfFile.QuadPart = size};
    return SetFileInformationByHandle(f, FileEndOfFileInfo, &eof, sizeof eof) ? 0 : win_result(GetLastError());
}

static uint32_t flush_file(uint64_t h) {
    HANDLE f;
    int mode;
    if (!file_of(h, &f, &mode)) return FS_R_INVALID_ARGUMENT;
    if (mode & FS_OPEN_WRITE) FlushFileBuffers(f);
    return 0;
}

static void close_file(uint64_t h) {
    if (romfs_handle(h)) {
        romfs_close(h);
        return;
    }
    if (!file_of(h, NULL, NULL)) rt_fatal("fs: close of invalid file handle %#llx", (unsigned long long)h);
    AcquireSRWLockExclusive(&lock);
    HANDLE f = files[(h - FILE_BASE) / 8].h;
    int save = files[(h - FILE_BASE) / 8].save;
    files[(h - FILE_BASE) / 8] = (File){0};
    ReleaseSRWLockExclusive(&lock);
    CloseHandle(f);
    if (save) {
        save_touched();
        InterlockedDecrement(&save_writers);
    }
}

static uint32_t create_file(const char *path, int64_t size) {
    char p[PATH_CAP];
    uint32_t r = save_change_path(path, p, sizeof p);
    if (r) return r;
    if (size < 0) return FS_R_INVALID_ARGUMENT;
    wchar_t *w = fs_wide(p);
    HANDLE f = CreateFileW(w, GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
    free(w);
    if (f == INVALID_HANDLE_VALUE) return win_result(GetLastError());
    FILE_END_OF_FILE_INFO eof = {.EndOfFile.QuadPart = size};
    int ok = !size || SetFileInformationByHandle(f, FileEndOfFileInfo, &eof, sizeof eof);
    DWORD e = GetLastError();
    CloseHandle(f);
    return ok ? 0 : win_result(e);
}

static uint32_t delete_file(const char *path) {
    char p[PATH_CAP];
    uint32_t r = save_change_path(path, p, sizeof p);
    if (r) return r;
    wchar_t *w = fs_wide(p);
    r = DeleteFileW(w) ? 0 : win_result(GetLastError());
    free(w);
    return r;
}

static uint32_t rename_file(const char *from, const char *to) {
    char a[PATH_CAP], b[PATH_CAP];
    uint32_t r = save_change_path(from, a, sizeof a);
    if (!r) r = host_path(to, b, sizeof b);
    if (r) return r;
    size_t n = strcspn(from, ":");
    if (strcspn(to, ":") != n || strncmp(from, to, n)) return FS_R_INVALID_PATH;   // across mounts
    if (attributes(b) != INVALID_FILE_ATTRIBUTES) return FS_R_ALREADY_EXISTS;
    wchar_t *wa = fs_wide(a), *wb = fs_wide(b);
    r = MoveFileExW(wa, wb, 0) ? 0 : win_result(GetLastError());
    free(wa);
    free(wb);
    return r;
}

static uint32_t entry_type(int *out, const char *path) {
    if (romfs_owns(path)) return romfs_entry_type(out, path);
    char p[PATH_CAP];
    uint32_t r = host_path(path, p, sizeof p);
    if (r) return r;
    if (!out) return FS_R_INVALID_ARGUMENT;
    DWORD a = attributes(p);
    if (a == INVALID_FILE_ATTRIBUTES) return FS_R_PATH_NOT_FOUND;
    *out = a & FILE_ATTRIBUTE_DIRECTORY ? FS_ENTRY_DIRECTORY : FS_ENTRY_FILE;
    return 0;
}

// ---- directories
typedef FsDirectoryEntry DirectoryEntry;   // fs.h
_Static_assert(sizeof(DirectoryEntry) == 0x310, "nn::fs::DirectoryEntry");

static uint32_t create_directory(const char *path) {
    char p[PATH_CAP];
    uint32_t r = save_change_path(path, p, sizeof p);
    if (r) return r;
    wchar_t *w = fs_wide(p);
    r = CreateDirectoryW(w, NULL) ? 0 : win_result(GetLastError());
    free(w);
    return r;
}

static uint32_t delete_recursive(const char *path) {
    char p[PATH_CAP];
    uint32_t r = save_change_path(path, p, sizeof p);
    if (r) return r;
    DWORD a = attributes(p);
    if (a == INVALID_FILE_ATTRIBUTES) return FS_R_PATH_NOT_FOUND;
    if (!(a & FILE_ATTRIBUTE_DIRECTORY)) return FS_R_PATH_NOT_FOUND;
    return remove_tree(p) ? FS_R_INVALID_ARGUMENT : 0;
}

static uint32_t open_directory(uint64_t *out, const char *path, int mode) {
    if (romfs_owns(path)) return romfs_open_directory(out, path, mode);
    char p[PATH_CAP];
    uint32_t r = host_path(path, p, sizeof p);
    if (r) return r;
    if (!out || mode < 1 || mode > 3) return FS_R_INVALID_ARGUMENT;
    DWORD a = attributes(p);
    if (a == INVALID_FILE_ATTRIBUTES || !(a & FILE_ATTRIBUTE_DIRECTORY)) return FS_R_PATH_NOT_FOUND;
    char pattern[PATH_CAP + 4];
    snprintf(pattern, sizeof pattern, "%s/*", p);
    wchar_t *w = fs_wide(pattern);
    WIN32_FIND_DATAW fd;
    HANDLE f = FindFirstFileW(w, &fd);
    if (f == INVALID_HANDLE_VALUE) {
        uint32_t e = win_result(GetLastError());
        free(w);
        return e;
    }
    AcquireSRWLockExclusive(&lock);
    for (int i = 0; i < NDIRS; i++)
        if (!dirs[i].find) {
            dirs[i] = (Dir){f, fd, 1, mode, w};
            *out = DIR_BASE + (uint64_t)i * 8;
            ReleaseSRWLockExclusive(&lock);
            return 0;
        }
    ReleaseSRWLockExclusive(&lock);
    free(w);
    rt_fatal("fs: host directory handle table full (%d)", NDIRS);
}

// The name filter read_directory applies: not . or .., at most 768 bytes of UTF-8, of the kinds mode asks for.
static int dir_entry_listed(const WIN32_FIND_DATAW *e, int mode) {
    if (!wcscmp(e->cFileName, L".") || !wcscmp(e->cFileName, L"..")) return 0;
    int k = WideCharToMultiByte(CP_UTF8, 0, e->cFileName, -1, NULL, 0, NULL, NULL);
    if (k <= 0 || k > 769) return 0;
    return !!(mode & (e->dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY ? 1 : 2));
}

// nn::fs::GetDirectoryEntryCount: every entry read_directory would return from a fresh open, whatever the handle has
// read so far (the SDK counts with a second directory session, so the read position does not move).
static uint32_t directory_entry_count(int64_t *out, uint64_t h) {
    if (romfs_dir_handle(h)) return romfs_directory_entry_count(out, h);
    if (h < DIR_BASE || h >= DIR_BASE + NDIRS * 8 || ((h - DIR_BASE) & 7) || !out) return FS_R_INVALID_ARGUMENT;
    AcquireSRWLockShared(&lock);
    Dir d = dirs[(h - DIR_BASE) / 8];
    ReleaseSRWLockShared(&lock);
    if (!d.find || !d.pattern) return FS_R_INVALID_ARGUMENT;
    WIN32_FIND_DATAW fd;
    HANDLE f = FindFirstFileW(d.pattern, &fd);
    if (f == INVALID_HANDLE_VALUE) return win_result(GetLastError());
    int64_t n = 0;
    do n += dir_entry_listed(&fd, d.mode);
    while (FindNextFileW(f, &fd));
    FindClose(f);
    *out = n;
    return 0;
}

static uint32_t read_directory(int64_t *count, DirectoryEntry *out, uint64_t h, int64_t max) {
    if (romfs_dir_handle(h)) return romfs_read_directory(count, out, h, max);
    if (h < DIR_BASE || h >= DIR_BASE + NDIRS * 8 || ((h - DIR_BASE) & 7) || !count || max < 0 || (!out && max)) return FS_R_INVALID_ARGUMENT;
    Dir *d = &dirs[(h - DIR_BASE) / 8];
    if (!d->find) return FS_R_INVALID_ARGUMENT;
    int64_t n = 0;
    while (n < max) {
        if (!d->has_next && !FindNextFileW(d->find, &d->next)) break;
        d->has_next = 0;
        const WIN32_FIND_DATAW *e = &d->next;
        if (!dir_entry_listed(e, d->mode)) continue;   // over 768 bytes: nn::fs cannot name it
        char name[800];
        int k = WideCharToMultiByte(CP_UTF8, 0, e->cFileName, -1, name, sizeof name, NULL, NULL);
        int is_dir = !!(e->dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
        memset(&out[n], 0, sizeof *out);
        memcpy(out[n].name, name, (size_t)k);
        out[n].type = is_dir ? FS_ENTRY_DIRECTORY : FS_ENTRY_FILE;
        out[n].size = is_dir ? 0 : (int64_t)((uint64_t)e->nFileSizeHigh << 32 | e->nFileSizeLow);
        n++;
    }
    *count = n;
    return 0;
}

static void close_directory(uint64_t h) {
    if (romfs_dir_handle(h)) {
        romfs_close_directory(h);
        return;
    }
    if (h < DIR_BASE || h >= DIR_BASE + NDIRS * 8 || ((h - DIR_BASE) & 7) || !dirs[(h - DIR_BASE) / 8].find)
        rt_fatal("fs: close of invalid directory handle %#llx", (unsigned long long)h);
    AcquireSRWLockExclusive(&lock);
    HANDLE f = dirs[(h - DIR_BASE) / 8].find;
    wchar_t *pattern = dirs[(h - DIR_BASE) / 8].pattern;
    dirs[(h - DIR_BASE) / 8] = (Dir){0};
    ReleaseSRWLockExclusive(&lock);
    FindClose(f);
    free(pattern);
}

// ---- overrides: AAPCS64 thunks. FileHandle, DirectoryHandle, CacheStorageListHandle
// and ApplicationId are 8-byte structs passed in one register; Uid, ReadOption and WriteOption by reference.
#define I32(i) ((int32_t)arg_x(c, i))
#define I64(i) ((int64_t)arg_x(c, i))
// Horizon's fs takes '\\' as a path separator and drops "." components, and this game spells its paths with both
// ("rom:\\StartupUI.pak", "rom:\\./Worlds"); the host's path code knows '/' only and refuses ".", so every path argument
// is copied with its backslashes turned into slashes and its "." components removed first ("rom:/." stays the
// root, "rom:/"). ".." is still refused further down. A path too long for the buffer is passed as it is and fails
// the host's own length check.
static const char *slashed(const char *path, char *buf, size_t cap) {
    if (!path || (!strchr(path, '\\') && !strstr(path, "/."))) return path;
    size_t n = strlen(path);
    if (n >= cap) return path;
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        char ch = path[i] == '\\' ? '/' : path[i];
        if (ch == '.' && k && buf[k - 1] == '/' && (i + 1 == n || path[i + 1] == '/' || path[i + 1] == '\\')) {
            i++;   // "/." at the end, or "/./": the component goes, with the separator that follows it
            continue;
        }
        buf[k++] = ch;
    }
    buf[k] = 0;
    return buf;
}
#define PATH(i, buf) slashed(arg_p(c, i), buf, sizeof buf)
static void ov_open_file(Cpu *c) {
    char pb[PATH_CAP];
    uint32_t r = open_file(arg_p(c, 0), PATH(1, pb), I32(2));
    if (rt_trace & RT_TRACE_FS) {   // every open, as this game's loader keeps its own request objects (ov_oot.c)
        char lr[96];
        rt_log("fs: OpenFile(\"%s\", mode %d) -> %#x from %s", (const char *)arg_p(c, 1), I32(2), r,
               ldr_describe(c->x[30] - 4, lr, sizeof lr));
    }
    ret_x(c, r);
}
static void ov_file_size(Cpu *c) { ret_x(c, file_size(arg_p(c, 0), arg_x(c, 1))); }
void kern_backtrace(const Cpu *c);   // kernel.h
static void ov_close_file(Cpu *c) {
    uint64_t h = arg_x(c, 0);
    if (!romfs_handle(h) && !file_of(h, NULL, NULL)) {   // before the fatal in close_file: where the game got the handle
        rt_log("fs: CloseFile(%#llx) with no such handle; guest backtrace:", (unsigned long long)h);
        kern_backtrace(c);
    }
    close_file(h);
}
static void ov_entry_type(Cpu *c) {
    char pb[PATH_CAP];
    uint32_t r = entry_type(arg_p(c, 0), PATH(1, pb));
    if (rt_trace & RT_TRACE_FS)
        rt_log("fs: GetEntryType(\"%s\") -> %#x, type %d", (const char *)arg_p(c, 1), r,
               arg_p(c, 0) ? *(int32_t *)arg_p(c, 0) : -1);
    ret_x(c, r);
}
static void ov_read_file(Cpu *c) { ret_x(c, read_file(arg_p(c, 0), arg_x(c, 1), I64(2), arg_p(c, 3), arg_x(c, 4))); }
static void ov_read_options(Cpu *c) { ret_x(c, read_file(arg_p(c, 0), arg_x(c, 1), I64(2), arg_p(c, 3), arg_x(c, 4))); }
static void ov_read_exact(Cpu *c) { ret_x(c, read_exact(arg_x(c, 0), I64(1), arg_p(c, 2), arg_x(c, 3))); }
static void ov_write_file(Cpu *c) { ret_x(c, write_file(arg_x(c, 0), I64(1), arg_p(c, 2), arg_x(c, 3), arg_p(c, 4))); }
static void ov_set_file_size(Cpu *c) { ret_x(c, set_file_size(arg_x(c, 0), I64(1))); }
static void ov_flush_file(Cpu *c) { ret_x(c, flush_file(arg_x(c, 0))); }
static void ov_create_file(Cpu *c) { char pb[PATH_CAP]; ret_x(c, create_file(PATH(0, pb), I64(1))); }
static void ov_delete_file(Cpu *c) { char pb[PATH_CAP]; ret_x(c, delete_file(PATH(0, pb))); }
static void ov_rename_file(Cpu *c) { char pa[PATH_CAP], pb[PATH_CAP]; ret_x(c, rename_file(PATH(0, pa), PATH(1, pb))); }
static void ov_create_directory(Cpu *c) { char pb[PATH_CAP]; ret_x(c, create_directory(PATH(0, pb))); }
static void ov_delete_recursive(Cpu *c) { char pb[PATH_CAP]; ret_x(c, delete_recursive(PATH(0, pb))); }
static void ov_open_directory(Cpu *c) {
    char pb[PATH_CAP];
    uint32_t r = open_directory(arg_p(c, 0), PATH(1, pb), I32(2));
    if (rt_trace & RT_TRACE_FS)
        rt_log("fs: OpenDirectory(\"%s\", mode %#x) -> %#x", (const char *)arg_p(c, 1), (unsigned)I32(2), r);
    ret_x(c, r);
}
static void ov_read_directory(Cpu *c) {
    uint32_t r = read_directory(arg_p(c, 0), arg_p(c, 1), arg_x(c, 2), I64(3));
    if (rt_trace & RT_TRACE_FS)
        rt_log("fs: ReadDirectory(%#llx, max %lld) -> %#x, %lld entries", (unsigned long long)arg_x(c, 2), (long long)I64(3), r,
               arg_p(c, 0) ? (long long)*(int64_t *)arg_p(c, 0) : -1);
    ret_x(c, r);
}
static void ov_close_directory(Cpu *c) { close_directory(arg_x(c, 0)); }
static void ov_directory_entry_count(Cpu *c) { ret_x(c, directory_entry_count(arg_p(c, 0), arg_x(c, 1))); }
static void ov_mount_save(Cpu *c) { ret_x(c, mount_save(arg_p(c, 0), arg_p(c, 1))); }
static void ov_mount_save_ro(Cpu *c) { ret_x(c, mount_save_ro(arg_p(c, 0), arg_x(c, 1), arg_p(c, 2))); }
static void ov_save_exists(Cpu *c) { ret_x(c, save_exists(arg_x(c, 0), arg_p(c, 1))); }
static void ov_mount_temp(Cpu *c) { ret_x(c, mount_temp(arg_p(c, 0))); }
static void ov_unmount(Cpu *c) { unmount(arg_p(c, 0)); }
static void ov_commit(Cpu *c) { ret_x(c, commit(arg_p(c, 0))); }
// EnsureSaveData(uid): the save's host folder appears with the first mount (mount_save), so there is nothing to make.
static void ov_ensure_save(Cpu *c) { ret_x(c, 0); }
// Result GetSaveDataSize(int64_t *size, int64_t *journal, const Uid &) (this game): the sizes the game's control data
// declares for its user save (10 MiB data, 4 MiB journal, read from control.nacp in Phase 0); the host folder has
// no quota. Another user has no save (as mount_save answers).
static void ov_save_data_size(Cpu *c) {
    int64_t *size = arg_p(c, 0), *journal = arg_p(c, 1);
    if (!my_uid(arg_p(c, 2))) { ret_x(c, FS_R_TARGET_NOT_FOUND); return; }
    if (size) *size = 10 << 20;
    if (journal) *journal = 4 << 20;
    ret_x(c, 0);
}
// SetAllocator: the allocator nn::fs uses for its own buffers; the host keeps none in guest memory.
static void ov_set_allocator(Cpu *c) { (void)c; }
static void ov_create_cache(Cpu *c) { ret_x(c, create_cache(I32(0), I64(1), I64(2))); }
static void ov_delete_cache(Cpu *c) { ret_x(c, delete_cache(I32(0))); }
static void ov_mount_cache(Cpu *c) { ret_x(c, mount_cache(arg_p(c, 0), I32(1))); }
static void ov_open_cache_list(Cpu *c) { ret_x(c, open_cache_list(arg_p(c, 0))); }
static void ov_read_cache_list(Cpu *c) { ret_x(c, read_cache_list(arg_p(c, 0), arg_p(c, 1), arg_x(c, 2), I32(3))); }
static void ov_close_cache_list(Cpu *c) { close_cache_list(arg_x(c, 0)); }
// The fs priority only orders I/O inside the system fs process; host I/O is unordered.
static void ov_set_priority(Cpu *c) { (void)c; }
#undef I32
#undef I64

#define S_OPEN_FILE "_ZN2nn2fs8OpenFileEPNS0_10FileHandleEPKci"
#define S_FILE_SIZE "_ZN2nn2fs11GetFileSizeEPlNS0_10FileHandleE"
#define S_CLOSE_FILE "_ZN2nn2fs9CloseFileENS0_10FileHandleE"
#define S_ENTRY_TYPE "_ZN2nn2fs12GetEntryTypeEPNS0_18DirectoryEntryTypeEPKc"
#define S_READ_FILE "_ZN2nn2fs8ReadFileEPmNS0_10FileHandleElPvm"
#define S_READ_OPTIONS "_ZN2nn2fs8ReadFileEPmNS0_10FileHandleElPvmRKNS0_10ReadOptionE"
#define S_READ_EXACT "_ZN2nn2fs8ReadFileENS0_10FileHandleElPvm"
#define S_WRITE_FILE "_ZN2nn2fs9WriteFileENS0_10FileHandleElPKvmRKNS0_11WriteOptionE"
#define S_SET_FILE_SIZE "_ZN2nn2fs11SetFileSizeENS0_10FileHandleEl"
#define S_FLUSH_FILE "_ZN2nn2fs9FlushFileENS0_10FileHandleE"
#define S_CREATE_FILE "_ZN2nn2fs10CreateFileEPKcl"
#define S_DELETE_FILE "_ZN2nn2fs10DeleteFileEPKc"
#define S_RENAME_FILE "_ZN2nn2fs10RenameFileEPKcS2_"
#define S_CREATE_DIRECTORY "_ZN2nn2fs15CreateDirectoryEPKc"
#define S_DELETE_RECURSIVE "_ZN2nn2fs26DeleteDirectoryRecursivelyEPKc"
#define S_OPEN_DIRECTORY "_ZN2nn2fs13OpenDirectoryEPNS0_15DirectoryHandleEPKci"
#define S_READ_DIRECTORY "_ZN2nn2fs13ReadDirectoryEPlPNS0_14DirectoryEntryENS0_15DirectoryHandleEl"
#define S_CLOSE_DIRECTORY "_ZN2nn2fs14CloseDirectoryENS0_15DirectoryHandleE"
#define S_MOUNT_SAVE "_ZN2nn2fs13MountSaveDataEPKcRKNS_7account3UidE"
#define S_MOUNT_SAVE_RO "_ZN2nn2fs21MountSaveDataReadOnlyEPKcNS_13ApplicationIdERKNS_7account3UidE"
#define S_SAVE_EXISTS "_ZN2nn2fs18IsSaveDataExistingENS_13ApplicationIdERKNS_7account3UidE"
#define S_MOUNT_TEMP "_ZN2nn2fs21MountTemporaryStorageEPKc"
#define S_UNMOUNT "_ZN2nn2fs7UnmountEPKc"
#define S_COMMIT "_ZN2nn2fs6CommitEPKc"
// Older SDKs (FE3H's 6.5.0): CommitSaveData is Commit by its earlier name.
#define S_COMMIT_SAVE "_ZN2nn2fs14CommitSaveDataEPKc"
#define S_ENSURE_SAVE "_ZN2nn2fs14EnsureSaveDataERKNS_7account3UidE"
#define S_SAVE_DATA_SIZE "_ZN2nn2fs15GetSaveDataSizeEPlS1_RKNS_7account3UidE"
#define S_SET_ALLOCATOR "_ZN2nn2fs12SetAllocatorEPFPvmEPFvS1_mE"
#define S_DIR_ENTRY_COUNT "_ZN2nn2fs22GetDirectoryEntryCountEPlNS0_15DirectoryHandleE"
#define S_CREATE_CACHE "_ZN2nn2fs18CreateCacheStorageEill"
#define S_DELETE_CACHE "_ZN2nn2fs18DeleteCacheStorageEi"
#define S_MOUNT_CACHE "_ZN2nn2fs17MountCacheStorageEPKci"
#define S_OPEN_CACHE_LIST "_ZN2nn2fs20OpenCacheStorageListEPNS0_22CacheStorageListHandleE"
#define S_READ_CACHE_LIST "_ZN2nn2fs20ReadCacheStorageListEPiPNS0_16CacheStorageInfoENS0_22CacheStorageListHandleEi"
#define S_CLOSE_CACHE_LIST "_ZN2nn2fs21CloseCacheStorageListENS0_22CacheStorageListHandleE"
#define S_SET_PRIORITY "_ZN2nn2fs26SetPriorityOnCurrentThreadENS0_8PriorityE"
// romfs.c
#define S_MOUNT_ROM "_ZN2nn2fs8MountRomEPKcPvm"
#define S_QUERY_CACHE "_ZN2nn2fs22QueryMountRomCacheSizeEPm"
#define S_INITIALIZE "_ZN2nn2fs24InitializeForApplicationEv"

HOST_OVERRIDE(S_OPEN_FILE, ov_open_file);
HOST_OVERRIDE(S_FILE_SIZE, ov_file_size);
HOST_OVERRIDE(S_CLOSE_FILE, ov_close_file);
HOST_OVERRIDE(S_ENTRY_TYPE, ov_entry_type);
HOST_OVERRIDE(S_READ_FILE, ov_read_file);
HOST_OVERRIDE(S_READ_OPTIONS, ov_read_options);
HOST_OVERRIDE(S_READ_EXACT, ov_read_exact);
HOST_OVERRIDE(S_WRITE_FILE, ov_write_file);
HOST_OVERRIDE(S_SET_FILE_SIZE, ov_set_file_size);
HOST_OVERRIDE(S_FLUSH_FILE, ov_flush_file);
HOST_OVERRIDE(S_CREATE_FILE, ov_create_file);
HOST_OVERRIDE(S_DELETE_FILE, ov_delete_file);
HOST_OVERRIDE(S_RENAME_FILE, ov_rename_file);
HOST_OVERRIDE(S_CREATE_DIRECTORY, ov_create_directory);
HOST_OVERRIDE(S_DELETE_RECURSIVE, ov_delete_recursive);
HOST_OVERRIDE(S_OPEN_DIRECTORY, ov_open_directory);
HOST_OVERRIDE(S_READ_DIRECTORY, ov_read_directory);
HOST_OVERRIDE(S_CLOSE_DIRECTORY, ov_close_directory);
HOST_OVERRIDE(S_MOUNT_SAVE, ov_mount_save);
HOST_OVERRIDE(S_MOUNT_SAVE_RO, ov_mount_save_ro);
HOST_OVERRIDE(S_SAVE_EXISTS, ov_save_exists);
HOST_OVERRIDE(S_MOUNT_TEMP, ov_mount_temp);
HOST_OVERRIDE(S_UNMOUNT, ov_unmount);
HOST_OVERRIDE(S_COMMIT, ov_commit);
HOST_OVERRIDE(S_COMMIT_SAVE, ov_commit);
HOST_OVERRIDE(S_ENSURE_SAVE, ov_ensure_save);
HOST_OVERRIDE(S_SAVE_DATA_SIZE, ov_save_data_size);
HOST_OVERRIDE(S_SET_ALLOCATOR, ov_set_allocator);
HOST_OVERRIDE(S_DIR_ENTRY_COUNT, ov_directory_entry_count);
HOST_OVERRIDE(S_CREATE_CACHE, ov_create_cache);
HOST_OVERRIDE(S_DELETE_CACHE, ov_delete_cache);
HOST_OVERRIDE(S_MOUNT_CACHE, ov_mount_cache);
HOST_OVERRIDE(S_OPEN_CACHE_LIST, ov_open_cache_list);
HOST_OVERRIDE(S_READ_CACHE_LIST, ov_read_cache_list);
HOST_OVERRIDE(S_CLOSE_CACHE_LIST, ov_close_cache_list);
HOST_OVERRIDE(S_SET_PRIORITY, ov_set_priority);

// ---- --probe-fs
// Every check calls the overrides the way guest code does: by synthetic address through rt_call, with AAPCS64
// arguments in a Cpu, on this (probe) thread. Prints counts only, never RomFS paths.
static const char *const probe_symbols[] = {
    S_OPEN_FILE, S_FILE_SIZE, S_CLOSE_FILE, S_ENTRY_TYPE, S_READ_FILE, S_READ_OPTIONS, S_READ_EXACT, S_WRITE_FILE,
    S_SET_FILE_SIZE, S_FLUSH_FILE, S_CREATE_FILE, S_DELETE_FILE, S_RENAME_FILE, S_CREATE_DIRECTORY, S_DELETE_RECURSIVE,
    S_OPEN_DIRECTORY, S_READ_DIRECTORY, S_CLOSE_DIRECTORY, S_MOUNT_SAVE, S_MOUNT_SAVE_RO, S_SAVE_EXISTS, S_MOUNT_TEMP,
    S_UNMOUNT, S_COMMIT, S_CREATE_CACHE, S_DELETE_CACHE, S_MOUNT_CACHE, S_OPEN_CACHE_LIST, S_READ_CACHE_LIST,
    S_CLOSE_CACHE_LIST, S_SET_PRIORITY, S_MOUNT_ROM, S_QUERY_CACHE, S_INITIALIZE};
#define NPROBE ((int)(sizeof probe_symbols / sizeof *probe_symbols))
#define SAMPLES 64
#define HEAD_BYTES (8u << 20)      // a sampled file is compared whole up to this size, else head and tail
#define TAIL_BYTES (64u << 10)

static uint64_t probe_addr[NPROBE];
static Cpu *pcpu;
static int failures, checks;
#define CHECK(cond, ...) \
    do { checks++; if (!(cond)) { failures++; printf("  FAIL: " __VA_ARGS__); putchar('\n'); } } while (0)

static uint64_t U(const void *p) { return (uint64_t)(uintptr_t)p; }

// The override registered for sym, called through its synthetic address with up to 6 arguments. Returns x0.
static uint64_t fs_call(const char *sym, const uint64_t a[6]) {
    for (int i = 0; i < NPROBE; i++)
        if (!strcmp(probe_symbols[i], sym)) {
            for (int r = 0; r < 19; r++) pcpu->x[r] = 0xdead0000 + r;   // caller-saved garbage
            return rt_call(pcpu, probe_addr[i], 6, a[0], a[1], a[2], a[3], a[4], a[5]);
        }
    rt_fatal("probe-fs: %s is not a probed symbol", sym);
}
#define CALL(sym, ...) fs_call(sym, (const uint64_t[6]){__VA_ARGS__})
#define RES(sym, ...) ((uint32_t)CALL(sym, __VA_ARGS__))

static int64_t independent_read(HANDLE f, uint8_t *buf, uint64_t n, uint64_t off) {
    uint64_t done = 0;
    while (done < n) {
        int64_t r = host_read_at(f, buf + done, (uint32_t)(n - done), off + done);
        if (r <= 0) return -1;
        done += (uint64_t)r;
    }
    return (int64_t)done;
}

// Read [pos, pos+n) of the open RomFS handle h through one of the three ReadFile overloads, and compare it with an
// independent read of romfs.bin at the index offset.
static int compare_range(HANDLE image, uint64_t h, uint64_t offset, uint64_t pos, uint64_t n, int variant,
                         uint8_t *a, uint8_t *b) {
    uint64_t got = ~0ull;
    uint32_t r;
    int32_t option = 0;
    if (variant == 0) r = RES(S_READ_FILE, U(&got), h, pos, U(a), n);
    else if (variant == 1) r = RES(S_READ_OPTIONS, U(&got), h, pos, U(a), n, U(&option));
    else r = RES(S_READ_EXACT, h, pos, U(a), n), got = r ? ~0ull : n;
    if (r || got != n) return 0;
    if (independent_read(image, b, n, offset + pos) != (int64_t)n) return 0;
    return !memcmp(a, b, n);
}

static void probe_paths(void) {
    static const struct { const char *path, *want; } cases[] = {
        {"rom:/a/b.bin", "a/b.bin"}, {"rom://a/b.bin", "a/b.bin"}, {"rom:///a/b.bin", "a/b.bin"},
        {"rom:/a", "a"},             {"rom:/../main", NULL},       {"rom://../a", NULL},
        {"rom:/a/../b", NULL},       {"rom:/a/./b", NULL},         {"rom:/a//b", "a/b"},
        {"rom:/a\\b", NULL},         {"rom:/", NULL},              {"rom://", NULL},
        {"rom:a", NULL},             {"ro:/a", NULL},              {"romx:/a", NULL},
        {"rom:/.", NULL},            {"rom:/..", NULL},            {"ROM:/a", NULL},
        {"rom://Voice/Resource//Stream/x.bwav", "Voice/Resource/Stream/x.bwav"}, {"rom:/a///b//c", "a/b/c"},
        {"rom:/a//../b", NULL},      {"rom:/a//./b", NULL},
    };
    int bad = 0;
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; i++) {
        const char *got = romfs_path_relative("rom", cases[i].path);
        if (cases[i].want ? !got || strcmp(got, cases[i].want) : got != NULL) {
            bad++;
            printf("  FAIL: romfs_path_relative(\"rom\", \"%s\") = %s%s%s\n", cases[i].path, got ? "\"" : "",
                   got ? got : "NULL", got ? "\"" : "");
        }
    }
    checks++;
    failures += bad > 0;
    CHECK(!romfs_path_relative(NULL, "rom:/a") && !romfs_path_relative("", ":/a") && !romfs_path_relative("rom", NULL),
          "romfs_path_relative accepts an empty mount or path");
    printf("paths: %u romfs_path.h normalisation cases: %s\n", (unsigned)(sizeof cases / sizeof *cases),
           bad ? "FAIL" : "ok");
}

static uint64_t probe_romfs(void) {
    uint64_t count = romfs_file_count(), cache = ~0ull;
    CHECK(RES(S_QUERY_CACHE, U(&cache)) == 0 && cache == 0, "QueryMountRomCacheSize");
    CHECK(RES(S_MOUNT_ROM, U("rom"), 0, 0) == 0, "MountRom(\"rom\")");
    CHECK(RES(S_MOUNT_ROM, U("rom"), 0, 0) == 0, "MountRom(\"rom\") again");
    CHECK(RES(S_MOUNT_ROM, U("../x"), 0, 0) == FS_R_INVALID, "MountRom with a bad name");
    probe_paths();

    wchar_t *w = fs_wide(romfs_image_path());
    HANDLE image = CreateFileW(w, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    free(w);
    if (image == INVALID_HANDLE_VALUE) rt_fatal("probe-fs: cannot open the RomFS image (error %lu)", GetLastError());
    uint8_t *a = malloc(HEAD_BYTES), *b = malloc(HEAD_BYTES);
    need(a && b, "probe buffers");
    uint64_t step = count / SAMPLES, bytes = 0, sampled = 0, deep = 0, high = 0, largest = 0, lsize = 0;
    char path[4200], sample[4200] = "";
    for (uint64_t i = 0, size; i < count; i++)
        if (romfs_entry(i, NULL, &size) && size > lsize) largest = i, lsize = size;
    for (uint64_t k = 0; k <= SAMPLES && step; k++) {   // k == SAMPLES: the largest file, beyond the 64
        uint64_t i = k < SAMPLES ? k * step : largest, off, size, h = 0;
        int64_t fsize = -1;
        int type = -1;
        const char *name = romfs_entry(i, &off, &size);
        snprintf(path, sizeof path, "rom:/%s", name);
        if (!*sample && strchr(name, '/') && size >= 256) snprintf(sample, sizeof sample, "%s", name);
        int ok = RES(S_ENTRY_TYPE, U(&type), U(path)) == 0 && type == FS_ENTRY_FILE;
        ok = ok && RES(S_OPEN_FILE, U(&h), U(path), FS_OPEN_READ) == 0 && romfs_handle(h);
        ok = ok && RES(S_FILE_SIZE, U(&fsize), h) == 0 && (uint64_t)fsize == size;
        uint64_t head = size < HEAD_BYTES ? size : HEAD_BYTES;
        ok = ok && compare_range(image, h, off, 0, head, (int)(k % 3), a, b);
        bytes += head;
        if (size > HEAD_BYTES) {
            uint64_t tail = size - TAIL_BYTES;
            ok = ok && compare_range(image, h, off, tail, TAIL_BYTES, (int)((k + 1) % 3), a, b);
            bytes += TAIL_BYTES;
            deep += k < SAMPLES;
        }
        high += k < SAMPLES && off >= (1ull << 32);
        uint64_t got = ~0ull;
        ok = ok && RES(S_READ_FILE, U(&got), h, size, U(a), 1) == 0 && got == 0;          // at EOF: 0 bytes
        ok = ok && RES(S_READ_FILE, U(&got), h, size + 1, U(a), 1) == FS_R_INVALID;      // past EOF
        if (h) CALL(S_CLOSE_FILE, h);
        CHECK(ok, "sampled RomFS file %llu (index row %llu)", (unsigned long long)k, (unsigned long long)i);
        sampled += ok && k < SAMPLES;
    }
    printf("romfs: %llu of %d sampled files equal romfs.bin at their index offset (%llu at offsets past 4 GiB, %llu "
           "over 8 MiB compared by head and tail, %.1f MiB compared), EOF reads ok\n", (unsigned long long)sampled,
           SAMPLES, (unsigned long long)high, (unsigned long long)deep, bytes / 1048576.0);
    printf("romfs: largest file (%.1f MiB): head and tail equal romfs.bin\n", lsize / 1048576.0);

    // The same file under "rom:/", "rom://" and "rom:///"; its folder is a directory; traversal and writes refused.
    uint64_t h = 0, off, size, got = 0;
    int type = -1;
    need(*sample, "no sampled file in a folder");
    uint64_t si = 0;
    while (strcmp(romfs_entry(si, &off, &size), sample)) si++;
    int same = 1;
    for (int slashes = 1; slashes <= 3; slashes++) {
        snprintf(path, sizeof path, "rom:%.*s%s", slashes, "///", sample);
        same &= RES(S_ENTRY_TYPE, U(&type), U(path)) == 0 && type == FS_ENTRY_FILE;
        if (RES(S_OPEN_FILE, U(&h), U(path), FS_OPEN_READ)) {
            same = 0;
            continue;
        }
        same &= RES(S_READ_FILE, U(&got), h, 0, U(a), 256) == 0 && got == 256;
        same &= independent_read(image, b, 256, off) == 256 && !memcmp(a, b, 256);
        CALL(S_CLOSE_FILE, h);
    }
    CHECK(same, "rom:/, rom:// and rom:/// do not open the same file");
    snprintf(path, sizeof path, "rom:/%.*s", (int)(strrchr(sample, '/') - sample), sample);
    CHECK(RES(S_ENTRY_TYPE, U(&type), U(path)) == 0 && type == FS_ENTRY_DIRECTORY, "a sampled file's folder");
    snprintf(path, sizeof path, "rom://../%s", sample);
    CHECK(RES(S_OPEN_FILE, U(&h), U(path), FS_OPEN_READ) == FS_R_PATH_NOT_FOUND, "rom://../<file> opens");
    snprintf(path, sizeof path, "rom:/%s.absent", sample);
    CHECK(RES(S_OPEN_FILE, U(&h), U(path), FS_OPEN_READ) == FS_R_PATH_NOT_FOUND, "an absent file opens");
    uint32_t traversal = RES(S_OPEN_FILE, U(&h), U("rom:/../main"), FS_OPEN_READ);
    uint32_t write_open = RES(S_OPEN_FILE, U(&h), U("rom:/absent"), FS_OPEN_WRITE);
    snprintf(path, sizeof path, "rom:/%s", sample);
    uint32_t write_existing = RES(S_OPEN_FILE, U(&h), U(path), FS_OPEN_READ | FS_OPEN_WRITE);
    printf("romfs: rom:/, rom://, rom:/// same file: %s; rom:/../main -> %#x; write open -> %#x, %#x\n",
           same ? "ok" : "FAIL", traversal, write_open, write_existing);
    CHECK(traversal == FS_R_PATH_NOT_FOUND, "rom:/../main -> %#x, expected 0x202", traversal);
    CHECK(write_open == FS_R_UNSUPPORTED_OPEN && write_existing == FS_R_UNSUPPORTED_OPEN,
          "write open -> %#x/%#x, expected 0x6402", write_open, write_existing);

    // Overlays (OOT_ROMFS_PATCHES, OOT_ROMFS_MODS): served in place of the original, added files and their folders
    // exist, bytes equal the overlay file.
    size_t nov = romfs_overlay_count(), ovok = 0;
    for (size_t k = 0; k < nov; k++) {
        const char *file;
        uint64_t osize;
        int is_added;
        const char *name = romfs_overlay(k, &file, &osize, &is_added);
        snprintf(path, sizeof path, "rom:/%s", name);
        int ok = RES(S_ENTRY_TYPE, U(&type), U(path)) == 0 && type == FS_ENTRY_FILE;
        char *slash = strrchr(path, '/');
        if (is_added && slash > path + 5) {
            *slash = 0;
            ok = ok && RES(S_ENTRY_TYPE, U(&type), U(path)) == 0 && type == FS_ENTRY_DIRECTORY;
            *slash = '/';
        }
        int64_t fsize = -1;
        uint64_t n = osize < HEAD_BYTES ? osize : HEAD_BYTES;
        ok = ok && RES(S_OPEN_FILE, U(&h), U(path), FS_OPEN_READ) == 0;
        if (ok) {
            ok = RES(S_FILE_SIZE, U(&fsize), h) == 0 && (uint64_t)fsize == osize &&
                 RES(S_READ_FILE, U(&got), h, 0, U(a), n) == 0 && got == n;
            CALL(S_CLOSE_FILE, h);
        }
        wchar_t *wf = fs_wide(file);
        HANDLE of = CreateFileW(wf, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        free(wf);
        ok = ok && of != INVALID_HANDLE_VALUE && independent_read(of, b, n, 0) == (int64_t)n && !memcmp(a, b, n);
        if (of != INVALID_HANDLE_VALUE) CloseHandle(of);
        CHECK(ok, "overlay %zu (%s)", k, is_added ? "added" : "replacement");
        ovok += ok;
    }
    if (nov) printf("romfs: %zu of %zu overlay files served from their overlay\n", ovok, nov);
    CloseHandle(image);
    free(a);
    free(b);
    return count;
}

// ---- save transactions: restarts, uncommitted and committed changes, a crash at every step of a Commit
// The process dies here: the save mount goes without Unmount, and the next start runs save_recover and mounts again.
static void probe_restart(const uint64_t *uid) {
    AcquireSRWLockExclusive(&lock);
    for (int i = 0; i < NMOUNTS; i++)
        if (mounts[i].kind == M_SAVE) mounts[i].kind = M_FREE;
    ReleaseSRWLockExclusive(&lock);
    save_mounts = tx_crash_at = tx_dead = 0;
    InterlockedExchange(&save_dirty, 0);
    save_recover();
    CHECK(RES(S_MOUNT_SAVE, U("save"), U(uid)) == 0, "MountSaveData after a restart");
}

// Commit through the override, then wait for its background thread, as the checks read the disk next.
static uint32_t tx_commit(void) {
    uint32_t r = RES(S_COMMIT, U("save"));
    fs_save_flush();
    return r;
}

#define TX_SIZE 4096
static uint8_t tx_buf[TX_SIZE];

// Fill a save file, created when absent, with byte g through the overrides.
static int tx_write(const char *path, int g) {
    uint64_t h = 0;
    int32_t no_flush = 0;
    memset(tx_buf, g, sizeof tx_buf);
    uint32_t r = RES(S_CREATE_FILE, U(path), TX_SIZE);
    if ((r && r != FS_R_ALREADY_EXISTS) || RES(S_OPEN_FILE, U(&h), U(path), FS_OPEN_WRITE)) return 0;
    int ok = RES(S_WRITE_FILE, h, 0, U(tx_buf), TX_SIZE, U(&no_flush)) == 0;
    CALL(S_CLOSE_FILE, h);
    return ok;
}

static int tx_byte(int ok) {
    for (int i = 1; ok && i < TX_SIZE; i++) ok = tx_buf[i] == tx_buf[0];
    return ok ? tx_buf[0] : -2;
}

// The byte a save file is filled with, read through the overrides; -1 absent, -2 short or mixed.
static int tx_read(const char *path) {
    uint64_t h = 0, got = 0;
    if (RES(S_OPEN_FILE, U(&h), U(path), FS_OPEN_READ)) return -1;
    int ok = RES(S_READ_FILE, U(&got), h, 0, U(tx_buf), TX_SIZE) == 0 && got == TX_SIZE;
    CALL(S_CLOSE_FILE, h);
    return tx_byte(ok);
}

// The same, read from the committed save on disk.
static int tx_read_committed(const char *name) {
    wchar_t *n = fs_wide(name), *p = wjoin(w_save, n);
    HANDLE f = CreateFileW(p, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    free(n);
    free(p);
    if (f == INVALID_HANDLE_VALUE) return -1;
    int ok = independent_read(f, tx_buf, TX_SIZE, 0) == TX_SIZE;
    CloseHandle(f);
    return tx_byte(ok);
}

// Hard links of a file of the committed save on disk; 0 when absent.
static int tx_links_committed(const char *name) {
    wchar_t *n = fs_wide(name), *p = wjoin(w_save, n);
    HANDLE f = CreateFileW(p, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    free(n);
    free(p);
    BY_HANDLE_FILE_INFORMATION fi;
    int links = f != INVALID_HANDLE_VALUE && GetFileInformationByHandle(f, &fi) ? (int)fi.nNumberOfLinks : 0;
    if (f != INVALID_HANDLE_VALUE) CloseHandle(f);
    return links;
}

static int count_dirs(const wchar_t *dir) {
    wchar_t *pattern = wjoin(dir, L"*");
    WIN32_FIND_DATAW fd;
    HANDLE f = FindFirstFileW(pattern, &fd);
    free(pattern);
    int n = 0;
    if (f != INVALID_HANDLE_VALUE) {
        do n += (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && fd.cFileName[0] != L'.';
        while (FindNextFileW(f, &fd));
        FindClose(f);
    }
    return n;
}

static void probe_save_tx(const uint64_t *uid) {
    const char *a = "save:/tx_a.sav", *b = "save:/tx_b.sav";
    uint64_t h = 0;
    CHECK(tx_write(a, 1) && tx_write(b, 1) && tx_commit() == 0 && tx_read_committed("tx_a.sav") == 1,
          "Commit of generation 1");

    // uncommitted changes: the game reads them; the committed save and the next start do not see them
    CHECK(tx_write(a, 2) && tx_write("save:/tx_c.sav", 2), "uncommitted writes");
    CHECK(tx_read(a) == 2 && tx_read("save:/tx_c.sav") == 2, "reads do not see the game's uncommitted writes");
    CHECK(tx_read_committed("tx_a.sav") == 1 && tx_read_committed("tx_c.sav") == -1,
          "an uncommitted write reached the committed save");
    probe_restart(uid);
    CHECK(tx_read(a) == 1 && tx_read("save:/tx_c.sav") == -1, "an uncommitted write survived a restart");

    // committed changes survive a restart; an Unmount without Commit drops changes
    CHECK(tx_write(a, 3) && tx_write(b, 3) && tx_commit() == 0, "Commit of generation 3");
    probe_restart(uid);
    CHECK(tx_read(a) == 3 && tx_read(b) == 3, "a commit lost in a restart");
    CHECK(tx_write(a, 4), "write before Unmount");
    CALL(S_UNMOUNT, U("save"));
    CHECK(RES(S_MOUNT_SAVE, U("save"), U(uid)) == 0 && tx_read(a) == 3, "Unmount without Commit kept the change");

    // a crash at step k of a Commit, k = 1, 2, ... until a Commit completes: the next start sees all of the old
    // generation or all of the new one
    int committed = 3, gen = 4, steps = 0, olds = 0, news = 0, mixed = 0;
    for (int k = 1; k < 64; k++) {
        int next = ++gen, ok = tx_write(a, next) && tx_write(b, next);
        tx_crash_at = k;
        uint32_t r = tx_commit();
        int died = tx_dead;
        tx_crash_at = 0;
        if (!died) {
            steps = k - 1;
            CHECK(ok && r == 0 && tx_read_committed("tx_a.sav") == next && tx_read_committed("tx_b.sav") == next,
                  "Commit after the crash points");
            committed = next;
            break;
        }
        probe_restart(uid);
        int va = tx_read(a), vb = tx_read(b);
        if (va == committed && vb == committed) olds++;
        else if (va == next && vb == next) news++, committed = next;
        else mixed++;
        CHECK(ok && (va == vb) && (va == committed), "crash at commit step %d: tx_a %d, tx_b %d (old %d, new %d)", k,
              va, vb, committed, next);
        CHECK(!exists_w(w_new) && !exists_w(w_old), "crash at commit step %d: staging left after recovery", k);
    }
    CHECK(steps >= 4 && olds && news && !mixed, "crash points: %d (%d old, %d new, %d mixed)", steps, olds, news,
          mixed);
    int backups = count_dirs(w_backup);
    CHECK(backups == SAVE_BACKUPS, "save-backup holds %d saves", backups);

    // a Commit writes the changed file and links the unchanged one to the committed save it replaces
    int kept = committed;
    CHECK(tx_write(a, ++gen) && tx_commit() == 0 && tx_read_committed("tx_a.sav") == gen &&
          tx_read_committed("tx_b.sav") == kept && tx_links_committed("tx_b.sav") >= 2 &&
          tx_links_committed("tx_a.sav") == 1, "Commit of one changed file: tx_a %d links, tx_b %d links",
          tx_links_committed("tx_a.sav"), tx_links_committed("tx_b.sav"));
    committed = gen;

    // the background commit: a write right after Commit waits for it and stays out of it; an Unmount right after
    // Commit waits for it and keeps it
    int queued = ++gen;
    CHECK(tx_write(a, queued) && RES(S_COMMIT, U("save")) == 0 && tx_write(a, ++gen) && tx_read(a) == gen &&
          tx_read_committed("tx_a.sav") == queued, "a write right after Commit: committed tx_a %d, expected %d",
          tx_read_committed("tx_a.sav"), queued);
    CHECK(RES(S_COMMIT, U("save")) == 0, "Commit before Unmount");
    CALL(S_UNMOUNT, U("save"));
    CHECK(tx_read_committed("tx_a.sav") == gen && RES(S_MOUNT_SAVE, U("save"), U(uid)) == 0 && tx_read(a) == gen,
          "an Unmount right after Commit lost it");
    committed = gen;

    // the read-only mount shows the committed save and refuses changes
    CHECK(tx_write(a, 99) && RES(S_MOUNT_SAVE_RO, U("ro"), OOT_APPLICATION_ID, U(uid)) == 0 &&
          tx_read("ro:/tx_a.sav") == committed && RES(S_OPEN_FILE, U(&h), U("ro:/tx_a.sav"), FS_OPEN_WRITE) ==
          FS_R_UNSUPPORTED_OPEN && RES(S_CREATE_FILE, U("ro:/x"), 0) == FS_R_UNSUPPORTED_OPEN &&
          RES(S_RENAME_FILE, U("save:/tx_a.sav"), U("ro:/y")) == FS_R_INVALID_PATH, "MountSaveDataReadOnly");
    CALL(S_UNMOUNT, U("ro"));
    CHECK(RES(S_DELETE_FILE, U(a)) == 0 && RES(S_DELETE_FILE, U(b)) == 0 && tx_commit() == 0 &&
          tx_read_committed("tx_a.sav") == -1, "Commit of deletions");
    printf("save tx: uncommitted writes visible to the game and dropped by a restart and by Unmount, commits kept; a "
           "crash at each of %d commit steps recovers to the old save %d times, the new one %d times, mixed %d "
           "times; %d backups\n", steps, olds, news, mixed, backups);
}

static void probe_storage(void) {
    wchar_t tmpw[MAX_PATH + 1];
    char tmp[MAX_PATH * 3], dir[MAX_PATH * 3 + 64];
    DWORD n = GetTempPathW(MAX_PATH + 1, tmpw);
    need(n && WideCharToMultiByte(CP_UTF8, 0, tmpw, -1, tmp, sizeof tmp, NULL, NULL) > 0, "temporary directory");
    snprintf(dir, sizeof dir, "%soot-probe-fs-%lu", tmp, GetCurrentProcessId());
    remove_tree(dir);
    fs_host_set_profile(dir);
    uint64_t uid[2], other[2], h = 0, h2 = 0, d = 0, l = 0;
    account_uid(uid);
    other[0] = uid[0] ^ 1, other[1] = uid[1];
    static uint8_t data[65536], back[65536];
    for (unsigned i = 0; i < sizeof data; i++) data[i] = (uint8_t)(i * 7 + (i >> 8));
    int32_t no_flush = 0, flush = 1;
    int64_t size = -1, count = -1;
    uint64_t got = 0;
    int type = -1;

    CHECK(RES(S_MOUNT_SAVE, U("save"), U(other)) == FS_R_TARGET_NOT_FOUND, "MountSaveData for another user");
    CHECK(RES(S_MOUNT_SAVE, U("save"), U(uid)) == 0, "MountSaveData");
    CHECK(RES(S_MOUNT_SAVE, U("save"), U(uid)) == FS_R_ALREADY_EXISTS, "MountSaveData twice");
    CHECK((CALL(S_SAVE_EXISTS, OOT_APPLICATION_ID, U(uid)) & 0xff) == 1 &&
          (CALL(S_SAVE_EXISTS, OOT_APPLICATION_ID + 0x1000, U(uid)) & 0xff) == 0, "IsSaveDataExisting");
    CHECK(RES(S_MOUNT_SAVE_RO, U("other"), OOT_APPLICATION_ID + 0x1000, U(uid)) == FS_R_TARGET_NOT_FOUND,
          "MountSaveDataReadOnly of another title");
    CHECK(!fs_save_busy(), "idle at start");

    // create, write, commit
    CHECK(RES(S_CREATE_FILE, U("save:/probe.sav"), 0) == 0 && fs_save_busy(), "CreateFile");
    CHECK(RES(S_CREATE_FILE, U("save:/probe.sav"), 0) == FS_R_ALREADY_EXISTS, "CreateFile twice");
    CHECK(RES(S_OPEN_FILE, U(&h), U("save:/probe.sav"), FS_OPEN_WRITE) == 0, "OpenFile for writing");
    CHECK(RES(S_WRITE_FILE, h, 0, U(data), 16, U(&no_flush)) == FS_R_APPEND_REQUIRED, "a growing write without append");
    CALL(S_CLOSE_FILE, h);
    CHECK(RES(S_OPEN_FILE, U(&h), U("save:/probe.sav"), FS_OPEN_WRITE | FS_OPEN_APPEND) == 0, "OpenFile for append");
    CHECK(RES(S_WRITE_FILE, h, 0, U(data), sizeof data / 2, U(&no_flush)) == 0 &&
          RES(S_WRITE_FILE, h, sizeof data / 2, U(data + sizeof data / 2), sizeof data / 2, U(&flush)) == 0,
          "WriteFile");
    CHECK(RES(S_FILE_SIZE, U(&size), h) == 0 && size == sizeof data, "GetFileSize after writing");
    CHECK(RES(S_READ_FILE, U(&got), h, 0, U(back), 16) == FS_R_OPEN_MODE, "ReadFile on a write-only handle");
    CHECK(RES(S_FLUSH_FILE, h) == 0, "FlushFile");
    CHECK(fs_save_busy(), "a save file open for writing does not count as a save in progress");
    CALL(S_CLOSE_FILE, h);
    CHECK(fs_save_busy(), "closed but uncommitted save data does not count as a save in progress");
    CHECK(tx_commit() == 0 && !fs_save_busy(), "Commit");
    CHECK(RES(S_COMMIT, U("nomount")) == FS_R_PATH_NOT_FOUND, "Commit of an unknown mount");

    // paths
    CHECK(RES(S_CREATE_FILE, U("save:/../x"), 0) == FS_R_INVALID_PATH, "save:/../x");
    CHECK(RES(S_CREATE_FILE, U("save:/a\\b"), 0) == FS_R_PATH_NOT_FOUND, "save:/a\\b: a backslash separates, and a is no directory");
    CHECK(RES(S_CREATE_FILE, U("save:/a:b"), 0) == FS_R_INVALID_PATH, "save:/a:b");
    CHECK(RES(S_CREATE_FILE, U("save:/a."), 0) == FS_R_INVALID_PATH, "save:/a.");
    CHECK(RES(S_CREATE_FILE, U("nomount:/x"), 0) == FS_R_PATH_NOT_FOUND, "a path on no mount");

    // directories
    CHECK(RES(S_CREATE_DIRECTORY, U("save:/dir")) == 0 && RES(S_CREATE_FILE, U("save:/dir/inner.bin"), 8) == 0,
          "CreateDirectory");
    CHECK(RES(S_ENTRY_TYPE, U(&type), U("save:/dir")) == 0 && type == FS_ENTRY_DIRECTORY, "GetEntryType (directory)");
    static DirectoryEntry ents[8];
    CHECK(RES(S_OPEN_DIRECTORY, U(&d), U("save:/"), 3) == 0, "OpenDirectory");
    CHECK(RES(S_READ_DIRECTORY, U(&count), U(ents), d, 8) == 0 && count == 2, "ReadDirectory count %lld",
          (long long)count);
    int seen = 0;
    for (int i = 0; i < count && i < 8; i++) {
        if (!strcmp(ents[i].name, "dir") && ents[i].type == FS_ENTRY_DIRECTORY) seen |= 1;
        if (!strcmp(ents[i].name, "probe.sav") && ents[i].type == FS_ENTRY_FILE && ents[i].size == sizeof data)
            seen |= 2;
    }
    CHECK(seen == 3, "ReadDirectory entries");
    CHECK(RES(S_READ_DIRECTORY, U(&count), U(ents), d, 8) == 0 && count == 0, "ReadDirectory at the end");
    CALL(S_CLOSE_DIRECTORY, d);
    CHECK(RES(S_OPEN_DIRECTORY, U(&d), U("save:/"), 2) == 0 && RES(S_READ_DIRECTORY, U(&count), U(ents), d, 8) == 0 &&
          count == 1 && ents[0].type == FS_ENTRY_FILE, "ReadDirectory, files only");
    CALL(S_CLOSE_DIRECTORY, d);

    CHECK(tx_commit() == 0, "Commit of the directories");

    // reopen after unmount: the bytes come back through the overrides and are on disk
    CALL(S_UNMOUNT, U("save"));
    CHECK(RES(S_OPEN_FILE, U(&h), U("save:/probe.sav"), FS_OPEN_READ) == FS_R_PATH_NOT_FOUND, "open after Unmount");
    CHECK(RES(S_MOUNT_SAVE, U("save"), U(uid)) == 0, "MountSaveData again");
    CHECK(RES(S_OPEN_FILE, U(&h), U("save:/probe.sav"), FS_OPEN_READ) == 0, "reopen");
    memset(back, 0, sizeof back);
    CHECK(RES(S_FILE_SIZE, U(&size), h) == 0 && size == sizeof data &&
          RES(S_READ_EXACT, h, 0, U(back), sizeof back) == 0 && !memcmp(back, data, sizeof data), "reopened bytes");
    CHECK(RES(S_READ_FILE, U(&got), h, sizeof data - 10, U(back), 100) == 0 && got == 10, "short read at the end");
    CHECK(RES(S_WRITE_FILE, h, 0, U(data), 1, U(&no_flush)) == FS_R_OPEN_MODE, "WriteFile on a read-only handle");
    CALL(S_CLOSE_FILE, h);
    char host[1200];
    snprintf(host, sizeof host, "%s/save/%016llx%016llx/probe.sav", dir, (unsigned long long)uid[1],
             (unsigned long long)uid[0]);
    wchar_t *w = fs_wide(host);
    HANDLE f = CreateFileW(w, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    free(w);
    memset(back, 0, sizeof back);
    CHECK(f != INVALID_HANDLE_VALUE && independent_read(f, back, sizeof back, 0) == sizeof back &&
          !memcmp(back, data, sizeof data), "the save file on disk");
    if (f != INVALID_HANDLE_VALUE) CloseHandle(f);

    // resize, rename, delete
    CHECK(RES(S_OPEN_FILE, U(&h), U("save:/probe.sav"), FS_OPEN_READ | FS_OPEN_WRITE) == 0 &&
          RES(S_SET_FILE_SIZE, h, 100) == 0 && RES(S_FILE_SIZE, U(&size), h) == 0 && size == 100, "SetFileSize");
    CHECK(RES(S_OPEN_FILE, U(&h2), U("save:/probe.sav"), FS_OPEN_READ) == 0, "a second handle on one file");
    CALL(S_CLOSE_FILE, h2);
    CALL(S_CLOSE_FILE, h);
    CHECK(RES(S_RENAME_FILE, U("save:/probe.sav"), U("save:/dir/inner.bin")) == FS_R_ALREADY_EXISTS,
          "RenameFile over an existing file");
    CHECK(RES(S_RENAME_FILE, U("save:/probe.sav"), U("save:/renamed.sav")) == 0 &&
          RES(S_ENTRY_TYPE, U(&type), U("save:/renamed.sav")) == 0 && type == FS_ENTRY_FILE &&
          RES(S_ENTRY_TYPE, U(&type), U("save:/probe.sav")) == FS_R_PATH_NOT_FOUND, "RenameFile");
    CHECK(RES(S_DELETE_FILE, U("save:/renamed.sav")) == 0 &&
          RES(S_DELETE_FILE, U("save:/renamed.sav")) == FS_R_PATH_NOT_FOUND, "DeleteFile");
    CHECK(RES(S_DELETE_RECURSIVE, U("save:/dir")) == 0 &&
          RES(S_ENTRY_TYPE, U(&type), U("save:/dir")) == FS_R_PATH_NOT_FOUND, "DeleteDirectoryRecursively");
    CHECK(tx_commit() == 0 && !fs_save_busy(), "Commit after the changes");
    probe_save_tx(uid);

    // temporary and cache storage
    CHECK(RES(S_MOUNT_TEMP, U("temp")) == 0 && RES(S_CREATE_FILE, U("temp:/t.bin"), 4) == 0 && !fs_save_busy(),
          "temporary storage");
    CHECK(RES(S_CREATE_CACHE, 3, 1 << 20, 1 << 20) == 0 && RES(S_CREATE_CACHE, 3, 1 << 20, 1 << 20) ==
          FS_R_ALREADY_EXISTS, "CreateCacheStorage");
    CHECK(RES(S_MOUNT_CACHE, U("cache"), 3) == 0 && RES(S_MOUNT_CACHE, U("cache9"), 9) == FS_R_TARGET_NOT_FOUND,
          "MountCacheStorage");
    CacheStorageInfo infos[4];
    int ncache = -1;
    CHECK(RES(S_OPEN_CACHE_LIST, U(&l)) == 0 && RES(S_READ_CACHE_LIST, U(&ncache), U(infos), l, 4) == 0 &&
          ncache == 1 && infos[0].index == 3, "cache storage list");
    CALL(S_CLOSE_CACHE_LIST, l);
    CALL(S_SET_PRIORITY, 1);
    CALL(S_UNMOUNT, U("cache"));
    CALL(S_UNMOUNT, U("temp"));
    CHECK(RES(S_DELETE_CACHE, 3) == 0, "DeleteCacheStorage");
    CALL(S_UNMOUNT, U("save"));

    int removed = !remove_tree(dir) && attributes(dir) == INVALID_FILE_ATTRIBUTES;
    CHECK(removed, "the temporary profile was not deleted");
    printf("storage: save create/write/commit/reopen, directories, rename/delete, temporary and cache storage in a "
           "temporary profile: %s; profile deleted: %s\n", failures ? "see FAIL lines" : "ok", removed ? "yes" : "no");
}

int probe_fs(void) {
    fs_init();
    for (int i = 0; i < NPROBE; i++) {
        GuestFn fn = hostfn_override(probe_symbols[i]);
        if (!fn) rt_fatal("probe-fs: no HOST_OVERRIDE for %s", probe_symbols[i]);
        probe_addr[i] = hostfn_register(probe_symbols[i], fn);
    }
    printf("overrides: %d nn::fs overrides registered\n", NPROBE);
    pcpu = _aligned_malloc(sizeof(Cpu), 64);
    need(pcpu != NULL, "probe Cpu");
    memset(pcpu, 0, sizeof *pcpu);
    uint8_t *stack = mem_kernel_alloc(0x10000, 16);
    pcpu->sp = (uint64_t)(uintptr_t)stack + 0x10000;
    pcpu->excl_addr = ~0ull;
    for (int r = 19; r < 30; r++) pcpu->x[r] = 0x1900 + r;
    CALL(S_INITIALIZE, 0);   // nn::fs::InitializeForApplication: fs_init again, idempotent
    uint64_t files = probe_romfs();
    probe_storage();
    int kept = 1;
    for (int r = 19; r < 30; r++) kept &= pcpu->x[r] == 0x1900u + r;
    CHECK(kept && pcpu->sp == (uint64_t)(uintptr_t)stack + 0x10000, "an override lost a callee-saved register");
    mem_kernel_free(stack, 0x10000);
    _aligned_free(pcpu);
    if (failures) {
        printf("probe-fs: FAILED (%d of %d checks)\n", failures, checks);
        return 1;
    }
    printf("probe-fs: passed (%llu files, %d sampled)\n", (unsigned long long)files, SAMPLES);
    return 0;
}
