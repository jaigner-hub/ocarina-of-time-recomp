// Runs this program again, with the same arguments, and captures what it prints: the probes that check a fatal stub
// (nvn_probe.c) or a service's behaviour in a child process (ov_probe.c). Windows: CreateProcessW with our command line,
// the pipe on stdout and stderr (as the probes always had it). Linux: posix_spawn of /proc/self/exe with the arguments
// of /proc/self/cmdline, the same way plat_reexec runs the program again.
// The caller sets the environment variable that tells the child what to do (plat_setenv) before the call.
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "platform/platform.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

// Output (stdout and stderr together, NUL-terminated, at most cap - 1 bytes) into out; the child's exit code. Stops with
// a fatal naming tag when the child cannot start.
static inline uint32_t child_run_self(const char *tag, const char *env_name, const char *mode, char *out, size_t cap) {
    if (!plat_setenv(env_name, mode)) rt_fatal("%s: cannot set %s", tag, env_name);
#ifdef _WIN32
    SECURITY_ATTRIBUTES sa = {sizeof sa, NULL, TRUE};
    HANDLE rd, wr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) rt_fatal("%s: CreatePipe (error %lu)", tag, GetLastError());
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si = {.cb = sizeof si, .dwFlags = STARTF_USESTDHANDLES, .hStdOutput = wr, .hStdError = wr};
    PROCESS_INFORMATION pi;
    wchar_t *cmdline = _wcsdup(GetCommandLineW());
    BOOL ok = CreateProcessW(NULL, cmdline, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
    plat_setenv(env_name, NULL);
    free(cmdline);
    CloseHandle(wr);
    if (!ok) rt_fatal("%s: CreateProcess for the %s child (error %lu)", tag, mode, GetLastError());
    size_t n = 0;
    DWORD got;
    while (n + 1 < cap && ReadFile(rd, out + n, (DWORD)(cap - 1 - n), &got, NULL) && got) n += got;
    out[n] = 0;
    char sink[512];
    while (ReadFile(rd, sink, sizeof sink, &got, NULL) && got) {}
    CloseHandle(rd);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (uint32_t)code;
#else
    static char args[16384];
    char *argv[256];
    int argc = 0;
    FILE *f = fopen("/proc/self/cmdline", "rb");
    size_t len = f ? fread(args, 1, sizeof args - 1, f) : 0;
    if (f) fclose(f);
    args[len] = 0;
    for (size_t i = 0; i < len && argc < 255; i += strlen(args + i) + 1) argv[argc++] = args + i;
    argv[argc] = NULL;
    int fds[2];
    if (pipe(fds)) {
        plat_setenv(env_name, NULL);
        rt_fatal("%s: pipe (error %u)", tag, plat_last_error());
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_addclose(&fa, fds[1]);
    pid_t pid = 0;
    int err = posix_spawn(&pid, "/proc/self/exe", &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);
    plat_setenv(env_name, NULL);
    if (err) {
        close(fds[0]);
        rt_fatal("%s: spawn for the %s child (error %d)", tag, mode, err);
    }
    size_t n = 0;
    while (n + 1 < cap) {
        ssize_t r = read(fds[0], out + n, cap - 1 - n);
        if (r <= 0) break;
        n += (size_t)r;
    }
    out[n] = 0;
    char sink[512];
    while (read(fds[0], sink, sizeof sink) > 0) {}
    close(fds[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return 1;
    return WIFEXITED(status) ? (uint32_t)WEXITSTATUS(status) : 128u + (uint32_t)(WIFSIGNALED(status) ? WTERMSIG(status) : 0);
#endif
}
