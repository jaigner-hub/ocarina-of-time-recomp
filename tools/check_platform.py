#!/usr/bin/env python3
# From metroid-prime-4-recomp tools/check_platform.py @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied unchanged.
# From fe3h-pc-recomp tools/check_platform.py @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied unchanged.
# From totk-pc-recomp tools/check_platform.py @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), copied unchanged.
"""Fails when runtime code outside runtime/platform/ uses the Win32 API directly.

New runtime code goes through runtime/platform/platform.h. Files that still use Win32 are listed in ALLOWED with the
reason; the list shrinks as the platform layer takes over.
A file in ALLOWED that no longer matches is noted, so the entry gets removed.

Usage: python tools/check_platform.py [--root DIR]   (DIR holds runtime/; default: the repository root)
"""
import argparse
import re
import sys
from pathlib import Path

# Per-file allowlist. "window", "input", "audio" and "probe" are permanent (OS-specific by nature); the rest wait for
# 2.7a (memory, faults, page tracking) or 2.7b (threads, time, locks, FP state).
ALLOWED = {
    "gfx_window.c": "window",
    "gfx_window_probe.c": "probe",
    "input_win.c": "input",
    "prof_win.c": "profiler",
    "rdoc_win.c": "RenderDoc loader",
    "dpi_probe.c": "DPI probe: its whole body is Win32 DPI, behind _WIN32; other systems get a stub that fails",
    "audren.c": "audio",
    "kernel_probe.c": "probe",
    "nvn_probe.c": "probe",
    "ov_probe.c": "probe",
    "tex_probe.c": "probe",
    "vk_exec_probe.c": "probe",
    "gfx_shader_probe.c": "probe",
    "gfx_shader.c": "2.7b: threads, SRW locks, interlocked counters, tick count",
    "main.c": "2.7a/b",
    "dispatch.c": "2.7b",
    "svc.c": "2.7b",
    "vk_texture.c": "2.7b",
    "audren.h": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "boot.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "flight.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "fs_host.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "ipc.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "nvn_api_cmd.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "nvn_api_core.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "nvn_api_obj.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "nvn_procs.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "ov_gfx.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "ov_input.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "ov_media.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "ov_system.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "romfs.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "tex_astc_mt.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "vk_backend.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "vk_device.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "vk_dump.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "vk_pipeline.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
    "vk_sync.c": "pending: SRW locks, threads, files; not in 2.7a/b/c scope yet",
}

INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](windows|winsock2|ws2tcpip|synchapi|memoryapi|processthreadsapi|'
                     r'fileapi|handleapi|libloaderapi|winbase|winnt)\.h[>"]', re.M)
CALLS = [
    "CreateFile[AW]?", "ReadFile", "WriteFile", "CloseHandle", "DeleteFile[AW]?", "MoveFileEx[AW]?", "CreateDirectory[AW]?",
    "FindFirstFile[AW]?", "GetFileAttributes[AW]?", "GetFileSizeEx", "SetFilePointerEx", "SetEndOfFile",
    "CreatePipe", "CreateProcess[AW]?", "TerminateProcess", "GetModuleFileName[AW]?", "GetEnvironmentVariable[AW]?",
    "VirtualAlloc(Ex)?", "VirtualFree", "VirtualProtect", "VirtualQuery", "MapViewOfFile[A-Za-z0-9]*", "CreateFileMapping[AW]?",
    "GetWriteWatch", "ResetWriteWatch", "AddVectoredExceptionHandler", "RemoveVectoredExceptionHandler",
    "CreateThread", "_beginthread(ex)?", "GetCurrentThread(Id)?", "SetThreadAffinityMask", "SetThreadDescription",
    "SetThreadPriority", "WaitForSingleObject(Ex)?", "WaitForMultipleObjects", "CreateEvent[AW]?", "CreateSemaphore[AW]?",
    "CreateMutex[AW]?", "CreateWaitableTimer(Ex)?[AW]?", "SetWaitableTimer(Ex)?", "Sleep(Ex)?", "SwitchToThread",
    "QueryPerformanceCounter", "QueryPerformanceFrequency", "GetTickCount(64)?", "GetSystemTimePreciseAsFileTime",
    "GetSystemInfo", "GetLogicalProcessorInformation(Ex)?", "SRWLOCK", "CONDITION_VARIABLE", "CRITICAL_SECTION",
    "InitializeCriticalSection[A-Za-z]*", "EnterCriticalSection", "LeaveCriticalSection",
    "(Acquire|Release)SRWLock(Exclusive|Shared)", "WakeAllConditionVariable", "WakeConditionVariable",
    "SleepConditionVariableSRW", "InitOnceExecuteOnce", "INIT_ONCE", "Interlocked[A-Za-z0-9]*", "TlsAlloc", "FlsAlloc",
    "LoadLibrary[AW]?", "GetProcAddress", "FreeLibrary", "HANDLE", "DWORD", "LPVOID", "WINAPI",
]
CALL_RE = re.compile(r"\b(" + "|".join(CALLS) + r")\b")


def strip(text):
    """Blank comments and string literals, keeping line numbers."""
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))
    text = re.sub(r"/\*.*?\*/", blank, text, flags=re.S)
    text = re.sub(r"//[^\n]*", blank, text)
    return re.sub(r'"(?:\\.|[^"\\\n])*"', blank, text)


def scan(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    hits = [("include", m.group(0).strip()) for m in INCLUDE.finditer(raw)]
    code = strip(raw)
    hits += [("call", m.group(1)) for m in CALL_RE.finditer(code)]
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--runtime", default=None, help="runtime dir (default ROOT/runtime)")
    a = ap.parse_args()
    rt = Path(a.runtime) if a.runtime else Path(a.root) / "runtime"
    bad, remaining = [], {}
    for p in sorted(list(rt.glob("*.c")) + list(rt.glob("*.h")) + list(rt.glob("*.cpp"))):
        hits = scan(p)
        if p.name in ALLOWED:
            remaining[p.name] = len(hits)
        elif hits:
            bad.append((p.name, hits))
    for name, hits in bad:
        kinds = sorted({h[1] for h in hits})
        print(f"FAIL runtime/{name}: {len(hits)} Win32 uses outside runtime/platform/ ({', '.join(kinds[:8])})")
    stale = [n for n, c in remaining.items() if c == 0]
    for n in stale:
        print(f"note runtime/{n}: allowlisted but no Win32 use left; remove it from ALLOWED")
    print("remaining Win32 uses per allowlisted file:")
    for n, c in sorted(remaining.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {c:5d}  runtime/{n}  ({ALLOWED[n]})")
    print(f"total {sum(remaining.values())} in {sum(1 for c in remaining.values() if c)} files")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
