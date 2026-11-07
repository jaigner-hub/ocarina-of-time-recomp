# From metroid-prime-4-recomp tools/build_game.py @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied with mp4 renamed to oot.
# From fe3h-pc-recomp tools/build_game.py @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), copied with fe3h renamed to mp4; MP4's version names; openh264 headers (the library is loaded at run time, runtime/ov_movie.c).
# From totk-pc-recomp tools/build_game.py @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh); renamed to fe3h, FE3H version names, expected hashes from the extraction manifest, no zstd, no icon, one profile folder.
"""Compile the generated C for one game version and link it with the runtime into build/<version>/oot.exe.

Writes build/<version>/build.ninja and runs MSYS2 UCRT64 Ninja on it. Ninja keeps the build incremental: a chunk is
recompiled only when it, a header it includes or its flags change.

    .venv\\Scripts\\python tools\\build_game.py                     # generated/v327680 -> build/v327680/oot.exe
    .venv\\Scripts\\python tools\\build_game.py --modules subsdk0   # one module (links if it has no calls into others)
    .venv\\Scripts\\python tools\\build_game.py --jobs 8
    .venv\\Scripts\\python tools\\build_game.py --out build/v327680-t2   # a separate build directory
    .venv\\Scripts\\python tools\\build_game.py --toolchain toolchain/llvm-mingw-<version>/llvm-mingw-<version>-ucrt-x86_64 \\
        --out build/v327680-llvmmingw                                    # llvm-mingw instead of MSYS2 UCRT64

Every runtime/*.c is compiled. build/<version>/modules.c carries the layout, the linked function tables, the
expected SHA-256 of each ExeFS file (from build/extracted/<version>/extraction-manifest.json, written by
tools/extract_nup.py, unless --expected names a file) and the default game paths
(--exefs, --romfs, --profile), which oot.exe uses unless its own command line names others. tools/romfs_index.py
writes build/<version>/romfs-index.bin and romfs_config.h from the RomFS directory's index.json (runtime/romfs.c reads
the index next to oot.exe).

Private builds reuse the main build's objects. When --out differs from build/<version>, the main build's generated
objects are up to date (checked with a dry run of its build.ninja) and every header they include from runtime/ is
byte-identical in --runtime (the list comes from the main build's Ninja deps log), build_game.py hardlinks those
objects into <out>/obj and compiles only runtime/*.c. A hardlink is a snapshot: clang and Ninja replace an object by
renaming a new file over it, so a later main rebuild leaves the linked copies alone. Otherwise (a header differs, a
main object is missing or stale, another --gen) it compiles everything and prints one line saying why. --full-build
skips the check.

Object manifest. A flipped bit in an object written by an earlier build stays forever, because Ninja rebuilds an object
only when its inputs change, and forks hardlink the object into every private build. So each build dir keeps
obj/objects.sha256 (SHA-256, size and mtime of every object it compiled). After Ninja finishes and before the link,
build_game.py hashes the objects, compares each one whose size and mtime match its manifest entry, and records the
objects Ninja just rebuilt (new mtime). A mismatch deletes the object, Ninja recompiles it, and the build prints the
summary line. A fork checks the hardlinked generated objects against build/<version>/obj/objects.sha256 (read-only; a
bad shared object stops the fork and asks for a main build run, which recompiles it and breaks the shared inode).

Compile jobs share a memory budget (MEM_BUDGET, default the smaller of 50 GiB and 80% of RAM) at JOB_MEM per job, so a
big machine does not run out of memory on the largest chunks. Each phase reports its wall time and the peak memory
of its processes (Windows job object accounting: committed private bytes).

--ninja PATH names the Ninja to run; without MSYS2 the bin/ninja.exe of the release folder (tools/make_release.py) is used.
--toolchain DIR builds with an llvm-mingw tree (DIR/bin/clang.exe) instead of MSYS2 UCRT64 clang. Ninja still comes
from MSYS2. The libraries and headers MSYS2 supplies (Vulkan, SDL3, Opus, libvpx, VMA) then come from --deps DIR
(default toolchain/deps-mingw: include/, lib/ with the import libraries and libvpx.a, bin/ with the DLLs), and the C++
runtime is libc++ from llvm-mingw. Such a build never reuses another build's generated objects.

Split build. --make-runtime-lib compiles only runtime/ into <out>/libootrt.a (what CI prebuilds). --runtime-lib PATH
then compiles only the generated C and the install-time files (modules.c and INSTALL_SIDE, which bake the player's
game hashes and version in) and links them with that library. The library holds no game-derived data.

Standard library only.
"""
import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == 'win32':
    import ctypes.wintypes as wt

import romfs_index
from build_shaders import stale as stale_shaders
from check_content import GENERATED_MARKER

ROOT = Path(__file__).resolve().parent.parent
# In a git worktree (tools/worktree.py) build/ and generated/ are junctions into the main checkout, and resolve() follows
# them. MAIN_ROOT is where they land, so resolved paths still print and compare relative to a root.
MAIN_ROOT = (ROOT / 'build').resolve().parent
def find_tool(name):
    """The executable called name on PATH, else name itself as a path that does not exist, so main() says which tool
    is missing."""
    return Path(shutil.which(name) or name)


# Flags for generated code as LAS Recomp builds it (Makefile GENFLAGS, Windows), plus -march for the FP/SIMD lowering.
GEN_FLAGS = ['-O1', '-fno-strict-aliasing', '-Wno-return-type', '-Wno-unused-label', '-fvisibility=hidden',
             '-march=x86-64-v3']
DEFAULT_DEPS = ROOT / 'toolchain' / 'deps-mingw'
# Runtime sources compiled per install, not in the library: romfs.c includes romfs_config.h, patches.c the patches.h
# of the player's game version (its patch page is the page after that version's main).
INSTALL_SIDE = {'romfs.c', 'patches.c'}
LIB_NAME = 'libootrt.a'
# --toolchain: an llvm-mingw tree. The libraries, headers and DLLs MSYS2 supplies come from the deps directory instead.
if sys.platform == 'win32':
    UCRT = Path(r'C:\msys64\ucrt64\bin')
    CLANG = UCRT / 'clang.exe'
    NINJA = UCRT / 'ninja.exe'
    SDL3_DLLS = [UCRT / 'SDL3.dll', UCRT / 'libiconv-2.dll', UCRT / 'libopus-0.dll',
                 UCRT / 'libopenh264-8.dll']   # OOT: the movies' H.264 decoder for a development build; an install gets
                                               # Cisco's DLL from the installer instead (runtime/ov_movie.c h264_lib)
    # _WIN32_WINNT: Windows 10, the port's minimum. MSYS2's headers default to it, llvm-mingw's (the release toolchain)
    # to Windows 7, which hides the per-monitor DPI functions gfx_window.c and dpi_probe.c call.
    RT_FLAGS = ['-O2', '-g', '-Wall', '-fno-strict-aliasing', '-march=x86-64-v3', '-D_WIN32_WINNT=0x0A00']
    # -Xlinker passes the map path as one argument; -Wl, would split it at any comma in the build dir path
    LINK_FLAGS = ['-fuse-ld=lld', '-Wl,--stack,16777216', '-Xlinker', '"--Map=$builddir/oot.map"']   # the map feeds tools/prof_report.py
    LINK_LIBS = ['-lpsapi', '-lbcrypt', '-lvulkan-1', '-luser32', '-lgdi32', '-ldwmapi', '-lwinmm', '-lSDL3', '-lopus', '-l:libvpx.a', '-l:libdav1d.a', '-l:libstdc++.a', '-l:libwinpthread.a']   # Vulkan back end; libvpx and dav1d static (movies)
    VMA_INCLUDE = UCRT.parent / 'include' / 'vma'   # vk_mem_alloc.h; MSYS2 package vulkan-memory-allocator
    LLVM_LINK_LIBS = ['-lpsapi', '-lbcrypt', '-luser32', '-lgdi32', '-ldwmapi', '-lwinmm', '-lvulkan-1', '-lSDL3', '-lopus',
                      '-l:libvpx.a', '-l:libdav1d.a', '-l:libc++.a', '-l:libc++abi.a', '-l:libunwind.a', '-l:libwinpthread.a']
    AR = UCRT / 'llvm-ar.exe' if (UCRT / 'llvm-ar.exe').is_file() else UCRT / 'ar.exe'   # MSYS2 without the llvm package has binutils' ar
    WINDRES = UCRT / 'windres.exe'   # runtime/oot.rc: the manifest of oot.exe
    EXE_NAME = 'oot.exe'
else:
    # Linux: clang, ninja and binutils from PATH (Fedora: clang, ninja-build, binutils). No DLLs are copied beside the
    # program. The link libraries are a first guess until runtime/platform/linux.c exists (the runtime is not ported).
    UCRT = None
    CLANG = find_tool('clang')
    NINJA = find_tool('ninja')
    SDL3_DLLS = []
    RT_FLAGS = ['-O2', '-g', '-Wall', '-fno-strict-aliasing', '-march=x86-64-v3']
    LINK_FLAGS = ['-Xlinker', '"--Map=$builddir/oot.map"']
    LINK_LIBS = ['-lvulkan', '-lSDL3', '-lopus', '-lm', '-lpthread', '-ldl']
    VMA_INCLUDE = Path('/usr/include')   # vk_mem_alloc.h from the VulkanMemoryAllocator-devel package
    LLVM_LINK_LIBS = LINK_LIBS
    AR = find_tool('llvm-ar') if shutil.which('llvm-ar') else find_tool('ar')
    WINDRES = Path('windres')   # unused: runtime_sources() leaves out runtime/*.rc on Linux
    EXE_NAME = 'oot'
TOOLCHAIN_BIN = None   # llvm-mingw bin directory, set by main() for --toolchain
DEPS = None            # the deps directory of an llvm-mingw build
# Display version of each build version name (the update's version number, as extract_exefs.py names the folder).
DEFAULT_VERSION = 'v131072'
DISPLAY_VERSIONS = {'v131072': '1.1.1'}

GiB = 1 << 30
# Peak commit of one compile, TotK 1.2.1 at -O1 (2026-10-20): 120-180 MiB per ~3 MiB chunk, 568 MiB for main_table.c
# (295k-entry tables). 768 MiB leaves headroom for the largest file.
JOB_MEM = 768 << 20
MEM_BUDGET = 50 * GiB  # total for all compile jobs


def below_root(p):
    """Resolved p relative to ROOT, else to MAIN_ROOT (the same relative path through the worktree's junctions), else
    absolute."""
    p = Path(p).resolve()
    for root in (ROOT, MAIN_ROOT):
        if p.is_relative_to(root):
            return p.relative_to(root)
    return p


def ninja_path(p):
    """A path for a build line: relative to the repo root where possible, forward slashes, Ninja-escaped."""
    return below_root(p).as_posix().replace('$', '$$').replace(' ', '$ ').replace(':', '$:')


def flag_path(p):
    return below_root(p).as_posix()


def cmd_arg(s):
    """s for a Ninja command line: Ninja-escaped, and quoted when it holds a space (a user or folder name), as clang and
    lld split their Windows command line at unquoted spaces."""
    s = s.replace('$', '$$')
    return f'"{s}"' if ' ' in s else s


def cmd_path(p):
    return cmd_arg(flag_path(p))


def read_layout(gen):
    """[(name, base, size)] from layout.json (same format as recomp/layout.py read(), without its NSO dependency)."""
    data = json.loads((gen / 'layout.json').read_text())
    return [(m['name'], int(m['base'], 16), int(m['size'], 16)) for m in data]


def chunk_files(gen, mod):
    """The C files of one module: the chunks listed in <mod>/manifest.json, then <mod>_table.c."""
    d = gen / mod
    data = json.loads((d / 'manifest.json').read_text())
    entries = data if isinstance(data, list) else data.get('chunks', data.get('files'))
    if entries is None:
        sys.exit(f'{d / "manifest.json"}: no "chunks" or "files" list')
    names = []
    for e in entries:
        name = e if isinstance(e, str) else e.get('file', e.get('path', e.get('name')))
        names.append(Path(name).name)
    table = f'{mod}_table.c'
    if table not in names:
        names.append(table)
    files = [d / n for n in names]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        sys.exit(f'{mod}: {len(missing)} files listed in manifest.json are missing, first {missing[0]}')
    return files


def write_if_changed(path, text):
    """Keep the mtime of an unchanged file so Ninja does not rebuild from it."""
    if path.exists() and path.read_text(encoding='utf-8', errors='replace') == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')   # Ninja and clang read UTF-8; paths may be non-ASCII


def c_string(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


RUNTIME = ROOT / 'runtime'   # --runtime: a copy of runtime/ lets a task build without touching the shared tree


def runtime_sources(skip=()):
    """Every runtime/*.c, *.cpp, platform/*.c, *.rc, the in-game menu (ui/*.cpp) and Dear ImGui except the names in skip,
    so a task adds a file without editing this script."""
    pats = ('*.c', '*.cpp', 'platform/*.c', '*.rc', 'ui/*.cpp', 'third_party/imgui/*.cpp',
            'third_party/imgui/backends/*.cpp')
    return sorted(p for pat in pats for p in RUNTIME.glob(pat)
                  if p.name not in skip and (sys.platform == 'win32' or p.suffix != '.rc'))


def runtime_object(out, src):
    """obj/runtime/<path below runtime>.o, so platform/win32.c cannot collide with a top-level win32.c."""
    src = Path(src)
    rel = src.relative_to(RUNTIME) if src.is_relative_to(RUNTIME) else Path(src.name)
    return out / 'obj' / 'runtime' / rel.with_suffix('.o')


def module_glue(layout, linked, hashes, paths, version):
    """C for build/<version>/modules.c: the layout table, the linked modules' function tables (dispatch.h), the
    expected ExeFS hashes, the game version and the default game paths (host.h)."""
    out = [f'// Module glue written by tools/build_game.py; do not edit. {GENERATED_MARKER}',
           '#include "dispatch.h"', '#include "host.h"', '#include "layout.h"', '']
    for name, base, size in layout:
        u = name.upper()
        out.append(f'_Static_assert(MOD_{u}_BASE == {base:#x}ull && MOD_{u}_SIZE == {size:#x}ull, '
                   f'"layout.h and layout.json disagree on {name}");')
    out += ['', 'const RtLayoutEntry rt_layout[] = {']
    out += [f'    {{"{n}", MOD_{n.upper()}_BASE, MOD_{n.upper()}_SIZE}},' for n, _, _ in layout]
    out += ['};', f'const int rt_layout_count = {len(layout)};', '']
    for name in linked:
        out += [f'extern const uint64_t {name}_fn_addrs[];', f'extern const GuestFn {name}_fn_ptrs[];',
                f'extern const uint64_t {name}_fn_count;']
    out += ['', 'int rt_linked_modules(RtModule *out, int max) {', '    int n = 0;']
    for name in linked:
        u = name.upper()
        out.append(f'    if (n < max) out[n++] = (RtModule){{"{name}", MOD_{u}_BASE, MOD_{u}_SIZE, {name}_fn_addrs, '
                   f'{name}_fn_ptrs, {name}_fn_count}};')
    out += ['    return n;', '}', '']
    out += ['const RtFileHash rt_exefs_sha256[] = {']
    out += [f'    {{{c_string(f)}, {c_string(h)}}},' for f, h in sorted(hashes.items())]
    out += ['};', f'const int rt_exefs_sha256_count = {len(hashes)};', '']
    for name in ('exefs', 'romfs', 'profile'):
        out.append(f'const char rt_default_{name}[] = {c_string(paths[name].as_posix())};')
    number = int(DISPLAY_VERSIONS[version].replace('.', '')) if version in DISPLAY_VERSIONS else 0   # 1.1.1 -> 111
    out += [f'const int rt_game_version = {number};', f'const char rt_port_version[] = {c_string(port_version())};', '']
    return '\n'.join(out)


def port_version():
    """VERSION at the root: the repository's semver, or in a release folder the release string (make_release.py
    version)."""
    path = ROOT / 'VERSION'
    return path.read_text(encoding='utf-8').strip() if path.is_file() else 'unknown'


def ninja_file(gen, out, mods, pool_depth, skip=(), prebuilt=False, runtime_lib=None, make_lib=False):
    """build.ninja text, and the object lists per module. With prebuilt the generated objects already exist in
    out/obj and are link inputs without a compile edge. runtime_lib: a prebuilt library replaces the runtime objects
    except INSTALL_SIDE. make_lib: build only that library, from the runtime objects except INSTALL_SIDE."""
    inc = f'-I {cmd_path(RUNTIME)} -I {cmd_path(gen)}'
    vma = f'{cmd_path(DEPS / "include")} -I {cmd_path(DEPS / "include" / "vma")}' if DEPS else cmd_path(VMA_INCLUDE)
    imgui = cmd_path(RUNTIME / 'third_party' / 'imgui')
    rt_inc = f'{inc} -I {cmd_path(out)} -I {vma} -I {imgui}'   # romfs_config.h, vk_mem_alloc.h (+ deps headers), imgui.h
    # llvm-mingw links against its libc++; MSYS2 UCRT64 clang (installer/layout.py's dev fallback, with the MSYS2 tree
    # as --deps) against libstdc++: by the toolchain's libraries, not by whether --deps was given.
    llvm_libs = DEPS and TOOLCHAIN_BIN and (TOOLCHAIN_BIN.parent / 'x86_64-w64-mingw32' / 'lib' / 'libc++.a').is_file()
    libs = ([f'-L {cmd_path(DEPS / "lib")}'] if DEPS else []) + (LLVM_LINK_LIBS if llvm_libs else LINK_LIBS)
    lines = [
        '# Written by tools/build_game.py; do not edit.',
        'ninja_required_version = 1.10',
        f'builddir = {ninja_path(out)}',
        f'cc = {cmd_arg(CLANG.as_posix())}',
        f'genflags = {" ".join(GEN_FLAGS)} {inc}',
        f'rtflags = {" ".join(RT_FLAGS)} {rt_inc}',
        f'ldflags = {" ".join(LINK_FLAGS)}',
        f'libs = {" ".join(libs)}',
        '',
        'pool gen',
        f'  depth = {pool_depth}',
        '',
        'rule cc_gen',
        '  command = $cc $genflags -MD -MF $out.d -c $in -o $out',
        '  deps = gcc',
        '  depfile = $out.d',
        '  description = CC $in',
        '',
        'rule cc_rt',
        '  command = $cc $rtflags -MD -MF $out.d -c $in -o $out',
        '  deps = gcc',
        '  depfile = $out.d',
        '  description = CC $in',
        '',
        'rule rc',
        f'  command = {cmd_arg(WINDRES.as_posix())} -I {cmd_path(RUNTIME)} -O coff $in -o $out',
        '  description = RC $in',
        '',
        'rule link',
        '  command = $cc $ldflags @$out.rsp $libs -o $out',
        '  rspfile = $out.rsp',
        '  rspfile_content = $in_newline',
        '  description = LINK $out',
        '',
    ]
    objs = {}
    for mod, files in mods.items():
        objs[mod] = []
        for f in files:
            o = out / 'obj' / mod / (f.stem + '.o')
            if not prebuilt:
                lines.append(f'build {ninja_path(o)}: cc_gen {ninja_path(f)}')
                lines.append('  pool = gen')
            objs[mod].append(o)
    rt = []
    sources = runtime_sources(skip)
    if make_lib:
        sources = [p for p in sources if p.name not in INSTALL_SIDE]
    elif runtime_lib:
        sources = [p for p in sources if p.name in INSTALL_SIDE] + [out / 'modules.c']
    else:
        sources += [out / 'modules.c']
    for src in sources:
        o = runtime_object(out, src)
        if src.suffix == '.rc':   # windres writes no depfile: the files oot.rc embeds are listed by hand
            lines.append(f'build {ninja_path(o)}: rc {ninja_path(src)} | {ninja_path(RUNTIME / "oot.manifest")}')
        else:
            lines.append(f'build {ninja_path(o)}: cc_rt {ninja_path(src)}')
        rt.append(o)
    objs['runtime'] = rt
    every = [o for v in objs.values() for o in v]
    exe = out / EXE_NAME
    if make_lib:
        exe = out / LIB_NAME
        lines += ['', 'rule ar', f'  command = {cmd_arg(AR.as_posix())} rcs $out $in', '  description = AR $out', '', f'build {ninja_path(exe)}: ar ' + ' '.join(ninja_path(o) for o in every),
                  'build objects: phony ' + ' '.join(ninja_path(o) for o in every), f'default {ninja_path(exe)}', '']
        return '\n'.join(lines), objs, exe
    rule = 'link'
    if runtime_lib:
        # Whole archive: the runtime's weak defaults (nvn_procs.c, nvn_api_core.c, vk_sync.c) would otherwise satisfy
        # references before the linker pulls the members with the strong definitions, which would stay out of oot.exe.
        rule = 'link_lib'
        lines += ['', 'rule link_lib',
                  f'  command = $cc $ldflags @$out.rsp -Wl,--whole-archive {cmd_path(runtime_lib)} -Wl,--no-whole-archive $libs -o $out',
                  '  rspfile = $out.rsp', '  rspfile_content = $in_newline', '  description = LINK $out']
    lines += ['', f'build {ninja_path(exe)}: {rule} ' + ' '.join(ninja_path(o) for o in every) +
              (f' | {ninja_path(runtime_lib)}' if runtime_lib else ''),
              'build objects: phony ' + ' '.join(ninja_path(o) for o in every), f'default {ninja_path(exe)}', '']
    return '\n'.join(lines), objs, exe


def main_objects(gen, main_out, rt_dir, mods):
    """({relative object path: the main build's object}, '') when the main build's generated objects can stand in for
    ours, else (None, why). That holds when the main build.ninja compiles the same sources with the same flags, its
    objects are up to date, and every header they include from runtime/ is byte-identical in rt_dir."""
    main_ninja = main_out / 'build.ninja'
    if not main_ninja.is_file():
        return None, f'{flag_path(main_ninja)} not found'
    if gen != (ROOT / 'generated' / main_out.name).resolve():
        return None, 'generated C directory is not the main one'
    lines = main_ninja.read_text(encoding='utf-8').splitlines()
    sources = dict(l[6:].split(': cc_gen ') for l in lines if l.startswith('build ') and ': cc_gen ' in l)
    flags = next((l for l in lines if l.startswith('genflags = ')), '')
    if not flags.startswith('genflags = ' + ' '.join(GEN_FLAGS) + ' -I '):
        return None, 'main build uses other compile flags'
    pairs = {}
    for mod, files in mods.items():
        for f in files:
            o = Path('obj') / mod / (f.stem + '.o')
            theirs = main_out / o
            if sources.get(ninja_path(theirs)) != ninja_path(f):
                return None, f'main build has no edge for {f.name}'
            if not theirs.is_file():
                return None, f'main object {flag_path(theirs)} is missing'
            pairs[o] = theirs
    # Ninja runs in the main checkout: the main build.ninja is relative to it, and the worktree's runtime/ files carry newer
    # mtimes that would make every object look stale. Header equality is checked byte for byte below.
    env = dict(os.environ, PATH=str(UCRT) + os.pathsep + os.environ.get('PATH', ''))
    targets = [ninja_path(t) for t in pairs.values()]
    for i in range(0, len(targets), 200):
        r = subprocess.run([str(NINJA), '-f', str(main_ninja), '-n', '-d', 'explain'] + targets[i:i + 200], cwd=MAIN_ROOT,
                           env=env, capture_output=True, text=True)
        if r.returncode or 'no work to do' not in r.stdout:
            return None, f'main generated objects are stale ({flag_path(main_out)} needs a build)'
    r = subprocess.run([str(NINJA), '-f', str(main_ninja), '-t', 'deps'], cwd=MAIN_ROOT, env=env, capture_output=True,
                       text=True)
    ours = {flag_path(t) for t in pairs.values()}
    headers, cur = set(), False
    for l in r.stdout.splitlines():
        if l.startswith(' '):
            if cur:
                headers.add(l.strip())
        else:
            cur = l.split(': #deps')[0] in ours
    main_rt = MAIN_ROOT / 'runtime'
    for h in sorted(headers):
        p = Path(h) if Path(h).is_absolute() else MAIN_ROOT / h
        try:
            rel = p.resolve().relative_to(main_rt)
        except ValueError:
            continue
        mine = rt_dir / rel
        if not mine.is_file() or mine.read_bytes() != p.read_bytes():
            return None, f'header {rel.as_posix()} differs in {flag_path(rt_dir)}'
    return pairs, ''


def link_objects(out, pairs):
    """Hardlink the main build's objects into out. Clang and Ninja replace an object by renaming a new file over it,
    so a link keeps the old bytes after a main rebuild. Copies where linking fails."""
    for o, theirs in pairs.items():
        dst = out / o
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if os.path.samefile(dst, theirs):
                continue
            dst.unlink()
        try:
            os.link(theirs, dst)
        except OSError:
            shutil.copy2(theirs, dst)


def unlink_shared(out, mods):
    """Drop generated objects still hardlinked to another build, so the next compile writes a new file."""
    for mod, files in mods.items():
        for f in files:
            o = out / 'obj' / mod / (f.stem + '.o')
            if o.exists() and o.stat().st_nlink > 1:
                o.unlink()


MANIFEST = 'objects.sha256'


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb', buffering=0) as f:
        buf = bytearray(1 << 20)
        while n := f.readinto(buf):
            h.update(memoryview(buf)[:n])
    return h.hexdigest()


def read_manifest(path):
    """{object path below obj/: (sha256, size, mtime_ns)}; empty when the file is missing."""
    man = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            sha, size, mtime, rel = line.split(' ', 3)
            man[rel] = (sha, int(size), int(mtime))
    return man


def write_manifest(path, man):
    write_if_changed(path, ''.join(f'{sha} {size} {mtime} {rel}\n' for rel, (sha, size, mtime) in sorted(man.items())))


def check_objects(out, objects, man, record):
    """Compare objects with the manifest man. An object whose size and mtime match its entry is hashed and must equal
    it; any other object was just (re)built, so with record its hash enters man. Returns (verified, bad, unrecorded):
    the objects with a wrong hash, and with record False the ones without a current entry."""
    verified, bad, unrecorded = 0, [], []
    for o in objects:
        rel = o.relative_to(out / 'obj').as_posix()
        st = o.stat()
        e = man.get(rel)
        if e and e[1] == st.st_size and e[2] == st.st_mtime_ns:
            if file_sha256(o) == e[0]:
                verified += 1
            else:
                bad.append(o)
        elif record:
            man[rel] = (file_sha256(o), st.st_size, st.st_mtime_ns)
            verified += 1
        else:
            unrecorded.append(o)
    return verified, bad, unrecorded


def ninja_build(base, targets, env):
    for i in range(0, len(targets), 200):
        code = subprocess.run(base + targets[i:i + 200], cwd=ROOT, env=env).returncode
        if code:
            sys.exit(code)


def verify_objects(out, objs, borrowed, main_out, pairs, base, env):
    """Verify every object before the link; see the module docstring. borrowed: the hardlinked generated objects of
    main_out (checked against its manifest, never recorded); the others belong to out."""
    t0 = time.monotonic()
    owned = [o for mod, v in objs.items() for o in v if not (borrowed and mod != 'runtime')]
    shared = [o for mod, v in objs.items() for o in v if borrowed and mod != 'runtime']
    man_path = out / 'obj' / MANIFEST
    man = read_manifest(man_path)
    verified, bad, _ = check_objects(out, owned, man, True)
    recompiled = 0
    if bad:
        for o in bad:
            print(f'build_game: object {flag_path(o)} does not match its hash; deleting it to recompile')
            o.unlink()
        ninja_build(base, [ninja_path(o) for o in bad], env)
        recompiled = len(bad)
        verified += check_objects(out, bad, man, True)[0]
    write_manifest(man_path, man)
    if shared:
        main_man = read_manifest(main_out / 'obj' / MANIFEST)
        theirs = [pairs[o.relative_to(out)] for o in shared]
        v, bad, unrecorded = check_objects(main_out, theirs, main_man, False)
        verified += v
        if bad:
            for b in bad:
                print(f'build_game: shared object {flag_path(b)} does not match its hash in '
                      f'{flag_path(main_out / "obj" / MANIFEST)}', file=sys.stderr)
            sys.exit(f'build_game: run tools/build_game.py for {flag_path(main_out)} (it recompiles them), then rebuild '
                     f'{flag_path(out)}')
        if unrecorded:
            print(f'build_game: warning: {len(unrecorded)} shared objects have no current hash in '
                  f'{flag_path(main_out / "obj" / MANIFEST)}; build {flag_path(main_out)} once to record them')
    print(f'objects verified: {verified}, recompiled after a hash mismatch: {recompiled} ({time.monotonic() - t0:.1f} s)')


if sys.platform == 'win32':
    # ---- Windows job object: peak memory of a process tree
    class IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ('r_ops', 'w_ops', 'o_ops', 'r_bytes', 'w_bytes', 'o_bytes')]


    class BasicLimits(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                    ('LimitFlags', wt.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wt.DWORD),
                    ('Affinity', ctypes.c_size_t), ('PriorityClass', wt.DWORD), ('SchedulingClass', wt.DWORD)]


    class ExtendedLimits(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', BasicLimits), ('IoInfo', IoCounters),
                    ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    CREATE_SUSPENDED = 0x4


    def run_measured_win(cmd, env):
        """Run cmd in its own job object. Returns (exit code, seconds, peak bytes of one process, peak bytes of all)."""
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.CreateJobObjectW.restype = wt.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
        k32.QueryInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p]
        k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
        k32.CloseHandle.argtypes = [wt.HANDLE]
        ntdll = ctypes.WinDLL('ntdll')
        ntdll.NtResumeProcess.argtypes = [wt.HANDLE]

        job = k32.CreateJobObjectW(None, None)
        info = ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE   # Ctrl+C stops every compile
        k32.SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info))
        t0 = time.monotonic()
        # Suspended until it is in the job, so the compilers it starts are counted from the first one.
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, creationflags=CREATE_SUSPENDED)
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):
            print(f'build_game: cannot track memory (error {ctypes.get_last_error()})', file=sys.stderr)
        ntdll.NtResumeProcess(int(proc._handle))
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            k32.CloseHandle(job)
            raise
        dt = time.monotonic() - t0
        k32.QueryInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                      ctypes.sizeof(info), None)
        k32.CloseHandle(job)
        return code, dt, info.PeakProcessMemoryUsed, info.PeakJobMemoryUsed


def run_measured_posix(cmd, env):
    """Linux: no job object, so wait4 gives the peak resident size of the largest process in ninja's tree (ninja and
    the compilers it waited for). That is one process, not the sum; the sum is not reported here."""
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    return proc.returncode, time.monotonic() - t0, usage.ru_maxrss * 1024, None   # ru_maxrss is in KiB on Linux


def run_measured(cmd, env):
    """Run cmd. Returns (exit code, seconds, peak bytes of one process, peak bytes of all processes or None)."""
    if sys.platform == 'win32':
        return run_measured_win(cmd, env)
    return run_measured_posix(cmd, env)


def total_ram():
    if sys.platform != 'win32':
        return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    class MemStatus(ctypes.Structure):
        _fields_ = [('dwLength', wt.DWORD), ('dwMemoryLoad', wt.DWORD)] + \
                   [(n, ctypes.c_ulonglong) for n in ('total', 'avail', 'tpf', 'apf', 'tv', 'av', 'aev')]
    s = MemStatus()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.total


def pe_sections(path):
    """{section name: virtual size} and SizeOfImage of a PE32+ file."""
    data = path.read_bytes()[:4096]
    pe = int.from_bytes(data[0x3C:0x40], 'little')
    nsec = int.from_bytes(data[pe + 6:pe + 8], 'little')
    opt = pe + 24
    image = int.from_bytes(data[opt + 56:opt + 60], 'little')
    table = opt + int.from_bytes(data[pe + 20:pe + 22], 'little')
    out = {}
    for i in range(nsec):
        s = data[table + 40 * i:table + 40 * (i + 1)]
        out[s[:8].rstrip(b'\0').decode(errors='replace')] = int.from_bytes(s[8:12], 'little')
    return out, image


def mib(n):
    return f'{n / (1 << 20):,.1f} MiB'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--version', default=DEFAULT_VERSION, help=f'game version (default {DEFAULT_VERSION}, {DISPLAY_VERSIONS[DEFAULT_VERSION]})')
    ap.add_argument('--gen', type=Path, help='generated C directory (default generated/<version>)')
    ap.add_argument('--out', type=Path, help='build directory (default build/<version>)')
    ap.add_argument('--modules', help='comma-separated subset of the layout modules (default all)')
    ap.add_argument('--jobs', type=int, help='parallel compiles (default: by memory budget and CPU count)')
    ap.add_argument('--compile-only', action='store_true', help='compile, do not link')
    ap.add_argument('--runtime', type=Path, help='runtime source directory (default runtime)')
    ap.add_argument('--full-build', action='store_true',
                    help="compile the generated C even where build/<version>'s objects could be reused")
    ap.add_argument('--skip-runtime', default='',
                    help='comma-separated runtime/*.c names to leave out (the link needs another definition of what they '
                         'define)')
    ap.add_argument('--exefs', type=Path,
                    help="oot.exe's default ExeFS directory (default build/extracted/<version>/exefs)")
    ap.add_argument('--romfs', type=Path, help="oot.exe's default RomFS directory with romfs.bin and index.json "
                                               '(default build/extracted/<version>/romfs)')
    ap.add_argument('--profile', type=Path, help="oot.exe's default profile directory (default build/profile)")
    ap.add_argument('--toolchain', type=Path, help='llvm-mingw directory (with bin/clang.exe) to build with instead of '
                                                   'MSYS2 UCRT64 clang')
    ap.add_argument('--ninja', type=Path, help='ninja.exe to build with (default: MSYS2 UCRT64, else bin/ninja.exe of the '
                                               'release folder)')
    ap.add_argument('--deps', type=Path, help=f'with --toolchain: include/, lib/ and bin/ of Vulkan, SDL3, Opus, libvpx '
                                              f'and VMA (default {DEFAULT_DEPS.relative_to(ROOT).as_posix()})')
    ap.add_argument('--make-runtime-lib', action='store_true',
                    help=f'build only <out>/{LIB_NAME} from runtime/ (no generated C, no oot.exe)')
    ap.add_argument('--runtime-lib', type=Path, help=f'link against this {LIB_NAME} instead of compiling runtime/ '
                                                    f'({", ".join(sorted(INSTALL_SIDE))} and modules.c still compile)')
    ap.add_argument('--expected', type=Path,
                    help='JSON with {"exefs_sha256": {name: sha256}} (default: the files of '
                         'build/extracted/<version>/extraction-manifest.json)')
    args = ap.parse_args()

    gen = (args.gen or ROOT / 'generated' / args.version).resolve()
    out = (args.out or ROOT / 'build' / args.version).resolve()
    extracted = ROOT / 'build' / 'extracted' / args.version
    paths = {'exefs': (args.exefs or extracted / 'exefs').resolve(),
             'romfs': (args.romfs or extracted / 'romfs').resolve(),
             'profile': (args.profile or ROOT / 'build' / 'profile').resolve()}
    expected = args.expected
    if expected:
        hashes = json.loads(Path(expected).read_text())['exefs_sha256']
    else:
        expected = extracted / 'extraction-manifest.json'
        if not expected.is_file():
            sys.exit(f'{expected} not found: run tools/extract_nup.py for {args.version}, or pass --expected')
        hashes = {f['name']: f['sha256'] for f in json.loads(expected.read_text())['files']}
    global CLANG, AR, WINDRES, NINJA, TOOLCHAIN_BIN, DEPS
    if args.ninja:
        NINJA = args.ninja.resolve()
    elif not NINJA.is_file() and (ROOT / 'bin' / 'ninja.exe').is_file():
        NINJA = ROOT / 'bin' / 'ninja.exe'   # the release folder (tools/make_release.py)
    dlls = SDL3_DLLS
    if args.toolchain:
        TOOLCHAIN_BIN = (args.toolchain / 'bin').resolve()
        CLANG = TOOLCHAIN_BIN / 'clang.exe'
        AR = TOOLCHAIN_BIN / 'llvm-ar.exe'
        WINDRES = TOOLCHAIN_BIN / 'llvm-windres.exe'
        DEPS = (args.deps or DEFAULT_DEPS).resolve()
        dlls = sorted((DEPS / 'bin').glob('*.dll'))
    for tool in (CLANG, NINJA):
        if not tool.is_file():
            sys.exit(f'{tool} not found: install MSYS2 UCRT64 clang, lld and ninja on Windows, or clang and ninja on Linux (docs/building.md)')
    if not (gen / 'layout.json').is_file():
        sys.exit(f'{gen / "layout.json"} not found: run the generator first')

    layout = read_layout(gen)
    names = [n for n, _, _ in layout]
    wanted = args.modules.split(',') if args.modules else names
    unknown = [m for m in wanted if m not in names]
    if unknown:
        sys.exit(f'unknown modules {unknown}; the layout has {names}')
    linked = [m for m in names if m in wanted]
    mods = {m: chunk_files(gen, m) for m in linked}

    cpus = os.cpu_count() or 1
    budget = min(MEM_BUDGET, int(total_ram() * 0.8))
    jobs = args.jobs or max(1, min(cpus, budget // JOB_MEM))
    missing = [n for n in names if n not in hashes]
    if missing:
        sys.exit(f'{expected}: no exefs_sha256 for {missing}')
    write_if_changed(out / 'modules.c', module_glue(layout, linked, hashes, paths, args.version))
    if (paths['romfs'] / 'index.json').is_file():
        count, size, dt = romfs_index.write(paths['romfs'], out)
        print(f'romfs index: {count} files, {mib(size)}, {dt:.1f} s')
    else:
        romfs_index.write_placeholder(out)
        print(f'build_game: warning: no {paths["romfs"] / "index.json"}; oot.exe will have no RomFS', file=sys.stderr)
    global RUNTIME
    RUNTIME = (args.runtime or RUNTIME).resolve()
    stale = stale_shaders(RUNTIME)
    if stale:
        sys.exit(f'{stale[0]}: its SPIR-V header no longer matches; run python tools/build_shaders.py')
    skip = {s for s in args.skip_runtime.split(',') if s}
    main_out = (ROOT / 'build' / args.version).resolve()
    pairs = None
    if args.make_runtime_lib:
        mods, linked, why = {}, [], ''
    elif args.full_build:
        why = '--full-build'
    elif args.toolchain:
        why = f'--toolchain {flag_path(args.toolchain)}'
    elif out == main_out:
        why = ''
    else:
        pairs, why = main_objects(gen, main_out, RUNTIME, mods)
    if pairs is not None:
        link_objects(out, pairs)
        print(f'build_game: reusing {len(pairs)} generated objects of {flag_path(main_out)} (hardlinks)')
    else:
        if out != main_out:   # forks hold hardlinks to the main build's objects; the main build keeps its own
            unlink_shared(out, mods)
        if why:
            print(f'build_game: compiling all generated C: {why}')
    runtime_lib = args.runtime_lib.resolve() if args.runtime_lib else None
    if runtime_lib and not runtime_lib.is_file():
        sys.exit(f'{runtime_lib} not found: build it with --make-runtime-lib')
    text, objs, exe = ninja_file(gen, out, mods, jobs, skip, pairs is not None, runtime_lib,
                                 args.make_runtime_lib)
    write_if_changed(out / 'build.ninja', text)

    env = dict(os.environ, PATH=os.pathsep.join([str(d) for d in (TOOLCHAIN_BIN, UCRT) if d] + [os.environ.get('PATH', '')]))
    base = [str(NINJA), '-f', str(out / 'build.ninja'), '-j', str(jobs)]
    nfiles = sum(len(v) for v in mods.values())
    print(f'build_game: {nfiles} generated files in {len(linked)} modules ({", ".join(linked)}), {jobs} jobs', flush=True)

    code, dt, peak_one, peak_all = run_measured(base + ['objects'], env)
    print(f'compile: {dt:.1f} s, peak {mib(peak_one)} per process' + (f', {mib(peak_all)} all processes' if peak_all else ''))
    if code:
        sys.exit(code)
    for mod, files in objs.items():
        size = sum(o.stat().st_size for o in files)
        print(f'  {mod:8} {len(files):5} objects {mib(size):>14}')
    verify_objects(out, objs, pairs is not None, main_out, pairs, base, env)
    if args.compile_only:
        return
    if args.make_runtime_lib:
        exe.unlink(missing_ok=True)   # ar rcs would keep members of deleted sources
    code, dt, peak_one, _ = run_measured(base + [flag_path(exe)], env)
    print(f'link: {dt:.1f} s, peak {mib(peak_one)}')
    if code:
        print(f'build_game: link failed, {flag_path(exe)} is the old one (a running oot.exe from this dir locks it)',
              file=sys.stderr)
        sys.exit(code)
    if args.make_runtime_lib:
        print(f'{below_root(exe)}: {mib(exe.stat().st_size)}')
        return
    for dll in dlls:   # dynamic SDL3 (zlib) and its libiconv (LGPL, unmodified DLL), found next to oot.exe
        shutil.copy2(dll, exe.parent / dll.name)
    if sys.platform != 'win32':
        print(f'{below_root(exe)}: file {mib(exe.stat().st_size)}')
        return
    sections, image = pe_sections(exe)
    print(f'{below_root(exe)}: file {mib(exe.stat().st_size)}, '
          f'image {mib(image)}; ' + ', '.join(f'{k} {mib(v)}' for k, v in sections.items() if k.startswith('.') and v))
    if image > 3 * GiB // 2:
        print('build_game: warning: image over 1.5 GiB; RIP-relative code reaches only 2 GiB', file=sys.stderr)


if __name__ == '__main__':
    main()
