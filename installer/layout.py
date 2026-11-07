# From metroid-prime-4-recomp installer/layout.py @ 586e91e1b6f38c9e512e39f2d47a046fdf273cf0 (MIT, Copyright (c) 2026 Jeff Aigner), copied with mp4 renamed to oot.
# From fe3h-pc-recomp installer/layout.py @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c) 2026 Jeff Aigner), renamed to mp4; MP4 changes are noted where made.
# From totk-pc-recomp installer/layout.py @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), renamed to mp4; MP4 changes are noted where made.
"""Where the installer finds what it runs: the one place that maps the release folder and a repository checkout.

Release (docs/installer.md "Release layout"), root = the unpacked folder:

    python/python.exe       runs every tool (numpy, lz4, cryptography, compression.zstd); its ._pth puts .., ../tools
                            and ../recomp on sys.path, so `python -m installer` imports from the release folder
    recomp/, tools/         the lifter and the scripts; recomp/patches-<v>.txt, tools/data/expected-v<version>.json for
                            each supported update (SUPPORTED)
    llvm/                   llvm-mingw (bin/clang.exe)
    runtime/                libootrt.a, the headers, install-side sources (romfs.c, patches.c, ov_zstd.c) and oot.ico; runtime/deps/ holds
                            include/ + lib/ of Vulkan, SDL3, Opus, libvpx and VMA, and openh264's headers
                            (build_game.py --deps)
    bin/                    oot_xlate.exe (+ oot_xlate.id), hactool.exe, ninja.exe, SDL3.dll, libopus-0.dll
    data/shader-list.bin    the shader variants play met (tools/prepare_shaders.py; OOT change: shipped with the release,
                            not kept in the repository)
    oot.ini.example
    VERSION                 the release version, one line
    temp/                   the default temp folder (created)

Dev mode, root = a repository checkout (no python/python.exe): build/py314-embed/py/python.exe (the embeddable
Python the release ships, with the same packages), else this Python; toolchain/llvm-mingw-*/ and toolchain/deps-mingw;
runtime/ as the sources with build/install-rtlib/libootrt.a (tools/build_game.py --make-runtime-lib --out
build/install-rtlib); MSYS2 UCRT64 ninja.exe; toolchain/hactool; build/xlate/oot_xlate.exe; build/shader-list.bin when
there is one; build/install-temp. The version is VERSION with -dev and the commit: 0.1.0-dev+973c626.
"""
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# MSYS2: OOT_MSYS2 if set, else a copy inside the repository (toolchain/msys64, ignored), else C:\msys64.
MSYS2_ROOT = Path(os.environ.get('OOT_MSYS2') or next(
    (d for d in [Path(__file__).resolve().parents[1] / 'toolchain' / 'msys64'] if d.is_dir()),
    Path(r'C:\msys64')))

EXE = '.exe' if os.name == 'nt' else ''
UCRT = (MSYS2_ROOT / r'ucrt64\bin')
# Updates the installer builds; the check identifies the player's by hash. OOT: 1.1.0, the last Switch 1 update.
SUPPORTED = ('1.1.0',)
# OOT names a version by its CNMT number (tools/extract_exefs.py): the folder of build_game.py and the patch table.
VERSION_TAGS = {'1.0.0': 'v0', '1.1.0': 'v131072'}


def version_tag(version):
    """'1.1.0' -> 'v131072': the version name of build_game.py and the patch tables."""
    return VERSION_TAGS[version]


@dataclass(frozen=True)
class Layout:
    root: Path
    dev: bool
    release: str
    python: Path          # runs the tools
    tools: Path
    recomp: Path
    llvm: Path            # llvm-mingw tree
    deps: Path            # include/ and lib/ for the link
    runtime: Path         # runtime headers and install-side sources
    runtime_lib: Path     # libootrt.a
    ninja: Path
    hactool: Path
    xlate: Path           # oot_xlate.exe; a oot_xlate.id beside it is copied too
    dlls: tuple = field(default_factory=tuple)
    ini_example: Path | None = None
    temp: Path | None = None
    shader_list: Path | None = None   # OOT: None when there is none (dev); the shaders are then translated in play

    @property
    def clang(self):
        return self.llvm / 'bin' / ('clang' + EXE)

    def patches(self, version):
        """The code patch table for recomp/gen.py."""
        return self.recomp / f'patches-{version_tag(version)}.txt'

    def expected(self, version):
        """The expected hashes of the update."""
        return self.tools / 'data' / f'expected-v{version}.json'

    def missing(self):
        """Paths the steps need that do not exist."""
        need = [self.python, self.tools / 'extract.py', self.tools / 'extract_exefs.py', self.tools / 'extract_romfs.py',
                self.tools / 'build_game.py', self.recomp / 'gen.py',
                self.clang,
                self.deps / 'include', self.deps / 'lib', self.runtime, self.runtime_lib, self.ninja, self.hactool,
                self.xlate, *self.dlls, *(f(v) for v in SUPPORTED for f in (self.patches, self.expected)),
                self.expected('1.0.0'), self.tools / 'prepare_shaders.py', self.tools / 'shader_inventory.py',
                self.tools / 'pack_shader_cache.py']
        if self.shader_list and not self.dev:   # a release always carries its list
            need.append(self.shader_list)
        return [str(p) for p in need if not p.exists()]


def is_release(root):
    return (root / 'python' / ('python' + EXE)).is_file()


def release_layout(root):
    version = root / 'VERSION'
    return Layout(
        root=root, dev=False,
        release=version.read_text(encoding='utf-8').strip() if version.is_file() else 'unknown',
        python=root / 'python' / ('python' + EXE),
        tools=root / 'tools', recomp=root / 'recomp',
        llvm=root / 'llvm', deps=root / 'runtime' / 'deps', runtime=root / 'runtime',
        runtime_lib=root / 'runtime' / 'libootrt.a',
        ninja=root / 'bin' / ('ninja' + EXE), hactool=root / 'bin' / ('hactool' + EXE),
        xlate=root / 'bin' / ('oot_xlate' + EXE),
        dlls=(root / 'bin' / 'SDL3.dll', root / 'bin' / 'libopus-0.dll'),
        ini_example=root / 'oot.ini.example', temp=root / 'temp', shader_list=root / 'data' / 'shader-list.bin')


def dev_layout(root):
    commit = subprocess.run(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                            text=True).stdout.strip()
    embed = root / 'build' / 'py314-embed' / 'py' / ('python' + EXE)
    trees = sorted(p.parent.parent for p in (root / 'toolchain').glob('llvm-mingw-*/llvm-mingw-*/bin/clang' + EXE))
    deps = root / 'toolchain' / 'deps-mingw'
    # OOT: without the release toolchain, the MSYS2 UCRT64 tree the usual build uses (tools/build_game.py): its
    # bin/clang.exe, include/ and lib/, and the DLLs the exe loads from its bin/.
    if not trees and (UCRT / ('clang' + EXE)).is_file():
        trees = [UCRT.parent]
    if not deps.is_dir() and (UCRT.parent / 'include').is_dir():
        deps = UCRT.parent
    version_file = root / 'VERSION'
    semver = version_file.read_text(encoding='utf-8').strip() if version_file.is_file() else '0.0.0'
    return Layout(
        root=root, dev=True, release=f'{semver}-dev+{commit or "unknown"}',
        python=embed if embed.is_file() else Path(sys.executable),
        tools=root / 'tools', recomp=root / 'recomp',
        llvm=trees[-1] if trees else root / 'toolchain' / 'llvm-mingw-missing', deps=deps,
        runtime=root / 'runtime', runtime_lib=root / 'build' / 'install-rtlib' / 'libootrt.a',
        ninja=UCRT / ('ninja' + EXE), hactool=root / 'toolchain' / 'hactool' / ('hactool' + EXE),
        xlate=root / 'build' / 'xlate' / ('oot_xlate' + EXE),
        dlls=tuple(deps / 'bin' / n for n in ('SDL3.dll', 'libopus-0.dll', 'libiconv-2.dll') if (deps / 'bin' / n).is_file())
        if deps == UCRT.parent else tuple(sorted((deps / 'bin').glob('*.dll'))),
        ini_example=root / 'oot.ini.example', temp=root / 'build' / 'install-temp',
        shader_list=(root / 'build' / 'shader-list.bin') if (root / 'build' / 'shader-list.bin').is_file() else None)


def find(root=None):
    """The layout of root (default: the folder above this package)."""
    root = Path(root or Path(__file__).resolve().parent.parent).resolve()
    return release_layout(root) if is_release(root) else dev_layout(root)
