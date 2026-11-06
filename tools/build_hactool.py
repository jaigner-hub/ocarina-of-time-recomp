"""Build hactool: toolchain/hactool/hactool[.exe] (SciresM/hactool at a pinned commit + tools/hactool/*.patch).

Copied unchanged from fe3h-pc-recomp tools/build_hactool.py @ 47ef99126793dbd13e4a475feb910a13ef6639b8 (MIT, Copyright (c)
2026 Jeff Aigner), apart from this header. That file is adapted from totk-pc-recomp tools/build_hactool.py @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT, Copyright (c) 2026
Valiantsin Dzerakh). Changes: it also builds on Linux (WSL) with the system gcc, producing a native `hactool`, so the
extraction tools can run beside the game files on the Linux filesystem; on Windows it builds `hactool.exe` with MSYS2
UCRT64 gcc as the original does.

Fetches the pinned commit into toolchain/src/hactool, resets it (local changes and build output there are discarded),
applies portable.patch (MacNativeGames' changes that are not macOS specific) and unicode.patch (a Windows manifest; the
Makefile skips it elsewhere), and builds; mbedtls comes with hactool's tree. The binary and hactool's LICENSE go to
toolchain/hactool.

  python3 tools/build_hactool.py [--src DIR] [--out DIR]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = 'https://github.com/SciresM/hactool.git'
COMMIT = '1d64a83450e025622f3468c28fc4164dad2c5ef6'
PATCHES = [ROOT / 'tools' / 'hactool' / n for n in ('portable.patch', 'unicode.patch')]
WINDOWS = os.name == 'nt'
MSYS_BASH = Path(r'C:\msys64\usr\bin\bash.exe')
EXE = 'hactool.exe' if WINDOWS else 'hactool'
# Windows links statically so the exe needs only KERNEL32 and the UCRT. On Linux the mbedtls archives are linked in
# anyway and glibc stays dynamic.
CONFIG_MK = ('CC = gcc\nEXEEXT = .exe\nCFLAGS = -O2 -Wall -Wextra -pedantic -std=gnu11\n'
             'LDFLAGS = -lmbedtls -lmbedx509 -lmbedcrypto -static\n') if WINDOWS else \
            ('CC = gcc\nEXEEXT =\nCFLAGS = -O2 -Wall -Wextra -pedantic -std=gnu11 -fPIC\n'
             'LDFLAGS = -lmbedtls -lmbedx509 -lmbedcrypto\n')


def git(src, *args):
    subprocess.run(['git', '-C', str(src), *args], check=True)


def fetch(src):
    """src holds COMMIT with no local changes or untracked files."""
    if not (src / '.git').is_dir():
        src.mkdir(parents=True, exist_ok=True)
        git(src, 'init', '-q')
        git(src, 'remote', 'add', 'origin', URL)
    head = subprocess.run(['git', '-C', str(src), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        git(src, 'fetch', '-q', '--depth', '1', 'origin', COMMIT)
    git(src, 'checkout', '-q', '-f', COMMIT)
    git(src, 'clean', '-q', '-fdx')


def make(src):
    """Runs make in src; returns the completed process."""
    jobs = f'-j{min(4, os.cpu_count() or 4)}'   # at most 4 jobs: the portable laptop has 15 GB of RAM
    if WINDOWS:
        env = dict(os.environ, MSYSTEM='UCRT64', CHERE_INVOKING='1')
        return subprocess.run([str(MSYS_BASH), '-lc', f'make {jobs}'], cwd=src, env=env,
                              capture_output=True, text=True)
    return subprocess.run(['make', jobs], cwd=src, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', type=Path, default=ROOT / 'toolchain' / 'src' / 'hactool', help='source checkout')
    ap.add_argument('--out', type=Path, default=ROOT / 'toolchain' / 'hactool', help='where the binary goes')
    args = ap.parse_args()
    if WINDOWS and not MSYS_BASH.is_file():
        sys.exit(f'{MSYS_BASH} not found: install MSYS2 with the UCRT64 gcc and make')
    if not WINDOWS and not (shutil.which('gcc') and shutil.which('make')):
        sys.exit('gcc and make not found: install build-essential')
    src, out = args.src.resolve(), args.out.resolve()

    fetch(src)
    for p in PATCHES:
        git(src, 'apply', '--whitespace=nowarn', str(p))
    (src / 'config.mk').write_text(CONFIG_MK, encoding='utf-8', newline='\n')

    r = make(src)
    (src / 'build.log').write_text(r.stdout + r.stderr, encoding='utf-8')
    if r.returncode:
        sys.exit(f'make failed ({r.returncode}); log: {src / "build.log"}')

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / EXE, out / EXE)
    shutil.copy2(src / 'LICENSE', out / 'LICENSE')
    banner = subprocess.run([str(out / EXE)], capture_output=True, text=True)
    if 'hactool (c) SciresM' not in banner.stdout + banner.stderr:
        sys.exit(f'{out / EXE} does not start')
    print(f'{out / EXE}: {COMMIT[:7]} + {", ".join(p.name for p in PATCHES)}')


if __name__ == '__main__':
    main()
