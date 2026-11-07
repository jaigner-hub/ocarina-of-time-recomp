"""Build nso_unzbic: toolchain/zbic/nso_unzbic[.exe] (upstream zstd's decoder at a pinned commit + tools/zbic/).

Switch 2 NSOs store their segments as ZBIC, zstd with its own FSE table headers (tools/zbic/zbic_ncount.c, which
describes the format). This fetches facebook/zstd at the v1.5.7 release commit into toolchain/src/zstd, resets it
(local changes there are discarded), applies tools/zbic/zbic.patch (with ZSTD_ZBIC defined, the ZBIC frame magic, and
FSE_readNCount_body handing table headers to ZBIC_readNCount), and compiles zstd's decompressor with
tools/zbic/zbic_ncount.c and tools/zbic/nso_unzbic.c: gcc on Linux (WSL), MSYS2 UCRT64 gcc on Windows, as
tools/build_hactool.py does. No GPL code is involved: zstd is BSD-licensed (its LICENSE goes beside the binary), the
ZBIC parts are ours. tools/extract_nup.py runs the result on ZBIC modules.

  python3 tools/build_zbic.py [--src DIR] [--out DIR]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# MSYS2: OOT_MSYS2 if set, else a copy inside the repository (toolchain/msys64, ignored), else C:\msys64.
MSYS2_ROOT = Path(os.environ.get('OOT_MSYS2') or next(
    (d for d in [Path(__file__).resolve().parents[1] / 'toolchain' / 'msys64'] if d.is_dir()),
    Path(r'C:\msys64')))

ROOT = Path(__file__).resolve().parent.parent
URL = 'https://github.com/facebook/zstd.git'
COMMIT = 'f8745da6ff1ad1e7bab384bd1f9d742439278e99'   # v1.5.7
OURS = ROOT / 'tools' / 'zbic'
PATCH = OURS / 'zbic.patch'
WINDOWS = os.name == 'nt'
MSYS_GCC = (MSYS2_ROOT / r'ucrt64\bin\gcc.exe')
EXE = 'nso_unzbic.exe' if WINDOWS else 'nso_unzbic'
ZSTD_SOURCES = ['common/entropy_common.c', 'common/error_private.c', 'common/fse_decompress.c', 'common/zstd_common.c',
                'common/xxhash.c', 'common/debug.c', 'decompress/huf_decompress.c', 'decompress/zstd_ddict.c',
                'decompress/zstd_decompress.c', 'decompress/zstd_decompress_block.c']
# Decoder only, no assembly (the amd64 Huffman loop is a .S file MSYS2 and Linux would build differently), no legacy
# formats, no threads.
DEFINES = ['-DZSTD_ZBIC', '-DZSTD_DISABLE_ASM', '-DZSTD_LEGACY_SUPPORT=0', '-DZSTD_MULTITHREAD=0']


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', type=Path, default=ROOT / 'toolchain' / 'src' / 'zstd', help='zstd source checkout')
    ap.add_argument('--out', type=Path, default=ROOT / 'toolchain' / 'zbic', help='where the binary goes')
    args = ap.parse_args()
    cc = str(MSYS_GCC) if WINDOWS else shutil.which('gcc')
    if WINDOWS and not MSYS_GCC.is_file():
        sys.exit(f'{MSYS_GCC} not found: install MSYS2 with the UCRT64 gcc')
    if not cc:
        sys.exit('gcc not found: install build-essential')
    src, out = args.src.resolve(), args.out.resolve()

    fetch(src)
    git(src, 'apply', '--whitespace=nowarn', str(PATCH))
    lib = src / 'lib'
    out.mkdir(parents=True, exist_ok=True)
    cmd = [cc, '-O2', '-std=gnu11', '-Wall', *DEFINES, f'-I{lib}', f'-I{lib / "common"}', '-o', str(out / EXE),
           str(OURS / 'nso_unzbic.c'), str(OURS / 'zbic_ncount.c'), *(str(lib / s) for s in ZSTD_SOURCES)]
    if WINDOWS:
        cmd.append('-static')
    env = dict(os.environ, PATH=str(MSYS_GCC.parent) + os.pathsep + os.environ.get('PATH', '')) if WINDOWS else None
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    (out / 'build.log').write_text(r.stdout + r.stderr, encoding='utf-8')
    if r.returncode:
        sys.exit(f'compile failed ({r.returncode}); log: {out / "build.log"}')
    shutil.copy2(src / 'LICENSE', out / 'LICENSE.zstd')
    usage = subprocess.run([str(out / EXE)], capture_output=True, text=True)
    if 'usage: nso_unzbic' not in usage.stderr:
        sys.exit(f'{out / EXE} does not start')
    print(f'{out / EXE}: zstd {COMMIT[:7]} + {PATCH.name}')


if __name__ == '__main__':
    main()
