#!/usr/bin/env python3
"""How far the Linux port has got, in percent, from the state of the build tree.

Stages (each worth the same share of the total):
  1. tools      the Python tools run on Linux (tools/build_game.py finds its compilers)
  2. runtime    runtime/*.c, *.cpp compile on Linux (an object exists and is newer than its source)
  3. game C     the generated C of main, sdk and subsdk0 compiles
  4. link       build/<version>/oot links
  5. selftest   `oot --selftest` passes (marker file written by whoever ran it, see --mark)
  6. title      the title cinematic shows on a Linux run (marker file, see --mark)

The platform layer line is extra information: how many functions declared in runtime/platform/platform.h
linux.c defines. Usage: linux_port_status.py [--version v131072] [--write docs/LINUX_PORT.md]
       linux_port_status.py --mark selftest|title
"""
import argparse, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))


def runtime_sources():
    skip = {'win32.c', 'rdoc_win.c', 'prof_win.c', 'gfx_window_probe.c'}
    twins = {'swkbd_win.c', 'input_win.c', 'gfx_window.c'}   # replaced by *_sdl.c on Linux
    rt = ROOT / 'runtime'
    pats = ('*.c', '*.cpp', 'platform/*.c', 'ui/*.cpp', 'third_party/imgui/*.cpp', 'third_party/imgui/backends/*.cpp')
    return sorted(p for pat in pats for p in rt.glob(pat)
                  if p.name not in skip and p.name not in twins and not p.stem.endswith('_win') and p.name != 'oot.rc')


def compiled(src, obj):
    return obj.is_file() and obj.stat().st_mtime >= src.stat().st_mtime


def platform_api():
    h = (ROOT / 'runtime/platform/platform.h').read_text(errors='replace')
    c = (ROOT / 'runtime/platform/linux.c').read_text(errors='replace')
    declared = set(re.findall(r'\b(plat_[a-z0-9_]+)\s*\(', re.sub(r'//.*', '', h)))
    inline = set(re.findall(r'static\s+inline[^;{]*?\b(plat_[a-z0-9_]+)\s*\(', h))
    need = declared - inline
    have = {n for n in need if re.search(r'\b' + n + r'\s*\(', c)}
    return len(have), len(need)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='v131072')
    ap.add_argument('--write')
    ap.add_argument('--mark', choices=('selftest', 'title'))
    a = ap.parse_args()
    out = ROOT / 'build' / a.version
    if a.mark:
        (out / f'.linux-{a.mark}-ok').write_text('ok\n')
        print('marked', a.mark)
        return
    stages = []
    bg = (ROOT / 'tools/build_game.py').read_text(errors='replace')
    stages.append(('tools run on Linux', 100 if 'find_tool' in bg and shutil.which('clang') else 0, 'build_game.py finds clang, ninja, ar on PATH'))
    srcs = runtime_sources()
    done = sum(compiled(s, out / 'obj' / 'runtime' / s.relative_to(ROOT / 'runtime').with_suffix('.o')) for s in srcs)
    stages.append(('runtime compiles', 100 * done / max(len(srcs), 1), f'{done} of {len(srcs)} runtime files'))
    gen_total = gen_done = 0
    for mod in ('main', 'sdk', 'subsdk0'):
        d = ROOT / 'generated' / a.version / mod
        for c in sorted(d.glob('*.c')) if d.is_dir() else []:
            gen_total += 1
            gen_done += compiled(c, out / 'obj' / mod / (c.stem + '.o'))
    stages.append(('game C compiles', 100 * gen_done / max(gen_total, 1) if gen_total else 0, f'{gen_done} of {gen_total} generated files'))
    exe = out / 'oot'
    stages.append(('links', 100 if exe.is_file() else 0, 'build/%s/oot %s' % (a.version, 'exists' if exe.is_file() else 'not built')))
    stages.append(('selftest passes', 100 if (out / '.linux-selftest-ok').is_file() else 0, 'oot --selftest'))
    stages.append(('title cinematic shows', 100 if (out / '.linux-title-ok').is_file() else 0, 'a Linux run reaches the title'))
    total = sum(p for _, p, _ in stages) / len(stages)
    have, need = platform_api()
    lines = [f'# Linux port status', '', f'**{total:.0f}% of the stages**', '',
             '| Stage | Done | Detail |', '|---|---:|---|']
    lines += [f'| {n} | {p:.0f}% | {d} |' for n, p, d in stages]
    lines += ['', f'Platform layer: linux.c defines {have} of the {need} functions platform.h declares ({100 * have / max(need, 1):.0f}%).', '']
    text = '\n'.join(lines)
    print(text)
    if a.write:
        (ROOT / a.write).write_text(text)


if __name__ == '__main__':
    main()
