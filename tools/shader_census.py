"""Census of every NVN2 (SM86) shader program the game ships, with an SM86 opcode histogram, the SM86-to-Maxwell
lowering of every program (runtime/sm86_lower.c through tools/sm86_lower_main.c) and an offline translation pass.

Written for this game from totk-pc-recomp tools/shader_census.py @ c944f9f94520874f2a669add30dd08d29ee9f7b5 (MIT,
Copyright (c) 2026 Valiantsin Dzerakh; itself a port of MacNativeGames games/totk/scripts/prepare_shader_seed.py @
b75be8cf, MIT, Copyright (c) 2026 MacNativeGames contributors): sarc(), bnsh_shaders() and the translator worker are
theirs with the NVN2 control layout in place of the Switch 1 one; sharcb_shaders() finds records by their control
pointer instead of by the 0x880 size word. Replaces MP4's tools/shader_inventory.py (Retro's SNVN chunks).

Where the programs are (docs/phase3.md): the RomFS's `Shader/*.Nin_Ounce_NVN2.bfsha.zs` and
`Layout2Shader/*.bfsha.zs` (BFSHA files embedding BNSH files, the Switch 1 container unchanged), and the two agl
packages (`AglShader.sharcb`, `FoliageShader.sharcb`) inside `Shader/ApplicationPackage.Nin_Ounce_NVN2.release.sarc.zs`,
which hold the compute programs. All are zstd with the dictionaries of `Pack/ZsDic.pack.zs`. The ExeFS modules carry
none (no BNSH, no control magic).

The NVN2 control section (measured over the corpus, 2026-11-05; see docs/phase3.md): magic 0x98761234, its own size at
+0xd4 (0xe0, 0xf8, 0x108, 0x118, ...; the BNSH archives repeat it at +0x30 and +0x98, the agl packages leave those 0),
the end of the instructions at +0x38 (header + instructions), the length of the jump-table constants at +0x3c, their
offset in the code at +0x40, the whole code block's size at +0x44, the stage at byte +0x56 (NVN's enum: 0 vertex,
1 fragment, 2 geometry, 3 tess control, 4 tess eval, 5 compute, plus 6 task and 7 mesh), and for compute the workgroup
size at +0x58/+0x5c/+0x60. The bytes past +0xe0 describe the jump tables: a u32 count, then 16-byte records {BRX
instruction offset in the code, entry count, 0x10000, byte offset of the table in c1}; +0xdc is the size they use.
Graphics code starts with the 0x80-byte Turing+ program header (Maxwell's is 0x50); compute code starts with its first
instruction.

Reads build/extracted/<version>/romfs (romfs.bin plus index.json). Writes under build/<version>/shaders/ only:
    ctl/<key>.ctl, code/<key>.code   the control section and the whole code block as the archive holds them
    programs.json                    key -> stage, source, sizes
    opcodes.txt                      SM86 opcode histogram (word0 & 0xfff): instructions, programs, the share of
                                     programs that use only opcodes above this row (the work list for the lowering)
    lowered/<key>.code, .c1          with --lower: the Maxwell program (0x50-byte header + instructions) and its c1
    lowered.txt                      with --lower: failures grouped by message
    spv/<key>.spv, failures.txt      with --translate (implies --lower)

The key is sha256(control + code). Needs Python 3.14 (compression.zstd); on Linux, --lower needs cc (gcc) and
--translate runs build/xlate/oot_xlate.exe through WSL interop:

    python3.14 tools/shader_census.py --version v131072 [--lower] [--translate]
"""
import argparse
import compression.zstd as zstd
import concurrent.futures
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
XLATE = ROOT / 'build' / 'xlate' / 'oot_xlate.exe'
LOWER_SOURCES = [ROOT / 'runtime' / 'sm86_lower.c', ROOT / 'tools' / 'sm86_lower_main.c']
ZSTD = b'\x28\xb5\x2f\xfd'
CONTROL_MAGIC = b'\x34\x12\x76\x98'                # u32 0x98761234, as on Switch 1
STAGES = {0: 'vs', 1: 'ps', 2: 'gs', 3: 'tcs', 4: 'tes', 5: 'cs', 6: 'task', 7: 'mesh'}
SPH_SIZE = 0x80                                    # Turing+ program header before the first graphics instruction
MAX_WORKERS = 20


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(data, offset, size):
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError('shader container range outside file')
    return data[offset:offset + size]


def sarc(data):
    if data[:4] != b'SARC':
        raise ValueError('expected SARC')
    endian = '<' if data[6:8] == b'\xff\xfe' else '>'
    header, size, start = struct.unpack_from(endian + 'HxxII', data, 4)
    if size != len(data) or header < 20 or checked(data, header, 4) != b'SFAT':
        raise ValueError('invalid SARC header')
    count = struct.unpack_from(endian + 'H', data, header + 6)[0]
    names = header + 12 + count * 16
    if checked(data, names, 4) != b'SFNT':
        raise ValueError('invalid SARC string table')
    names += 8
    result = {}
    for i in range(count):
        _, attribute, lo, hi = struct.unpack_from(endian + '4I', checked(data, header + 12 + i * 16, 16))
        off = names + (attribute & 0xffffff) * 4
        if attribute >> 24 != 1 or not names <= off < start or hi < lo or start + lo < start:
            raise ValueError('invalid SARC member')
        name = data[off:data.index(0, off, start)].decode('utf-8')
        if name.startswith('/') or '\\' in name or any(p in ('', '.', '..') for p in name.split('/')) or name in result:
            raise ValueError('unsafe or duplicate SARC name')
        result[name] = checked(data, start + lo, hi - lo)
    return result


def control_fields(control):
    """The NVN2 control section's fields the host needs (runtime/gfx_shader.c ctl_parse reads the same)."""
    if control[:4] != CONTROL_MAGIC:
        raise ValueError('NVN control section without its magic')
    _, _, code_end, c1_len, c1_off, code_size = struct.unpack_from('<6I', control, 0x30)
    size = struct.unpack_from('<I', control, 0xd4)[0]   # +0x30 repeats it in the BNSH archives and is 0 in agl's
    stage = control[0x56]
    local = struct.unpack_from('<III', control, 0x58)
    return dict(size=size, code_end=code_end, c1_len=c1_len, c1_off=c1_off, code_size=code_size, stage=stage,
                local=list(local))


def program(control, code):
    """(key, stage name, control, code) of one archive program, checked against its control fields."""
    f = control_fields(control)
    if f['size'] != len(control) or f['stage'] not in STAGES:
        raise ValueError(f'unexpected NVN2 control section: size {f["size"]:#x} of {len(control):#x}, stage {f["stage"]}')
    header = 0 if f['stage'] == 5 else SPH_SIZE
    if not header < f['code_end'] <= f['c1_off'] <= f['c1_off'] + f['c1_len'] <= f['code_size'] <= len(code):
        raise ValueError(f'invalid NVN2 code bounds {f}')
    code = code[:f['code_size']]
    return digest(control + code), STAGES[f['stage']], control, code


def bnsh_shaders(data):
    """BNSH absolute offsets are relative to each embedded BNSH, not BFSHA. The variation and program records are
    the Switch 1 ones: a 64-byte variation with its program at +16, and per program the stage records at +8 (vs),
    +16 (gs), +24 (tcs), +32 (tes), +40 (ps), +48 (cs), each {+8 control offset, +16 code offset, +24 code size,
    +28 control size}."""
    if data[:4] != b'BNSH' or checked(data, 12, 2) != b'\xff\xfe' or checked(data, 96, 4) != b'grsc':
        raise ValueError('unsupported BNSH header')
    size = struct.unpack_from('<I', data, 28)[0]
    data = checked(data, 0, size)
    count, variations = struct.unpack_from('<IQ', data, 124)
    checked(data, variations, count * 64)
    for i in range(count):
        prog = struct.unpack_from('<Q', data, variations + i * 64 + 16)[0]
        checked(data, prog, 56)
        for relative in (8, 16, 24, 32, 40, 48):
            offset = struct.unpack_from('<Q', data, prog + relative)[0]
            if not offset:
                continue
            control_off, code_off, code_size, control_size = struct.unpack_from('<QQII', checked(data, offset + 8, 24))
            yield program(checked(data, control_off, control_size), checked(data, code_off, code_size))


def sharcb_shaders(data):
    """agl shader packages (.sharcb, magic BAHS): 16-byte records <u64 code offset in the pool, u32 control size,
    i32 offset of the control from the record> between the header and the code pool. A record is one whose control
    size is plausible and whose control pointer lands on a control magic (TotK's census found them by the 0x880 size
    word; the NVN2 sizes vary, so the magic is the anchor)."""
    if len(data) < 32 or data[:4] != b'BAHS':
        raise ValueError('expected an agl shader package')
    start, pool_size, pool = struct.unpack_from('<3I', data, 20)
    if not 32 <= start <= pool or pool + pool_size > len(data):
        raise ValueError('agl shader package layout outside the file')
    seen = set()
    for record in range(start, pool - 16, 4):
        code_off, control_size, relative = struct.unpack_from('<QIi', data, record)
        control_at = record + relative
        if not 0xe0 <= control_size <= 0x400 or control_size % 4 or not start <= control_at <= pool - control_size:
            continue
        if data[control_at:control_at + 4] != CONTROL_MAGIC or code_off >= pool_size:
            continue
        control = data[control_at:control_at + control_size]
        if struct.unpack_from('<I', control, 0xd4)[0] != control_size:
            continue
        code_size = struct.unpack_from('<I', control, 0x44)[0]
        if code_size > pool_size - code_off:
            continue
        key, stage, control, code = program(control, data[pool + code_off:pool + code_off + code_size])
        if key not in seen:
            seen.add(key)
            yield key, stage, control, code


def shader_file(name):
    return (name.endswith('.bfsha.zs') or name.endswith('.bfsha') or name.endswith('.bnsh') or name.endswith('.bnsh.zs')
            or name.endswith('.sharcb') or name.endswith('.sharcb.zs')
            or name.startswith('Shader/') and name.endswith('.sarc.zs'))


def inventory(romfs):
    """Every unique program: key -> dict(stage, control, code, source, origin). Also the file names visited."""
    index = json.loads((romfs / 'index.json').read_text(encoding='utf-8'))
    if index.get('status') != 'pass':
        raise ValueError('RomFS has not passed integrity verification')
    entries = {r['path']: r for r in index['files']}
    shaders, visited = {}, []

    def add(origin, name, found):
        for key, stage, control, code in found:
            shaders.setdefault(key, dict(stage=stage, control=control, code=code, source=name, origin=origin))

    with (romfs / 'romfs.bin').open('rb') as stream:
        def read(name):
            row = entries[name]
            stream.seek(row['offset'])
            raw = stream.read(row['size'])
            if len(raw) != row['size']:
                raise ValueError('short RomFS read')
            return raw

        dictionaries = {}
        for raw in sarc(zstd.decompress(read('Pack/ZsDic.pack.zs'))).values():
            d = zstd.ZstdDict(raw)
            dictionaries[d.dict_id] = d

        def decode(raw):
            info = zstd.get_frame_info(raw)
            if info.decompressed_size is None or info.decompressed_size > 512 << 20:
                raise ValueError('unbounded shader package')
            return zstd.decompress(raw, zstd_dict=dictionaries[info.dictionary_id] if info.dictionary_id else None)

        def visit(name, data, depth=0):
            if data[:4] == b'BAHS':
                add('sharcb', name, sharcb_shaders(data))
                return
            if data[:4] == b'SARC':
                if depth > 4:
                    return
                for member, raw in sarc(data).items():
                    visit(name + '/' + member, decode(raw) if raw[:4] == ZSTD else raw, depth + 1)
                return
            pos = 0   # BFSHA and BNSH files: complete, independently addressed BNSH files
            while (pos := data.find(b'BNSH\0\0\0\0', pos)) >= 0:
                size = struct.unpack_from('<I', checked(data, pos + 28, 4))[0]
                add('bnsh', name, bnsh_shaders(checked(data, pos, size)))
                pos += max(size, 8)

        for name in sorted(entries):
            if not shader_file(name):
                continue
            raw = read(name)
            visited.append(name)
            try:
                visit(name, decode(raw) if raw[:4] == ZSTD else raw)
            except ValueError as e:
                raise ValueError(f'{name}: {e}') from None
    return shaders, visited


def instruction_words(stage, code, fields):
    """The SM86 instructions of a program: 16-byte words from the header's end to the control's code end."""
    start = 0 if stage == 'cs' else SPH_SIZE
    body = code[start:fields['code_end']]
    return [struct.unpack_from('<4I', body, o) for o in range(0, len(body) - len(body) % 16, 16)]


def write_corpus(shaders, out):
    """Writes ctl/, code/, programs.json and opcodes.txt; returns the programs table."""
    (out / 'ctl').mkdir(parents=True, exist_ok=True)
    (out / 'code').mkdir(parents=True, exist_ok=True)
    programs = {}
    instructions, users = Counter(), Counter()
    per_program = {}
    for key, s in sorted(shaders.items()):
        f = control_fields(s['control'])
        for path, data in ((out / 'ctl' / f'{key}.ctl', s['control']), (out / 'code' / f'{key}.code', s['code'])):
            if not path.is_file() or path.read_bytes() != data:
                path.write_bytes(data)
        ops = Counter(w[0] & 0xfff for w in instruction_words(s['stage'], s['code'], f))
        instructions.update(ops)
        users.update(ops.keys())
        per_program[key] = set(ops)
        programs[key] = dict(stage=s['stage'], source=s['source'], origin=s['origin'], ctl_size=f['size'],
                             code_end=f['code_end'], c1_off=f['c1_off'], c1_len=f['c1_len'], code_size=f['code_size'],
                             local=f['local'], instructions=sum(ops.values()))
    (out / 'programs.json').write_text(json.dumps(programs, indent=0), encoding='utf-8')
    # The work list: opcodes by the number of programs using them; the last column is the share of programs that
    # would lower if every opcode down to this row were supported.
    order = [op for op, _ in users.most_common()]
    supported, lines, total = set(), [], len(programs)
    lines.append(f'# {sum(instructions.values())} instructions in {total} programs, {len(order)} distinct opcodes')
    lines.append('# opcode  instructions  programs  cumulative share of programs covered')
    for op in order:
        supported.add(op)
        covered = sum(1 for ops in per_program.values() if ops <= supported)
        lines.append(f'{op:03x}\t{instructions[op]}\t{users[op]}\t{covered / total:.4f}')
    (out / 'opcodes.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return programs


def build_lowerer(out):
    """Compiles the lowering helper for this platform, when the sources changed; returns its path."""
    exe = out / ('sm86_lower.exe' if os.name == 'nt' else 'sm86_lower')
    stamp = out / 'sm86_lower.sha256'
    h = hashlib.sha256(b''.join(p.read_bytes() for p in LOWER_SOURCES)).hexdigest()
    if exe.is_file() and stamp.is_file() and stamp.read_text() == h:
        return exe
    cc = [r'C:\msys64\ucrt64\bin\clang.exe'] if os.name == 'nt' else ['cc']
    cmd = cc + ['-std=c11', '-O2', '-Wall', '-Wextra', '-I', str(ROOT / 'runtime'), '-o', str(exe)] + [str(p) for p in LOWER_SOURCES]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'building the lowering helper failed:\n{r.stderr}')
    stamp.write_text(h)
    return exe


def lower(programs, out):
    """Lowers every program through the helper's batch mode: one line per program in,
    `ok <key>` or `fail <key> <offset> <message>` out. Returns (ok, {key: message})."""
    exe = build_lowerer(out)
    low = out / 'lowered'
    low.mkdir(exist_ok=True)
    # Paths relative to out, where the helper runs: the request is split at spaces, and the repository's path may
    # hold one.
    request = ''.join(f'{key} {p["stage"]} ctl/{key}.ctl code/{key}.code lowered/{key}.code lowered/{key}.c1\n'
                      for key, p in sorted(programs.items()))
    r = subprocess.run([str(exe), '--batch'], input=request, capture_output=True, text=True, cwd=out)
    ok, failures = 0, {}
    for line in r.stdout.splitlines():
        parts = line.split(' ', 3)
        if parts[0] == 'ok':
            ok += 1
        elif parts[0] == 'fail' and len(parts) == 4:
            failures[parts[1]] = f'{parts[3]} (at +{parts[2]})'
        else:
            failures[parts[1] if len(parts) > 1 else '?'] = line
    missing = set(programs) - failures.keys()
    if ok != len(missing):
        sys.exit(f'the lowering helper answered for {ok + len(failures)} of {len(programs)} programs:\n{r.stderr[-2000:]}')
    write_groups(out / 'lowered.txt', [('lowering failures', failures)], programs, out)
    return ok, failures


def normalize(message):
    return re.sub(r'\b\d+\b|0x[0-9a-fA-F]+|\+[0-9a-fA-F]+', 'N', message)[:300]


def write_groups(path, tables, programs, out=None):
    """Failures grouped by message; a lowering failure's example lines also show the instruction (sm86_dis)."""
    lines = []
    for title, table in tables:
        groups = defaultdict(list)
        for key, message in table.items():
            groups[normalize(message)].append((key, message))
        lines.append(f'== {title}: {len(table)} in {len(groups)} groups')
        for _, members in sorted(groups.items(), key=lambda g: -len(g[1])):
            stages = Counter(programs[k]['stage'] for k, _ in members if k in programs)
            lines.append(f'{len(members)}\t{members[0][1][:300]}\t{dict(stages)}')
            for key, message in sorted(members)[:2]:
                lines.append(f'\texample {key}')
                m = re.search(r'\(at \+([0-9a-f]+)\)$', message)
                if m and out and key in programs:
                    import sm86_dis
                    code = (out / 'code' / f'{key}.code').read_bytes()
                    at = int(m.group(1), 16)
                    lines.extend('\t\t' + t for t in sm86_dis.listing(code, at, min(at + 16, len(code))))
    if any(t for _, t in tables):
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    else:
        path.unlink(missing_ok=True)


def host_path(p):
    """The path as the Windows translator sees it (WSL interop maps /mnt/<d>/ to <D>:\\)."""
    s = str(p)
    m = re.match(r'^/mnt/([a-z])/(.*)$', s)
    return f'{m.group(1).upper()}:\\{m.group(2).replace("/", "\\")}' if m else s


def worker(keys, jobs, spv_dir):
    """One --serve process for a slice of keys. Returns (translated, failures {key: message})."""
    failures, ok = {}, 0
    process = subprocess.Popen([str(XLATE), '--serve'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True)
    try:
        for key in keys:
            process.stdin.write(json.dumps({'job': jobs[key], 'out': host_path(spv_dir)}) + '\n')
            process.stdin.flush()
            reply = process.stdout.readline().rstrip('\n')
            if not reply:
                failures[key] = 'translator service exited'
                break
            if not reply.startswith('ok '):
                failures[key] = reply.split(' ', 2)[2] if reply.count(' ') >= 2 else reply
                continue
            ok += 1
    finally:
        process.stdin.close()
        process.wait()
    return ok, failures


def translate(programs, lowered_ok, out):
    """Translates every lowered program; the job layout is runtime/gfx_shader.c build_request's."""
    spv_dir = out / 'spv'
    spv_dir.mkdir(exist_ok=True)
    jobs = {}
    for key in sorted(lowered_ok):
        p = programs[key]
        stage = {'vs': 'vs', 'ps': 'ps', 'cs': 'cs'}.get(p['stage'])
        if not stage:
            continue
        c1 = (out / 'lowered' / f'{key}.c1').read_bytes()
        job = dict(key=key, stage=stage, code=host_path(out / 'lowered' / f'{key}.code'), c1=c1.hex(), debug=False)
        if stage == 'cs':
            job['local'] = [v or 1 for v in p['local']]
        jobs[key] = job
    keys = sorted(jobs)
    count = max(1, min(MAX_WORKERS, len(keys)))
    slices = [keys[i::count] for i in range(count)]
    ok, failures = 0, {}
    with concurrent.futures.ThreadPoolExecutor(count) as pool:
        for n, f in pool.map(lambda s: worker(s, jobs, spv_dir), slices):
            ok += n
            failures.update(f)
    write_groups(out / 'failures.txt', [('translate failures', failures)], programs)
    return len(jobs), ok, len(failures)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--version', required=True, help='v131072 (build/extracted/<version>)')
    ap.add_argument('--lower', action='store_true', help='lower every program to Maxwell and report the failures')
    ap.add_argument('--translate', action='store_true', help='also translate every lowered program (implies --lower)')
    ap.add_argument('--extract', action='store_true', help='read the archives again even when programs.json exists')
    args = ap.parse_args()
    t0 = time.monotonic()
    extracted = ROOT / 'build' / 'extracted' / args.version
    out = ROOT / 'build' / args.version / 'shaders'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'programs.json').is_file() and not args.extract:
        programs = json.loads((out / 'programs.json').read_text(encoding='utf-8'))
        print(f'{len(programs)} programs from programs.json (--extract reads the archives again)', flush=True)
    else:
        shaders, visited = inventory(extracted / 'romfs')
        programs = write_corpus(shaders, out)
        stages = Counter(p['stage'] for p in programs.values())
        print(f'{len(programs)} unique programs from {len(visited)} files: {dict(stages)}; '
              f'extracted in {time.monotonic() - t0:.1f} s', flush=True)
    if not args.lower and not args.translate:
        return 0
    ok, failures = lower(programs, out)
    print(f'lowered {ok} of {len(programs)} ({ok / len(programs):.1%}), {len(failures)} failed '
          f'(lowered.txt) in {time.monotonic() - t0:.1f} s', flush=True)
    if not args.translate:
        return 0
    if not XLATE.is_file():
        sys.exit(f'missing {XLATE} (python tools/build_xlate.py)')
    n, ok, bad = translate(programs, set(programs) - failures.keys(), out)
    print(f'translated {ok} of {n} lowered vs/ps/cs programs, {bad} failed (failures.txt), total {time.monotonic() - t0:.1f} s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
