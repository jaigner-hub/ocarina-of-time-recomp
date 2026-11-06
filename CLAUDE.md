@AGENTS.md

# Session guide

## What this project is

A static recompilation of *The Legend of Zelda: Ocarina of Time*, from the user's dump
`The Legend of Zelda Ocarina of TIme [04004860224DEC000].nup` (title id `04004860224DEC00`). The goal is the same as
the predecessors': lift the game's ARM64 code to C, compile it for x86-64 and link it with a host runtime that stands
in for the OS and GPU. Players bring their own dump and keys; nothing from the game is ever committed.

This is the fourth game on the totk-pc-recomp code, and the first from the Switch 2: the title id is in the `0400`
range, and the dump is a `.nup`, not an XCI or NSP. metroid-prime-4-recomp deliberately left its Switch 2 build
alone (its `docs/plan.md`, Phase 0), so everything below the container is unknown until Phase 0 answers it.

## Where to get code from

- `refs/metroid-prime-4-recomp/`: the MP4 port at `586e91e`, the direct predecessor. Every file in this tree was
  copied from there on 2026-11-05 with `mp4` renamed to `oot` (`MP4_` to `OOT_`); each file's first line names
  its source, and the lines after it carry the older attributions. Its `docs/plan.md` records what broke on the third
  game and why.
- `refs/fe3h-pc-recomp/` and `refs/totk-pc-recomp/`: the older ports. Read `docs/architecture.md` in TotK first.
- `CLAUDE.local.md` (ignored) names the local working checkouts.

Still MP4's and to be rewritten for this game once Phase 0 knows its modules and versions (they would mislead if
trusted): `runtime/boot.c` (modules, init_array sizes, stack, `nvnBootstrapLoader` return addresses, SDK init form),
the main stack in `runtime/kernel.c`, the version tables in `tools/build_game.py`, `installer/layout.py`,
`tools/make_release.py` and `tools/prepare_shaders.py` (`v131072`), the probe
table in `runtime/loader.c`, `runtime/frame_rate.c` (host half of MP4's frame-time patch), and `runtime/ov_oot.c`
(MP4's SDK 19.3 overrides: keep what this game reaches, drop the rest). Not taken: MP4's `recomp/patches-v131072.txt`
and `tools/data/expected-*.json`.

## Status

Phase 1 reached (2026-11-05): `oot.exe` builds (`.venv-win/Scripts/python.exe tools/build_game.py` from MP4's venv for
now), `--selftest` and the host probes pass, and `--boot` passes to the first `nvnBootstrapLoader` call. `docs/plan.md`
Phase 1 lists the Switch 2 and SDK 22 differences met on the way (GetInfo 29-33, the thread handle at TLR+0x110, the
PACIBSP prologue, nn::oe and nn::vi additions in `ov_oot.c`, `nv::InitializeGraphics(0, 0)`). Extraction:
`python3.14 tools/extract_nup.py --verify` (needs `toolchain/zbic/nso_unzbic` from `tools/build_zbic.py`) writes
`build/extracted/v131072/{exefs,romfs,extraction-manifest.json}`. Logs of runs go to `build/logs/`. Phase 2 (`oot.exe
--seconds 15`) runs the whole start-up without a crash: audio, saves, NPLN offline, 3,661 shader programs, three frames
presented; it then stalls because no SM86 shader translates (`docs/plan.md`, "Open: shader translation"), which is
Phase 3: `docs/phase3.md` is the hand-off for it (what blocks, the measured NVN2 control-section fields, the lowering
approach, the order of work). Start there, with its dated Status sections: as of the night of 2026-11-06 the SM86
lowering covers 94.1% of the corpus, the title cinematic decodes on the host (`runtime/ov_movie22.c`, dav1d) and
plays on screen, upscaled to 2560x1440 by the game's neural network through the emulated tensor-core instructions
(HMMA, LDSM, MOVM); HMMA runs on the host's tensor cores (VK_KHR_cooperative_matrix, translator patch 0016) and
the title plays at 60 fps. The shader cache keys carry the translator identity and `KEY_VERSION`
(`gfx_shader.c`): bump the latter after a lowering change. The first run after a translator or lowering change spends
a minute building the kernels' pipelines (the driver's disk cache keeps them). The start-up crash that killed
about one run in two right after the first frame is fixed (`docs/phase3.md`, "the start-up crash").

## Running and checking

- Run the game with `python3 tools/run.py` (`--seconds`, `--shot`, `--press-a`, `--env OOT_X=V`, `--fresh-cache`,
  `--dump`, `--expect REGEX`), never by hand: it names every `OOT_*` variable in WSLENV, retries a crash in the first 25 s,
  numbers the log (`build/logs/run-<n>.log`), kills `oot.exe` and `oot_xlate.exe` afterwards and refuses to overlap
  runs. `--expect` checks the log for a line proving a diagnostic switch applied.
- After a change to the lowering, the translator or anything a shader computes, follow the project skill
  `verify-shader-change` (`.claude/skills/`): probes, one kernel both ways, scored screenshots. A frame rate counts
  only next to a screenshot that `tools/shot_diff.py` scores as a picture.
- Shader tools: `tools/kernel.py` (list, stats, extract, lower, xlate, dis of a stored input),
  `tools/bisect_kernels.py` (which kernel breaks the picture), `tools/xlate_patch.py` (start / save a translator
  patch; `tools/build_xlate.py` takes any number of patches numbered without gaps). Image tools need Pillow and numpy:
  for now `/mnt/c/dev/metroid-prime-4-recomp/.venv/bin/python3`.
- What the game does on its own, found the slow way:
  - The title shows a grass loop first (at 30 s, a good fixed scene); idle, it plays the castle and horse
    cinematic. A at frame 600 (`--press-a`) brings up the menu.
  - Into the game without sitting through the opening: `python3 tools/run.py --seconds 150 --replay
    build/logs/a-spam.txt --skip-movies` (the script: A every 120 frames from frame 600, written by a one-line
    Python loop) reaches the level in about 75 s; the game itself cannot skip the opening. `--record FILE` writes a played session as a script for `--replay`.
  - Out of focus the game pauses itself, music included, as on the console under the HOME menu; `tools/run.py` pins
    the focus (`OOT_ASSUME_FOCUS=1`).
  - The movies have no sound track: the music is the game's own audio.
  - The input id known when a program is translated is not always its final input id (the `.in` name): a per-program
    switch matches the code hash instead (`OOT_TENSOR_ONLY`, `tools/kernel.py` prints it).
  - `--shot` builds pipelines synchronously, so screenshot runs are slower than normal ones.
  - A relative `OOT_DUMP_DIR` or `OOT_SHADER_CACHE` is relative to the repo (run.py passes them with `/p`).
  - `OOT_CS_TIME=1` logs the GPU time per compute shader; `OOT_TRACE=perf` the GPU time per frame and what bounds it. Discovery tools that paid off: `OOT_BREAK_AT`/`OOT_BREAK_BRIEF`/`OOT_BREAK_WATCH`, the
crash dump's memory and TLS peeks (`OOT_CRASH_PEEK`), the NVN2 logging stubs (`nvn_procs_stub.inc`),
`build/logs/nvn-proc-got.json` (GOT slot -> proc name, from the loader at `main+0x129e290`), `--probe-tensor` and
`--probe-cs` (a kernel in isolation), `OOT_DUMP=dispatch`, `OOT_SKIP_DISPATCH_FROM`, `OOT_MQ_LOG`, `OOT_THREAD_DUMP`.

## Environment notes

- The user works in WSL (Linux), where the interpreter is `python3`. Builds run with Windows Python and MSYS2 UCRT64
  Clang through interop, as in MP4 (`.venv-win/Scripts/python.exe tools/build_game.py`); make the venvs from
  `tools/requirements.txt`.
- The dump is 25 GB on the Windows desktop; `config.local.toml` points at it in place rather than copying it.
- `oot.exe` is a Windows program run from WSL through interop: WSL environment variables do NOT reach it unless named
  in `WSLENV`. Always run it as `WSLENV=OOT_TRACE:OOT_VOLUME:OOT_APP_POOL:OOT_CRASH_PEEK:OOT_BREAK_AT:OOT_BREAK_BRIEF:OOT_BREAK_WATCH
  OOT_TRACE=gfx build/v131072/oot.exe ...`
  (add every `OOT_*` variable used); without it the variable is silently ignored. Half a day of traces in this
  session were empty for this reason.
- Never commit without the content check passing; never bypass the hook.
