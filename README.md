# Ocarina of Time PC Recomp

Research project: static recompilation of *The Legend of Zelda: Ocarina of Time* (title id `04004860224DEC00`) into a
native Windows program, by the method of [totk-pc-recomp](https://github.com/valentderah/totk-pc-recomp) as carried
to further games by [fe3h-pc-recomp](https://github.com/jaigner-hub/fe3h-pc-recomp) and
[metroid-prime-4-recomp](https://github.com/jaigner-hub/metroid-prime-4-recomp). The game's code is lifted to C,
compiled for x86-64 and linked with a host runtime that stands in for the console's OS and graphics API on Vulkan.

This is the fourth game on that code and the first Switch 2 title, so the project also tests how much of the shared
core survives a new console generation.

## Status

As of 2026-11-06 (details and dates in [docs/plan.md](docs/plan.md), the shader work in
[docs/phase3.md](docs/phase3.md)):

- The dump extracts (`tools/extract_nup.py`, the Switch 2's `.nup` container and zbic-compressed NSOs), the three
  game modules lift to C and `oot.exe` builds and links (Phase 1).
- The game runs its whole start-up without a crash: audio, saves, the NPLN services offline, 3,661 shader programs
  loaded, and presents frames (Phase 2). On the way the port met the Switch 2 and SDK 22 differences listed in the
  plan (new kernel queries, thread-local layout, PAC prologues, NVN2 entry points).
- Phase 3, the shaders: the Switch 2's shaders are SM86 (Ampere) code in an NVN2 container, which the Maxwell
  translator from the earlier ports cannot read. `tools/shader_census.py` pulls the 14,363 programs out of the game's
  archives, `runtime/sm86_lower.c` lowers them to Maxwell in front of the unchanged translator (93.6% of the corpus
  lowers, and every lowered program translates), and the host binds the lowered jump tables itself. With that the
  boot logo renders at 60 fps through translated shaders.
- The title screen, which in this version is a looping cinematic, plays. SDK 22's movie library decodes through
  console services this host does not serve; `runtime/ov_movie22.c` replaces it and decodes the AV1 stream with
  dav1d, and the game's neural upscaler, which runs on the Switch 2's tensor cores (HMMA, LDSM, MOVM), upscales it
  to 2560x1440. HMMA runs on the host GPU's tensor cores through Vulkan cooperative matrices, and the title plays at
  60 fps. Next: the 656 skinning vertex shaders the lowering still
  refuses (register pressure), the mesh stages, and whatever lies past the cinematic.

Builds run with Windows Python and MSYS2 Clang (`tools/build_game.py`), the shader translator with .NET
(`tools/build_xlate.py`); see the plan for the set-up.

This repository contains no game content. You need your own dump of the game and the keys from your own console. The
content check (`tools/check_content.py`, run by the pre-commit hook and CI) keeps it that way.

## License

MIT, see [LICENSE](LICENSE). Code from other projects is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
