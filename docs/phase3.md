# Phase 3: SM86 shaders

Hand-off written 2026-11-05 at commit `b46d663`, for a fresh session. Read `CLAUDE.md` and `AGENTS.md` first; the
history is in `docs/plan.md` (Phases 0 to 2 and the NVN2 survey). This file says where the port stands, exactly what
blocks it, what is already measured, and the order to do the work in.

## Status (2026-11-05, later the same day)

Steps 1 to 3 of the order of work below are done; step 4 is at 86% of the corpus; step 5 is where the work is.

- The translator builds (`build/xlate/oot_xlate.exe`, `python tools/build_xlate.py` from the Windows venv, 20 s).
  `tools/xlate/Program.cs` now echoes the job's `c1` into the `.res.json` (the host binds the lowered jump tables).
- `tools/shader_census.py` (NVN2 version; `python3.14`, Linux) reads the archives and writes
  `build/v131072/shaders/{ctl,code}/<key>`, `programs.json` and `opcodes.txt`: 14,363 programs (10,994 pixel, 3,106
  vertex, 251 compute, 9 mesh, 3 task) from the four `Shader/*.bfsha.zs`, the `Layout2Shader` bfsha and the two agl
  `.sharcb` of the ApplicationPackage sarc; the ExeFS modules hold none. `--lower` builds `runtime/sm86_lower.c` with
  gcc (`tools/sm86_lower_main.c`) and lowers every program, grouping the failures with the failing instruction in
  `lowered.txt`; `--translate` runs the translator over the lowered ones through WSL interop (`failures.txt`).
  `tools/sm86_dis.py <key>` lists a program's fields. The corpus is regenerated from the RomFS in 2 min; later runs
  reuse `programs.json` (`--extract` reads the archives again).
- The control section's fields are settled (the table below, columns "Reading", and `sm86_control_parse`): size at
  `+0xd4` (the BNSH archives repeat it at `+0x30` and `+0x98`, the agl packages leave those 0), instruction end at
  `+0x38`, jump-table constants at `+0x40` with their length at `+0x3c`, code size at `+0x44`, the stage at byte
  `+0x56` (NVN's enum plus 6 task, 7 mesh), the compute workgroup size at `+0x58`; the tail past `+0xe0` holds a u32
  count and 16-byte records `{BRX instruction offset, entry count, 0x10000, byte offset of the table in c1}`.
  Graphics code starts with the 0x80-byte Turing+ header, whose first 0x50 bytes are the Maxwell header the
  translator reads. The early-fragment-tests flag is not identified (candidates: `+0xa4` bits 0 and 8).
- `runtime/sm86_lower.c` lowers 13,444 of 14,363 programs (93.6%). Registers: the lowering's uniform registers
  and scratch live above the program's own; a uniform register loaded by a single direct `ULDC` (or `UMOV`), or by
  one earlier in the same basic block, is forwarded as that constant and needs no register; when the program leaves
  no room, a liveness analysis over the basic blocks (`insn_regs`, exact destination widths, reads over-stated only
  for texture coordinates) finds scratch among the registers dead at each instruction and a uniform register's
  register among those dead over its span. Uniform predicates take free predicates, or are swapped through a
  register around each instruction naming them. What is left: 656 vertex shaders (skinning: up to 247 registers
  live at once, no adjacent pair free where a load needs one), the surface stores (`SUST`, 86 compute programs),
  `ISBERD`/`ISBEWR` (task and mesh stages), a few rare forms (`lowered.txt`).
- The host (`runtime/gfx_shader.c`, `vk_exec_draw.c`, `vk_dump.c`, the shader probe and the pipeline self-test)
  reads the NVN2 control through `sm86_control_parse`, hashes the control section (its own size) and the whole code
  block for the cache key (`KEY_VERSION` 5, `IN_VERSION` 2, `PRG_VERSION` 2), lowers before translating, and binds
  the lowered program's c1 (`gfx_shader_c1`, uploaded once per shader) as uniform slot 1 instead of the game's, since
  the jump tables' targets moved with the code.

## Status (2026-11-06)

The game plays its title cinematic through a host decoder, and the picture is still black for a reason outside the
shaders lowered so far: the game's video path runs on the tensor cores.

- `runtime/ov_movie22.c` decodes the cinematics with dav1d (linked statically from MSYS2, BSD-2, row in
  `THIRD_PARTY_NOTICES.md`): the title is "video/av01" 1920x1080 at 60 fps, a 64 s stream with a key frame every
  321 samples (`nn::movie::Buffer` flag 4 marks those, not the end of the stream, which is an empty flagged buffer).
  Frames come back to the game as NV12 in the buffer it registered (it reads planes 0 and 1 only), with the
  sample's time stamp. Audio is still dropped. The decode runs on the game's decode thread; the game then runs
  at about 14 fps there, to be profiled (dav1d on 4 threads, the NV12 interleave, or the game waiting on frames).
- The lowering grew: SUST (bindless store, the handle loaded from the constant word), LEA and IMAD.X carries
  through comparisons and predicate adds (the translator ignores Maxwell LEA's carry predicate and X), the
  uniform register budget counts registers read before any definition, uniform pairs are allocated adjacently
  in both the pool and the liveness mode, the swap register for uniform predicates comes from the liveness
  allocator when the pool is full, and only the predicate fields an op has are rewritten. Corpus: 13,514 of 14,363
  lower (94.1%), all translate.
- What blocks the picture: every shader of the video path (19 compute programs, dumped by the host to
  `shader-cache/<key>.fail.{ctl,code}` for `tools/sm86_dis.py`) uses the tensor-core instructions: HMMA (0x03c,
  4,478 instructions), LDSM (0x03b, 1,670) and MOVM (0x03a, 24). They are the game's neural upscaler (the SDK
  list names "SMAA_For_NDSR"; the swapchain is 2560x1440 while the movie is 1080p), and they are not in the RomFS
  archives the census reads (nor in the ExeFS modules; the raw `romfs.bin` has no match either, so they sit in a
  compressed file the census does not visit, or the game builds them). NVN2 device property `0x67`, kept by the
  game as a capability flag and required by its video surface setup, is most likely "supports tensor cores".
  Next: lower HMMA.16816 / 1688 f16 (LDSM's matrix loads into per-lane fragments, the warp-cooperative product
  through SHFL and FFMA, MOVM's transpose) in front of the translator, or find the non-tensor path if the game has
  one (with `0x67` = 0 the player aborted in its surface setup; the shader's decision may lie elsewhere).
- Reaching the game proper: `OOT_INPUT_SCRIPT` (lines "<ticks> A 10") passes the title; the file select and the
  opening are cinematics too, so they need the same path.
- The intermittent death right after the first frame (one run in three, exit 40 or 5): the crash handler's dump
  (`main.c`) caught two. One executed from the thread's own stack while logging a guest `Break` from
  `sdk+0x1a5560` (the SDK abort), the log line itself garbled; the other executed from the heap region of the
  `Cpu` structures with `rt_svc+0x59` (after `svc_body`: the host-depth bookkeeping, `kern_preempt_pending` /
  `kern_host_left`) as the return address and lifted `main+0x3779bc0` below it. Both point at the asynchronous
  preemption (`kernel.c`, the interrupt trampoline that pushes the interrupted thread's registers on its stack).
  Test: `OOT_ASYNC_PREEMPT=0` over several runs; then read the trampoline against a thread suspended inside host
  code.

## Status (2026-11-06, later): the title frame decodes, the upscaler does not

The movie frame reaches the GPU and converts correctly: the first compute pass of the game's video path writes a
1920x1080 RGBA8 image of the title cinematic (Link on Epona by the river; dump `build/logs/dump-93`, pass 0000,
147,050 distinct colours). What follows it, the neural upscaler's twelve kernels and the final 2560x1440 pass, still
produces a near-constant image and then zeros, so the screen stays black. Found and done on the way:

- The tensor-core instructions are emulated in the lowering (`runtime/sm86_lower.c`, "tensor-core instructions"):
  LDSM (1, 2 or 4 matrices, plain or transposed) through SHFL.IDX of the row addresses and LDS; HMMA.16816.F16.F16
  through the PTX fragment layouts (lane = 4g + t), SHFL.IDX and HFMA2 (f16 accumulation, the tensor core accumulates
  wider); MOVM through two SHFL and a PRMT. Also added: SUST (bindless store, the handle a dword index in the
  constant word as TEX's), HFMA2 (all forms, .RELU through HSET2.GT.BF and HMUL2), F2FP (two F2F.F16.F32 and a
  PRMT), HMNMX2 (HSET2, HADD2, HFMA2), BAR.SYNC. All 19 kernels lower and translate; the corpus stays at 13,514
  lowered, 0 translation failures.
- NVN2's constant bank 0 for compute is not Maxwell's driver bank. `nvnCommandBufferPushComputeConstants(cmd, 0,
  offset, bytes, data)` (an NVN2 entry point the host only stubbed) pushes the kernel's parameter block, and the bank
  reads as: 0x10 reserved bytes (the dispatch size is put there), the pushed constants, then the bound resources in
  slot order, storage buffers {u64 address, u32 size, u32 0} for control[0xb2] slots, texture handles (u64) for
  control[0xb3] slots, image handles (u64) for control[0xb6] slots (the counts are bytes of the control section;
  control[0xb8..0xba] holds the bank's size). Measured on the kernels: the final pass pushes 0x54 bytes and reads its
  storage slot 2 at 0x90, textures 0 and 3 at 0xa0 and 0xb8, image 1 at 0xc8; the NV12 pass pushes 0x40 bytes and
  reads textures 1 and 2 at 0x58 and 0x60, image 0 at 0x68; the kernels without a push read storage slots 0 and 1 at
  0x10 and 0x20. The host records the push as OP_PUSH_CC, builds the bank at each dispatch (`vk_exec_draw.c`
  nvn2_c0_fill_cs), consumes the push with the dispatch, and resolves the shader's bank-0 words back to the bound
  slots for its descriptors (nvn2_c0_slot, cbuf_handle, ssbo_info). The graphics stages still get Maxwell's driver
  bank; the game's vertex shaders read 64-bit addresses at c0+0x60, which is the same scheme and will need the push
  (nvnCommandBufferPushGraphicsConstants is still a stub) when a draw depends on it.
- Diagnostics added: `OOT_BIND_LOG=1` logs the compute stage's binds, pushes, dispatches and inline uniform updates
  from frame 290 on; `OOT_PROG_DUMP=1` hex-dumps the memory around compute programs' control sections at
  SetShaders; the frame dump prints each stage's bound texture, image and uniform slots per pass and resolves texture
  declarations as the replay does (`vk_exec_res_handle`), so storage images written by dispatches get their PNGs.
- The death after the first frame, one run in three, is the SDK's own abort in
  `nn::os::detail::MultiWaitImpl::InternalWaitAnyImpl` (sdk+0x167be0): after a holder's virtual IsSignaled (vtable
  +0x18) returns true, it aborts when its count of found holders reaches the caller's capacity (the compare before
  the jump to sdk+0x167ec4, which calls `nn::diag::detail::AbortImpl()` through the GOT). The host then dies with
  the abort's return address as the instruction pointer (`executing 0x7005367ec8`, the LR of that call, rcx holding
  the Cpu), before `AbortImpl(const char *, ...)` is reached (an observer on it, `ov_oot.c` obs_abort_impl, never
  fires). Two bugs: more holders report signalled than the game expects (an nn::os event or timer the host leaves
  signalled at start-up, to find), and the abort path crashing instead of reporting. `OOT_ASYNC_PREEMPT=0` does not
  prevent it.

Next (as of that section): find where the upscaler's values go wrong. Resolved in the section below.

## Status (2026-11-06, night): the cinematic plays

The title cinematic renders: the AV1 frame decoded by dav1d, upscaled to 2560x1440 by the game's neural network
through the emulated tensor-core instructions, composited with its subtitles (`build/logs/run-184.png`, 27,548
distinct colours). At 21 fps while the cinematic plays, 60 fps before it. Six distinct defects stood between the
black picture and this, found one after the other with the tools listed at the end:

- The translator's dead-node removal (`Optimizer.RemoveNode`) followed into any producer whose use count reached
  zero, even one still live through another destination. SHFL defines a value and a validity predicate; dropping the
  dead predicate copy stripped the use lists of the shuffle's lane index, and the next pass deleted the index
  computation: every emulated LDSM and HMMA shuffled from an undefined lane. Patch 0013 follows only into producers
  that are dead through every destination.
- PRMT, the byte permute, was an unimplemented logging stub in Ryujinx in every form (R, I, C, Rc). Half the lowered
  programs use it (packing halves), and the emulation builds every half pair with it. Patch 0014 implements it; a
  constant selector (the usual case) is resolved at translation time, the half-word packs 0x5410 and 0x7632 to two
  instructions.
- With those two fixed the kernels' SPIR-V grew to 16-22 MB (over a million instructions; the NVIDIA driver took 20
  to 30 s to build each pipeline): every emulated shuffle was a helper-function call with four pointer arguments.
  Patch 0015 emits SHFL.IDX with the plain mask (clamp 31, no segment) on a 32-lane subgroup as the subgroup shuffle
  itself, the lane index masked to the subgroup. The pipelines still take 0.5 to 30 s to build the first time (the
  driver's disk cache keeps them); the emulation's instruction count is the next performance item.
- The uniform-predicate swap (`sm86_lower.c` lower_one): when a program uses all seven predicates, a uniform
  predicate is swapped into the predicate of the same number around each instruction naming it, and the game's
  predicate restored after. The first neural-network layer ends its tile loop on `PLOP3 P0 = UP0`: the restore put
  the old P0 back over the result, the loop ran until its counters wrapped (2.2 s per dispatch, measured with the
  kernel probe) and the GPU hit Windows' 2 s timeout, reported as a device loss with the instruction pointers of
  every killed warp. The swap now leaves the instruction's own predicate outputs alone.
- The upscaler's final pass stores its 2560x1440 picture into the game's sRGB output texture, which Vulkan cannot
  make a storage image; the host bound a dummy. `vk_texture.c` now creates such images with EXTENDED_USAGE and the
  STORAGE usage, and `vk_exec_draw.c` get_view stores through a view of the UNORM sibling (`vk_unorm_of_srgb`).
- The lowering put the scheduling words of graphics programs at absolute multiples of 32 bytes from the block
  start, but Ryujinx counts them from the end of the 0x50-byte header: every third instruction of every vertex and
  fragment shader sat where the decoder skips a scheduling word, and the scheduling word was decoded as an
  instruction. The compositing quad's texture fetch was one of them. `addr_of` now counts from the header's end.
  (The pipeline check `module_binding_missing` in `vk_pipeline.c` also had to learn that a declared but unreferenced
  binding is not a layout error: the translator declares a constant buffer whose binding it reserved even when the
  load was removed.)

Also on the way: the game's movie worker feeds samples whenever it has them without consulting the input event;
decoding each as it came ran the whole stream through dav1d in seconds, the ready ring dropped pictures and the game
waited forever for the frame of a sample it had sent. `ov_movie22.c` now queues the samples and decodes them as the
ring has room. The SHFL emitted by the emulation sends its validity predicate to PT (it clobbered P0). The device
fault handler keeps the vendor's crash blob (`fault-vendor.bin` next to the log), compute pipeline builds over 100
ms are logged with their SPIR-V size.

Tools that paid off, all kept: `oot.exe --probe-tensor` (a hand-assembled SM86 kernel: LDSM x4, LDSM x2 transposed,
HMMA.16816.F16 against a CPU reference, through the lowering, the translator and a real dispatch);
`oot.exe --probe-cs <dir>` (one of the game's compute kernels dispatched in isolation with stand-in resources and
timing: ctl, code, bindings.txt; the kernel's weights from an `OOT_DUMP_BUF` snapshot); `OOT_DUMP=dispatch` (the
frame dump armed on the first compute dispatch); `OOT_SKIP_DISPATCH_FROM=<n>` (bisecting a device loss down to a
dispatch); `OOT_CS_DUMMY_TEX=1`; `OOT_MQ_LOG=1` (nn::os LightMessageQueue and EventType traffic with threads and
call sites, which found the movie worker's stall); `OOT_SM86_STUB=hmma|ldsm|all` (the emulation replaced by zeros,
for timing); `OOT_THREAD_DUMP=s<seconds>` with host backtraces (the render thread inside `nvgpucomp64.dll`).

Open: the frame rate (21 fps in the cinematic: the emulated kernels, and the decode on the game's thread); the
start-up crash, now about one run in two (the SDK abort in MultiWait, see the previous section); the 656 skinning
vertex shaders the lowering refuses; the mesh stages. The game has not yet been taken past the cinematic.

## Status (2026-11-06, late night): the title screen as the player sees it

Watched with sound for the first time (the user), three things looked wrong. All three are settled:

- **The game froze and its music stopped for a while.** It was out of focus: the host reports the window's focus to
  the game as the console reports the HOME menu, and the game pauses itself. By design. `OOT_ASSUME_FOCUS=1` pins the
  focus for watched test runs. The title movie has no sound track: no run has ever created a movie AudioDecoder.
- **The cinematic ran at a third of its speed.** The game takes one movie picture per game frame and does not look at
  the time stamps, so at 21 fps the 60 fps movie played at a third of its speed (a pacing line every 300 pictures:
  5 s of video in 15 s). `ov_movie22.c` now hands pictures back on a playback clock and skips the ones already
  late. dav1d runs on a host thread per decoder, which signals the output event; the events are interprocess
  system events for that. The frame rate stays at about 22 fps. Decoding was not what limited it.
- **The logo was pixelated.** Its ASTC blocks were decoded right (ARM's astcenc agrees texel for texel) but put in the
  wrong places. Inside a GOB the Switch 2 puts the 16-byte column at 64 bytes and the second row bit at 32, the
  reverse of the Tegra X1. This was found by matching each block's edges against every other block in the stored
  order, and holds for the ASTC logo (16-byte blocks) and the BC4 masks (8-byte blocks). `tex_deswizzle.c` applies it
  to elements of 8 and 16 bytes. Smaller elements keep the X1 order: no game data has settled them yet.
  `OOT_GOB_SWITCH2_MIN=1` (or 2, 4) tries the Switch 2 order for them, and `OOT_TEX_RAW=<w>x<h>` writes a texture's
  raw blocks for such checks. Frame dumps now also write the alpha of 8-bit RGBA images (`..._a.png`).

Getting to the title menu needs an A press (`OOT_INPUT_SCRIPT` with `600 A 10`).

## Status (2026-11-06, end of the night): the title runs at 60 fps on the tensor cores

The cinematic was GPU-bound: 48 ms of GPU time per frame, all of it the upscaler's kernels, each emulated HMMA about
116 Maxwell instructions (32 shuffles, 32 half FMAs that the translator expands further). `OOT_CS_TIME=1` showed the
time spread over seven kernels. HMMA now runs on the host's tensor cores through VK_KHR_cooperative_matrix:

- `--probe-coopmat` found f16 16x8x16 subgroup matrices on the RTX 4070 and measured their per-lane element order:
  HMMA's register fragments, except that A's four registers go in as 0, 2, 1, 3. No data moves between lanes.
- The lowering (`sm86_lower.c lower_hmma`) emits a marker for an unconditional HMMA of a compute program with
  whole-subgroup workgroups; translator patch 0016 decodes it into one IR operation with two results and emits
  `OpCooperativeMatrixMulAddKHR` with element inserts and extracts. `vk_caps.c tensor_cores` turns it on when the
  device has the shape, half floats and 32-lane compute subgroups (`OOT_TENSOR_CORES=0` keeps the emulation); the
  choice is mixed into the cache keys.
- The first full run drew garbage at 60 fps. Bisection (`tools/bisect_kernels.py`, after a first attempt whose
  allow-list never matched) found kernel 5d415995, and the isolated kernel on random data found the cause in the
  translator: its AST optimizer moved single-use memory loads to their use, past later stores, so the kernel read its
  weights after it had reused that shared memory. Patch 0017 keeps memory reads in place. The emulation hid the bug
  by luck of scheduling.
- Result: 4.3 ms of GPU time per frame, 60 fps (the game's cap), the movie on its clock with no skipped pictures,
  and screenshots scored as pictures (`build/logs/check-on.png` against the emulated `check-off.png`).

Open: spirv-val asks for the VulkanMemoryModel capability next to CooperativeMatrixKHR (the driver accepts the
modules; switching the translator's memory model touches every shader). The tools made tonight are in CLAUDE.md
("Running and checking") and the project skill `verify-shader-change`.

## Status (2026-11-06, first play): into a game, then black

The user played from the title into a new game with a Switch Pro Controller (`tools/run.py --seconds 3600`,
build/logs/run-243.log). The title, menu and opening cutscene run at a steady 60 fps. Two problems:

- **The opening's voice-over is silent.** The game is voiced: about 7,700 voice streams in nine languages, an English
  voice bank for 99 cutscenes including `Dm_Kokiri_0000`, the Deku Tree's opening. No audio service was refused
  and the renderer runs. Next: a run with `OOT_TRACE=fs` to see whether the game opens the voice bank and streams
  (if not, suspect the language the host reports, `ov_system.c ov_language`).
- **Gameplay is black.** When the opening hands over to the game, the screen goes black while the game keeps
  presenting frames at 60 fps (no crash, no stall). In the same moment the game starts calling
  `nvnCommandBufferDrawMeshTasks`, which is a stub that draws nothing, and only 51 distinct shaders were seen in the
  whole session. The world is most likely drawn with mesh shaders: the mesh stages (listed as open since the
  start of Phase 3) are now the blocker. Next: a frame dump of a gameplay frame (`tools/run.py --dump`) to confirm
  what is drawn and how much goes through mesh tasks. Reproduced unattended (run-244): A pressed every two seconds
  from the title and `OOT_MOVIE_SKIP=2` (`tools/run.py --skip-movies`) reach the mesh draws in about 25 s, and the
  screenshot is pure black.
- **Faint rectangle outlines in the cinematic** (the user, on a frame filled with blue sky): the upscaler's tiles
  show at their edges. Not yet measured whether the emulated HMMA shows them too (tensor cores round once, the
  emulation per step) or whether a tile border is handled wrongly.

## Status (2026-11-06, past the opening): the level renders, then the GPU times out

The black screen after the opening cutscene was not the mesh stages. The game marks a movie's end with an empty
buffer of flag 8 (flag 4 marks the first sample of a stream); `ov_movie22.c` took 4 for the end, so no movie ever
ended and the game waited for the end-of-stream frame forever (run-248 logged the flag-8 buffer). Past it, the game
aborted in `nn::prepo::AsyncContext::GetCompletionEvent` (a play report's context; `ov_system.c` now completes it at
once with a signalled kernel event attached through the SDK). With both, the Kokiri Forest level renders: about 80
passes, 744 draws and 126 dispatches a frame (run-257), for about ten frames, until a device loss with the GPU
timeout's pattern (one instruction pointer per killed warp: a shader that runs too long, as the upscaler's loop did).

Open, in order:
- (Done, the same night) The GPU timeout on the first level frames. `OOT_SKIP_CS=all` removed it (a compute kernel);
  `OOT_SKIP_CS=!<list>` (run only those) and leaving one out at a time found a pair, eb67c232 (one thread copying
  light data from its uniform block, sampling two textures) and c82d3cba (a volume kernel with counted loops), that
  hung only together. The frame dump armed on eb67c232 (`OOT_DUMP=cs:<id>`) showed its two textures unbound: the
  host placed NVN2's bank-0 resource tables after the pushed constants by the length the game pushed, but the shader's
  layout ends the tables at the u16 at control +0xba (eb67c232 ends at 0x40 with 1 storage and 2 texture slots and
  reads its storage slot at 0x20). With the base from +0xba every compute texture slot of the level frame resolves,
  the level runs (662 frames, A/B against the old rule: device lost after 14) and the Deku Tree's dialogue plays
  (subtitles over a still black scene). Also translator patch 0018: NMin/NMax/NClamp and a saturating, NaN-to-0 F2I,
  as Maxwell defines them (not the cause, but SPIR-V left them undefined).
- The scene behind the dialogue was black for two reasons (2026-11-06, dumps of frame 1500 with
  `OOT_DUMP_SKIP_RUNS=1`; the dump's pass cap is now 1200 and it notes draws and dispatches skipped for a missing
  shader, with the code hash `tools/kernel.py` prints):
  - The last steps of the frame, the 1080p-to-1440p kernel 0a9afab0 and the kernel after it 7034cdb8, did not
    lower; the frame showed only the subtitles drawn over an empty target. The lowering now has SULD (23 level
    kernels), REDUX (27), HSETP2, HMUL2, negated HMNMX2 operands and TLD4 with per-tap offsets and fp16 results.
    With them the level shows: the sky and the forest's shapes (`build/logs/level-up.png`), the geometry still
    mostly black.
  - The volumetric fog: kernel 1a62084a (8x4x4 groups, the 64x36x96 in-scattering volumes) writes only 0 and NaN,
    and the integration along depth and the fog's full-screen pass carry the NaN into every pixel of the scene.
    Its uniform blocks, textures and storage buffers are bound and plausible; the cause is open.
    `OOT_SKIP_CS=1a62084a` leaves the fog out for now.
  - The materials: most fragment shaders take their texture handles from memory (bindless: a handle loaded into a
    register, R2UR, then TEX through a uniform register), which Ryujinx 1.3.2 cannot resolve; they sample the
    placeholder. True bindless (a descriptor array over the texture pool, indexed in the shader) is the next big
    piece.
- 605 programs the lowering refuses in the level (`gfx_shader: translation failed`), so about 280 of 744 draws are
  skipped: unknown opcodes 261, `TEX unpacked operands` 128, scratch and predicate pressure 53, `RED address` 37,
  uniform datapath opcodes 35, and smaller groups (TLD, TLD4, HMNMX2, PLOP3, STG of wide zeros).
- The mesh stages: the level draws with `DrawMeshTasksIndirect` too (stubs).
- The movie start stutters for about 8 s (every other frame ~160 ms while the GPU needs 5-13 ms: CPU side), and the
  cutscene's audio stream runs 10-20 s off the picture (the user); the opening's voice lines are silent.
- (Done) `OOT_MOVIE_SKIP`: the cutscene follows the time stamps of the pictures it shows, not the movie's end alone
  nor its audio stream (the user: the cut to the game came before the lagging audio ended). The skip now hands back,
  once the game has sent its end buffer, one black picture stamped with the last sample's time, then the end; the
  quick route (`tools/run.py --replay build/logs/a-spam.txt --skip-movies`) reaches the level in about 75 s.
  The game has no button to skip the opening (the user).

## Where the port stands

`oot.exe --seconds 15` runs the whole start-up of OoT 1.1.1 without a crash: 82 threads, 10,988 IPC messages, 1,485
file opens, saves mounted, NPLN offline, the audio renderer at 48 kHz, 3,661 shader programs set up through NVN2's
program builder, three frames presented at 2560x1440 (`build/logs/run-45.log`). Then it stalls in the third frame
(51.7 s) because no shader translates:

```
[host] gfx_shader: translation failed (-): bad control section
[host] vk: a stage 0 shader of program 4136 failed to translate: draws with it are skipped
```

Draws with failed programs are skipped, the game waits on GPU work that never completes, nothing is drawn. Everything
below the graphics API is believed to work; every remaining known problem is in this one area or is minor (see "After
the shaders").

## Why every shader fails

The host's shader path (`runtime/gfx_shader.c`, `runtime/vk_exec_draw.c`, `tools/xlate/`) is TotK's, for Switch 1:

- `gfx_shader.c` `req_init()` reads the c1 offset at `control+0x700` and the code length at `control+0x704`, and
  `CONTROL_SIZE` is `0x880`: the cache key, the `.in` files, `build_code_hash()` and the async job copies all take
  0x880 bytes from the control pointer. `vk_dump.c:585` reads `+0x700` too.
- The NVN2 control section is 224, 248 or 280 bytes (`0xe0`, `0xf8`, `0x118`). Reading `+0x700` reads whatever lies
  after it, so the length check fails and every program is rejected before translation.
- The code is Ampere SM86 (the Switch 2's T239 GPU): 128-bit instructions, opcode in the low 12 bits of word 0, no
  Maxwell scheduling words. The vendored translator (`tools/vendor/ryujinx`, Ryujinx 1.3.2, MIT) decodes Maxwell and
  Pascal 64-bit forms only (`src/Ryujinx.Graphics.Shader/Decoders/InstDecoders.cs`).
- The translator itself is not built here yet: the log says `translator identity 0000000000000000 (none: no
  translator)`. `.NET 10.0.403` is installed; `python tools/build_xlate.py` (Windows venv) builds
  `build/xlate/oot_xlate.exe`. MP4's `mp4_xlate.exe` next door is the same source and would do for a first smoke test.

So there are two separate jobs: the control-section layout (small, data-driven) and an SM86 front end (large).

## What is measured about the NVN2 control section

From the decompressed `Shader/Model.Product.110.product.Nin_Ounce_NVN2.bfsha` (320 MB; zstd with the `zs.zsdic`
dictionary from `Pack/ZsDic.pack.zs`): 8,517 blocks found by the magic `0x98761234`, 20,000 of them surveyed by
per-dword variance. The first 0xe0 bytes are the fixed part; the block's own size sits at `+0x30` and is repeated at
`+0x98` and `+0xd4`; the tail after `+0xe0` is 0, 0x18 or 0x38 bytes of extra records.

| Offset | Seen | Reading |
|---|---|---|
| `+0x00` | `0x98761234` | magic, as on Switch 1 |
| `+0x04`, `+0x08`, `+0x0c` | `0xc9`, `0x14`, `0xe0170` in every block | constants (version words) |
| `+0x18`, `+0x1c` | `0x220014`, `0x288040` in every block | constants (SDK 22.x / NVN2 version?) |
| `+0x20..+0x27` | unique per block | a 64-bit hash |
| `+0x28..+0x2f` | 7,832 distinct over 8,517 | a second 64-bit hash, shared by a few blocks |
| `+0x30` | `0xf8` (5,900), `0xe0` (2,177), `0x118` (315) | **control section size**; repeated at `+0x98`, `+0xd4` |
| `+0x38` | 402 values, `0x6e80..0x7000` typical | code size candidate (multiple of 0x80) |
| `+0x3c` | `0x14` (5,900), `0` (1,906), `0x20` (394) | tracks `+0x30`; `+0xdc` has the same classes |
| `+0x40`, `+0x44` | rise with `+0x38` (`0x9e80, 0x9f00, 0xa000`) | sizes with headers, or offsets of c1 and end |
| `+0x4c` | 150 values, `0x5a..0x60` common | register count or similar small per-shader number |
| `+0x50..+0x60` | small flags, `0x10000` common | stage/feature flags |
| `+0x64..+0x70` | `0x3f800000` in a few blocks | floats (1.0f) |
| `+0xa4` | `0x1000001` or `0x1000101` | flags |
| `+0xbc`, `+0xc4`, `+0xc8`, `+0xd8` | `0xe0` always | size of the fixed part |

The survey script is a few lines: find each magic, measure to the next, tabulate each dword. Not yet known and needed
first: which fields are the **code length**, the **c1 offset** (the constant bank the translator needs for `BRX`
jump tables) and the **stage**, and whether an early-fragment-tests flag exists (Switch 1: `control[0x718] & 1`).
Two ways to pin them, both cheap:

1. The archive's own tables. BNSH's shader code section records per program the code and control offsets and sizes
   (TotK's `refs/totk-pc-recomp/tools/shader_census.py`, from MacNativeGames' `prepare_shader_seed.py`, MIT, parses
   Switch 1 BNSH; the container is unchanged here). Match those sizes against `+0x38`/`+0x40`/`+0x44`.
2. The game at run time. `prog_set_shaders()` (`nvn_api_obj.c`) logs `data0`; add a once-only dump of the 0x120 bytes
   behind `control` and the size of the pool mapping behind `data`, and compare with the bytes that follow the code
   (the program's own end is where the code size says).

Note the game hands over the Switch 1 `NVNshaderData` form, `{data, control}` of 16 bytes
(`nvnProgramBuilderSetShaders`, built on the stack at `main+0x14e2468`), so the host side needs no ABI change.

## What is known about the code

- Every stage ends in a 128-bit `EXIT`, opcode `0x94d`; none has Maxwell's. Instruction words are 16 bytes, the opcode
  is `word0 & 0xfff` (Lichii's `instruction.h` uses exactly this `Bits(0, 12)`).
- Phase 0 measured a 0x60-byte program header before the instructions, against Maxwell's 0x50-byte SPH. Lichii's
  environment reads `sizeof(sph)` (0x50) from the program base for Switch 2 titles too, so settle the header size on
  the corpus before anything else: the first instruction is where a plausible opcode stream begins.
- The translator job format (`tools/xlate/Program.cs`) is "graphics: 0x50-byte SPH + instructions, compute:
  instructions"; `gfx_shader.c` skips `CODE_SKIP 0x30` of NVN code-block header before that for graphics stages.
  Both constants must be checked against the NVN2 layout.

## Approach: lower SM86 to Maxwell in front of the existing translator

Recommended. Lichii (`src/shader_recompiler/frontend/ampere/lower_to_maxwell.h`, GPL-3.0, read for behaviour only) runs
four Switch 2 titles this way: it does not reinterpret the bytes, it emits new Maxwell instructions for each SM86 one,
inserts the scheduling word every three instructions, maps the uniform registers (`UR0..`) onto high general registers
(R216 and up), keeps an address map so branch targets and `BRX` tables survive the re-layout, and hands the result to
the unchanged yuzu Maxwell front end. The alternative, a native SM86 decoder inside Ryujinx's translator, would mean
new decoder structs and emitter signatures for every instruction (Ryujinx's emitters are bound to the 64-bit field
layout), many times the work for the same output.

Ours would be a C module with no dependency on the translator, usable both in `runtime/gfx_shader.c` (lower, then
request the translation of the Maxwell code; cache keys stay over the original bytes) and in an offline tool over the
whole corpus. Sources of truth that may be copied, with a header and a `THIRD_PARTY_NOTICES.md` row:

- Mesa's NAK compiler, `src/nouveau/compiler/nak/sm70_encode.rs` (MIT): the Volta/Turing/Ampere encoding, field by
  field, and the reference Lichii itself names for its encoding knowledge.
- Ryujinx's `Decoders/InstDecoders.cs` and `InstTable.cs` (MIT, already vendored): the Maxwell target encoding.

From Lichii, only behaviour: which SM86 forms Nintendo's compiler emits for these engines, the pitfalls it records in
`NX2_NOTES.md` (row 54: widen until the whole dump lowers; row 93: save and restore P6 around uniform guards; row 121:
translation time), and its test cases under `src/tests/video_core/ampere_*.cpp` as a checklist of forms (uniform
datapath `R2UR`/`UP2UR`/`ULDC`, `LDSM`, `IPA` modes, formatted surfaces, `CALL.REL.NOINC`, 64-bit `UIADD3` carry,
`TMML`, attribute `0x5c`).

## Order of work

1. **Build the translator** (`python tools/build_xlate.py` from the Windows venv), so a lowered shader can be checked
   at once. Confirm with MP4's census shaders if needed.
2. **Corpus tool**: port `refs/totk-pc-recomp/tools/shader_census.py` to the NVN2 archives (`Model.bfsha`, the
   `Layout2Shader`, `ApplicationPackage` sarc, `.vsetb`, fonts; `tools/shader_inventory.py` is MP4's Retro-format
   tool and is replaced, not extended). Output under `build/v131072/shaders/`: control and code per program, plus an
   opcode histogram (`word0 & 0xfff`) over all code. The histogram is the work list for step 4.
3. **Control section** in `gfx_shader.c`: one `ctl_parse()` giving size, code length, c1 offset, stage and flags, used
   by `req_init`, `build_code_hash`, the `.in` writer/reader, the async job copies and `vk_dump.c`; `CONTROL_SIZE`
   becomes the block's own size. Bump `KEY_VERSION`/`IN_VERSION` (the formats change). The `--probe-gfx`-style census
   self-test in `vk_pipeline.c` (`build/v121/shaders`) and `gfx_shader_probe.c` need the new corpus path.
4. **SM86 lowering**, corpus-driven: implement by opcode frequency, with the offline tool reporting the share of
   programs that lower and translate; stop at every unknown form rather than guessing. Hard parts in order: the
   address map (sched words every three instructions change every branch target; `BRX` tables live in c1), the
   uniform datapath, predicates and the P6 guard, then texture and surface forms.
5. **Run the game** with `OOT_TRACE=gfx`. Expect the first pixels, then new failures: NVN2 procs still stubbed
   (`nvn_procs_stub.inc` logs the first three calls of each; 160 return 0, only three were reached so far), state
   objects whose NVN2 layout differs, and whatever frame 3 was waiting for.

## After the shaders

Known and parked, none blocking: the C++ unwinder reads past `subsdk0` into the gap before `sdk` (`QueryMemory`
should report it free); `--probe-ipc`'s sdk heap constructor address is MP4's (`ipc.c` `ctors_143`); the
`--probe-services` override list; installer tables (`installer/layout.py`, `tools/make_release.py`,
`tools/prepare_shaders.py`); `tests/diff/func.py` and `tests/jumptables_check.py` not run; MP4's stale `fnprof_hit`
stub and `"\mp4"` escape to report upstream.

## Working facts

- Build: `/mnt/c/dev/metroid-prime-4-recomp/.venv-win/Scripts/python.exe tools/build_game.py` (257 s), output
  `build/v131072/oot.exe`. Lift: `recomp/gen.py --exefs build/extracted/v131072/exefs --out generated/v131072
  --modules main,subsdk0,sdk`.
- Run, always with WSLENV or every `OOT_*` variable is silently dropped:
  `export WSLENV=OOT_TRACE:OOT_VOLUME:OOT_APP_POOL:OOT_CRASH_PEEK:OOT_BREAK_AT:OOT_BREAK_BRIEF:OOT_BREAK_WATCH;
  OOT_VOLUME=0 OOT_TRACE=gfx build/v131072/oot.exe --seconds 15 > build/logs/run-N.log 2>&1`.
- Debug: `OOT_BREAK_AT=main+0x...`, `OOT_BREAK_WATCH`, `OOT_CRASH_PEEK`, `build/logs/nvn-proc-got.json` (GOT slot to
  proc name), `OOT_TRACE=svc,host,ipc,fs,gfx`.
- Extraction: `python3.14 tools/extract_nup.py --verify` (needs `toolchain/zbic/nso_unzbic` from `tools/build_zbic.py`).
- Lichii is at `/mnt/c/dev/Lichii` (GPL-3.0): `NX2_NOTES.md`, `src/shader_recompiler/frontend/ampere/`,
  `src/video_core/shader_environment.cpp`. Behaviour only, never code.
- Nothing from the game is committed; everything extracted or decompressed stays under `build/` (the decompressed
  bfsha of this session lived in the scratchpad and is gone; regenerate it from the RomFS image).

## Status (2026-11-06, lowering refusals of the first level)

A play session to the first level logged 192 refused draws, which were 33 distinct programs (`.fail.ctl` / `.fail.code`
files in the shader cache; `tools/sm86_dis.py` reads the code). Seven of them are not in the RomFS corpus the census
reads (a 3D fragment shader, two fragment shaders with IPA 0x526, four compute programs). Changes, each checked by
lowering, translating and `spirv-val` offline (no game run):

- TEX with bit 60 clear (128 of the 192 draws, one program): Ra holds the coordinates, Rb the LOD then the packed
  offsets (x, y, z in nibbles), as for TEX.B. Seen only on 3D fetches with LOD and offsets; other combinations are
  assumed to follow the same rule.
- IPA 0x526: Ra is a uniform register (UR4, always 0 there), ignored as the offset-free modes ignore Ra.
- The skinning vertex shaders (660 corpus programs: up to 247 live registers, no run of four free): a wide LDG with a
  uniform base keeps the base in its own destination, IMAD.WIDE writes Rd, Rd+1 directly or reuses the multiplier's
  scratch. Checked by interpreting old and new lowerings of random LDG / IMAD.WIDE sequences over symbolic registers
  (0 differences in 4000 programs; a deliberate bug is caught).
- RED with a forwarded uniform pair aligns its scratch pair; PLOP3 tables are matched only on the rows consistent with
  PT / !PT inputs; STG of wide zeros stores from zeroed scratch.
- Corpus: 14,306 of 14,363 lower (was 13,569 after the previous commit), every newly lowered vs/ps/cs program
  translates and validates. Left: mesh and task stages (12), UPLOP3 (13), PLOP3 reading a swapped predicate (10), the
  uniform-predicate budget of 10 vertex shaders, FCHK, IABS with an immediate, one TLD.

Update (2026-11-06, the Great Deku Tree scene): a real session (build/logs/run-51.log) had 24 distinct programs refused,
all compute kernels but one vertex shader. All 24 now lower, translate and pass `spirv-val` offline. What did it:
UPLOP3 (through the PLOP3 lowering); a PLOP3 whose real input is the predicate a uniform one is swapped into (copied to a
free predicate first); PLOP3 tables in any input order and through PSETP's second destination; a retry with the uniform
predicates swapped through a register when the free predicates run out; opcode 0x243 (F2IP.U8.F32, clamp and pack two
floats into bytes; the rounding numbering is an assumption), IABS from a constant, LEA with a subtracted addend, TLD with
texel offsets (2D: x and y in Ra, Rb the LOD then the offsets; seen on 40 fetches), and SURED (0x59f). The register that
holds the uniform predicates could not be register 0 before. Corpus: 14,332 of 14,363 lower; left are the mesh and
task stages (12), six compute programs stopped by one more opcode, ten vertex shaders short of a register for the
uniform predicates and three short of a uniform register pair.
