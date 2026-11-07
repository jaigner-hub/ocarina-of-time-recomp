#!/usr/bin/env python3
# Written for ocarina-of-time-recomp (MIT, Copyright (c) 2026 Jeff Aigner).
"""Run oot.exe from WSL for a test: the environment passed through, the log numbered, the game killed afterwards.

    python3 tools/run.py [--seconds 60] [--shot PNG] [--press-a] [--env OOT_X=1 ...] [--fresh-cache NAME]
                         [--dump FRAME] [--trace gfx] [--volume 60] [--retry 3] [--expect REGEX ...] [-- extra args]

What it takes care of, each a mistake made by hand at least once:
  - Every OOT_* variable in the child's environment is named in WSLENV (path-valued ones with /p). oot.exe is a
    Windows program and silently ignores the rest.
  - oot.exe and oot_xlate.exe are killed after every run (the exit path can hang), and a run refuses to start while
    one is alive (two runs share the shader cache and the window; --kill-existing kills it first).
  - The start-up crash (an SDK abort in MultiWait, about one run in two, docs/phase3.md) is retried: a run that exits
    non-zero within its first 25 s is run again, up to --retry times. A later crash is reported, not retried.
  - The window keeps focus as far as the game knows (OOT_ASSUME_FOCUS=1): out of focus the game pauses itself, as on
    the console under the HOME menu. --no-focus-pin to test that.
  - --press-a presses A at frame 600, which brings up the title menu over the cinematic. --record FILE writes what
    the player does (pad and keyboard) as an input script, --replay FILE plays one back; --skip-movies ends every
    movie after the title loop at once (OOT_MOVIE_SKIP=2), so a replay reaches the game without the opening.
  - --fresh-cache NAME translates into an empty build/logs/sc-NAME, for runs whose lowering differs in ways the
    cache keys do not show (OOT_TENSOR_ONLY, OOT_TENSOR_HMMA_MASK).
  - --expect REGEX checks the log for each pattern afterwards (did the switch apply?) and fails the run if one is
    missing; a diagnostic that never applied cost a whole bisection on 2026-11-06.
Logs: build/logs/run-<n>.log; --dump writes build/logs/dump-<n>/. Prints a summary of the log (frame rate, GPU time,
movie pacing, tensor cores, crashes) and exits non-zero when the run failed or an --expect was not met.
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Windows: oot.exe, also when this script runs in WSL, where it starts the program through interop (WSLENV carries the
# OOT_* variables). Native Linux: the Linux binary.
IS_WSL = "microsoft" in platform.uname().release.lower()
WIN = sys.platform == "win32" or IS_WSL
EXE = ROOT / "build" / "v131072" / ("oot.exe" if WIN else "oot")
LOGS = ROOT / "build" / "logs"
PATH_VARS = {"OOT_SHADER_CACHE", "OOT_INPUT_SCRIPT", "OOT_INPUT_RECORD", "OOT_DUMP_DIR", "OOT_XLATE", "OOT_PROFILE_DIR", "OOT_SHOTS_DIR",
             "OOT_FLIGHT_DIR", "OOT_RDOC_DIR"}
PROCS = ("oot.exe", "oot_xlate.exe") if WIN else ("oot", "oot_xlate")


def running() -> list[str]:
    if not WIN:
        return [p for p in PROCS if subprocess.run(["pgrep", "-x", p], capture_output=True).returncode == 0]
    out = subprocess.run(["tasklist.exe"], capture_output=True, text=True).stdout.lower()
    return [p for p in PROCS if p in out]


def kill_all() -> None:
    for p in PROCS:
        if WIN:
            subprocess.run(["taskkill.exe", "/F", "/IM", p, "/T"], capture_output=True)
        else:
            subprocess.run(["pkill", "-KILL", "-x", p], capture_output=True)
    for _ in range(20):
        if not running():
            return
        time.sleep(0.25)


def next_log() -> tuple[int, Path]:
    LOGS.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for p in LOGS.glob("run-*.log") if (m := re.fullmatch(r"run-(\d+)\.log", p.name))]
    n = max(nums, default=0) + 1
    return n, LOGS / f"run-{n}.log"


def wslenv(env: dict) -> str:
    names = [k for k in env if k.startswith("OOT_")]
    keep = [w for w in env.get("WSLENV", "").split(":") if w and w.split("/")[0] not in names]
    return ":".join(keep + [k + ("/p" if k in PATH_VARS else "") for k in sorted(names)])


def summary(log: Path) -> list[str]:
    text = log.read_text(errors="replace") if log.exists() else ""
    out = []
    pick = [r"vk caps: tensor cores.*", r"frametime: 60 s:.*", r"exception 0x[0-9a-f]+.*", r"FATAL.*",
            r"device lost.*", r"boot: frames \d+ presented.*", r"OOT_TENSOR_ONLY: program .*"]
    for pat in pick:
        hits = re.findall(r"^(?:\[host\] )?(" + pat + ")$", text, re.M)
        out += hits[:3] if pat.startswith("frametime") or pat.startswith("OOT_TENSOR") else hits[:1]
    for pat in (r"movie22: pacing:.*", r"perf: frame \d+: gpu avg [\d.]+ ms"):
        hits = re.findall(r"(" + pat + ")", text)
        if hits:
            out.append(hits[-1])
    return out


def run_game(seconds: int = 60, shot: str | None = None, env: dict | None = None, press_a: bool = False,
             fresh_cache: str | None = None, dump: int | None = None, trace: str | None = None, volume: int = 0,
             focus_pin: bool = True, retry: int = 3, expect: list[str] | None = None, extra: list[str] | None = None,
             kill_existing: bool = False, quiet: bool = False, record: str | None = None, replay: str | None = None,
             skip_movies: bool = False) -> dict:
    """Runs the game once (retrying the start-up crash); returns {ok, log, shot, exit, summary, missing}."""
    if not EXE.exists():
        sys.exit(f"run: {EXE} is not built (tools/build_game.py)")
    alive = running()
    if alive:
        if not kill_existing:
            sys.exit(f"run: {', '.join(alive)} is running; refusing to overlap runs (--kill-existing kills it)")
        kill_all()
    child = dict(os.environ)
    child.setdefault("OOT_VOLUME", str(volume))
    child["OOT_VOLUME"] = str(volume)
    if focus_pin:
        child["OOT_ASSUME_FOCUS"] = "1"
    if trace:
        child["OOT_TRACE"] = trace
    if press_a:
        script = LOGS / "press-a.txt"
        LOGS.mkdir(parents=True, exist_ok=True)
        script.write_text("600 A 10\n")
        child["OOT_INPUT_SCRIPT"] = str(script.relative_to(ROOT))
    if record:
        child["OOT_INPUT_RECORD"] = record
    if replay:
        if not (ROOT / replay).exists() and not Path(replay).exists():
            sys.exit(f"run: no input script {replay}")
        child["OOT_INPUT_SCRIPT"] = replay
    if skip_movies:
        child["OOT_MOVIE_SKIP"] = "2"
    for kv in env or {}:
        child[kv] = str(env[kv])
    result = {"ok": False}
    for attempt in range(1, max(1, retry) + 1):
        n, log = next_log()
        if fresh_cache:
            cache = LOGS / f"sc-{fresh_cache}"
            shutil.rmtree(cache, ignore_errors=True)
            cache.mkdir(parents=True)
            child["OOT_SHADER_CACHE"] = str(cache.relative_to(ROOT))
        if dump is not None:
            child["OOT_DUMP"] = str(dump)
            child["OOT_DUMP_DIR"] = f"build/logs/dump-{n}"
        if WIN:
            child["WSLENV"] = wslenv(child)
        shot_path = None
        if shot:
            shot_path = Path(shot) if Path(shot).is_absolute() else ROOT / shot
            shot_path.unlink(missing_ok=True)
        args = [str(EXE.relative_to(ROOT)), "--seconds", str(seconds)]
        if shot_path:
            args += ["--shot", str(shot_path.relative_to(ROOT))]
        args += extra or []
        if not quiet:
            print(f"run: run-{n} (attempt {attempt}), {seconds} s" + (f", shot {shot}" if shot else ""), flush=True)
        t0 = time.monotonic()
        try:
            with open(log, "w") as f:
                rc = subprocess.run(args, cwd=ROOT, env=child, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=seconds + 240).returncode
        except subprocess.TimeoutExpired:
            rc = -1
        finally:
            kill_all()
        took = time.monotonic() - t0
        left = running()
        with open(log, "a") as f:
            f.write(f"exit {rc}\nprocesses left: {len(left)}\n")
        # The start-up crash comes within the first frames; a crash later is a result, not a reason to run again.
        early = rc != 0 and took < min(seconds, 25)
        missing_shot = bool(shot_path) and not shot_path.exists()
        result = {"ok": not early and not missing_shot and rc in (0,), "log": str(log), "exit": rc,
                  "shot": str(shot_path) if shot_path and shot_path.exists() else None, "seconds_run": round(took),
                  "summary": summary(log), "run": n}
        if result["ok"] or not early:
            break
        if not quiet:
            print(f"run: run-{n} ended early (exit {rc} after {took:.0f} s{', no shot' if missing_shot else ''}); "
                  "retrying", flush=True)
    text = Path(result["log"]).read_text(errors="replace")
    result["missing"] = [p for p in expect or [] if not re.search(p, text, re.M)]
    if result["missing"]:
        result["ok"] = False
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[2:]))
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--shot", help="PNG path (relative to the repo) for a screenshot at the end; builds pipelines "
                                   "synchronously, so the frame does not depend on build timing")
    ap.add_argument("--press-a", action="store_true", help="press A at frame 600 (the title menu)")
    ap.add_argument("--env", action="append", default=[], metavar="OOT_X=V")
    ap.add_argument("--fresh-cache", metavar="NAME")
    ap.add_argument("--dump", type=int, metavar="FRAME", help="frame dump into build/logs/dump-<run>/")
    ap.add_argument("--trace", help="OOT_TRACE value (gfx, perf, ...)")
    ap.add_argument("--volume", type=int, default=0)
    ap.add_argument("--no-focus-pin", action="store_true")
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--expect", action="append", default=[], metavar="REGEX")
    ap.add_argument("--kill-existing", action="store_true")
    ap.add_argument("--record", metavar="FILE", help="write the player's input as a script (relative to the repo)")
    ap.add_argument("--replay", metavar="FILE", help="play an input script back (from --record or by hand)")
    ap.add_argument("--skip-movies", action="store_true", help="end every movie after the title loop at once")
    ap.add_argument("extra", nargs="*", help="arguments for oot.exe after --")
    a = ap.parse_args()
    env = {}
    for kv in a.env:
        k, _, v = kv.partition("=")
        if not k.startswith("OOT_"):
            sys.exit(f"run: --env {kv}: only OOT_* variables reach the game")
        env[k] = v
    r = run_game(a.seconds, a.shot, env, a.press_a, a.fresh_cache, a.dump, a.trace, a.volume, not a.no_focus_pin,
                 a.retry, a.expect, a.extra, a.kill_existing, record=a.record, replay=a.replay,
                 skip_movies=a.skip_movies)
    print(f"run: {'ok' if r['ok'] else 'FAILED'}: {Path(r['log']).relative_to(ROOT)}, exit {r['exit']}, "
          f"{r.get('seconds_run')} s" + (f", shot {r['shot']}" if r.get("shot") else ""))
    for line in r["summary"]:
        print("  " + line[:200])
    for p in r["missing"]:
        print(f"  EXPECTED BUT MISSING: {p}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
