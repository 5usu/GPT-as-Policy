"""Terminal driver for bimanual YAM with robocurve/pi0.5-yam.

    python -m hybrid_rollout.robodojo.yam.cli preflight [--pi05-url ...]
    python -m hybrid_rollout.robodojo.yam.cli phase1 --episode DIR
    python -m hybrid_rollout.robodojo.yam.cli run    --episode DIR (--chunks-file F | --pi05-url U)
    python -m hybrid_rollout.robodojo.yam.cli hold   --hold
    python -m hybrid_rollout.robodojo.yam.cli live   --hold --pi05-url U --cameras top:4,left:0,right:2
    python -m hybrid_rollout.robodojo.yam.cli doctor | set | cameras | calibrate-grippers
                                               | record-pose | tracking-test
    python -m hybrid_rollout.robodojo.yam.cli execute --execute --pi05-url U

The same staging as the KUKA branch's cli, and the same defaults: nothing is
sent to Astra without --astra-live and the Qwen monitor records without gating
unless --monitor-gate. `hold`, `live` and `record-pose` connect to the arms --
which enables the motors -- and refuse without --hold. Only `execute`,
`tracking-test` and `calibrate-grippers` move them; each refuses without
--execute AND an operator typing the rig id at this terminal.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from dataclasses import replace as _dc_replace
from pathlib import Path

from .contract import (CAMERA_NAMES, CHECKPOINT, CHUNK_STEPS, ROBOT_MODEL,
                       eef_execution_gate)
from .experiment import DEFAULT_EXPERIMENT, flatten_config, load_config, stop_conditions
from .safety import ASTRA_DIRECT_EXTRA, REQUIRED_CONFIG, Mode, missing_config

ASTRA_URL = "https://api.openai.com/v1/responses"
#: Every default output lands here (git-ignored), never in the repo root.
RUNS = Path(__file__).parent / "runs"


def _out(name: str) -> str:
    RUNS.mkdir(parents=True, exist_ok=True)
    return str(RUNS / name)


def _banner(title: str, mode: Mode, cfg_missing: list[str]) -> None:
    print(f"YAM BIMANUAL -- {title}")
    print(f"  robot     : {ROBOT_MODEL}")
    print(f"  policy    : {CHECKPOINT['repo_id']}@{CHECKPOINT['revision'][:8]}")
    print(f"  mode      : {mode.value}")
    print(f"  config    : {len(cfg_missing)} required value(s) still unmeasured")
    ok, missing, _ = eef_execution_gate()
    print(f"  eef       : {'ENABLED' if ok else 'accepted but REFUSED'} "
          f"({len(missing)} prerequisite(s) missing)\n")


def _episode(args):
    from .episode import YamRecordedEpisode
    d = Path(args.episode)
    return YamRecordedEpisode.open(
        d / "manifest.json", media_root=d,
        state_rows=json.loads((d / "state.json").read_text()),
        traj_rows=json.loads((d / "trajectory.json").read_text()))


def _get_json(url: str, timeout: float = 3.0) -> dict:
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _health_url(infer_url: str) -> str:
    return infer_url.rsplit("/", 1)[0] + "/health"


# ------------------------------------------------------------------ preflight
def cmd_preflight(args) -> int:
    from .loop import EDIT_EXECUTION_ENABLED, EXECUTABLE_DECISION_MODES
    from .transports import check_checkpoint_meta
    cfg = load_config(args.experiment)
    flat = flatten_config(cfg)
    print(f"PREFLIGHT -- experiment {args.experiment!r}\n")
    print("CHECKPOINT CONTRACT")
    for k in ("repo_id", "revision", "action_horizon", "action_dim", "action_space",
              "gripper_convention", "cameras", "norm_stats_sha256"):
        print(f"  {k:20s} {CHECKPOINT[k]}")
    print(f"  {'validation':20s} {CHECKPOINT['validation']}\n")
    if args.pi05_url:
        try:
            h = _get_json(_health_url(args.pi05_url))
            bad = check_checkpoint_meta(h.get("meta"))
            print(f"pi0.5 server : {'CONTRACT OK' if bad is None else 'MISMATCH'}"
                  f" -- {bad or h['meta'].get('checkpoint')}")
        except Exception as exc:                             # noqa: BLE001
            print(f"pi0.5 server : UNREACHABLE ({type(exc).__name__}: {exc})")
    if args.monitor_url or args.monitor_backend != "unconfigured":
        from ..kuka.vlm_backends import VlmConfig, make_backend
        vcfg = VlmConfig.jetson()
        if args.monitor_url:
            vcfg = _dc_replace(vcfg, endpoint=args.monitor_url)
        pr = make_backend(args.monitor_backend, config=vcfg).probe()
        print(f"monitor      : {'reachable' if pr.available else 'UNAVAILABLE'} -- "
              f"{pr.reason[:80]}")
    print("\nEXECUTION LOCKS")
    print(f"  emittable modes  : {sorted(EXECUTABLE_DECISION_MODES)}")
    print(f"  edit execution   : {EDIT_EXECUTION_ENABLED}")
    print("  review loop      : ShadowGateway only (the single-step envelope path)")
    print("  continuous motion: `execute` only -- --execute + operator arming + "
          "every EXECUTE_REQUIRED value measured\n")
    for mode in (Mode.REVIEWED_EXECUTION, Mode.ASTRA_DIRECT):
        miss = missing_config(flat, mode)
        print(f"{mode.value}: {len(miss)} missing")
        for k in miss:
            print(f"    {k:34s} {(REQUIRED_CONFIG.get(k) or ASTRA_DIRECT_EXTRA.get(k, ''))[:70]}")
        print()
    print(f"stop conditions: {stop_conditions(cfg)}")
    return 0


# --------------------------------------------------------------------- phase1
def cmd_phase1(args) -> int:
    from .kinematics import make_fk
    from .packet import build_packet
    cfg = load_config(args.experiment)
    _banner("PHASE 1: recorded trajectory review", Mode.REPLAY,
            missing_config(flatten_config(cfg), Mode.REVIEWED_EXECUTION))
    ep = _episode(args)
    fk = make_fk()
    samples = ep.sample_ticks(every=args.every, limit=args.limit)
    out = Path(args.out or _out("phase1_packets.jsonl"))
    with out.open("w") as f:
        for s in samples:
            pkt = build_packet(
                task_instruction=ep.m.instruction, observation_id=s.observation_id,
                state=s.state, chunk=s.recorded_chunk, provenance="recorded_demo",
                frames=s.frames, fk_preview=fk.preview(s.recorded_chunk).to_log())
            f.write(json.dumps(pkt) + "\n")
            print(f"  {s.observation_id:30s} chunk={len(s.recorded_chunk)} "
                  f"frames={sorted(s.frames)}")
    print(f"\n  {len(samples)} review packet(s) -> {out}. NOTHING SENT.")
    return 0


# ------------------------------------------------------------ shared builders
def _astra(args):
    from .transports import YamAstraReviewSource
    return YamAstraReviewSource(
        base_url=args.astra_url, model=args.astra_model,
        api_key_env=args.astra_key_env, enabled=args.astra_live,
        dry_run=not args.astra_live, reasoning=args.astra_effort,
        background=args.astra_background, deadline_s=args.astra_deadline)


def _monitor_backend(args):
    from ..kuka.vlm_backends import VlmConfig, make_backend
    vcfg = VlmConfig.jetson(timeout_s=args.monitor_timeout)
    if args.monitor_url:
        vcfg = _dc_replace(vcfg, endpoint=args.monitor_url)
    if args.monitor_frames:
        vcfg = _dc_replace(vcfg, max_frames=args.monitor_frames)
    return make_backend(args.monitor_backend, config=vcfg)


def _escalator(review, counter: dict):
    def escalate(packet):
        counter["n"] += 1
        print(f"      ESCALATING to Astra (#{counter['n']})")
        return review.review(packet)
    return escalate


def _reference(args):
    if not getattr(args, "task_reference", None):
        return None
    from .task_reference import load
    ref = load(args.task_reference)
    print(f"  reference    : {ref.episode_id} ({len(ref.phases)} phases), "
          f"verified by {ref.verified_by}")
    return ref


# ------------------------------------------------------------------------ run
def cmd_run(args) -> int:
    """Recorded episode through pi0.5 -> Qwen -> Astra -> sanitize -> validate."""
    from .kinematics import make_fk
    from .loop import AuditLog, YamReviewLoop
    from .pipeline import (YamPolicyPipeline, describe_trajectory, resolve_mode,
                           state_text)
    from .transports import Pi05ChunkReplay, Pi05HttpProposalSource, ShadowGateway
    cfg = load_config(args.experiment)
    flat = flatten_config(cfg)
    _banner("RECORDED LOOP: pi0.5 + Qwen + Astra", Mode.LIVE_SHADOW,
            missing_config(flat, Mode.REVIEWED_EXECUTION))
    if args.chunks_file:
        src = Pi05ChunkReplay.from_file(args.chunks_file)
        print(f"  proposals    : {args.chunks_file} ({len(src.chunks)} chunk(s))")
    elif args.pi05_url:
        src = Pi05HttpProposalSource(args.pi05_url, timeout_s=args.pi05_timeout)
        print(f"  proposals    : LIVE inference via {args.pi05_url}")
    else:
        print("  REFUSED: pass --chunks-file (from yam.make_chunks) or --pi05-url "
              "(a running yam.pi05_serve).", file=sys.stderr)
        return 2

    policy_mode = resolve_mode(args.policy_mode)
    review = _astra(args)
    ok, why = review.preflight()
    print(f"  astra        : {'LIVE (PAID)' if args.astra_live else 'DRY RUN'} -- {why}")
    if args.astra_live and not ok:
        print(f"  REFUSED: {why}", file=sys.stderr)
        return 2
    ref = _reference(args)
    gate, escalations = None, {"n": 0}
    if policy_mode.value != "pi05_only":
        backend = _monitor_backend(args)
        probe = backend.probe()
        print(f"  monitor      : {policy_mode.value} via {backend.name} "
              f"({getattr(backend, 'model', '?')}) -- "
              f"{'reachable' if probe.available else 'UNAVAILABLE'}")
        if not probe.available and not args.monitor_shadow:
            print("  REFUSED: gating requested but no monitor is reachable.",
                  file=sys.stderr)
            return 2
        mon_f = Path(args.monitor_audit or _out("monitor.jsonl")).open("a")
        gate = YamPolicyPipeline(
            policy_mode, backend=backend, shadow=args.monitor_shadow,
            task_reference=ref,
            on_record=lambda row: (mon_f.write(json.dumps(row, default=str) + "\n"),
                                   mon_f.flush()))
        gate.astra_review = (None if policy_mode.value == "pi05_local_monitor"
                             else _escalator(review, escalations))
        print(f"  monitor mode : {'SHADOW (records only)' if gate.shadow else 'GATING'}")
    astra_per_tick = gate is None

    ep = _episode(args)
    samples = ep.sample_ticks(every=args.every, limit=args.limit)
    missing = [f"{s.observation_id}/{c}" for s in samples for c in CAMERA_NAMES
               if not (s.frames.get(c) or {}).get("frame_present")]
    if missing:
        print(f"  REFUSED: {len(missing)} frame(s) not extracted, e.g. {missing[0]}",
              file=sys.stderr)
        return 2
    audit = AuditLog(args.audit or _out("run_audit.jsonl"))
    loop = YamReviewLoop(
        mode=Mode.LIVE_SHADOW, config=flat, raw_config=cfg, proposal_source=src,
        review_source=review if astra_per_tick else None, gateway=ShadowGateway(),
        fk=make_fk(), audit=audit, task_reference=ref)
    print()
    prev: dict[str, str] = {}
    for s in samples:
        named = {}
        for cam in CAMERA_NAMES:
            p = Path(s.frames[cam]["frame_path"])
            mime = "image/png" if p.suffix == ".png" else "image/jpeg"
            named[cam] = f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
        obs = {"observation_id": s.observation_id, "state": s.state,
               "epoch": time.time(), "task": ep.m.instruction, "frames": s.frames,
               "images": named, "image_data_urls": [named[c] for c in CAMERA_NAMES]}
        rec = loop.step(obs)
        print(f"  {s.observation_id:30s} {rec.outcome:16s} {rec.reason[:70]}")
        if gate is not None and rec.proposal:
            rows = rec.proposal["values"]
            labelled = []
            for cam in CAMERA_NAMES:
                if cam in prev:
                    labelled.append((f"t-1 {cam}", prev[cam]))
                labelled.append((f"t {cam}", named[cam]))
            g = gate.step(state=s.state, proposed_steps=len(rows), frames=labelled,
                          state_text=state_text(s.state),
                          intent_text=describe_trajectory(rows, s.state),
                          episode_id=ep.m.episode_id, task=ep.m.instruction,
                          proposed_chunk=rows, frames_meta=s.frames,
                          fk_preview=rec.fk_preview,
                          image_data_urls=[u for _, u in labelled])
            print(f"      monitor: {g.gate['disposition'] if g.gate else 'n/a'} "
                  f"steps={g.executed_steps}{' SHADOW' if g.shadow else ''}"
                  f"{'  ESCALATED' if g.escalated else ''}")
        prev = named
    if gate is not None:
        m = gate.metrics()
        print(f"\n  monitor: {m['cycles']} cycle(s), {m['escalations']} escalation(s), "
              f"{m['rejected_monitor_outputs']} rejected; astra via escalation "
              f"{escalations['n']}x")
    print(f"\n  audit -> {audit.path}\n  commands sent: 0 (shadow)")
    return 0


# ----------------------------------------------------------------- hold / live
def _open_arms(args, cfg):
    from .robot import HeldArms
    rig = cfg.get("rig") or {}
    left, right = args.left_can or rig.get("left_can"), args.right_can or rig.get("right_can")
    limits = {"left": rig.get("gripper_limits_left"), "right": rig.get("gripper_limits_right")}
    if args.gripper_limits:
        limits = json.loads(args.gripper_limits)
    return HeldArms(left_can=left, right_can=right, gripper_limits=limits)


HOLD_WARNING = """  CONNECTING ENABLES THE MOTORS. Both arms hold the pose they are in.
  ON EXIT THE MOTORS ARE DISABLED AND THE ARMS GO LIMP -- start from, and
  end in, a pose they can rest in. E-stop within reach."""


def cmd_hold(args) -> int:
    """Connect, hold, read both arms; command nothing. Proves the rig reports."""
    if not args.hold:
        print("REFUSED: connecting enables the motors. Pass --hold.\n" + HOLD_WARNING,
              file=sys.stderr)
        return 2
    cfg = load_config(args.experiment)
    arms = _open_arms(args, cfg)
    print("HOLD on " + json.dumps(arms.channels) + "\n" + HOLD_WARNING + "\n")
    arms.open()
    t0, n, errors = time.time(), 0, 0
    try:
        while args.seconds <= 0 or time.time() - t0 < args.seconds:
            try:
                st = arms.read()
                n += 1
            except Exception:                                  # noqa: BLE001
                errors += 1
                continue
            if n % 50 == 1:
                print(f"  {time.time() - t0:6.1f}s  L {[round(v, 3) for v in st.measured[:7]]}"
                      f"  R {[round(v, 3) for v in st.measured[7:]]}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        arms.close()
        dt = time.time() - t0
        print(f"\n  reads {n} ({n / dt if dt else 0:.0f} Hz), errors {errors}. "
              f"Motors disabled.")
    return 0 if n and not errors else 1


def cmd_live(args) -> int:
    """Three-model loop against the REAL arms. They hold; nothing is commanded."""
    from .cameras import LiveCameras, parse_mapping
    from .kinematics import make_fk
    from .live import YamLiveObservationRun
    from .pipeline import YamPolicyPipeline, resolve_mode
    from .transports import Pi05ChunkReplay, Pi05HttpProposalSource
    if not args.hold:
        print("REFUSED: live connects to the arms, which enables the motors. "
              "Pass --hold.\n" + HOLD_WARNING, file=sys.stderr)
        return 2
    cfg = load_config(args.experiment)
    args.audit = args.audit or _out(f"live_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    task = args.task or (cfg.get("task") or {}).get("instruction", "")
    if args.chunks_file:
        src = Pi05ChunkReplay.from_file(args.chunks_file, sequential=True)
        desc = f"REPLAY from {args.chunks_file} (NOT live inference)"
    elif args.pi05_url:
        src = Pi05HttpProposalSource(args.pi05_url, timeout_s=args.pi05_timeout)
        desc = f"LIVE via {args.pi05_url}"
    else:
        print("REFUSED: pass --pi05-url or --chunks-file.", file=sys.stderr)
        return 2
    mapping_text = args.cameras or ""
    if not mapping_text and (cfg.get("rig") or {}).get("camera_mapping"):
        m = cfg["rig"]["camera_mapping"]
        mapping_text = ",".join(f"{k}:{v}" for k, v in
                                (json.loads(m) if isinstance(m, str) else m).items())
    if not mapping_text:
        print("REFUSED: no camera mapping. Pass --cameras top:N,left:N,right:N "
              "after confirming each stream by looking at it.", file=sys.stderr)
        return 2

    review = _astra(args)
    backend = _monitor_backend(args)
    pipe = YamPolicyPipeline(resolve_mode(args.policy_mode), backend=backend,
                             shadow=True, task_reference=_reference(args),
                             astra_review=(review.review if args.astra_live else None))
    print(f"LIVE OBSERVATION -- {ROBOT_MODEL}")
    print("  motion   : IMPOSSIBLE -- hold-only reader, no command method")
    print(f"  pi0.5    : {desc}")
    print(f"  monitor  : {backend.name} ({getattr(backend, 'model', '?')})")
    print(f"  astra    : {'LIVE (PAID)' if args.astra_live else 'DRY RUN'}")
    print(f"  cameras  : {mapping_text}")
    print(f"  task     : {task!r}")
    print(f"  audit    : {args.audit}\n{HOLD_WARNING}\n")

    cams = LiveCameras(parse_mapping(mapping_text)).start()
    arms = _open_arms(args, cfg)
    arms.open()
    run = YamLiveObservationRun(arms, pi05_infer=src.propose, pipeline=pipe,
                                grab_frames=lambda: cams.snapshot(args.frame_max_age),
                                fk=make_fk(), audit_path=args.audit,
                                min_model_interval_s=args.model_interval, task=task)
    t0 = time.time()
    try:
        run.read_one()
        run.start_models()
        while True:
            run.read_one()
            if sys.stdout.isatty() and run.reads % 50 == 0:
                c = run.cycles[-1] if run.cycles else None
                sys.stdout.write(
                    f"\r  reads {run.reads:>7} | model {len(run.cycles):>4} | "
                    f"last {c.total_s if c else '-'}s "
                    f"{('ERR ' + c.error[:40]) if c and c.error else ''} | "
                    f"astra {sum(1 for x in run.cycles if x.astra_called):>3} | "
                    f"{time.time() - t0:6.1f}s  ")
                sys.stdout.flush()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        run.stop()
        arms.close()
        cams.stop()
        print("\n" + json.dumps(run.summary(time.time() - t0), indent=2))
        print("  Motors disabled.")
    return 0


# ------------------------------------------------------------- rig setup
def _rig(cfg) -> dict:
    return cfg.get("rig") or {}


def _channels(args, cfg) -> tuple[str, str]:
    rig = _rig(cfg)
    return (getattr(args, "left_can", None) or rig.get("left_can"),
            getattr(args, "right_can", None) or rig.get("right_can"))


def _can_state(channel: str) -> tuple[bool, str]:
    import subprocess
    try:
        out = subprocess.run(["ip", "-j", "-d", "link", "show", "dev", channel],
                             capture_output=True, text=True, timeout=5)
    except Exception as exc:                                   # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    if out.returncode != 0:
        return False, (out.stderr.strip() or "not found")[:120]
    link = json.loads(out.stdout)[0]
    up = "UP" in link.get("flags", [])
    rate = ((link.get("linkinfo") or {}).get("info_data") or {}).get("bittiming", {}).get("bitrate")
    return up and rate in (None, 1000000), f"{'UP' if up else 'DOWN'} bitrate={rate}"


def _operator_arms(rig_id: str, what: str) -> bool:
    """A human at THIS terminal types the rig id. Nothing else arms motion."""
    if not sys.stdin.isatty():
        print("REFUSED: arming needs an operator at an interactive terminal.",
              file=sys.stderr)
        return False
    print(f"\n  {what}\n  E-stop in reach, workspace clear, both arms free to move.")
    try:
        typed = input(f"  Type the rig id ({rig_id}) to arm, anything else aborts: ")
    except (EOFError, KeyboardInterrupt):
        return False
    return typed.strip() == rig_id


def cmd_doctor(args) -> int:
    """Everything the rig needs, checked without enabling a motor."""
    import importlib
    import platform
    import shutil
    from .execute import execution_blockers
    cfg = load_config(args.experiment)
    flat = flatten_config(cfg)
    checks: list[tuple[str, str, str]] = []

    def add(name, ok, detail, warn=False):
        checks.append((name, "PASS" if ok else ("WARN" if warn else "FAIL"), detail))

    add("python>=3.10", sys.version_info >= (3, 10), platform.python_version())
    for mod, why in (("numpy", "arrays"), ("scipy", "FK preview"),
                     ("cv2", "cameras"), ("PIL", "images"), ("ruckig", "motion"),
                     ("i2rt", "YAM driver")):
        try:
            importlib.import_module(mod)
            add(f"import {mod}", True, why)
        except Exception as exc:                               # noqa: BLE001
            add(f"import {mod}", False, f"{why}: {type(exc).__name__}: {exc}"[:90])
    left, right = _channels(args, cfg)
    for arm, ch in (("left", left), ("right", right)):
        if not ch:
            add(f"can {arm}", False, "not configured (rig.left_can / rig.right_can)")
        else:
            ok, d = _can_state(ch)
            add(f"can {arm} {ch}", ok, d)
    import glob
    vids = sorted(glob.glob("/dev/video*"))
    add("video devices", len(vids) >= 3, " ".join(vids) or "none")
    gpu = shutil.which("nvidia-smi") or Path("/etc/nv_tegra_release").exists()
    add("gpu", bool(gpu), "nvidia-smi or Jetson present" if gpu else
        "no local GPU: pi0.5 must be served from another host", warn=True)
    if args.pi05_url:
        from .transports import check_checkpoint_meta
        try:
            h = _get_json(_health_url(args.pi05_url))
            bad = check_checkpoint_meta(h.get("meta"))
            add("pi0.5 server", bad is None, bad or f"contract OK at {args.pi05_url}")
        except Exception as exc:                               # noqa: BLE001
            add("pi0.5 server", False, f"unreachable: {type(exc).__name__}")
    else:
        add("pi0.5 server", False, "pass --pi05-url to check", warn=True)
    if args.monitor_url:
        from ..kuka.vlm_backends import VlmConfig, make_backend
        pr = make_backend("local", config=_dc_replace(
            VlmConfig.jetson(), endpoint=args.monitor_url)).probe()
        add("qwen monitor", pr.available, pr.reason[:90])
    else:
        add("qwen monitor", False, "pass --monitor-url to check", warn=True)
    import os
    add("astra key", bool(os.environ.get(args.astra_key_env)),
        f"${args.astra_key_env} {'set' if os.environ.get(args.astra_key_env) else 'empty'}",
        warn=True)
    add("rig_id", bool(_rig(cfg).get("rig_id")), str(_rig(cfg).get("rig_id") or "unset"))
    blockers = execution_blockers(flat)
    add("execute config", not blockers, "; ".join(blockers)[:300] or "all measured")

    if args.json:
        print(json.dumps([{"check": c, "result": r, "detail": d} for c, r, d in checks],
                         indent=1))
    else:
        for c, r, d in checks:
            print(f"  [{r}] {c:24s} {d}")
    return 0 if all(r != "FAIL" for _, r, _ in checks) else 1


def cmd_set(args) -> int:
    """Write one measured value into rig.local.toml: `set rig.left_can can_l`."""
    from .experiment import save_local
    section, _, key = args.key.partition(".")
    if not key:
        print("key must be section.name, e.g. rig.left_can", file=sys.stderr)
        return 2
    try:
        value = json.loads(args.value)
    except ValueError:
        value = args.value
    p = save_local({section: {key: value}}, args.experiment, note=f"set {args.key}")
    print(f"  {args.key} = {json.dumps(value)} -> {p}")
    return 0


def cmd_cameras(args) -> int:
    """Save one frame from every capturing /dev/video node, for identification."""
    from ..kuka.cameras import detect_cameras
    from .cameras import HEIGHT, WIDTH
    import cv2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    found = []
    for idx in detect_cameras():
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        for _ in range(5):
            ok, img = cap.read()
        cap.release()
        if ok:
            p = out / f"video{idx}.jpg"
            cv2.imwrite(str(p), img)
            found.append((idx, p, img.shape[1], img.shape[0]))
    for idx, p, w, h in found:
        print(f"  /dev/video{idx}: {w}x{h} -> {p}")
    print("\n  Look at each image and decide which is top, left wrist and right "
          "wrist.\n  Then: yam.cli set rig.camera_mapping "
          "'{\"top\": N, \"left\": N, \"right\": N}'")
    return 0 if found else 1


def cmd_calibrate_grippers(args) -> int:
    from .experiment import save_local
    from .robot import calibrate_gripper
    cfg = load_config(args.experiment)
    left, right = _channels(args, cfg)
    if not args.execute:
        print("REFUSED: calibration SWEEPS each gripper to its hard stops. "
              "Pass --execute.", file=sys.stderr)
        return 2
    if not _operator_arms(str(_rig(cfg).get("rig_id") or "yam"),
                          "Each gripper will close and open fully; the arm holds, "
                          "then goes limp when done."):
        return 2
    vals = {}
    for arm, ch in (("left", left), ("right", right)):
        vals[f"gripper_limits_{arm}"] = calibrate_gripper(ch)
        print(f"  {arm}: {vals[f'gripper_limits_{arm}']}")
    print(f"  saved -> {save_local({'rig': vals}, args.experiment, note='calibrate-grippers')}")
    return 0


def _gravity_factory(channel, limits):
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType
    import numpy as np
    return get_yam_robot(channel=channel, gripper_type=GripperType.LINEAR_4310,
                         zero_gravity_mode=True,
                         gripper_limits_override=np.asarray(limits, float))


def cmd_record_pose(args) -> int:
    """Hand-guide both arms (gravity compensation), press Enter, save the pose."""
    from .experiment import save_local
    from .motion import goal_problem
    from .robot import HeldArms
    if not args.hold:
        print("REFUSED: this enables the motors in gravity compensation. Pass "
              "--hold.", file=sys.stderr)
        return 2
    cfg = load_config(args.experiment)
    left, right = _channels(args, cfg)
    rig = _rig(cfg)
    arms = HeldArms(left_can=left, right_can=right,
                    gripper_limits={"left": rig.get("gripper_limits_left"),
                                    "right": rig.get("gripper_limits_right")},
                    factory=_gravity_factory).open()
    try:
        print(f"  Gravity compensation ON. Move both arms by hand to the "
              f"{args.name}, then press Enter.")
        input("  ")
        pose = [round(v, 4) for v in arms.read().measured]
    finally:
        arms.close()
    bad = goal_problem(pose)
    if bad:
        print(f"  REFUSED: {bad}", file=sys.stderr)
        return 2
    print(f"  {args.name} = {pose}")
    print(f"  saved -> {save_local({'rig': {args.name: pose}}, args.experiment, note='record-pose')}")
    return 0


def _motion_owner(args, cfg, limits, tolerance):
    from .motion import MotionOwner
    from .robot import open_for_motion
    left, right = _channels(args, cfg)
    rig = _rig(cfg)
    robots = open_for_motion(left_can=left, right_can=right,
                             gripper_limits={"left": rig.get("gripper_limits_left"),
                                             "right": rig.get("gripper_limits_right")})
    return MotionOwner(robots, limits, tolerance_rad=tolerance)


def _close_owner(owner) -> None:
    owner.stop()
    for r in owner.robots.values():
        try:
            r.close()
        except Exception:                                      # noqa: BLE001
            pass


def cmd_tracking_test(args) -> int:
    """Move each wrist joint +/-0.1 rad slowly; measure tracking error."""
    from .experiment import save_local
    from .execute import SLOW
    cfg = load_config(args.experiment)
    if not args.execute:
        print("REFUSED: this MOVES both wrists by 0.1 rad. Pass --execute.",
              file=sys.stderr)
        return 2
    if not _operator_arms(str(_rig(cfg).get("rig_id") or "yam"),
                          "Both arms' joint6 will rotate +/-0.1 rad at 0.3 rad/s."):
        return 2
    owner = _motion_owner(args, cfg, SLOW, tolerance=0.5).start()
    try:
        home = list(owner.measured)
        for sign in (1, -1, 0):
            q = list(home)
            for j in (5, 12):
                q[j] = home[j] + 0.1 * sign
            owner.set_goal(q)
            if not owner.wait_settled(15.0):
                break
            time.sleep(0.5)
        st = owner.status()
    finally:
        owner.brake()
        _close_owner(owner)
    dev = st["max_deviation_rad"]
    suggest = round(max(0.05, 3.0 * dev), 3)
    print(f"\n  max tracking deviation {dev} rad (fault: {st['fault']})")
    print(f"  suggested commanded_observed_tolerance_rad = {suggest}")
    if args.save and not st["fault"]:
        p = save_local({"stop_conditions": {"commanded_observed_tolerance_rad": suggest}},
                       args.experiment, note="tracking-test")
        print(f"  saved -> {p}")
    return 0 if not st["fault"] else 1


def cmd_execute(args) -> int:
    """pi0.5 drives the real arms. Gated, bounded, operator-armed."""
    from .cameras import LiveCameras, parse_mapping
    from .execute import (SLOW, ExecSettings, ExecutionRun, MonitorWorker,
                          execution_blockers, limits_from_config, _pose)
    from .pipeline import YamPolicyPipeline, resolve_mode
    from .transports import Pi05HttpProposalSource, check_checkpoint_meta
    cfg = load_config(args.experiment)
    flat = flatten_config(cfg)
    rig = _rig(cfg)
    task = args.task or (cfg.get("task") or {}).get("instruction", "")
    blockers = execution_blockers(flat)
    for ch in _channels(args, cfg):
        ok, d = _can_state(ch) if ch else (False, "unset")
        if not ok:
            blockers.append(f"CAN {ch}: {d}")
    if not rig.get("rig_id"):
        blockers.append("rig.rig_id is not set")
    if not args.pi05_url:
        blockers.append("--pi05-url is required")
    else:
        try:
            bad = check_checkpoint_meta(_get_json(_health_url(args.pi05_url)).get("meta"))
            if bad:
                blockers.append(bad)
        except Exception as exc:                               # noqa: BLE001
            blockers.append(f"pi0.5 server unreachable: {type(exc).__name__}")
    if blockers or not args.execute:
        print("EXECUTE -- NOT ARMED")
        for b in blockers:
            print(f"  - {b}")
        if not args.execute:
            print("  - pass --execute to command the arms")
        return 2

    settings = ExecSettings(steps_per_chunk=args.steps, max_seconds=args.max_seconds,
                            stop_file=args.stop_file, monitor_gate=args.monitor_gate,
                            monitor_max_age_s=args.monitor_max_age,
                            frame_max_age_s=min(args.frame_max_age,
                                                float(flat["observation_freshness_s"])))
    limits = limits_from_config(flat)
    mapping = rig["camera_mapping"]
    mapping = json.loads(mapping) if isinstance(mapping, str) else mapping
    src = Pi05HttpProposalSource(args.pi05_url, timeout_s=args.pi05_timeout)
    monitor = None
    if resolve_mode(args.policy_mode).value != "pi05_only":
        review = _astra(args)
        pipe = YamPolicyPipeline(resolve_mode(args.policy_mode),
                                 backend=_monitor_backend(args),
                                 shadow=not args.monitor_gate,
                                 task_reference=_reference(args),
                                 astra_review=(review.review if args.astra_live else None))
        monitor = MonitorWorker(pipe)

    print(f"EXECUTE -- {ROBOT_MODEL}, rig {rig['rig_id']}")
    print(f"  policy   : {args.pi05_url}")
    print(f"  task     : {task!r}")
    print(f"  prefix   : {settings.steps_per_chunk} of {CHUNK_STEPS} steps per chunk "
          f"({'monitor-GATED' if args.monitor_gate else 'fixed; monitor shadow'})")
    print(f"  mode     : {args.policy_mode}; astra {'LIVE (PAID)' if args.astra_live else 'dry run'}")
    print(f"  limits   : {limits.velocity} rad/s, {limits.acceleration} rad/s^2; "
          f"deviation stop {flat['commanded_observed_tolerance_rad']} rad")
    print(f"  stop     : Ctrl-C, `touch {settings.stop_file}`, or the E-stop")
    Path(settings.stop_file).unlink(missing_ok=True)
    if not _operator_arms(str(rig["rig_id"]),
                          "pi0.5 WILL MOVE BOTH ARMS."):
        print("  not armed.")
        return 2

    cams = LiveCameras(parse_mapping(",".join(f"{k}:{v}" for k, v in mapping.items()))).start()
    owner = _motion_owner(args, cfg, limits, float(flat["commanded_observed_tolerance_rad"]))
    audit = args.audit or _out(f"execute_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    summary = None
    try:
        owner.start()
        time.sleep(0.5)
        start_pose = _pose(flat.get("start_pose"))
        if start_pose is not None:
            print("  moving to start_pose (slow) ...")
            owner.park(start_pose, SLOW)
            owner.retune(limits)                    # task limits from here
        if monitor is not None:
            monitor.start()
        run = ExecutionRun(owner, propose=src.propose,
                           grab_frames=lambda: cams.snapshot(settings.frame_max_age_s),
                           settings=settings, config=flat, task=task, monitor=monitor,
                           audit_path=audit)

        def show(rec):
            print(f"  c{rec.cycle:04d} {rec.outcome:9s} steps={rec.steps_executed:2d} "
                  f"pi05={rec.pi05_s}s dev={rec.motion['max_deviation_rad']} "
                  f"{rec.reason[:60]}")
        try:
            summary = run.run(on_cycle=show)
        except KeyboardInterrupt:
            run.stop_reason = "operator Ctrl-C"
            summary = run.summary(0.0)
    finally:
        owner.brake()
        if monitor is not None:
            monitor.stop()
        rest = _pose(flat.get("rest_pose"))
        try:
            park = rest is not None and not args.no_park
            if park and owner.fault:
                park = _operator_arms(str(rig["rig_id"]),
                                      f"MOTION FAULTED ({owner.fault}). Park to rest_pose anyway?")
            if park:
                print("  parking to rest_pose (slow) ...")
                owner.park(rest, SLOW, after_fault=bool(owner.fault))
        except Exception as exc:                               # noqa: BLE001
            print(f"  park failed: {exc}")
        _close_owner(owner)
        cams.stop()
        print("  motors disabled.")
    print(json.dumps(summary, indent=2, default=str))
    print(f"  audit -> {audit}")
    return 0


def cmd_phase3(args) -> int:
    miss = missing_config(flatten_config(load_config(args.experiment)), Mode.ASTRA_DIRECT)
    print(f"REFUSED: Astra-direct is not implemented for YAM, and would need "
          f"{len(miss)} more value(s) and completed earlier phases: {miss}")
    return 2


# ----------------------------------------------------------------------- main
def _astra_args(sp) -> None:
    sp.add_argument("--astra-url", default=ASTRA_URL)
    sp.add_argument("--astra-model", default="gpt-6-astra")
    sp.add_argument("--astra-key-env", default="OPENAI_API_KEY")
    sp.add_argument("--astra-effort", default="low")
    sp.add_argument("--astra-live", action="store_true",
                    help="MAKE PAID API CALLS. Off by default.")
    sp.add_argument("--astra-background", action="store_true", default=True)
    sp.add_argument("--astra-no-background", dest="astra_background",
                    action="store_false")
    sp.add_argument("--astra-deadline", type=float, default=300.0)


def _monitor_args(sp, default_mode: str, default_backend: str) -> None:
    sp.add_argument("--policy-mode", default=default_mode,
                    help="pi05_only | pi05_local_monitor | pi05_local_monitor_astra")
    sp.add_argument("--monitor-backend", default=default_backend,
                    help="local | mock | unconfigured")
    sp.add_argument("--monitor-url", help="OpenAI-compatible endpoint (Qwen on the Jetson)")
    sp.add_argument("--monitor-timeout", type=float, default=12.0,
                    help="seconds; the KUKA branch measured 5.1-9.1 s on the Orin")
    sp.add_argument("--monitor-frames", type=int, default=2)
    sp.add_argument("--task-reference")


def _arm_args(sp) -> None:
    sp.add_argument("--hold", action="store_true",
                    help="REQUIRED. Connects to both arms, enabling the motors.")
    sp.add_argument("--left-can"); sp.add_argument("--right-can")
    sp.add_argument("--gripper-limits",
                    help='JSON {"left": [closed, open], "right": [closed, open]}')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="yam-experiment", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def exp(sp):
        sp.add_argument("--experiment", default=DEFAULT_EXPERIMENT)

    pf = sub.add_parser("preflight"); exp(pf)
    pf.add_argument("--pi05-url")
    pf.add_argument("--monitor-url")
    pf.add_argument("--monitor-backend", default="unconfigured")
    pf.set_defaults(func=cmd_preflight)

    p1 = sub.add_parser("phase1"); exp(p1)
    p1.add_argument("--episode", required=True)
    p1.add_argument("--every", type=int, default=CHUNK_STEPS)
    p1.add_argument("--limit", type=int)
    p1.add_argument("--out")
    p1.set_defaults(func=cmd_phase1)

    rn = sub.add_parser("run", help="recorded episode through the three-model loop")
    exp(rn)
    rn.add_argument("--episode", required=True)
    rn.add_argument("--every", type=int, default=CHUNK_STEPS)
    rn.add_argument("--limit", type=int)
    rn.add_argument("--chunks-file")
    rn.add_argument("--pi05-url", help="e.g. http://127.0.0.1:18840/infer")
    rn.add_argument("--pi05-timeout", type=float, default=30.0)
    rn.add_argument("--monitor-audit")
    rn.add_argument("--monitor-shadow", action="store_true", default=True)
    rn.add_argument("--monitor-gate", dest="monitor_shadow", action="store_false")
    rn.add_argument("--audit")
    _monitor_args(rn, "pi05_only", "unconfigured")
    _astra_args(rn)
    rn.set_defaults(func=cmd_run)

    hd = sub.add_parser("hold", help="connect, hold and read both arms")
    exp(hd); _arm_args(hd)
    hd.add_argument("--seconds", type=float, default=10.0, help="0 = until Ctrl-C")
    hd.set_defaults(func=cmd_hold)

    lv = sub.add_parser("live", help="three-model loop on the real arms, holding")
    exp(lv); _arm_args(lv)
    lv.add_argument("--pi05-url"); lv.add_argument("--chunks-file")
    lv.add_argument("--pi05-timeout", type=float, default=30.0)
    lv.add_argument("--cameras", help="top:N,left:N,right:N")
    lv.add_argument("--frame-max-age", type=float, default=0.25)
    lv.add_argument("--model-interval", type=float, default=1.0)
    lv.add_argument("--task", default="")
    lv.add_argument("--audit")
    _monitor_args(lv, "pi05_local_monitor_astra", "local")
    _astra_args(lv)
    lv.set_defaults(func=cmd_live)

    dr = sub.add_parser("doctor", help="check the rig and services; moves nothing")
    exp(dr); dr.add_argument("--left-can"); dr.add_argument("--right-can")
    dr.add_argument("--pi05-url"); dr.add_argument("--monitor-url")
    dr.add_argument("--astra-key-env", default="OPENAI_API_KEY")
    dr.add_argument("--json", action="store_true")
    dr.set_defaults(func=cmd_doctor)

    st = sub.add_parser("set", help="write a measured value into rig.local.toml")
    exp(st); st.add_argument("key"); st.add_argument("value")
    st.set_defaults(func=cmd_set)

    cm = sub.add_parser("cameras", help="save one frame per camera to identify them")
    exp(cm); cm.add_argument("--out", default=str(RUNS / "cameras"))
    cm.set_defaults(func=cmd_cameras)

    cg = sub.add_parser("calibrate-grippers", help="i2rt gripper sweep (MOTION)")
    exp(cg); cg.add_argument("--execute", action="store_true")
    cg.add_argument("--left-can"); cg.add_argument("--right-can")
    cg.set_defaults(func=cmd_calibrate_grippers)

    rp = sub.add_parser("record-pose", help="hand-guide the arms, save a pose")
    exp(rp); rp.add_argument("--name", choices=("rest_pose", "start_pose"), required=True)
    rp.add_argument("--hold", action="store_true")
    rp.add_argument("--left-can"); rp.add_argument("--right-can")
    rp.set_defaults(func=cmd_record_pose)

    tt = sub.add_parser("tracking-test", help="measure tracking error (MOTION, small)")
    exp(tt); tt.add_argument("--execute", action="store_true")
    tt.add_argument("--save", action="store_true")
    tt.add_argument("--left-can"); tt.add_argument("--right-can")
    tt.set_defaults(func=cmd_tracking_test)

    ex = sub.add_parser("execute", help="pi0.5 drives the real arms (MOTION)")
    exp(ex)
    ex.add_argument("--execute", action="store_true", help="REQUIRED to move the arms")
    ex.add_argument("--pi05-url", help="e.g. http://127.0.0.1:18840/infer")
    ex.add_argument("--pi05-timeout", type=float, default=10.0)
    ex.add_argument("--task", default="")
    ex.add_argument("--steps", type=int, default=8, help="executed rows per chunk, 1-16")
    ex.add_argument("--max-seconds", type=float, default=120.0)
    ex.add_argument("--stop-file", default="/tmp/yam_stop")
    ex.add_argument("--frame-max-age", type=float, default=0.25)
    ex.add_argument("--monitor-gate", action="store_true",
                    help="let the Qwen/Astra decision set the prefix (default: shadow)")
    ex.add_argument("--monitor-max-age", type=float, default=20.0)
    ex.add_argument("--no-park", action="store_true", help="do not move to rest_pose at the end")
    ex.add_argument("--left-can"); ex.add_argument("--right-can")
    ex.add_argument("--audit")
    _monitor_args(ex, "pi05_local_monitor_astra", "local")
    _astra_args(ex)
    ex.set_defaults(func=cmd_execute)

    p3 = sub.add_parser("phase3"); exp(p3)
    p3.set_defaults(func=cmd_phase3)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
