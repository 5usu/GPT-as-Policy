"""Terminal driver for the three-phase KUKA dishwasher experiment.

    python -m hybrid_rollout.robodojo.kuka.cli phase1 --manifest ... [--send]
    python -m hybrid_rollout.robodojo.kuka.cli phase2 --manifest ... [--arm]
    python -m hybrid_rollout.robodojo.kuka.cli phase3 ...
    python -m hybrid_rollout.robodojo.kuka.cli preflight

Phases, as agreed in the inference group chat:
  1  show Astra a recorded trajectory and the matching observation videos
  2  replay that trajectory on the arm under full gating while Astra watches the
     cameras, to establish what a successful episode looks like
  3  let Astra propose the actions itself, through the identical gate chain

WITHOUT --send NOTHING IS TRANSMITTED and without --arm NOTHING IS EXECUTED. Both
default off; both refuse unless every gate passes. `--arm` additionally refuses
outright in phases that are structurally observation-only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .contract import ROBOT_MODEL, eef_execution_gate
from .episode import RecordedEpisode
from .experiment import (flatten_config, load_config, load_manifest,
                         retry_limits, stop_conditions, validate_manifest)
from .kinematics import make_fk
from .loop import AuditLog, KukaReviewLoop, Outcome
from .packet import build_packet
from .safety import (ASTRA_DIRECT_EXTRA, REQUIRED_CONFIG, CommandLedger, Mode,
                     RobotIdentity, Supervisor, missing_config)
from .transports import (BoundedAstraProposalSource, RecordedTrajectorySource,
                         ShadowGateway, a800_live_review_enabled)


def _banner(phase: str, mode: Mode, cfg_missing: list[str]) -> None:
    print(f"KUKA DISHWASHER EXPERIMENT -- {phase}")
    print(f"  robot     : {ROBOT_MODEL}")
    print(f"  mode      : {mode.value}")
    print(f"  config    : {len(cfg_missing)} required value(s) still unmeasured")
    ok, missing, _ = eef_execution_gate()
    print(f"  eef       : {'ENABLED' if ok else 'accepted but REFUSED'} "
          f"({len(missing)} prerequisite(s) missing)")
    live, why = a800_live_review_enabled()
    print(f"  A800 live : {live} -- {why}")
    print()


def cmd_preflight(args: argparse.Namespace) -> int:
    cfg = load_config(args.experiment)
    flat = flatten_config(cfg)
    print(f"PREFLIGHT -- experiment {args.experiment!r}\n")
    for mode in (Mode.REVIEWED_EXECUTION, Mode.ASTRA_DIRECT):
        miss = missing_config(flat, mode)
        print(f"{mode.value}: {len(miss)} missing")
        for k in miss:
            src = REQUIRED_CONFIG.get(k) or ASTRA_DIRECT_EXTRA.get(k, "")
            print(f"    {k:34s} {src}")
        print()
    ok, missing, _ = eef_execution_gate()
    print(f"eef execution: {ok}; missing {missing}")
    print(f"stop conditions: {stop_conditions(cfg)}")
    print(f"retry limits   : {retry_limits(cfg)}")
    tol = (cfg.get('stop_conditions') or {}).get('commanded_observed_tolerance_deg')
    if tol in ("", None):
        print("\nNOTE: commanded_observed_tolerance_deg is unset, so the "
              "commanded-vs-observed check is UNEVALUABLE and any execution "
              "cycle will perform a controlled stop after the first step.")
    return 0


def _episode(args) -> RecordedEpisode:
    state = traj = None
    if args.state_json:
        state = json.loads(Path(args.state_json).read_text())
    if args.trajectory_json:
        traj = json.loads(Path(args.trajectory_json).read_text())
    return RecordedEpisode.open(args.manifest, media_root=args.media_root,
                                state_rows=state, traj_rows=traj)


def cmd_phase1(args: argparse.Namespace) -> int:
    """Show Astra a recorded trajectory + the matching observation frames."""
    cfg = load_config(args.experiment)
    flat = flatten_config(cfg)
    _banner("PHASE 1: recorded trajectory review", Mode.REPLAY,
            missing_config(flat, Mode.REVIEWED_EXECUTION))
    ep = _episode(args)
    print(f"  episode   : {ep.m.episode_id}  ticks={ep.n_ticks} @ {ep.control_hz:.0f} Hz")
    print(f"  task      : {ep.m.instruction}")
    print(f"  outcome   : {ep.m.outcome.get('success')} "
          f"(source: {ep.m.outcome.get('source')})\n")
    fk = make_fk()
    samples = ep.sample_ticks(every=args.every, chunk_steps=args.chunk_steps,
                              limit=args.limit)
    out = Path(args.out or "phase1_packets.jsonl")
    n_frames_missing = 0
    with out.open("w") as f:
        for s in samples:
            prev = fk.preview([r[:6] for r in s.recorded_chunk]).to_log()
            pkt = build_packet(
                task_instruction=ep.m.instruction, observation_id=s.observation_id,
                state=s.state, chunk=s.recorded_chunk, provenance="recorded_demo",
                frames=s.frames, fk_preview=prev)
            n_frames_missing += sum(1 for v in s.frames.values()
                                    if not v.get("frame_present"))
            f.write(json.dumps(pkt) + "\n")
            print(f"  tick {s.tick:6d}  {s.observation_id:28s} "
                  f"chunk={len(s.recorded_chunk):3d}  "
                  f"frames={sorted(s.frames)}")
    print(f"\n  {len(samples)} review packet(s) -> {out}")
    if n_frames_missing:
        print(f"  {n_frames_missing} frame file(s) unresolved; extract frames into "
              f"<media_root>/frames/ before sending")
    print("  NOTHING SENT. Review the packets, then send them with your own "
          "approved call path.")
    return 0


def cmd_phase2(args: argparse.Namespace) -> int:
    """Replay the recorded trajectory under full gating while Astra watches."""
    cfg = load_config(args.experiment)
    flat = flatten_config(cfg)
    miss = missing_config(flat, Mode.REVIEWED_EXECUTION)
    _banner("PHASE 2: gated replay of the recorded trajectory", 
            Mode.REVIEWED_EXECUTION, miss)
    ep = _episode(args)
    samples = ep.sample_ticks(every=args.every, chunk_steps=args.chunk_steps,
                              limit=args.limit)
    chunks = {s.observation_id: s.recorded_chunk for s in samples}
    src = RecordedTrajectorySource(chunks, episode_id=ep.m.episode_id)
    print(f"  proposals : {src.name} (provenance={src.provenance})")
    print(f"  gateway   : {'REAL (--arm)' if args.arm else 'ShadowGateway (no socket)'}")
    if args.arm and miss:
        print(f"\n  REFUSED: --arm requires all {len(REQUIRED_CONFIG)} measured "
              f"values; {len(miss)} are missing.", file=sys.stderr)
        return 2
    if args.arm:
        print("\n  REFUSED: no real gateway is wired in this build. The Jetson RSI "
              "transport must be supplied and reviewed separately.", file=sys.stderr)
        return 2
    gw = ShadowGateway()
    audit = AuditLog(args.audit or "phase2_audit.jsonl")
    loop = KukaReviewLoop(
        mode=Mode.REVIEWED_EXECUTION, config=flat, raw_config=cfg,
        proposal_source=src, review_source=None, gateway=gw,
        fk=make_fk(), audit=audit,
        target=RobotIdentity(ROBOT_MODEL, args.serial or "UNSET", "172.17.255.2", 59152),
        allowlist=[], supervisor=Supervisor(), ledger=CommandLedger(), secret=None)
    print()
    for s in samples:
        rec = loop.step({"observation_id": s.observation_id, "state": s.state,
                         "epoch": time.time()})
        print(f"  tick {s.tick:6d}  {rec.stage_reached:10s} {rec.outcome:16s} "
              f"{rec.reason[:56]}")
        if rec.outcome == Outcome.STOPPED.value:
            break
    print(f"\n  audit -> {audit.path}   commands sent: "
          f"{sum(1 for e in gw.emitted if e.get('sent'))}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Live loop: pi0.5 (real checkpoint) -> FK -> Astra -> sanitize -> validate.

    Commands are SHADOW unless every gate passes, and Astra is DRY-RUN unless
    --astra-live is passed. Both are off by default so a first run costs nothing
    and touches nothing.
    """
    from .transports import AstraReviewSource, Pi05HttpProposalSource
    cfg = load_config(args.experiment)
    flat = flatten_config(cfg)
    miss = missing_config(flat, Mode.REVIEWED_EXECUTION)
    _banner("LIVE LOOP: pi0.5 + Astra", Mode.LIVE_SHADOW, miss)

    src = Pi05HttpProposalSource(args.pi05_url,
                                 checkpoint_id=cfg["checkpoint"]["id"],
                                 expected_sha256=cfg["checkpoint"].get("sha256"))
    try:
        h = src.health()
        m = h.get("meta") or {}
        print(f"  pi0.5 server : {args.pi05_url}  ok={h.get('ok')}")
        print(f"  checkpoint   : {m.get('checkpoint')}")
        print(f"  use_rel_act  : {m.get('use_relative_actions')} "
              f"{'OK' if m.get('use_relative_actions') is True else '<-- WRONG CHECKPOINT'}")
        print(f"  chunk        : {m.get('chunk_size')}/{m.get('n_action_steps')}")
    except Exception as exc:
        print(f"  pi0.5 server UNREACHABLE at {args.pi05_url}: "
              f"{type(exc).__name__}. Start it on the A800 with "
              f"`python -m hybrid_rollout.robodojo.kuka.pi05_serve`.", file=sys.stderr)
        return 2

    review = AstraReviewSource(
        base_url=args.astra_url, model=args.astra_model,
        api_key_env=args.astra_key_env, enabled=args.astra_live,
        dry_run=not args.astra_live, reasoning=args.astra_effort)
    ok, why = review.preflight()
    print(f"  astra        : {'LIVE (PAID)' if args.astra_live else 'DRY RUN'} -- {why}")
    if args.astra_live and not ok:
        print(f"  REFUSED: {why}", file=sys.stderr)
        return 2

    ep = _episode(args)
    samples = ep.sample_ticks(every=args.every, chunk_steps=args.chunk_steps,
                              limit=args.limit)
    fk = make_fk()
    audit = AuditLog(args.audit or "live_loop_audit.jsonl")
    loop = KukaReviewLoop(
        mode=Mode.LIVE_SHADOW, config=flat, raw_config=cfg,
        proposal_source=src, review_source=review, gateway=ShadowGateway(),
        fk=fk, audit=audit,
        target=RobotIdentity(ROBOT_MODEL, args.serial or "UNSET",
                             "172.17.255.2", 59152),
        allowlist=[], supervisor=Supervisor(), ledger=CommandLedger(), secret=None)
    print()
    for s in samples:
        obs = {"observation_id": s.observation_id, "state": s.state,
               "epoch": time.time(), "task": ep.m.instruction}
        rec = loop.step(obs)
        d = rec.decision or {}
        print(f"  {s.observation_id:26s} {rec.outcome:16s} "
              f"mode={d.get('mode','-'):8s} safe={rec.execution_safe} "
              f"{rec.reason[:44]}")
    print(f"\n  audit -> {audit.path}")
    print("  commands sent: 0 (shadow). Astra: "
          f"{'live calls made' if args.astra_live else 'dry run, nothing sent'}")
    return 0


def cmd_phase3(args: argparse.Namespace) -> int:
    """Astra proposes the actions itself, through the identical gate chain."""
    cfg = load_config(args.experiment)
    flat = flatten_config(cfg)
    miss = missing_config(flat, Mode.ASTRA_DIRECT)
    _banner("PHASE 3: Astra-direct", Mode.ASTRA_DIRECT, miss)
    src = BoundedAstraProposalSource(
        bounded_action_space=(cfg.get("astra_direct") or {}).get("bounded_action_space"))
    r = src.propose({"observation_id": "probe"})
    print(f"  bounded action space: "
          f"{'configured' if r.get('ok') else 'NOT configured'}")
    print(f"  proposal probe      : {r.get('error', 'ok')}")
    print(f"\n  REFUSED: phase 3 needs {len(miss)} more measured value(s) and a "
          f"completed phase 2. Missing:")
    for k in miss:
        print(f"    {k}")
    return 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kuka-experiment", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--experiment", default="dishwasher_door_open")
        sp.add_argument("--manifest", required=True)
        sp.add_argument("--media-root")
        sp.add_argument("--state-json", help="JSON list of state rows")
        sp.add_argument("--trajectory-json", help="JSON list of recorded targets")
        sp.add_argument("--every", type=int, default=50)
        sp.add_argument("--chunk-steps", type=int, default=50)
        sp.add_argument("--limit", type=int)

    pf = sub.add_parser("preflight"); pf.add_argument("--experiment",
                                                      default="dishwasher_door_open")
    pf.set_defaults(func=cmd_preflight)

    p1 = sub.add_parser("phase1"); common(p1)
    p1.add_argument("--out"); p1.add_argument("--send", action="store_true",
                                              help="reserved; refuses in this build")
    p1.set_defaults(func=cmd_phase1)

    p2 = sub.add_parser("phase2"); common(p2)
    p2.add_argument("--audit"); p2.add_argument("--serial")
    p2.add_argument("--arm", action="store_true",
                    help="attempt real execution; refuses unless every gate passes")
    p2.set_defaults(func=cmd_phase2)

    rn = sub.add_parser("run", help="live pi0.5 + Astra loop (shadow commands)")
    common(rn)
    rn.add_argument("--pi05-url", default="http://127.0.0.1:8710")
    rn.add_argument("--astra-url", default="https://api.openai.com/v1/responses")
    rn.add_argument("--astra-model", default="gpt-6-astra")
    rn.add_argument("--astra-key-env", default="OPENAI_API_KEY")
    rn.add_argument("--astra-effort", default="low")
    rn.add_argument("--astra-live", action="store_true",
                    help="MAKE PAID API CALLS. Off by default.")
    rn.add_argument("--audit"); rn.add_argument("--serial")
    rn.set_defaults(func=cmd_run)

    p3 = sub.add_parser("phase3")
    p3.add_argument("--experiment", default="dishwasher_door_open")
    p3.set_defaults(func=cmd_phase3)

    args = p.parse_args(argv)
    if getattr(args, "send", False):
        print("--send is not wired in this build: sending requires an explicit, "
              "separately approved call path.", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
