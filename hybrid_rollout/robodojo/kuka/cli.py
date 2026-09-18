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
import base64
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
    from . import interfaces as I
    from .loop import EDIT_EXECUTION_ENABLED, EXECUTABLE_DECISION_MODES
    from .rsi_gateway import RSIGateway, check_rsi_elements
    cfg = load_config(args.experiment)
    flat = flatten_config(cfg)
    print(f"PREFLIGHT -- experiment {args.experiment!r}\n")

    print('THE THREE THINGS CALLED "TCP" HERE -- do not conflate them')
    for f in I.describe():
        where = (f"{f['protocol']}/{f['port']}" if f["port"] else "geometry, no port")
        print(f"  {f['meaning']:32s} {where:12s} "
              f"carries motion: {f['carries_motion_commands']}")
        print(f"      {f['description'].splitlines()[0][:96]}")
    print()

    print("RSI RECEIVE CONFIGURATION (verified deployment fact)")
    print(f"  accepts          : {', '.join(I.RSI_ACCEPTED_ELEMENTS)}")
    print(f"  RKorr present    : {I.RSI_HAS_RKORR}")
    print(f"  cartesian usable : {I.cartesian_capability()[0]}")
    ok, why = check_rsi_elements()
    print(f"  element check    : {'match' if ok else '; '.join(why)}")
    print()

    print("TOOL CENTER POINT (geometry, unrelated to any port)")
    print(f"  $TOOL known      : {I.TOOL_TRANSFORM_KNOWN}")
    print(f"  FK frame         : {I.FK_FRAME}  (tool tip: {I.FK_IS_TOOL_TIP})")
    print(f"  {I.TOOL_NOTE[:96]}")
    print()

    print("EXECUTION LOCKS")
    print(f"  emittable modes  : {sorted(EXECUTABLE_DECISION_MODES)}")
    print(f"  edit execution   : {EDIT_EXECUTION_ENABLED}")
    for m, whyl in I.modes_locked().items():
        print(f"    {m:14s} LOCKED -- {whyl[:80]}")
    print()

    print("GATEWAY READINESS (HOLD only; says nothing about motion)")
    r = RSIGateway(host=args.rsi_host, port=I.RSI_UDP_PORT).readiness()
    okh, blockers = r.ready_for_hold()
    print(f"  ready for HOLD   : {okh}")
    for b in blockers:
        print(f"    - {b}")
    print(f"  {I.SILENCE_IS_UNSAFE[:96]}")
    print(f"  {I.ESTOP_AUTHORITATIVE[:96]}")
    print()
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



def cmd_measure(args: argparse.Namespace) -> int:
    """Guided collection of operator-measured cell values.

    COLLECTS AND VALIDATES. NEVER INFERS. Every prompt is a question for a human
    who has measured the thing; there is no path that fills one in from a similar
    cell, a previous run, or a plausible range. Blank input leaves the value null
    and the corresponding gate stays NO-GO.
    """
    import json as _json
    from pathlib import Path as _P
    from . import deployment_config as dc

    out = _P(args.out)
    cfg = dc.load(out) if out.exists() and not args.fresh else dc.blank()
    if args.checklist:
        print(dc.checklist(cfg))
        return 0

    print("GUIDED MEASUREMENT -- values you have MEASURED on the real cell.")
    print("Blank leaves a value unset. Nothing here is inferred or defaulted.")
    print(f"Writing to {out}\n")
    cfg.setdefault("cell", args.cell)
    cfg["measured_by"] = args.operator or cfg.get("measured_by")
    cfg["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    only = set(args.only.split(",")) if args.only else None
    section = None
    for key, (sec, what, why) in dc.FIELDS.items():
        if only and sec not in only and key not in only:
            continue
        if sec != section:
            section = sec
            print(f"\n[{sec.upper()}]")
        cur = (cfg["values"] or {}).get(key)
        shown = "unset" if cur in (None, "", [], {}) else _json.dumps(cur)[:60]
        print(f"\n  {key}")
        print(f"    measure: {what}")
        print(f"    why    : {why}")
        print(f"    current: {shown}")
        if args.non_interactive:
            continue
        try:
            raw = input("    value (JSON, blank to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  interrupted; keeping what was already set")
            break
        if not raw:
            continue
        try:
            cfg["values"][key] = _json.loads(raw)
        except Exception:
            cfg["values"][key] = raw       # a plain string is a legitimate answer
        print(f"    recorded: {_json.dumps(cfg['values'][key])[:70]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(cfg, indent=1) + "\n")
    rep = dc.report(cfg)
    print(f"\nwrote {out}")
    print(f"overall: {rep['overall']}")
    for g, d in rep["gates"].items():
        print(f"  {g:28s} {d['decision']:6s} ({d['n_missing_values']} missing)")
    return 0


def cmd_gonogo(args: argparse.Namespace) -> int:
    """Machine-readable GO/NO-GO naming every failed gate."""
    import json as _json
    from pathlib import Path as _P
    from . import cell as C
    from . import deployment_config as dc
    from . import interfaces as I

    p = _P(args.config)
    cfg = dc.load(p) if p.exists() else dc.blank()

    # Gates that are about the CELL rather than measured values.
    cart_ok, cart_why = I.cartesian_capability()
    extra = {
        "hold_handshake": (
            C.RSI_PROGRAM_RUNNING and C.JETSON_RSI_SOCKET_BOUND,
            [b for b in (
                None if C.RSI_PROGRAM_RUNNING else
                f"controller-side RSI program not running (tcp/{C.PORT_EXT_TRIGGER_TCP} closed)",
                None if C.JETSON_RSI_SOCKET_BOUND else
                f"Jetson udp/{C.PORT_RSI_UDP} socket unbound") if b]),
        "supervised_student_prefix": (
            False, ["CLI-to-RSI execution requires an operator arming action "
                    "and a live HOLD handshake first"]),
        "astra_direct": (cart_ok, cart_why),
    }
    rep = dc.report(cfg, extra_gates=extra)
    rep["cell_state"] = {"ext_trigger_open": C.EXT_TRIGGER_OPEN,
                         "rsi_program_running": C.RSI_PROGRAM_RUNNING,
                         "rsi_socket_bound": C.JETSON_RSI_SOCKET_BOUND}
    rep["locked_modes"] = I.modes_locked()
    if args.json:
        print(_json.dumps(rep, indent=1))
        return 0
    print(f"GO / NO-GO   config={p if p.exists() else '(none: all values null)'}")
    print(f"overall: {rep['overall']}\n")
    for g, d in rep["gates"].items():
        print(f"  {g:28s} {d['decision']}")
        for b in d["blockers"][:6]:
            print(f"      - {b}")
        if len(d["blockers"]) > 6:
            print(f"      ... and {len(d['blockers']) - 6} more")
    return 0


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
    from .transports import AstraReviewSource, LocalPi05ProposalSource
    cfg = load_config(args.experiment)
    flat = flatten_config(cfg)
    miss = missing_config(flat, Mode.REVIEWED_EXECUTION)
    _banner("LIVE LOOP: pi0.5 + Astra", Mode.LIVE_SHADOW, miss)

    if not args.chunks_file:
        print("  --chunks-file is required: this build does not serve the model. "
              "Run pi0.5 locally and write {observation_id: 50x7} to JSON, with "
              "a _meta object carrying use_relative_actions.", file=sys.stderr)
        return 2
    try:
        src = LocalPi05ProposalSource.from_file(
            args.chunks_file, checkpoint_id=cfg["checkpoint"]["id"])
    except Exception as exc:
        print(f"  cannot read {args.chunks_file}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    rel = src.meta.get("use_relative_actions")
    print(f"  proposals    : {args.chunks_file}  ({len(src.chunks)} chunk(s))")
    print(f"  checkpoint   : {src.checkpoint_id}")
    print(f"  use_rel_act  : {rel} "
          f"{'OK' if rel is True else '<-- WRONG CHECKPOINT, will refuse'}")

    review = AstraReviewSource(
        base_url=args.astra_url, model=args.astra_model,
        api_key_env=args.astra_key_env, enabled=args.astra_live,
        dry_run=not args.astra_live, reasoning=args.astra_effort,
        stream=not args.astra_no_stream, attempts=args.astra_attempts)
    ok, why = review.preflight()
    print(f"  astra        : {'LIVE (PAID)' if args.astra_live else 'DRY RUN'} -- {why}"
          f" [{'streamed' if review.stream else 'single response'}]")
    if args.astra_live and not ok:
        print(f"  REFUSED: {why}", file=sys.stderr)
        return 2

    ep = _episode(args)
    samples = ep.sample_ticks(every=args.every, chunk_steps=args.chunk_steps,
                              limit=args.limit)
    missing = [f"{s.observation_id}/{cam}" for s in samples
               for cam, f in s.frames.items() if not f.get("frame_present")]
    if missing:
        print(f"  REFUSED: the reviewer judges the chunk against the frames it was "
              f"proposed from, and {len(missing)} are not extracted "
              f"(e.g. {missing[0]}). Pass --media-root and extract "
              f"<media_root>/frames/<camera>_<frame:06d>.png first.", file=sys.stderr)
        return 2
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
    answered, tokens = 0, 0
    for s in samples:
        images = {cam: base64.b64encode(Path(f["frame_path"]).read_bytes()).decode()
                  for cam, f in sorted(s.frames.items())}
        obs = {"observation_id": s.observation_id, "state": s.state,
               "epoch": time.time(), "task": ep.m.instruction, "frames": s.frames,
               "images": images,
               "image_data_urls": [f"data:image/png;base64,{b}" for b in images.values()]}
        rec = loop.step(obs)
        d = rec.decision or {}
        rv = rec.review or {}
        if rv.get("ok") and rv.get("is_live"):
            answered += 1
            tokens += int((rv.get("usage") or {}).get("total_tokens") or 0)
        print(f"  {s.observation_id:26s} {rec.outcome:16s} "
              f"mode={d.get('mode','-'):8s} steps={d.get('steps','-')!s:3s} "
              f"safe={rec.execution_safe} {rec.reason[:60]}")
        if d.get("reason"):
            print(f"      astra: {str(d['reason'])[:150]}")
        elif rv.get("dry_run"):
            print("      review: dry run -- request built and hashed, not sent")
        elif rv.get("error"):
            print(f"      review: {str(rv['error'])[:150]}")
    print(f"\n  audit -> {audit.path}")
    print("  commands sent: 0 (shadow). Astra: "
          + (f"{answered} live review(s) answered, {tokens} total tokens"
             if args.astra_live else "dry run, nothing sent"))
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

    pf = sub.add_parser("preflight")
    pf.add_argument("--experiment", default="dishwasher_door_open")
    pf.add_argument("--rsi-host", default="172.17.255.2")
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
    rn.add_argument("--chunks-file",
                    help="JSON {observation_id: 50x7} from your local pi0.5 run, "
                         "plus a _meta object with use_relative_actions")
    rn.add_argument("--astra-url", default="https://api.openai.com/v1/responses")
    rn.add_argument("--astra-model", default="gpt-6-astra")
    rn.add_argument("--astra-key-env", default="OPENAI_API_KEY")
    rn.add_argument("--astra-effort", default="low")
    rn.add_argument("--astra-live", action="store_true",
                    help="MAKE PAID API CALLS. Off by default.")
    rn.add_argument("--astra-attempts", type=int, default=1,
                    help="how many times ONE review may be attempted when the "
                         "CONNECTION fails (default 1). An answered review is "
                         "never re-asked. Each attempt is billable.")
    rn.add_argument("--astra-no-stream", action="store_true",
                    help="do not stream the review. Still one attempt, but an "
                         "outbound proxy that closes idle tunnels (the Jetson's "
                         "does, at ~108s) will lose long reviews.")
    rn.add_argument("--audit"); rn.add_argument("--serial")
    rn.set_defaults(func=cmd_run)

    mz = sub.add_parser("measure", help="guided collection of measured cell values")
    mz.add_argument("--out", default="deployment_config.json")
    mz.add_argument("--cell", default="dishwasher_table")
    mz.add_argument("--operator")
    mz.add_argument("--only", help="comma-separated sections or field names")
    mz.add_argument("--checklist", action="store_true", help="print and exit")
    mz.add_argument("--fresh", action="store_true", help="start from a blank config")
    mz.add_argument("--non-interactive", action="store_true")
    mz.set_defaults(func=cmd_measure)

    gg = sub.add_parser("gonogo", help="machine-readable GO/NO-GO report")
    gg.add_argument("--config", default="deployment_config.json")
    gg.add_argument("--json", action="store_true")
    gg.set_defaults(func=cmd_gonogo)

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
