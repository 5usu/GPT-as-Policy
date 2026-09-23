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

    rt = I and __import__("hybrid_rollout.robodojo.kuka.cell", fromlist=["x"]).probe_runtime()
    print("CELL STATE")
    print(f"  baseline was taken WITH RSI OFF -- resting state, not an invariant")
    print(f"  live local probe : udp/{I.RSI_UDP_PORT} "
          f"{'in use' if rt['jetson_rsi_socket_bound'] else 'free'}")
    print(f"  trigger/program  : {rt['ext_trigger_open']} / {rt['rsi_program_running']}"
          f"  (None = not probed; run `gonogo --probe-network` on the Jetson)")
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

    # Gates about the CELL rather than measured values. These use a LIVE probe,
    # never the recorded baseline: that baseline was taken with RSI deliberately
    # off, so treating it as current would permanently report "not running" even
    # after the engineer starts it.
    cart_ok, cart_why = I.cartesian_capability()
    rt = C.probe_runtime(allow_network=args.probe_network)
    hold_blockers = []
    if rt["ext_trigger_open"] is None:
        hold_blockers.append(
            f"tcp/{C.PORT_EXT_TRIGGER_TCP} state UNKNOWN -- re-run with "
            f"--probe-network from the Jetson to observe it")
    elif not rt["ext_trigger_open"]:
        hold_blockers.append(
            f"controller-side trigger tcp/{C.PORT_EXT_TRIGGER_TCP} not reachable; "
            f"the RSI program is not started")
    if rt["rsi_program_running"] is not True:
        hold_blockers.append(
            "RSI program state is only proven by receiving a Rob frame; run the "
            "HOLD handshake to establish it")
    extra = {
        "hold_handshake": (not hold_blockers, hold_blockers),
        "supervised_student_prefix": (
            False, ["CLI-to-RSI execution requires an operator arming action "
                    "and a live HOLD handshake first"]),
        "astra_direct": (cart_ok, cart_why),
    }
    rep = dc.report(cfg, extra_gates=extra)
    rep["cell_state_live"] = rt
    rep["cell_state_baseline"] = {
        "ext_trigger_open": C.BASELINE_EXT_TRIGGER_OPEN,
        "rsi_program_running": C.BASELINE_RSI_PROGRAM_RUNNING,
        "rsi_socket_bound": C.BASELINE_JETSON_RSI_SOCKET_BOUND,
        "note": C.BASELINE_NOTE}
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




def cmd_live(args: argparse.Namespace) -> int:
    """Three-model loop against the REAL arm. Holds position; sends nothing.

    Live: the RSI session, the measured joint angles, the cameras, pi0.5, the
    Qwen monitor and Astra. Not live: actuation. See live.py for why motion is
    structurally impossible here rather than merely disabled.
    """
    import socket as _s
    from . import cell as C
    import base64 as _b64

    from .cameras import (ENCODE_MIME, LiveCameras, frames_to_data_urls,
                          frames_to_packet_refs)
    from .execution import ExecutionController, ProtocolAdapter
    from .live import LiveObservationRun, RSI_HEALTHY_HZ
    from .pipeline import PolicyPipeline, resolve_mode
    from .transports import AstraReviewSource
    from .vlm_backends import VlmConfig, make_backend

    if not args.bind:
        print("REFUSED: binding udp/%d is an action on the robot network.\n"
              "  Pass --bind to do it. This mode holds position and sends no\n"
              "  motion, but it DOES take the port, so the teleop RSI driver\n"
              "  must be stopped first." % C.PORT_RSI_UDP, file=sys.stderr)
        return 2

    # ---- pi0.5 proposal source ------------------------------------------
    if args.chunks_file:
        from .transports import LocalPi05ProposalSource
        src = LocalPi05ProposalSource.from_file(args.chunks_file)
        def pi05_infer(obs):
            return src.propose(obs)
        pi05_desc = f"REPLAY from {args.chunks_file} (NOT live inference)"
    elif args.pi05_url:
        import urllib.request as _u
        # A loopback model server must not be routed through the outbound
        # proxy: urllib does NOT bypass 127.0.0.1 unless no_proxy says so, and
        # sending frames through Clash to reach this machine would add the very
        # latency this measurement is trying to characterise.
        _direct = _u.build_opener(_u.ProxyHandler({}))
        _seen_contract = {"checked": False}

        def pi05_infer(obs):
            body = json.dumps({
                "state": obs["joints_deg"],
                "images": obs.get("images") or {},
                "task": obs.get("task", ""),
                "observation_id": obs["observation_id"]}).encode()
            req = _u.Request(args.pi05_url, data=body, method="POST",
                             headers={"Content-Type": "application/json"})
            with _direct.open(req, timeout=args.pi05_timeout) as r:
                out = json.loads(r.read().decode())
            # Enforce the checkpoint contract on the RESPONSE. The eight
            # look-alike finetunes report use_relative_actions=False and return
            # targets in the wrong space without erroring, so trusting the
            # endpoint's name is not enough.
            meta = (out or {}).get("meta") or {}
            rel = meta.get("use_relative_actions")
            if rel is not True:
                return {"ok": False,
                        "error": f"pi0.5 server reports use_relative_actions="
                                 f"{rel!r}, expected True; refusing the chunk "
                                 f"(checkpoint={meta.get('checkpoint')!r})"}
            if not _seen_contract["checked"]:
                _seen_contract["checked"] = True
                print(f"  pi0.5 contract OK: use_relative_actions=True, "
                      f"checkpoint={meta.get('checkpoint')}")
            return out
        pi05_desc = f"LIVE via {args.pi05_url}"
    else:
        print("REFUSED: no pi0.5 source. Pass --pi05-url (a running server) or\n"
              "  --chunks-file (replay, which is NOT live inference).\n"
              "  Note the teleop container serves inference AND owns udp/59152;\n"
              "  it cannot both be stopped for this test and serve pi0.5, so the\n"
              "  checkpoint must be served by a separate process.", file=sys.stderr)
        return 2

    # ---- cameras ---------------------------------------------------------
    cams = None
    if args.cameras:
        mapping = {}
        for pair in args.cameras.split(","):
            name, _, node = pair.partition(":")
            mapping[name.strip()] = int(node)
        cams = LiveCameras(mapping).start()

    def grab():
        if cams is None:
            return [], {}, {}
        frames = cams.capture()
        named = {name: f"data:{ENCODE_MIME};base64," +
                       _b64.b64encode(fr.png).decode()
                 for name, fr in sorted(frames.items()) if fr.png}
        return (frames_to_data_urls(frames), frames_to_packet_refs(frames),
                named)

    # ---- monitor + astra --------------------------------------------------
    cfg = VlmConfig.jetson(timeout_s=args.monitor_timeout)
    if args.monitor_url:
        cfg = VlmConfig(endpoint=args.monitor_url, model=cfg.model,
                        api_key_env=cfg.api_key_env,
                        timeout_s=args.monitor_timeout,
                        max_frames=cfg.max_frames)
    backend = make_backend(args.monitor_backend, config=cfg)
    review = AstraReviewSource(
        base_url=args.astra_url, model=args.astra_model,
        api_key_env=args.astra_key_env, enabled=args.astra_live,
        dry_run=not args.astra_live, reasoning=args.astra_effort,
        background=args.astra_background, deadline_s=args.astra_deadline)
    pipe = PolicyPipeline(resolve_mode(args.policy_mode), backend=backend,
                          shadow=True,
                          astra_review=(review.review if args.astra_live else None))

    # ---- RSI session ------------------------------------------------------
    class UdpTransport:
        def __init__(self, host, port, timeout):
            self.sock = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
            try:
                self.sock.bind((host, port))
            except OSError as exc:
                self.sock.close()
                raise SystemExit(
                    f"\nCANNOT BIND udp/{port} on {host}: {exc}\n"
                    f"  Stop the teleop RSI driver first: ss -lunp | grep {port}"
                    ) from None
            self.sock.settimeout(timeout)

        def receive(self, timeout_s):
            try:
                return self.sock.recvfrom(4096)
            except (TimeoutError, OSError):
                return None

        def send(self, payload, peer):
            self.sock.sendto(payload, peer)

        def close(self):
            self.sock.close()

    print(f"LIVE OBSERVATION on {args.host}:{C.PORT_RSI_UDP}")
    print(f"  motion   : IMPOSSIBLE -- holds position, every reply STOPFLAG=1")
    print(f"  pi0.5    : {pi05_desc}")
    print(f"  monitor  : {backend.__class__.__name__} ({cfg.model})")
    print(f"  astra    : {'LIVE (PAID)' if args.astra_live else 'DRY RUN'}")
    print(f"  cameras  : {args.cameras or 'NONE -- the monitor will be blind'}")
    print(f"  mode     : {args.policy_mode}")
    print(f"  audit    : {args.audit}\n")
    print("  Ctrl-C to stop. Keep this window open: if it exits, nothing\n"
          "  answers the controller and RSI faults.\n")

    tr = UdpTransport(args.host, C.PORT_RSI_UDP, args.timeout)
    ctrl = ExecutionController(ProtocolAdapter(tr), allow_motion=False)
    ctrl.open_session()
    run = LiveObservationRun(ctrl, pi05_infer=pi05_infer, pipeline=pipe,
                             grab_frames=grab, audit_path=args.audit,
                             min_model_interval_s=args.model_interval,
                             task=args.task or "")
    t0 = time.time()
    first = None
    live_tty = sys.stdout.isatty()
    win_t, win_n, rate = t0, 0, 0.0
    started_models = False
    try:
        while True:
            out = run.serve_one(timeout_s=args.timeout)
            now = time.time()
            if out is None:
                if first is None and now - t0 > args.wait:
                    print("\n  NO FRAMES RECEIVED -- the RSI program is not "
                          "running or is not pointed here.", file=sys.stderr)
                    return 2
                continue
            if first is None:
                first = out
                print(f"  RSI IS RUNNING -- first frame IPOC={out.ipoc}")
                print(f"  measured joints: "
                      f"{[round(v, 2) for v in (out.measured or [])]}\n")
            if not started_models:
                run.start_models()
                started_models = True
            win_n += 1
            if now - win_t >= 1.0:
                rate, win_t, win_n = win_n / (now - win_t), now, 0
            if live_tty and ctrl.adapter.frames_in % 25 == 0:
                a = ctrl.adapter
                c = run.cycles[-1] if run.cycles else None
                tag = "THINKING" if run._thinking.is_set() else "idle    "
                health = "OK " if rate >= RSI_HEALTHY_HZ else "LOW"
                sys.stdout.write(
                    f"\r  [{health}] {rate:6.1f} Hz | frames {a.frames_in:>7}"
                    f" | {tag} | model {len(run.cycles):>4}"
                    f" | last {c.total_s if c else '-'}s"
                    f" | astra {sum(1 for x in run.cycles if x.astra_called):>3}"
                    f" | {now - t0:6.1f}s  ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        run.stop()
        elapsed = time.time() - t0
        summary = run.summary(elapsed)
        print("\n" + json.dumps(summary, indent=2))
        tr.close()
        if cams is not None:
            try:
                cams.stop()
            except Exception:                                   # noqa: BLE001
                pass
    return 0


def cmd_hold(args: argparse.Namespace) -> int:
    """HOLD handshake: bind udp/59152, answer every frame, command NOTHING.

    This is the only safe way to prove the RSI program is running, and the
    reason is the protocol itself: the controller expects a reply within 4 ms,
    so a passive listener that binds and stays silent FAULTS IT. Checking by
    "just listening" is not a lighter-touch version of this -- it is worse.

    HOLD answers every frame with the pose the arm is already at and
    STOPFLAG=1. It keeps the session alive and moves nothing.
    """
    import socket as _s
    from . import cell as C
    from .execution import ExecutionController, ProtocolAdapter, State

    if not args.bind:
        print("REFUSED: binding udp/%d is an action on the robot network.\n"
              "  Pass --bind to do it. Before you do, understand that this "
              "process MUST keep answering:\n"
              "  a silent listener on this port faults the controller."
              % C.PORT_RSI_UDP, file=sys.stderr)
        return 2

    class UdpTransport:
        def __init__(self, host, port, timeout):
            self.sock = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
            try:
                self.sock.bind((host, port))
            except OSError as exc:
                self.sock.close()
                raise SystemExit(
                    f"\nCANNOT BIND udp/{port} on {host}: {exc}\n"
                    f"  Only ONE process may own this port. On this cell it is\n"
                    f"  normally held by the teleop inference container, which\n"
                    f"  must be stopped first -- and stopping it takes the pi0.5\n"
                    f"  production path offline for the duration of this test.\n"
                    f"  Check with:  ss -lunp | grep {port}\n"
                    f"  If the address itself is wrong, pass --host.") from None
            self.sock.settimeout(timeout)

        def receive(self, timeout_s):
            try:
                return self.sock.recvfrom(4096)
            except (TimeoutError, OSError):
                return None

        def send(self, payload, peer):
            self.sock.sendto(payload, peer)

        def close(self):
            self.sock.close()

    print(f"HOLD handshake on {args.host}:{C.PORT_RSI_UDP}")
    print("  motion: DISABLED. Every reply carries the measured pose and "
          "STOPFLAG=1.")
    print(f"  waiting up to {args.wait:.0f}s for the controller to connect ...\n")
    tr = UdpTransport(args.host, C.PORT_RSI_UDP, args.timeout)
    ctrl = ExecutionController(ProtocolAdapter(tr), allow_motion=False)
    ctrl.open_session()
    t0 = time.time()
    first = None
    unlimited = args.frames <= 0
    wait_forever = args.wait <= 0
    live = sys.stdout.isatty()
    win_t, win_n, rate = t0, 0, 0.0
    last_seen = None
    gaps = 0
    print("  Ctrl-C to stop. This window must stay open: the moment it exits,\n"
          "  nothing answers the controller and RSI faults.\n")
    try:
        while unlimited or len(ctrl.outcomes) < args.frames:
            out = ctrl.serve_cycle(timeout_s=args.timeout)
            now = time.time()
            if out is None:
                if first is None and not wait_forever and now - t0 > args.wait:
                    print("  NO FRAMES RECEIVED.", file=sys.stderr)
                    print("  The RSI program is not running, or it is not "
                          "pointed at this host.", file=sys.stderr)
                    print(f"  Start it via the trigger on "
                          f"tcp/{C.PORT_EXT_TRIGGER_TCP}, then retry.",
                          file=sys.stderr)
                    return 2
                if first is not None and last_seen and now - last_seen > 0.05:
                    gaps += 1               # >12 RSI cycles with no frame
                    last_seen = now
                continue
            if first is None:
                first = out
                print(f"  RSI IS RUNNING -- first frame IPOC={out.ipoc}")
                print(f"  measured joints: "
                      f"{[round(v, 2) for v in (out.measured or [])]}\n")
            last_seen = now
            win_n += 1
            if now - win_t >= 1.0:
                rate, win_t, win_n = win_n / (now - win_t), now, 0
            if live and len(ctrl.outcomes) % 25 == 0:
                a = ctrl.adapter
                health = "OK " if (rate >= 200 and not a.malformed) else "CHECK"
                sys.stdout.write(
                    f"\r  [{health}] {rate:6.1f} Hz | frames {a.frames_in:>7}"
                    f" in / {a.frames_out:>7} out | ipoc {a.last_ipoc:>8}"
                    f" | gaps {gaps:>4} jumps {a.ipoc_jumps:>4}"
                    f" | malformed {a.malformed:>3} stale {a.ipoc_regressions:>4}"
                    f" | {ctrl.state.value:<9} | {now - t0:6.1f}s  ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n  interrupted -- controller will see silence from here")
    finally:
        a = ctrl.adapter
        dt = time.time() - t0
        print(f"\n  frames in/out   : {a.frames_in} / {a.frames_out}")
        print(f"  malformed       : {a.malformed}")
        print(f"  IPOC regressions: {a.ipoc_regressions}   jumps: {a.ipoc_jumps}")
        print(f"  elapsed         : {dt:.2f}s "
              f"({a.frames_in / dt if dt else 0:.0f} Hz observed)")
        print(f"  state           : {ctrl.state.value}")
        tr.close()
    ok = (a.frames_in > 0 and a.frames_in == a.frames_out
          and a.ipoc_regressions == 0 and a.malformed == 0)
    print(f"\n  HOLD handshake: {'CLEAN' if ok else 'PROBLEMS -- see counters'}")
    return 0 if ok else 1


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


def _arming_identity(args, cfg: dict) -> tuple[list, object | None, list[str]]:
    """Build (allowlist, signing key, blockers) for a loop that may arm.

    Both were hardcoded to [] and None, which refuses everything -- correct as
    a default, but indistinguishable from a configuration bug. This makes the
    refusal explicit and tells the operator what to supply. The key is read
    from the environment BY NAME and never printed or stored.
    """
    from .safety import (ArmingRefused, identity_from_config,
                         signing_key_from_env)
    allowlist, secret, blockers = [], None, []
    src = dict(cfg or {})
    for f in ("robot_model", "controller_serial", "rsi_host", "rsi_port"):
        v = getattr(args, f, None)
        if v:
            src[f] = v
    try:
        allowlist = [identity_from_config(src)]
    except ArmingRefused as exc:
        blockers.append(str(exc))
    try:
        secret = signing_key_from_env(getattr(args, "signing_key_env", "") or
                                      "KUKA_SIGNING_KEY")
    except ArmingRefused as exc:
        blockers.append(str(exc))
    return allowlist, secret, blockers


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
    _allow, _secret, _arm_blockers = _arming_identity(args, flat)
    if _arm_blockers:
        print("  arming       : REFUSED")
        for b in _arm_blockers:
            print(f"                 - {b}")
    else:
        print(f"  arming       : identity {_allow[0].key()}, signing key present")
    loop = KukaReviewLoop(
        mode=Mode.REVIEWED_EXECUTION, config=flat, raw_config=cfg,
        proposal_source=src, review_source=None, gateway=gw,
        fk=make_fk(), audit=audit,
        target=RobotIdentity(ROBOT_MODEL, args.serial or "UNSET", "172.17.255.2", 59152),
        allowlist=_allow, supervisor=Supervisor(), ledger=CommandLedger(),
        secret=_secret)
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
    from .pipeline import PolicyPipeline, describe_trajectory, resolve_mode
    from .transports import AstraReviewSource, LocalPi05ProposalSource
    from .vlm_backends import VlmConfig, make_backend
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

    # --- the VLM gate, when a mode asks for it -----------------------------
    policy_mode = resolve_mode(args.policy_mode)
    gate = None
    if policy_mode.value != "pi05_only":
        vcfg = VlmConfig.jetson(timeout_s=args.monitor_timeout)
        if args.monitor_url:
            vcfg = VlmConfig(endpoint=args.monitor_url, model=vcfg.model,
                             timeout_s=args.monitor_timeout,
                             max_frames=vcfg.max_frames)
        backend = make_backend(args.monitor_backend, config=vcfg)
        probe = backend.probe()
        print(f"  monitor      : {policy_mode.value} via {backend.name} "
              f"({getattr(backend, 'model', '?')})")
        print(f"  monitor svc  : {'reachable' if probe.available else 'UNAVAILABLE'}"
              f" -- {probe.reason[:70]}")
        if not probe.available and not args.monitor_shadow:
            print("  REFUSED: gating was requested but no monitor service is "
                  "reachable. Start one (see vlm_backends.SERVER_EXAMPLES) or "
                  "run with --monitor-shadow.", file=sys.stderr)
            return 2
        # FIX 3: shadow mode is worthless if the records are printed and
        # dropped. One append-only JSONL row per evaluation.
        mon_path = Path(args.monitor_audit or "monitor_shadow.jsonl")
        mon_path.parent.mkdir(parents=True, exist_ok=True)
        mon_f = mon_path.open("a")

        def _persist(row):
            mon_f.write(json.dumps(row, default=str) + "\n")
            mon_f.flush()

        # BUG A: the escalation path was never wired -- PolicyPipeline was
        # built without astra_review, so at escalation the branch was skipped
        # silently. `_escalate_to_astra` is installed below, after the reviewer
        # exists.
        gate = PolicyPipeline(policy_mode, backend=backend,
                              shadow=args.monitor_shadow, on_record=_persist)
        print(f"  monitor log  : {mon_path} (append-only)")
        print(f"  monitor mode : {'SHADOW (records only)' if gate.shadow else 'GATING'}")

    review = AstraReviewSource(
        base_url=args.astra_url, model=args.astra_model,
        api_key_env=args.astra_key_env, enabled=args.astra_live,
        dry_run=not args.astra_live, reasoning=args.astra_effort,
        background=args.astra_background, deadline_s=args.astra_deadline,
        # background and stream are two answers to the same proxy cutoff and
        # are mutually exclusive; background wins because a stream was measured
        # hanging past the client timeout rather than failing.
        stream=(not args.astra_no_stream) and not args.astra_background,
        attempts=args.astra_attempts)
    ok, why = review.preflight()
    print(f"  astra        : {'LIVE (PAID)' if args.astra_live else 'DRY RUN'} -- {why}"
          f" [{'streamed' if review.stream else 'single response'}]")

    # BUG B: Astra was installed as the loop's per-tick review source regardless
    # of policy mode, so pi05_local_monitor -- the mode whose whole purpose is
    # NOT calling Astra -- called it on every tick. That inverts the design: the
    # triage layer became pure overhead on top of the cost it exists to avoid.
    #
    # Astra is now the loop's reviewer ONLY in the mode that reviews every tick.
    # In the monitored modes the loop runs unreviewed and Astra is reached
    # exclusively through escalation.
    astra_per_tick = policy_mode.value == "pi05_only" or gate is None
    escalations = {"n": 0}

    def _escalate_to_astra(packet):
        """Called by PolicyPipeline only on PERSISTENT adverse evidence."""
        escalations["n"] += 1
        print(f"      ESCALATING to Astra (#{escalations['n']})")
        r = review.review(packet)
        if isinstance(r, dict):
            return r
        return {"ok": bool(getattr(r, "decision", None)),
                "decision": getattr(r, "decision", None),
                "error": getattr(r, "error", None)}

    if gate is not None:
        gate.astra_review = _escalate_to_astra
        if policy_mode.value == "pi05_local_monitor":
            # This mode must never escalate. Make that structural rather than
            # relying on the caller not asking for it.
            gate.astra_review = None
        pok, pblockers = gate.preflight()
        print(f"  astra path   : "
              + ("per-tick review (pi05_only)" if astra_per_tick
                 else "escalation only -- Astra is NOT called per tick"))
        if not pok:
            print(f"  monitor preflight: {'; '.join(pblockers)[:100]}")
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
        proposal_source=src,
        review_source=review if astra_per_tick else None,
        gateway=ShadowGateway(),
        fk=fk, audit=audit,
        target=RobotIdentity(ROBOT_MODEL, args.serial or "UNSET",
                             "172.17.255.2", 59152),
        # cmd_run is shadow-only (ShadowGateway has no socket), so an empty
        # allowlist and absent key are correct here and stay explicit.
        allowlist=[], supervisor=Supervisor(), ledger=CommandLedger(),
        secret=None)
    print()
    answered, tokens = 0, 0
    prev_frames: list[tuple[str, str]] = []
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
        if gate is not None:
            # FIX 2: the monitor must see pi0.5's ACTUAL proposal. `rec.proposal`
            # is the model output; `s.recorded_chunk` is the human demonstration
            # and was what the first version sent.
            pi05_chunk = ((rec.proposal or {}).get("values")
                          or (rec.proposal or {}).get("full_chunk")
                          or [])
            # FIX 1: label by time AND viewpoint, and carry the previous tick
            # forward so there is a real temporal pair.
            cur = [(f"t base", u) for u in []]
            labelled = []
            for cam, url in sorted(zip(sorted(s.frames), 
                                       obs.get("image_data_urls") or [])):
                labelled.append((f"t {cam}", url))
            if prev_frames:
                labelled = [(f"t-1 {c}", u) for c, u in prev_frames] + labelled
            g = gate.step(
                state=s.state, proposed_steps=int(d.get("steps") or 0) or 5,
                frames=labelled,
                state_text=f"joints_deg={[round(v, 2) for v in s.state[:6]]} "
                           f"gripper={s.state[6]:.3f}",
                intent_text=describe_trajectory(pi05_chunk, s.state),
                episode_id=ep.m.episode_id, task=ep.m.instruction,
                proposed_chunk=pi05_chunk, frames_meta=s.frames,
                fk_preview=rec.fk_preview,
                image_data_urls=list(obs.get("image_data_urls") or []))
            prev_frames = [(c, u) for c, u in
                           zip(sorted(s.frames), obs.get("image_data_urls") or [])]
            print(f"      monitor: {g.gate['disposition'] if g.gate else 'n/a'} "
                  f"steps={g.executed_steps} (baseline {g.baseline_steps})"
                  f"{' SHADOW' if g.shadow else ''}"
                  f"{'  ESCALATED' if g.escalated else ''}")
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
    if gate is not None:
        m = gate.metrics()
        print(f"  astra reached via escalation: {escalations['n']} time(s)")
        print(f"\n  monitor: {m['evaluations'] if 'evaluations' in m else m['cycles']} "
              f"cycle(s), {m['escalations']} escalation(s), "
              f"{m['rejected_monitor_outputs']} rejected output(s)")
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
    rn.add_argument("--policy-mode", default="pi05_only",
                    help="pi05_only | pi05_local_monitor | "
                         "pi05_local_monitor_astra (aliases accepted)")
    rn.add_argument("--monitor-backend", default="unconfigured",
                    help="local | mock | unconfigured (a800/jetson are aliases "
                         "for local and are deprecated: the monitor runs on the "
                         "Jetson, the A800 cannot serve it)")
    rn.add_argument("--monitor-timeout", type=float, default=6.0,
                    help="seconds; set from a MEASURED reading on this device")
    rn.add_argument("--monitor-url", help="OpenAI-compatible endpoint")
    rn.add_argument("--monitor-audit", help="append-only JSONL of monitor records")
    rn.add_argument("--monitor-shadow", action="store_true", default=True,
                    help="record monitor decisions without gating (default)")
    rn.add_argument("--monitor-gate", dest="monitor_shadow",
                    action="store_false",
                    help="LET THE MONITOR GATE. Off by default.")
    rn.add_argument("--astra-background", action="store_true", default=True,
                    help="submit and poll instead of holding one connection "
                         "(default on; the proxy cuts held connections at ~60s)")
    rn.add_argument("--astra-no-background", dest="astra_background",
                    action="store_false")
    rn.add_argument("--astra-deadline", type=float, default=300.0)
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

    lv = sub.add_parser("live", help="three-model loop on the REAL arm; "
                                     "holds position, sends no motion")
    lv.add_argument("--host", default="172.17.255.2")
    lv.add_argument("--bind", action="store_true",
                    help="REQUIRED. Binds udp/59152 and answers every frame.")
    lv.add_argument("--timeout", type=float, default=0.05)
    lv.add_argument("--wait", type=float, default=300.0,
                    help="seconds to wait for the first controller frame")
    lv.add_argument("--pi05-url",
                    help="a RUNNING pi0.5 server, e.g. http://127.0.0.1:18830/infer. "
                         "It must be a separate process from the teleop container, "
                         "which owns udp/59152 and has to be stopped.")
    lv.add_argument("--pi05-timeout", type=float, default=30.0)
    lv.add_argument("--chunks-file",
                    help="replay precomputed chunks instead of inferring. NOT "
                         "live inference; recorded as such in the audit.")
    lv.add_argument("--cameras",
                    help="name:node pairs, e.g. base:0,wrist:2 (on this cell "
                         "only nodes 0 and 2 capture)")
    lv.add_argument("--policy-mode", default="pi05_local_monitor_astra")
    lv.add_argument("--monitor-backend", default="local")
    lv.add_argument("--monitor-url")
    lv.add_argument("--monitor-timeout", type=float, default=6.0)
    lv.add_argument("--astra-url", default="https://api.openai.com/v1/responses")
    lv.add_argument("--astra-model", default="gpt-6-astra")
    lv.add_argument("--astra-key-env", default="OPENAI_API_KEY")
    lv.add_argument("--astra-effort", default="low")
    lv.add_argument("--astra-live", action="store_true",
                    help="make PAID Astra calls; dry-run otherwise")
    lv.add_argument("--astra-background", action="store_true", default=True,
                    help="submit and poll instead of holding one connection. "
                         "DEFAULT ON: the Jetson's outbound proxy was measured "
                         "cutting a held connection at ~60s while a review "
                         "needs ~130s, so the non-background path loses an "
                         "answer it already paid for.")
    lv.add_argument("--astra-no-background", dest="astra_background",
                    action="store_false")
    lv.add_argument("--astra-deadline", type=float, default=300.0,
                    help="TRUE wall-clock bound across submit+poll")
    lv.add_argument("--model-interval", type=float, default=1.0,
                    help="minimum seconds between model cycles")
    lv.add_argument("--task", default="")
    lv.add_argument("--audit", default="live_observation.jsonl")
    lv.set_defaults(func=cmd_live)

    hd = sub.add_parser("hold", help="HOLD handshake: prove RSI is running, move nothing")
    hd.add_argument("--host", default="172.17.255.2")
    hd.add_argument("--frames", type=int, default=0,
                    help="0 = run until Ctrl-C (what you want for a wire test)")
    hd.add_argument("--wait", type=float, default=300.0,
                    help="seconds to wait for the FIRST frame while the "
                         "controller side is started; 0 = wait forever")
    hd.add_argument("--timeout", type=float, default=0.05)
    hd.add_argument("--bind", action="store_true",
                    help="REQUIRED. Binds udp/59152 and answers every frame.")
    hd.set_defaults(func=cmd_hold)

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
    gg.add_argument("--probe-network", action="store_true",
                    help="observe the controller trigger port (read-only "
                         "connect; opens nothing). Run this on the Jetson.")
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
