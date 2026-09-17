"""Shadow end-to-end: all four modes over synthetic fixtures. Sends nothing.

Run: python -m hybrid_rollout.robodojo.kuka.shadow_e2e
"""
from __future__ import annotations

import json, pathlib, sys, tempfile, time

from .experiment import CameraAssessment, flatten_config, load_config, retry_limits
from .loop import AuditLog, KukaReviewLoop, Outcome
from .safety import CommandLedger, Mode, RobotIdentity, Supervisor, missing_config
from .transports import OfflineProposalSource, RecordedReview, ShadowGateway

FIX = pathlib.Path(__file__).parent / "fixtures"
ROBOT = RobotIdentity("KUKA LBR iisy 11 R1300", "SN-SHADOW", "172.17.255.2", 59152)


def main() -> int:
    chunks = {k: v for k, v in json.loads((FIX / "chunks.json").read_text()).items()
              if not k.startswith("_")}
    decs = {k: v for k, v in json.loads((FIX / "decisions.json").read_text()).items()
            if not k.startswith("_")}
    raw = load_config()
    flat = flatten_config(raw)
    out = pathlib.Path(tempfile.mkdtemp()) / "shadow_audit.jsonl"

    print("SHADOW E2E -- synthetic fixtures, no robot / API / simulator / model")
    print(f"  shipped config missing values: "
          f"{len(missing_config(flat, Mode.REVIEWED_EXECUTION))} "
          f"(reviewed_execution) / "
          f"{len(missing_config(flat, Mode.ASTRA_DIRECT))} (astra_direct)")
    print(f"  audit: {out}\n")
    print(f"  {'mode':20s} {'stage':10s} {'outcome':16s} {'safe':5s} sent  reason")
    print("  " + "-" * 92)
    rows = 0
    for mode in Mode:
        lp = KukaReviewLoop(
            mode=mode, config=flat, raw_config=raw,
            proposal_source=OfflineProposalSource(chunks, checkpoint_id="synthetic"),
            review_source=RecordedReview(decs), gateway=ShadowGateway(),
            target=ROBOT, allowlist=[ROBOT],
            supervisor=Supervisor(time.time(), True, True),
            ledger=CommandLedger(), secret=b"shadow-only", audit=AuditLog(out))
        rec = lp.step({"observation_id": "obs-feasible",
                       "state": [-76.55, -94.75, 66.60, 8.53, 19.30, 5.19, 0.999],
                       "epoch": time.time()},
                      camera_assessment=CameraAssessment(
                          "door may be ajar", [{"camera": "base", "frame": 80}], 0.6))
        rows += 1
        sent = bool((rec.gateway or {}).get("sent"))
        print(f"  {mode.value:20s} {rec.stage_reached:10s} {rec.outcome:16s} "
              f"{str(rec.execution_safe):5s} {str(sent):5s} {rec.reason[:44]}")
    audit = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(audit) == rows
    shapes = {frozenset(r) for r in audit}
    print(f"\n  audit rows: {len(audit)}  | identical row shape across modes: "
          f"{len(shapes) == 1}")
    print(f"  commands actually sent: "
          f"{sum(1 for r in audit if (r.get('gateway') or {}).get('sent'))}")
    print(f"  success declared from prose: "
          f"{sum(1 for r in audit if (r.get('success') or {}).get('success') is True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
