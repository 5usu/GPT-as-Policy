"""HTTP surface for a locally loaded pi0.5 checkpoint. Proposal-only.

WHY THIS FILE EXISTS
  jetson/pi05_serve_jetson.py imports this module, overrides load_policy and
  predict with the production KUKA path (teleoperation/pi05_standalone.py), and
  calls main(). The module it imports was never committed, so that launcher has
  been failing with ImportError -- which is why "live pi0.5" has so far meant
  replaying precomputed chunks.

  This supplies only the serving surface: argument parsing, checkpoint contract
  enforcement, and a small stdlib HTTP server. It deliberately does NOT know how
  to load or run a policy. The deployment engineer's loader is the real one and
  is injected over the top.

WHAT IT REFUSES
  The checkpoint contract is enforced here rather than trusted. pi05_corrected_b8
  reports use_relative_actions=True; the eight finetunes under
  oss://i-robot-data/models/ look correct in every other respect and report
  False. Loading one does not crash -- it silently returns targets in the wrong
  space, which on a real arm is a wrong-place motion, not an error message.
  A checkpoint that does not report True is refused at load, not at use.

NOT A CONTROL PATH
  This serves proposals over HTTP. It holds no socket to the controller, and
  nothing here can command motion.
"""
from __future__ import annotations

import json
from typing import Any

SCHEMA = "kuka.pi05_serve.v1"

#: Populated by load_policy(); read by predict(). Both are overridden by the
#: Jetson launcher, which owns the real loader.
_STATE: dict[str, Any] = {"policy": None, "pre": None, "post": None,
                          "meta": {}, "device": None}

#: The distinguishing field. See module docstring.
REQUIRED_META = {"use_relative_actions": True}

EXPECTED_STEPS = 50
EXPECTED_DIMS = 7          # 6 joints + gripper, absolute degrees


class ContractViolation(RuntimeError):
    """The checkpoint is not the one this cell was validated against."""


def check_contract(meta: dict[str, Any]) -> None:
    """Raise unless the loaded checkpoint matches the validated contract."""
    for key, want in REQUIRED_META.items():
        got = meta.get(key)
        if got != want:
            raise ContractViolation(
                f"checkpoint reports {key}={got!r}, expected {want!r}. This is "
                f"almost certainly a different training run. Such a checkpoint "
                f"loads cleanly and returns targets in the WRONG SPACE, so it "
                f"is refused here rather than at use. checkpoint="
                f"{meta.get('checkpoint')!r}")


def load_policy(checkpoint: str, device: str = "cuda"):
    """Overridden by the Jetson launcher, which owns the real loader."""
    raise NotImplementedError(
        "no loader installed. This module serves HTTP only; run it through "
        "jetson/pi05_serve_jetson.py, which installs the production KUKA "
        "loader (teleoperation/pi05_standalone.load_pi05_policy).")


def predict(observation: dict[str, Any]) -> dict[str, Any]:
    """Overridden by the Jetson launcher."""
    return {"ok": False, "error": "no predict installed; see load_policy"}


def validate_rows(result: dict[str, Any]) -> dict[str, Any]:
    """Shape-check a proposal before it leaves the process.

    A malformed chunk is refused here so the consumer never has to guess
    whether a short or non-finite row meant anything.
    """
    if not result.get("ok"):
        return result
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        return {"ok": False, "error": "predict returned no rows"}
    if len(rows) != EXPECTED_STEPS:
        return {"ok": False,
                "error": f"expected {EXPECTED_STEPS} steps, got {len(rows)}"}
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != EXPECTED_DIMS:
            return {"ok": False,
                    "error": f"row {i} has {len(row) if isinstance(row, list) else '?'} "
                             f"values, expected {EXPECTED_DIMS}"}
        for v in row:
            if not isinstance(v, (int, float)) or v != v or v in (
                    float("inf"), float("-inf")):
                return {"ok": False, "error": f"row {i} contains {v!r}"}
    return result


def _handler_class():
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                                   # noqa: N802
            if self.path.rstrip("/") in ("/health", "/healthz"):
                loaded = _STATE.get("policy") is not None
                return self._send(200 if loaded else 503, {
                    "ok": loaded, "schema": SCHEMA,
                    "meta": _STATE.get("meta", {}),
                    "note": "proposal-only; this process holds no robot socket"})
            return self._send(404, {"ok": False, "error": "GET /health only"})

        def do_POST(self):                                  # noqa: N802
            if self.path.rstrip("/") != "/infer":
                return self._send(404, {"ok": False, "error": "POST /infer only"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                obs = json.loads(self.rfile.read(n).decode())
            except Exception as exc:                        # noqa: BLE001
                return self._send(400, {"ok": False,
                                        "error": f"bad request: {exc}"[:200]})
            if not isinstance(obs, dict) or "state" not in obs:
                return self._send(400, {
                    "ok": False,
                    "error": "body must be {state: [...], images: {...}, task: str}"})
            out = validate_rows(predict(obs))
            # The contract travels WITH every response, so a consumer can verify
            # what produced the chunk instead of trusting the endpoint's name.
            out.setdefault("meta", _STATE.get("meta", {}))
            out["schema"] = SCHEMA
            return self._send(200 if out.get("ok") else 500, out)

        def log_message(self, fmt, *a):                     # quieter
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    import argparse
    from http.server import ThreadingHTTPServer

    p = argparse.ArgumentParser(description="Serve pi0.5 proposals over HTTP.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="127.0.0.1",
                   help="loopback by default; this serves a model, not a robot")
    p.add_argument("--port", type=int, default=18830)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-contract-mismatch", action="store_true",
                   help="load a checkpoint that does NOT report "
                        "use_relative_actions=True. It will return targets in "
                        "the wrong space; for offline inspection only.")
    args = p.parse_args(argv)

    policy, pre, post, meta = load_policy(args.checkpoint, args.device)
    try:
        check_contract(meta)
    except ContractViolation as exc:
        if not args.allow_contract_mismatch:
            print(json.dumps({"event": "refused", "error": str(exc)}), flush=True)
            return 2
        print(json.dumps({"event": "contract_override",
                          "warning": str(exc)}), flush=True)
    _STATE.update(policy=policy, pre=pre, post=post, meta=meta)

    srv = ThreadingHTTPServer((args.host, args.port), _handler_class())
    print(json.dumps({"event": "loaded", "schema": SCHEMA,
                      "host": args.host, "port": args.port,
                      "use_relative_actions": meta.get("use_relative_actions"),
                      "checkpoint": meta.get("checkpoint")}), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
