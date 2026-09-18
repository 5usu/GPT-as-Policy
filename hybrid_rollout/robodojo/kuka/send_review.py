"""Send Phase 1 review packets to Astra. THIS SPENDS MONEY.

    # dry run: builds and hashes every body, sends nothing
    python -m hybrid_rollout.robodojo.kuka.send_review --episode /root/kuka_ep

    # live: one attempt per packet, no retries, no fallback model
    python -m hybrid_rollout.robodojo.kuka.send_review --episode /root/kuka_ep --send

DRY RUN IS THE DEFAULT and prints a body sha256 plus byte count per packet, so a
batch can be inspected and costed before any of it is paid for.

ONE ATTEMPT PER PACKET. A failure is recorded and the run continues to the next
packet; it is never retried. A retry on a review is not free and not idempotent,
and silently re-asking until you get an answer is how a batch stops meaning what
it claims to mean.

CREDENTIALS are read from the environment by name and never printed, logged or
placed in any output file. The written journal contains the request DIGEST, not
the request.

MEASURED: bodies run ~770 KB with two 640x480 frames attached. At that size the
endpoint throttled roughly every other call in a back-to-back batch, so
--gap-seconds defaults to a pause between calls.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

from .transports import AstraReviewSource

DEFAULT_ENDPOINT = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-6-astra"


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def attach_frames(packet: dict, episode: Path) -> int:
    urls = []
    for cam in sorted(packet.get("frames") or {}):
        f = packet["frames"][cam]
        idx = f.get("frame_index")
        if idx is None:
            continue
        p = episode / "frames" / f"{cam}_{int(idx):06d}.png"
        if p.exists():
            urls.append(data_url(p))
    packet["image_data_urls"] = urls
    return len(urls)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", required=True,
                   help="directory holding phase1_packets.jsonl and frames/")
    p.add_argument("--packets", default="phase1_packets.jsonl")
    p.add_argument("--send", action="store_true",
                   help="MAKE PAID API CALLS. Off by default.")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--key-env", default="OPENAI_API_KEY")
    p.add_argument("--effort", default="low")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--gap-seconds", type=float, default=3.0,
                   help="pause between calls; large bodies get throttled")
    p.add_argument("--out")
    a = p.parse_args(argv)

    ep = Path(a.episode)
    pkt_path = ep / a.packets
    if not pkt_path.exists():
        print(f"no packets at {pkt_path}. Run `cli phase1` first.", file=sys.stderr)
        return 2
    packets = [json.loads(l) for l in pkt_path.read_text().splitlines() if l.strip()]

    src = AstraReviewSource(base_url=a.endpoint, model=a.model,
                            api_key_env=a.key_env, enabled=a.send,
                            dry_run=not a.send, reasoning=a.effort,
                            timeout=a.timeout)
    if a.send:
        ok, why = src.preflight()
        if not ok:
            print(f"REFUSED: {why}", file=sys.stderr)
            return 2

    print(f"{'DRY RUN -- nothing sent' if not a.send else 'LIVE -- paid calls'}"
          f"   model={a.model}  effort={a.effort}  packets={len(packets)}")
    print(f"{'#':>2s} {'request_id':26s} {'body sha':>13s} {'mode':9s} "
          f"{'exec/intent':24s} {'s':>5s}")
    print("-" * 88)
    rows, tin, tout, failures = [], 0, 0, 0
    for i, pkt in enumerate(packets, 1):
        n_img = attach_frames(pkt, ep)
        if i > 1 and a.send and a.gap_seconds:
            time.sleep(a.gap_seconds)
        t0 = time.time()
        r = src.review(pkt)
        dt = time.time() - t0
        sha = r.get("body_sha256_12", "?")
        rid = pkt.get("request_id", "?")
        if not a.send:
            print(f"{i:2d} {rid:26s} {sha:>13s} {'DRY':9s} "
                  f"{str(r.get('would_send_bytes')) + ' B':24s} {dt:5.1f}")
            rows.append({"request_id": rid, "body_sha256_12": sha,
                         "bytes": r.get("would_send_bytes"), "images": n_img})
            continue
        if not r.get("ok"):
            failures += 1
            print(f"{i:2d} {rid:26s} {sha:>13s} {'ERROR':9s} "
                  f"{str(r.get('error'))[:24]:24s} {dt:5.1f}")
            rows.append({"request_id": rid, "body_sha256_12": sha,
                         "error": str(r.get("error"))[:300], "latency_s": round(dt, 2)})
            continue
        d = r["decision"]
        asmt = d.get("assessment") or {}
        u = r.get("usage") or {}
        tin += u.get("input_tokens", 0) or 0
        tout += u.get("output_tokens", 0) or 0
        print(f"{i:2d} {rid:26s} {sha:>13s} {d.get('mode', '?'):9s} "
              f"{asmt.get('execution_status', '?') + '/' + asmt.get('intent_status', '?'):24s}"
              f" {dt:5.1f}")
        rows.append({"request_id": rid, "body_sha256_12": sha, "decision": d,
                     "usage": u, "latency_s": round(dt, 2), "images": n_img})
    out = Path(a.out) if a.out else ep / (
        "phase1_astra.jsonl" if a.send else "phase1_dryrun.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if a.send:
        print(f"\ncalls={len(packets)} ok={len(packets) - failures} failed={failures} "
              f"input={tin} output={tout} "
              f"est_cost_usd={tin / 1e6 * 10 + tout / 1e6 * 50:.4f}")
        if failures:
            print(f"{failures} packet(s) failed and were NOT retried. Re-run them "
                  f"deliberately if you want them.")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
