"""The real KUKA RSI UDP/XML gateway. Stdlib only; runs on the Jetson.

THIS IS THE ONE MODULE IN THIS PACKAGE THAT CAN MOVE A REAL ARM.
Everything else refuses structurally. This one refuses by policy, which is a
weaker guarantee, so the policy is stated here and enforced in code.

PROTOCOL, verified against KUKA/teleoperation/udp_teleoperate.py
  The controller connects to us and sends a <Rob> frame every RSI cycle
  (4 ms, 250 Hz) carrying <AIPos A1..A6> (measured joint angles) and <IPOC>
  (its cycle counter). We MUST reply to every frame, within the cycle, echoing
  that exact IPOC. A missed or late reply faults the controller.

  That last point drives the whole design. "No command" does not mean "no
  reply" -- it means "reply with the position it is already at". A gateway that
  falls silent because nothing was commanded is not safe, it is a fault.

TWO MODES
  HOLD     reply = the measured position, every cycle. Keeps an RSI session
           alive and commands no motion. This is the default and it is what you
           run first on real hardware.
  COMMAND  reply = interpolated targets from ONE authorised envelope, then the
           gateway returns to HOLD. Requires enable_motion=True at construction
           AND a signed envelope AND a live session.

RATE BRIDGE
  Chunks are 30 Hz; RSI is 250 Hz. One commanded step is therefore spread over
  ~8 RSI cycles by linear interpolation from the measured start pose. Stepping
  straight to the target in one cycle would demand the whole 30 Hz displacement
  in 4 ms -- an 8x velocity overshoot.

WHAT THIS MODULE STILL CANNOT PROMISE
  It has never run against a real controller. Loopback tests prove the framing,
  the IPOC echo, the interpolation and the refusals; they do not prove 4 ms
  timing under load, behaviour on packet loss, or how this controller reacts to
  a late frame. Those are hardware facts and must be established on the cell,
  in HOLD mode, before motion is ever enabled.
"""
from __future__ import annotations

import re
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .contract import ARM_DIM, CONTROL_HZ, GRIPPER_SCALE, POSITION_LIMIT_DEG
from .safety import ArmingRefused, CommandEnvelope
from .transports import (RSI_CYCLE_TIME, RSI_DEFAULT_PORT, RSI_IPOC_FIELD,
                         RSI_JOINT_PRECISION, RSI_LOCAL_IP_DEFAULT,
                         RSI_RESPONSE_ROOT, RSI_SEN_TYPE, gripper_to_raw)

SCHEMA = "hybrid_rollout.robodojo.kuka.rsi_gateway.v1"

_IPOC_RE = re.compile(r"<IPOC>(\d+)</IPOC>")
# Scoped to the <AIPos> tag so it cannot match <ASol>/<AKorr>/<RKorr>, which
# carry the same attribute names -- the deployed stack hit exactly this bug.
_AIPOS_RE = re.compile(
    r"<AIPos\s+" + r"\s+".join(f'A{i + 1}="([-0-9.eE+]+)"' for i in range(ARM_DIM)))


def parse_frame(xml: str) -> tuple[int | None, list[float] | None]:
    """(IPOC, measured joints). Either may be None if the frame is malformed."""
    m_ipoc = _IPOC_RE.search(xml)
    m_pos = _AIPOS_RE.search(xml)
    ipoc = int(m_ipoc.group(1)) if m_ipoc else None
    joints = [float(m_pos.group(i + 1)) for i in range(ARM_DIM)] if m_pos else None
    return ipoc, joints


def build_frame(ipoc: int, joints_deg: Sequence[float], *, gripper_pos: int = 0,
                stop_flag: int = 0) -> str:
    nl = "\r\n"
    ak = " ".join(f'A{i + 1}="{v:.{RSI_JOINT_PRECISION}f}"'
                  for i, v in enumerate(list(joints_deg)[:ARM_DIM]))
    return (f'<{RSI_RESPONSE_ROOT} Type="{RSI_SEN_TYPE}">{nl}'
            f'<AK {ak}/>{nl}'
            f'<GRIPPER_POS>{gripper_pos}</GRIPPER_POS>{nl}'
            f'<Stopflag>{stop_flag}</Stopflag>{nl}'
            f'<{RSI_IPOC_FIELD}>{ipoc}</{RSI_IPOC_FIELD}>{nl}'
            f'</{RSI_RESPONSE_ROOT}>')


def interpolate(start: Sequence[float], target: Sequence[float],
                cycles: int) -> list[list[float]]:
    """Linear bridge from the measured pose to one commanded step."""
    if cycles < 1:
        raise ValueError("cycles must be >= 1")
    s, t = list(start)[:ARM_DIM], list(target)[:ARM_DIM]
    return [[s[j] + (t[j] - s[j]) * (k + 1) / cycles for j in range(ARM_DIM)]
            for k in range(cycles)]


@dataclass
class GatewayStats:
    frames_in: int = 0
    frames_out: int = 0
    malformed: int = 0
    ipoc_regressions: int = 0
    commanded_cycles: int = 0
    hold_cycles: int = 0
    late_replies: int = 0
    last_ipoc: int | None = None
    stopped_reason: str | None = None

    def to_log(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class RSIGateway:
    """Owns the UDP socket and the reply loop.

    can_move_robot is True ONLY when enable_motion was explicitly passed. The
    default construction is a HOLD-only gateway: it will keep an RSI session
    alive and refuse every envelope.
    """

    name = "jetson_rsi"

    def __init__(self, *, host: str = RSI_LOCAL_IP_DEFAULT,
                 port: int = RSI_DEFAULT_PORT,
                 enable_motion: bool = False,
                 secret: bytes | None = None,
                 control_hz: float = CONTROL_HZ,
                 cycle_time: float = RSI_CYCLE_TIME,
                 max_frame_gap_s: float = 0.050,
                 sock: Any = None) -> None:
        self.host, self.port = host, port
        self.enable_motion = bool(enable_motion)
        self.secret = secret
        self.control_hz = control_hz
        self.cycle_time = cycle_time
        self.max_frame_gap_s = max_frame_gap_s
        self.cycles_per_step = max(1, round((1.0 / control_hz) / cycle_time))
        self.stats = GatewayStats()
        self._sock = sock
        self._peer: tuple[str, int] | None = None
        self._pending: list[list[float]] = []
        self._pending_grip: int = 0
        self._delivered: list[dict[str, Any]] = []
        self._stop = False
        self._last_measured: list[float] | None = None

    # --------------------------------------------------------------- policy
    @property
    def can_move_robot(self) -> bool:
        return self.enable_motion and not self._stop

    def open(self) -> None:
        if self._sock is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((self.host, self.port))
            s.settimeout(self.max_frame_gap_s)
            self._sock = s

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def controlled_stop(self, reason: str) -> None:
        """Latch a stop. The loop keeps replying HOLD with Stopflag=1: the
        controller must still be answered, it just must not be moved."""
        self._stop = True
        self._pending.clear()
        self.stats.stopped_reason = reason

    # -------------------------------------------------------------- commands
    def send(self, envelope: CommandEnvelope) -> dict[str, Any]:
        """Queue ONE authorised step. Refuses unless everything lines up."""
        if self._stop:
            raise ArmingRefused("gateway_stopped",
                                f"gateway latched stopped: {self.stats.stopped_reason}")
        if not self.enable_motion:
            raise ArmingRefused(
                "motion_not_enabled",
                "gateway constructed without enable_motion=True; it will hold "
                "position and command nothing")
        if not envelope.approved_for_execution:
            raise ArmingRefused("unapproved_envelope", "envelope was never authorised")
        if self.secret is None or not envelope.verify(self.secret):
            raise ArmingRefused("bad_signature", "envelope signature invalid")
        if len(envelope.rows) != 1:
            raise ArmingRefused("not_single_step",
                                f"gateway accepts one step, got {len(envelope.rows)}")
        if self._last_measured is None:
            raise ArmingRefused("no_session",
                                "no RSI frame received yet; refusing to command "
                                "from an unknown pose")
        row = list(envelope.rows[0])
        for j in range(ARM_DIM):
            lo, hi = POSITION_LIMIT_DEG[j]
            if not lo <= row[j] <= hi:
                raise ArmingRefused("position_limit",
                                    f"joint {j + 1} target {row[j]} outside "
                                    f"[{lo}, {hi}]")
        self._pending = interpolate(self._last_measured, row, self.cycles_per_step)
        self._pending_grip = gripper_to_raw(row[ARM_DIM]) if len(row) > ARM_DIM else 0
        rec = {"command_id": envelope.command_id,
               "envelope_sha256": envelope.digest(),
               "cycles": len(self._pending)}
        self._delivered.append(rec)
        return {"ok": True, "sent": True, "queued_cycles": len(self._pending),
                "record": rec}

    # ------------------------------------------------------------------ loop
    def serve_once(self, *, now: float | None = None) -> dict[str, Any] | None:
        """Receive one controller frame and answer it. Returns what was sent."""
        if self._sock is None:
            raise RuntimeError("gateway socket not open")
        try:
            data, peer = self._sock.recvfrom(4096)
        except (socket.timeout, TimeoutError):
            return None
        t0 = time.monotonic() if now is None else now
        self._peer = peer
        self.stats.frames_in += 1
        ipoc, measured = parse_frame(data.decode("utf-8", "replace"))
        if ipoc is None or measured is None:
            self.stats.malformed += 1
            return None                      # never guess an IPOC
        if self.stats.last_ipoc is not None and ipoc < self.stats.last_ipoc:
            self.stats.ipoc_regressions += 1
        self.stats.last_ipoc = ipoc
        self._last_measured = measured

        if self._pending and self.can_move_robot:
            target = self._pending.pop(0)
            self.stats.commanded_cycles += 1
        else:
            target = measured                # HOLD: reply where it already is
            self.stats.hold_cycles += 1
        frame = build_frame(ipoc, target, gripper_pos=self._pending_grip,
                            stop_flag=1 if self._stop else 0)
        self._sock.sendto(frame.encode("ascii"), peer)
        self.stats.frames_out += 1
        elapsed = (time.monotonic() if now is None else now) - t0
        if elapsed > self.cycle_time:
            self.stats.late_replies += 1
        return {"ipoc": ipoc, "measured": measured, "target": target,
                "holding": target is measured, "late": elapsed > self.cycle_time}

    def serve(self, *, max_frames: int | None = None,
              on_frame: Callable[[dict[str, Any]], None] | None = None) -> GatewayStats:
        n = 0
        while max_frames is None or n < max_frames:
            r = self.serve_once()
            if r is None:
                if self.stats.frames_in and not self._stop:
                    self.controlled_stop("rsi frame gap exceeded watchdog")
                continue
            if on_frame:
                on_frame(r)
            n += 1
        return self.stats

    def to_log(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "name": self.name,
                "bind": f"{self.host}:{self.port}",
                "can_move_robot": self.can_move_robot,
                "enable_motion": self.enable_motion,
                "cycles_per_step": self.cycles_per_step,
                "delivered": self._delivered, "stats": self.stats.to_log()}


class FakeController:
    """Test double for the KUKA side. Speaks the controller half over loopback.

    Exists so the framing, the IPOC echo, the interpolation and the refusals can
    be exercised without a robot. It is NOT a simulator: it has no dynamics and
    it moves exactly where it is told.
    """

    def __init__(self, gateway_addr: tuple[str, int],
                 start: Sequence[float] | None = None) -> None:
        self.addr = gateway_addr
        self.joints = list(start or [0.0] * ARM_DIM)
        self.ipoc = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        self.received: list[str] = []

    def send_frame(self) -> int:
        """Controller -> gateway. Split from receive so a single-threaded test
        can drive both halves in lockstep without either side deadlocking."""
        self.ipoc += 1
        ak = " ".join(f'A{i + 1}="{v:.4f}"' for i, v in enumerate(self.joints))
        req = (f'<Rob Type="KUKA">\r\n<AIPos {ak}/>\r\n'
               f'<IPOC>{self.ipoc}</IPOC>\r\n</Rob>')
        self.sock.sendto(req.encode("ascii"), self.addr)
        return self.ipoc

    def recv_reply(self) -> dict[str, Any] | None:
        try:
            data, _ = self.sock.recvfrom(4096)
        except (socket.timeout, TimeoutError):
            return None
        return self._apply(data.decode())

    def exchange(self, gateway: "RSIGateway") -> dict[str, Any] | None:
        """One full RSI cycle: send, let the gateway answer, apply the reply."""
        self.send_frame()
        gateway.serve_once()
        return self.recv_reply()

    def tick(self) -> dict[str, Any] | None:
        self.send_frame()
        try:
            data, _ = self.sock.recvfrom(4096)
        except (socket.timeout, TimeoutError):
            return None
        return self._apply(data.decode())

    def _apply(self, xml: str) -> dict[str, Any]:
        self.received.append(xml)
        ipoc, _ = parse_frame(xml)
        m = re.search(r"<AK\s+" + r"\s+".join(
            f'A{i + 1}="([-0-9.eE+]+)"' for i in range(ARM_DIM)), xml)
        if m:
            self.joints = [float(m.group(i + 1)) for i in range(ARM_DIM)]
        return {"echoed_ipoc": ipoc, "sent_ipoc": self.ipoc, "joints": list(self.joints)}

    def close(self) -> None:
        self.sock.close()
