"""Verified live-cell facts, the gripper's separateness, and the interpolator lock.

Each test pins a fact reported from the Jetson. A failure means the cell changed
or a lock was relaxed -- both need a human.
"""
from __future__ import annotations

import pytest

from . import cell
from .cameras import CAPTURE_NODES, METADATA_NODES, NODE_PAIRS, CameraError, LiveCameras
from .gripper import GripperCapability, plan, stop_semantics
from .rsi_gateway import (INTERPOLATOR_DEPLOYED, LinearInterpolator, RSIGateway,
                          RuckigInterpolator, validate_cycles)
from .safety import ArmingRefused


class TestNetworkAndAuthority:
    def test_robot_nic_and_kli(self):
        assert cell.JETSON_ROBOT_IP == "172.17.255.2"
        assert cell.KUKA_KLI_IP == "172.17.255.1"

    def test_only_jetson_may_command(self):
        assert cell.AUTHORITY["jetson"]["may_command"] is True
        assert cell.AUTHORITY["a800"]["may_command"] is False
        assert cell.AUTHORITY["eng1"]["may_command"] is False

    def test_a800_has_no_route_to_the_robot_network(self):
        assert cell.AUTHORITY["a800"]["route_to_robot_network"] is False

    def test_wifi_is_isolated_from_the_robot_path(self):
        assert "wlp1s0" in cell.ISOLATED_NICS

    def test_current_state_is_unable_to_command(self):
        assert cell.JETSON_RSI_SOCKET_BOUND is False
        assert cell.RSI_PROGRAM_RUNNING is False


class TestPortsAreNotConfused:
    def test_three_distinct_ports(self):
        assert len({cell.PORT_RSI_UDP, cell.PORT_EXT_TRIGGER_TCP,
                    cell.PORT_OPCUA_TCP}) == 3

    def test_only_rsi_carries_robot_control(self):
        controls = [p for p, v in cell.PORT_FACTS.items() if v[2]]
        assert controls == [cell.PORT_RSI_UDP]

    def test_opcua_is_supervision_only(self):
        proto, purpose, control = cell.PORT_FACTS[cell.PORT_OPCUA_TCP]
        assert control is False and "supervision" in purpose.lower()

    def test_ext_trigger_is_closed(self):
        assert cell.EXT_TRIGGER_OPEN is False


class TestRsiDirectionAndEcho:
    def test_kuka_is_the_client(self):
        assert cell.RSI_CLIENT == "kuka_controller" and cell.RSI_SERVER == "jetson"

    def test_reply_carries_joints_stopflag_and_ipoc(self):
        for e in ("AK.A1", "AK.A6", "STOPFLAG", "IPOC"):
            assert e in cell.RSI_REPLY_ELEMENTS

    def test_reply_type_is_imfree(self):
        assert cell.RSI_REPLY_TYPE == "ImFree"

    def test_late_reply_faults_the_controller(self):
        assert cell.RSI_LATE_REPLY_FAULTS_CONTROLLER is True

    def test_gateway_frame_matches_the_cell_reply_contract(self):
        from .rsi_gateway import build_frame
        f = build_frame(7, [1, 2, 3, 4, 5, 6], stop_flag=1)
        assert 'Type="ImFree"' in f and "<IPOC>7</IPOC>" in f
        assert "<Stopflag>1</Stopflag>" in f


class TestInterpolationIsNotFaked:
    def test_deployed_method_is_ruckig(self):
        assert cell.INTERPOLATOR == "Ruckig" == INTERPOLATOR_DEPLOYED
        assert cell.RSI_DT == 0.004

    def test_linear_is_marked_not_deployable(self):
        li = LinearInterpolator()
        assert li.deployable is False and "discontinuous" in li.reason_not_deployable

    def test_motion_with_linear_is_refused(self):
        with pytest.raises(ArmingRefused) as e:
            RSIGateway(enable_motion=True, secret=b"x")
        assert e.value.code == "interpolator_not_deployable"

    def test_ruckig_without_a_generator_refuses_rather_than_substituting(self):
        with pytest.raises(RuntimeError) as e:
            RuckigInterpolator()([0] * 6, [1] * 6, 8)
        assert "does not reimplement" in str(e.value)

    def test_motion_allowed_with_a_deployable_interpolator(self):
        g = RSIGateway(enable_motion=True, secret=b"x",
                       interpolator=RuckigInterpolator(generator=lambda s, t, c: []))
        assert g.enable_motion is True


class TestPerCycleValidation:
    def test_every_cycle_is_checked_not_just_the_endpoints(self):
        """A chunk feasible on average can hide one infeasible 4 ms step."""
        cycles = [[0.0] * 6, [0.1] * 6, [50.0] * 6, [0.1] * 6]
        problems = validate_cycles([0.0] * 6, cycles, hz=250.0)
        assert any("cycle 2" in p for p in problems)

    def test_feasible_trajectory_passes(self):
        cycles = [[0.1 * i] * 6 for i in range(1, 9)]
        assert validate_cycles([0.0] * 6, cycles, hz=250.0) == []

    def test_position_limits_checked_per_cycle(self):
        problems = validate_cycles([0.0] * 6, [[9999.0] * 6], hz=250.0)
        assert any("outside" in p for p in problems)


class TestGripperIsSeparate:
    def test_not_via_the_controller(self):
        assert cell.GRIPPER_VIA_CONTROLLER is False
        assert cell.GRIPPER_PATH == "direct_modbus"

    def test_device_and_bus_recorded(self):
        assert cell.GRIPPER_DEVICE == "/dev/ttyUSB0"
        assert "Modbus" in cell.GRIPPER_BUS

    def test_rsi_stop_does_not_stop_the_gripper(self):
        s = stop_semantics()
        assert s["rsi_stop_affects_arm"] is True
        assert s["rsi_stop_affects_gripper"] is False
        assert "stays closed" in s["consequence"]

    def test_polarity_is_required_and_unknown(self):
        cap = GripperCapability()
        assert cap.allowed()[0] is False
        assert any("polarity" in m for m in cap.missing())

    def test_command_is_planned_and_audited_but_not_emitted(self):
        c = plan(1.0, GripperCapability())
        assert c.intent == "close" and c.emitted is False and c.refused_reason

    def test_even_fully_attested_does_not_emit_without_a_transport(self):
        cap = GripperCapability(polarity={"open": 0, "closed": 12000},
                                max_force=10, enabled=True)
        assert cap.allowed()[0] is True
        c = plan(1.0, cap)
        assert c.emitted is False and "no gripper transport" in c.refused_reason


class TestCameraNodes:
    def test_capture_and_metadata_nodes_recorded(self):
        assert CAPTURE_NODES == (0, 2) and METADATA_NODES == (1, 3)
        assert NODE_PAIRS == {0: 1, 2: 3}

    def test_metadata_node_mapping_is_refused(self):
        """A metadata node opens fine and yields nothing -- looks like a dead camera."""
        with pytest.raises(CameraError) as e:
            LiveCameras(mapping={"base": 0, "wrist": 1}).start()
        assert "metadata" in str(e.value)

    def test_still_refuses_to_guess(self):
        with pytest.raises(CameraError):
            LiveCameras().start()


class TestKnownAbsent:
    def test_absences_are_recorded_not_glossed(self):
        checks = {c.name for c in cell.verify()}
        assert "absent" in checks
        absent = [c for c in cell.verify() if c.name == "absent"]
        assert len(absent) == len(cell.KNOWN_ABSENT)
        assert all(c.passed is False for c in absent)

    def test_output_path_absence_is_explicit(self):
        assert any("CLI-to-RSI output path" in a for a in cell.KNOWN_ABSENT)
