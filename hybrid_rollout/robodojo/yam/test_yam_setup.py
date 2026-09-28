"""The setup runbook and scripts stay consistent with the CLI they drive."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from .cli import main

HERE = Path(__file__).parent
SCRIPTS = sorted((HERE / "setup").glob("*.sh"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_scripts_parse(script):
    if not shutil.which("bash"):
        pytest.skip("no bash")
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert script.stat().st_mode & 0o111, f"{script.name} is not executable"


def test_every_subcommand_in_setup_exists():
    text = (HERE / "SETUP.md").read_text() + (HERE / "README.md").read_text()
    used = set(re.findall(r"\$Y ([a-z][a-z0-9-]+)", text))
    assert used, "runbook lists no commands"
    for cmd in used:
        with pytest.raises(SystemExit) as e:
            main([cmd, "--help"])
        assert e.value.code == 0, cmd


def test_setup_names_the_contracted_revisions():
    text = (HERE / "SETUP.md").read_text()
    for f in SCRIPTS:
        text += f.read_text()
    from .contract import CHECKPOINT
    assert CHECKPOINT["revision"] in text
    assert CHECKPOINT["openpi_commit"] in text


def test_moving_commands_refuse_without_execute(capsys):
    assert main(["calibrate-grippers"]) == 2
    assert main(["tracking-test"]) == 2
    assert main(["execute"]) == 2
    assert main(["record-pose", "--name", "rest_pose"]) == 2


def test_set_writes_only_the_local_overlay(tmp_path, monkeypatch):
    local = tmp_path / "rig.local.toml"
    monkeypatch.setenv("YAM_RIG_LOCAL", str(local))
    committed = (HERE / "experiments" / "bimanual_blocks" / "config.toml").read_text()
    assert main(["set", "rig.left_can", "can_follower_l"]) == 0
    assert main(["set", "rig.camera_mapping", '{"top": 4, "left": 0, "right": 2}']) == 0
    from .experiment import flatten_config, load_config
    flat = flatten_config(load_config())
    assert flat["left_can"] == "can_follower_l"
    assert flat["camera_mapping"] == {"top": 4, "left": 0, "right": 2}
    assert (HERE / "experiments" / "bimanual_blocks" / "config.toml").read_text() == committed


def test_rig_local_is_git_ignored():
    text = (HERE / ".gitignore").read_text()
    assert "rig.local.toml" in text and "runs/" in text
