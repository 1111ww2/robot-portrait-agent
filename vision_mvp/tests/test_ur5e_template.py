import json
from pathlib import Path
import socket
import subprocess

import pytest


def test_ur5e_template_contains_no_transplanted_physical_calibration(monkeypatch):
    root=Path(__file__).resolve().parents[2]
    data=json.loads((root/'vision_mvp/config/physical-ur5e.template.json').read_text())
    def forbidden(*args,**kwargs):pytest.fail('An unconfigured UR5e must not contact devices')
    monkeypatch.setattr(socket,'socket',forbidden)
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    assert data['robot_model']=='ur5e'
    assert data['robot_ip'] is None
    assert data['installation']['reference'] is None
    assert data['installation']['joints'] is None
    assert data['installation']['tcp'] is None
    assert data['intro']['waypoints']==[] and data['intro_preview'] is None
    assert data['intro']['joint_speed']==.7 and data['intro']['joint_acceleration']==1.2
    assert data['max_joint_speed']==.7
    assert data['installation']['gesture']==dict(duration_s=3.3,repetitions=2,joint_speed_limit=.7)
    assert all(value is False for value in data['installation']['review'].values())
    assert all(value is False for value in data['intro']['review'].values())
    from leica_tracker.physical_runtime import reviewed_profile
    with pytest.raises(ValueError):reviewed_profile(data)


def test_ur5e_windows_entry_never_falls_back_to_ur10e_profile():
    root=Path(__file__).resolve().parents[2]
    text=(root/'Start-Leica-UR5e.cmd').read_text()
    assert 'start-canonical-ipad.ps1' in text
    assert '-RobotModel ur5e' in text
    assert 'ur10e' not in text.lower()
