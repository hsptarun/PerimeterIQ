import torch

from app.core.config import Settings


def test_auto_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert Settings(device="auto").resolved_device() == "cuda"


def test_auto_device_uses_mps_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert Settings(device="auto").resolved_device() == "mps"


def test_auto_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert Settings(device="auto").resolved_device() == "cpu"


def test_explicit_device_setting_is_preserved():
    assert Settings(device="mps").resolved_device() == "mps"
    assert Settings(device="cpu").resolved_device() == "cpu"
