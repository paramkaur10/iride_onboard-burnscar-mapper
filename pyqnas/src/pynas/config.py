"""The configuration: package defaults (config.ini next to this file), an experiment's file on top, and
per-run overrides {"Section.option": value} on top of that. Configurations are passed explicitly; there is no
global one.

    cfg = load("configs/burnscar.ini")
    cfg_run = load("configs/burnscar.ini", {"Train.seed": 1})
    text(cfg_run)                  # the effective configuration, as written into a run folder
"""
from __future__ import annotations

import os
from configparser import ConfigParser
from pathlib import Path

DEFAULTS = Path(__file__).with_name("config.ini")


def _parser():
    return ConfigParser(interpolation=None)


def load(path=None, overrides=None):
    """Defaults, then the file at path (if any), then overrides. Unknown sections or options in the file are
    an error, so that a misspelt option cannot be silently ignored."""
    cfg = _parser()
    cfg.read(DEFAULTS)
    if path:
        user = _parser()
        if not user.read(path):
            raise FileNotFoundError(path)
        _merge(cfg, user, str(path))
    return with_overrides(cfg, overrides)


def loads(text_, overrides=None):
    """As load(), from the text of a configuration (e.g. the one written into a run folder)."""
    cfg = _parser()
    cfg.read(DEFAULTS)
    user = _parser()
    user.read_string(text_)
    _merge(cfg, user, "<text>")
    return with_overrides(cfg, overrides)


def _merge(cfg, user, where):
    for sec in user.sections():
        if not cfg.has_section(sec):
            raise ValueError(f"{where}: unknown section [{sec}]")
        for opt, val in user.items(sec, raw=True):
            if not cfg.has_option(sec, opt):
                raise ValueError(f"{where}: unknown option {opt} in [{sec}]")
            cfg.set(sec, opt, val)


def with_overrides(cfg, overrides):
    """A copy of cfg with {"Section.option": value} set (known options only)."""
    out = _parser()
    out.read_string(text(cfg))
    for key, value in (overrides or {}).items():
        sec, _, opt = key.partition(".")
        if not out.has_option(sec, opt):
            raise ValueError(f"override {key}: unknown option")
        out.set(sec, opt, str(value))
    return out


def text(cfg):
    """The configuration as ini text."""
    lines = []
    for sec in cfg.sections():
        lines.append(f"[{sec}]")
        lines += [f"{k} = {v}" for k, v in cfg.items(sec, raw=True)]
        lines.append("")
    return "\n".join(lines)


def get_list(cfg, sec, opt, cast=str):
    raw = cfg.get(sec, opt, fallback="") or ""
    return [cast(v.strip()) for v in raw.split(",") if v.strip()]


def get_mapping(cfg, sec, opt, cast=float):
    """'a: 1, b: 2' -> {'a': 1.0, 'b': 2.0}."""
    out = {}
    for item in get_list(cfg, sec, opt):
        k, _, v = item.partition(":")
        out[k.strip()] = cast(v.strip())
    return out


def get_optional_float(cfg, sec, opt):
    raw = (cfg.get(sec, opt, fallback="") or "").strip()
    return float(raw) if raw else None


def path(cfg, sec, opt):
    """An option holding a path, with ~ expanded ('' stays '')."""
    raw = (cfg.get(sec, opt, fallback="") or "").strip()
    return os.path.expanduser(raw) if raw else ""
