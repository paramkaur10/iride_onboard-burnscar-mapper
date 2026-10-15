"""What a result depends on, written with it: versions, the PyNAS source it came from, the machine."""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def code_sha():
    """SHA-1 of the PyNAS source (every .py and .ini file of the package, by relative path and content)."""
    h = hashlib.sha1()
    for p in sorted(PACKAGE.rglob("*")):
        if p.suffix in (".py", ".ini") and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(PACKAGE)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def git_commit(where=None):
    """The git commit of the folder (and whether it has uncommitted changes), or None outside a repository."""
    try:
        cwd = str(where or PACKAGE)
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=10)
        return sha.stdout.strip() + ("+changes" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    """Versions and machine, for run.json."""
    from . import __version__
    out = dict(pynas=__version__, pynas_code_sha=code_sha(), git=git_commit(), python=sys.version.split()[0],
               platform=platform.platform(), host=platform.node(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        import torch
        out.update(torch=torch.__version__, cuda=getattr(torch.version, "cuda", None),
                   cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:                                            # noqa: BLE001 - versions are informative only
        pass
    for mod in ("segmentation_models_pytorch", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                                        # noqa: BLE001
            pass
    return out
