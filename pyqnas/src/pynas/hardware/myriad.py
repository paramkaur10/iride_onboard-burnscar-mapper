"""The Myriad X without the device: the files ArgOS takes, and whether they compile.

ArgOS runs models as OpenVINO 2022.3.1 FP16 IR (.xml + .bin + .mapping) on the Myriad X of the Unibap iX5-106.
This module makes exactly that and checks that it compiles for the Myriad, in environments of their own (old
packages that cannot live next to the training environment):

    [Hardware] ov_env      Python 3.8 + openvino-dev 2022.3.1 (pip): its Model Optimizer makes the FP16 IR
    [Hardware] ov_archive  Intel's OpenVINO 2022.3.1 C++ archive: its compile_tool compiles the IR for MYRIAD
                           with no device attached (the pip packages have no MYRIAD plugin)
    [Hardware] nnmeter_env Python 3.8 + nn-Meter 2.0, for the predicted latency (latency.py)

    setup(cfg)                                          # makes the missing environments once per machine
    recs = check({"name": model}, out_dir, (7, 256, 256), cfg)
    rec = export_ir(model, "models/onboard", "burnscar", (7, 256, 256), cfg)
    y = run_ir(xml, x, cfg)                             # the IR's outputs on the CPU plugin
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import get_optional_float, path as cfg_path

TOOLS = Path(__file__).resolve().parent / "tools"
OV_PACKAGES = ["openvino-dev[onnx]==2022.3.1", "setuptools<70", "openvino-telemetry<2023"]
NNM_PACKAGES = ["nn-meter==2.0", "onnx==1.9.0", "protobuf==3.20.3", "scikit-learn==0.23.1", "numpy==1.19.5",
                "setuptools<70"]
NNM_PACKAGES_LOOSE = ["nn-meter==2.0", "onnx<1.13", "protobuf<3.21", "setuptools<70"]
OV_ARCHIVE_URL = ("https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3.1/linux/"
                  "l_openvino_toolkit_ubuntu20_2022.3.1.9227.cf2c7da5689_x86_64.tgz")
# The archive is built for Ubuntu 20.04 and takes TBB 2020 (libtbb.so.2) from the system; newer Ubuntu releases
# do not have it, so it comes from the Ubuntu 22.04 package, checked against the hash Ubuntu publishes.
TBB2_DEB_URL = "http://archive.ubuntu.com/ubuntu/pool/universe/t/tbb/libtbb2_2020.3-1ubuntu3_amd64.deb"
TBB2_DEB_SHA256 = "93684d6b6a0871cbe27c4332aeed7f9b3209f342e72817d255fcf0d0f48dc321"


# ---- environments ------------------------------------------------------------------------------
def ov_env(cfg):
    return Path(cfg_path(cfg, "Hardware", "ov_env"))


def ov_archive(cfg):
    return Path(cfg_path(cfg, "Hardware", "ov_archive"))


def nnmeter_env(cfg):
    return Path(cfg_path(cfg, "Hardware", "nnmeter_env"))


def env_python(env):
    return Path(env) / "bin" / "python"


def compile_tool_path(root):
    return Path(root) / "tools" / "compile_tool" / "compile_tool"


def status(cfg):
    """Which of the three tools are there."""
    return dict(model_optimizer=env_python(ov_env(cfg)).exists(), compile_tool=compile_tool_path(ov_archive(cfg)).exists(),
                nnmeter=env_python(nnmeter_env(cfg)).exists())


def run(cmd, timeout=None, show=0, env=None):
    """Run a command; print the last `show` lines of its output; return the CompletedProcess."""
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, **(env or {})})
    lines = (p.stdout + p.stderr).strip().splitlines()
    if show and lines:
        print("\n".join(lines[-show:]))
    return p


def uv():
    exe = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    if not Path(exe).exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
        exe = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    return exe


def make_env(env, packages, python="3.8"):
    env = Path(env)
    if not env_python(env).exists():
        if run([uv(), "venv", env, "--python", python], show=5).returncode != 0:
            return False
    return run([uv(), "pip", "install", "--python", env_python(env), *packages], show=15).returncode == 0


def ensure_tbb2(root):
    """libtbb.so.2 for the archive, in <archive>/extra_libs, unless the archive or the system has it."""
    import ctypes
    import hashlib
    import urllib.request
    root = Path(root)
    extra = root / "extra_libs"
    if any(root.rglob("libtbb.so.2")):
        return extra
    try:
        ctypes.CDLL("libtbb.so.2")
        return extra
    except OSError:
        pass
    deb = root.parent / TBB2_DEB_URL.rsplit("/", 1)[-1]
    urllib.request.urlretrieve(TBB2_DEB_URL, deb)
    if hashlib.sha256(deb.read_bytes()).hexdigest() != TBB2_DEB_SHA256:
        deb.unlink()
        raise RuntimeError("the libtbb2 package does not have the expected SHA256; not used")
    tmp = root.parent / "libtbb2_unpacked"
    subprocess.run(["dpkg-deb", "-x", str(deb), str(tmp)], check=True)
    extra.mkdir(parents=True, exist_ok=True)
    for f in tmp.rglob("libtbb*.so*"):
        shutil.copy2(f, extra / f.name, follow_symlinks=True)
    shutil.rmtree(tmp)
    deb.unlink()
    return extra


def get_ov_archive(root, url=OV_ARCHIVE_URL):
    """Intel's OpenVINO 2022.3.1 C++ archive (MYRIAD plugin and compile_tool), downloaded and unpacked once."""
    import tarfile
    import urllib.request
    root = Path(root)
    if compile_tool_path(root).exists():
        ensure_tbb2(root)
        return root
    base = root.parent
    base.mkdir(parents=True, exist_ok=True)
    tgz = base / url.rsplit("/", 1)[-1]
    if not tgz.exists():
        urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz) as t:
        tops = {(n[2:] if n.startswith("./") else n).split("/")[0] for n in t.getnames()} - {"", "."}
        if len(tops) != 1:
            raise RuntimeError(f"unexpected archive layout: {sorted(tops)[:5]}")
        top = tops.pop()
        try:
            t.extractall(base, filter="tar")
        except TypeError:
            t.extractall(base)
    if (base / top) != root:
        if root.exists():
            shutil.rmtree(root)
        (base / top).rename(root)
    tgz.unlink()
    ensure_tbb2(root)
    return root


def setup(cfg, model_optimizer=True, compile_tool=True, nnmeter=True):
    """Make the missing environments. Returns status()."""
    if model_optimizer and not env_python(ov_env(cfg)).exists():
        make_env(ov_env(cfg), OV_PACKAGES)
    if compile_tool and not compile_tool_path(ov_archive(cfg)).exists():
        get_ov_archive(ov_archive(cfg))
    if nnmeter and not env_python(nnmeter_env(cfg)).exists():
        if not make_env(nnmeter_env(cfg), NNM_PACKAGES):
            make_env(nnmeter_env(cfg), NNM_PACKAGES_LOOSE)
    return status(cfg)


# ---- models ------------------------------------------------------------------------------------------
def onnx_module():
    """The onnx package, which torch.onnx.export needs; installed into this environment the first time."""
    try:
        import onnx
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx"])
        import onnx
    return onnx


def export_onnx(model, path, input_shape, cfg):
    """ONNX with a fixed input (1, bands, H, W), as the Myriad needs, with the TorchScript exporter ([Hardware]
    onnx_opset; the newer exporter writes opsets the 2022.3 Model Optimizer cannot read). The file's IR version
    is set back to [Hardware] onnx_ir_version when torch writes a newer one. The model is exported in FP32: the
    Model Optimizer makes the FP16 IR."""
    import torch
    opset = cfg.getint("Hardware", "onnx_opset")
    ir_version = cfg.getint("Hardware", "onnx_ir_version")
    onnx = onnx_module()
    model = copy.deepcopy(model).cpu().float().eval()
    x = torch.zeros(1, *input_shape)
    kw = dict(opset_version=opset, input_names=["input"], output_names=["logits"], do_constant_folding=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        try:
            torch.onnx.export(model, x, str(path), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(model, x, str(path), **kw)
    m = onnx.load(str(path))
    if ir_version and m.ir_version > ir_version:
        m.ir_version = ir_version
        onnx.save(m, str(path))
    return path


def fp16_mb(model):
    """Size of the weights in FP16, MB (about what the .bin of the IR holds)."""
    return sum(p.numel() for p in model.parameters()) * 2 / 1e6


def _records(stdout):
    return [json.loads(line[5:]) for line in stdout.splitlines() if line.startswith("JSON:")]


def _check_chunk(paths, out_dir, compile, cfg, timeout):
    py = env_python(ov_env(cfg))
    cmd = [py, TOOLS / "myriad_check.py", "--out", out_dir] + ([] if compile else ["--no-compile"])
    if compile and compile_tool_path(ov_archive(cfg)).exists():
        cmd += ["--archive", str(ov_archive(cfg))]
    layers = cfg_path(cfg, "Hardware", "argos_layers")
    if layers:
        cmd += ["--argos-layers", layers]
    p = run(cmd + [str(x) for x in paths], timeout=timeout, env={"OPENVINO_CONF_IGNORE": "YES"})
    recs = _records(p.stdout)
    if not recs:
        tail = (p.stdout + p.stderr)[-1500:]
        recs = [dict(model=Path(x).stem, ir=False, compiled=False, error=f"no record from the check: {tail}") for x in paths]
    return recs


def check_onnx(onnx_paths, out_dir, cfg, compile=True, timeout=7200):
    """FP16 IR and MYRIAD compilation of ONNX files, [Hardware] check_workers at a time; one record per file."""
    py = env_python(ov_env(cfg))
    if not py.exists():
        raise RuntimeError(f"the OpenVINO 2022.3.1 environment is missing ({py}); run pynas.hardware.myriad.setup(cfg)")
    paths = [str(p) for p in onnx_paths]
    k = max(1, min(cfg.getint("Hardware", "check_workers"), len(paths)))
    chunks = [paths[i::k] for i in range(k)]
    with ThreadPoolExecutor(k) as ex:
        parts = list(ex.map(lambda c: _check_chunk(c, out_dir, compile, cfg, timeout), chunks))
    return [r for part in parts for r in part]


def check(models, out_dir, input_shape, cfg, compile=True):
    """{name: model} -> {name: record}: ONNX, FP16 IR, MYRIAD compile. A record has ir, compiled, ir_mb, bin_mb,
    layers (IR layer types and counts), mapping, argos_unlisted (IR layer types missing from [Hardware]
    argos_layers, when given), error, params, fp16_mb, within_size ([Hardware] max_ir_mb)."""
    out_dir = Path(out_dir)
    paths, out = [], {}
    for name, model in models.items():
        try:
            paths.append(export_onnx(model, out_dir / "onnx" / f"{name}.onnx", input_shape, cfg))
        except Exception as e:                                   # noqa: BLE001 - recorded
            out[name] = dict(model=name, ir=False, compiled=False, error=f"ONNX export: {type(e).__name__}: {e}")
    if paths:
        for rec in check_onnx(paths, out_dir / "ir", cfg, compile=compile):
            out[rec["model"]] = rec
    limit = cfg.getfloat("Hardware", "max_ir_mb")
    for name, model in models.items():
        rec = out.setdefault(name, dict(model=name, ir=False, compiled=False, error="no record from the check"))
        rec["params"] = int(sum(p.numel() for p in model.parameters()))
        rec["fp16_mb"] = round(fp16_mb(model), 3)
        rec["within_size"] = bool(rec.get("ir") and rec.get("ir_mb", 1e9) <= limit)
    return out


def export_ir(model, out_dir, name, input_shape, cfg):
    """The ArgOS model files of a model: <out_dir>/<name>/<name>.xml, .bin, .mapping (FP16 IR of OpenVINO
    2022.3.1), the .onnx and the compiled .blob, and myriad_check.json. Returns the record of check()."""
    work = Path(out_dir) / "_work"
    rec = check({name: model}, work, input_shape, cfg, compile=True)[name]
    src, dst = work / "ir" / name, Path(out_dir) / name
    dst.mkdir(parents=True, exist_ok=True)
    for ext in ("xml", "bin", "mapping", "blob"):
        f = src / f"{name}.{ext}"
        if f.exists():
            shutil.copy2(f, dst / f.name)
    onnx_file = work / "onnx" / f"{name}.onnx"
    if onnx_file.exists():
        shutil.copy2(onnx_file, dst / onnx_file.name)
    shutil.rmtree(work, ignore_errors=True)
    rec["files"] = sorted(p.name for p in dst.iterdir())
    with open(dst / "myriad_check.json", "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def run_ir(xml, x, cfg, device="CPU", timeout=3600):
    """The outputs of an IR for inputs x ((N, bands, H, W), as the network takes them), computed in the OpenVINO
    2022.3.1 environment. The CPU plugin computes the IR's layers in FP32 with its FP16 weights: compared with
    PyTorch, this checks the conversion (folding, upsampling, padding)."""
    import numpy as np
    py = env_python(ov_env(cfg))
    if not py.exists():
        raise RuntimeError(f"the OpenVINO 2022.3.1 environment is missing ({py}); run pynas.hardware.myriad.setup(cfg)")
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "x.npy"), np.ascontiguousarray(x, dtype=np.float32))
        y = os.path.join(d, "y.npy")
        p = run([py, TOOLS / "ir_infer.py", "--xml", xml, "--input", os.path.join(d, "x.npy"), "--output", y,
                 "--device", device], timeout=timeout, env={"OPENVINO_CONF_IGNORE": "YES"})
        if p.returncode != 0 or not os.path.exists(y):
            raise RuntimeError("IR inference failed: " + (p.stdout + p.stderr)[-3000:])
        return np.array(np.load(y))


def limits(cfg):
    """The hardware limits of the configuration: max IR MB, max latency ms (or None), ArgOS list path (or '')."""
    return dict(max_ir_mb=cfg.getfloat("Hardware", "max_ir_mb"),
                max_latency_ms=get_optional_float(cfg, "Hardware", "max_latency_ms"),
                argos_layers=cfg_path(cfg, "Hardware", "argos_layers"))
