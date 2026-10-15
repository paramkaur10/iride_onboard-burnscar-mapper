"""Compile ONNX models for the Myriad X without the device.

Runs in the OpenVINO 2022.3.1 environment (~/envs/ov2022), the last OpenVINO with the MYRIAD
plugin and the version ArgOS uses. For each ONNX file: the Model Optimizer makes an FP16 IR
(.xml + .bin + .mapping, what ArgOS takes), then the IR is compiled into a Myriad X blob. Compiling
needs no stick: if it compiles, every layer is one the Myriad runs, and the blob is what the device
would load. The pip packages have no MYRIAD plugin, so with --archive the compiling is done by
compile_tool from Intel's C++ archive of the same version (pynas.hardware.myriad.get_ov_archive).
With --argos-layers, the IR's layer types are also compared with the list of the ArgOS test suite
(test_suite/src/myriad_compatible_layers.txt).

    ~/envs/ov2022/bin/python pynas/hardware/tools/myriad_check.py --out DIR [--archive DIR] model1.onnx ...

Prints one line per model, starting with JSON: and followed by a JSON record.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def layer_types(xml_path):
    """How many layers of each type the IR has (Parameter, Result and Const left out)."""
    out = {}
    for layer in ET.parse(xml_path).getroot().iter("layer"):
        t = layer.get("type")
        if t not in ("Parameter", "Result", "Const"):
            out[t] = out.get(t, 0) + 1
    return out


def allowed_layers(path):
    """Layer types listed in a text file (one or more per line, separated by spaces or commas; # comments)."""
    out = set()
    with open(path, errors="replace") as f:
        for line in f:
            line = line.split("#", 1)[0]
            for tok in line.replace(",", " ").split():
                out.add(tok.strip().lower())
    return out


def export_blob(compiled):
    """The compiled model as bytes (the 2022.3 Python API has two forms of export_model)."""
    try:
        data = compiled.export_model()
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
    except TypeError:
        pass
    buf = io.BytesIO()
    compiled.export_model(buf)
    return buf.getvalue()


def compile_with_tool(root, xml, blob):
    """compile_tool of the C++ archive: IR -> Myriad X blob. Returns (ok, message)."""
    tool = root / "tools" / "compile_tool" / "compile_tool"
    env = dict(os.environ)
    libs = [root / s for s in ("runtime/lib/intel64", "runtime/3rdparty/tbb/lib", "extra_libs") if (root / s).is_dir()]
    env["LD_LIBRARY_PATH"] = ":".join([str(p) for p in libs] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    env["INTEL_OPENVINO_DIR"] = str(root)
    if blob.exists():
        blob.unlink()
    conf = blob.parent / "myriad.conf"                   # compile only: don't look for a device to boot
    conf.write_text("MYRIAD_ENABLE_MX_BOOT NO\n")         # (Intel support article 000091771)
    p = subprocess.run([str(tool), "-m", str(xml), "-d", "MYRIAD", "-c", str(conf), "-o", str(blob)],
                       capture_output=True, text=True, env=env)
    if p.returncode == 0 and blob.exists() and blob.stat().st_size > 0:
        return True, ""
    return False, (p.stdout + "\n" + p.stderr).strip()[-1500:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="folder for the IRs and blobs")
    ap.add_argument("--no-compile", action="store_true", help="only make the FP16 IR")
    ap.add_argument("--archive", help="OpenVINO 2022.3.1 C++ archive: compile with its compile_tool")
    ap.add_argument("--argos-layers", help="ArgOS myriad_compatible_layers.txt: report IR layers not in it")
    ap.add_argument("onnx", nargs="+")
    a = ap.parse_args()

    os.environ.setdefault("OPENVINO_CONF_IGNORE", "YES")
    from openvino.runtime import Core, get_version
    core = Core()
    mo = str(Path(sys.executable).parent / "mo")
    allowed = allowed_layers(a.argos_layers) if a.argos_layers else None
    for path in a.onnx:
        name = Path(path).stem
        d = Path(a.out) / name
        d.mkdir(parents=True, exist_ok=True)
        rec = dict(model=name, openvino=get_version(), ir=False, compiled=False)
        t0 = time.time()
        proc = subprocess.run([mo, "--input_model", path, "--output_dir", str(d), "--model_name", name,
                               "--data_type", "FP16"], capture_output=True, text=True)
        rec["mo_s"] = round(time.time() - t0, 1)
        xml, binf = d / f"{name}.xml", d / f"{name}.bin"
        if proc.returncode != 0 or not xml.exists():
            rec["error"] = "Model Optimizer: " + (proc.stderr or proc.stdout)[-1500:]
            print("JSON:" + json.dumps(rec), flush=True)
            continue
        rec.update(ir=True, ir_mb=round((xml.stat().st_size + binf.stat().st_size) / 1e6, 3),
                   bin_mb=round(binf.stat().st_size / 1e6, 3), layers=layer_types(xml),
                   mapping=(d / f"{name}.mapping").exists())
        if allowed is not None:
            rec["argos_unlisted"] = sorted(t for t in rec["layers"] if t.lower() not in allowed)
        if not a.no_compile:
            t0 = time.time()
            blob_path = d / f"{name}.blob"
            if a.archive:
                ok, msg = compile_with_tool(Path(a.archive), xml, blob_path)
                if ok:
                    rec.update(compiled=True, blob_mb=round(blob_path.stat().st_size / 1e6, 3))
                else:
                    rec["error"] = "MYRIAD: " + msg
            else:
                try:
                    compiled = core.compile_model(core.read_model(str(xml)), "MYRIAD")
                    blob = export_blob(compiled)
                    blob_path.write_bytes(blob)
                    rec.update(compiled=True, blob_mb=round(len(blob) / 1e6, 3))
                except Exception as e:                   # the message says what the Myriad refused
                    rec["error"] = "MYRIAD: " + str(e)[-1500:]
            rec["compile_s"] = round(time.time() - t0, 1)
        print("JSON:" + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
