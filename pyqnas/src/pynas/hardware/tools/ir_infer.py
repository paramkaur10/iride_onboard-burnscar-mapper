"""Run an OpenVINO IR on a batch of inputs, one at a time, and save its outputs.

Runs in the OpenVINO 2022.3.1 environment (~/envs/ov2022). On the CPU plugin the IR's layers are computed in
FP32 with its FP16 weights, so comparing the outputs with PyTorch's checks the conversion itself (folded
BatchNorm, upsampling, padding, concatenation), apart from the FP16 arithmetic of the Myriad.

    ~/envs/ov2022/bin/python pynas/hardware/tools/ir_infer.py --xml M.xml --input x.npy --output y.npy

x.npy: (N, bands, H, W) float32. Prints one line starting with JSON: and followed by a JSON record.
"""
import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="CPU")
    a = ap.parse_args()

    os.environ.setdefault("OPENVINO_CONF_IGNORE", "YES")
    from openvino.runtime import Core, get_version
    x = np.load(a.input).astype(np.float32)
    t0 = time.time()
    compiled = Core().compile_model(a.xml, a.device)
    req = compiled.create_infer_request()
    out = None
    for i in range(len(x)):
        req.infer({0: x[i:i + 1]})
        y = np.array(req.get_output_tensor(0).data, dtype=np.float32, copy=True)
        if out is None:
            out = np.lib.format.open_memmap(a.output, mode="w+", dtype=np.float32, shape=(len(x),) + y.shape[1:])
        out[i] = y[0]
    if out is not None:
        out.flush()
    print("JSON:" + json.dumps(dict(xml=a.xml, device=a.device, n=int(len(x)), openvino=get_version(),
                                    seconds=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
