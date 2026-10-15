"""Predicted latency of ONNX models on the Myriad VPU, from nn-Meter.

Runs in the nn-Meter environment (~/envs/nnmeter). nn-Meter (Microsoft Research, MobiSys 2021)
splits a model into the kernels the device runs (a convolution with its batch norm and
activation is one kernel, for example) and predicts each kernel's time with a model fitted to
kernels timed on a real Intel Neural Compute Stick 2 (Myriad X, OpenVINO 2019 R2). The
published accuracy for that predictor is 83.4% of models within +-10%.

    ~/envs/nnmeter/bin/python pynas/hardware/tools/nnmeter_latency.py model1.onnx model2.onnx ...

Prints one line per model, starting with JSON: and followed by a JSON record.
"""
import argparse
import json
import time
from pathlib import Path

PREDICTOR = "myriadvpu_openvino2019r2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default=PREDICTOR)
    ap.add_argument("--version", type=float, default=1.0)
    ap.add_argument("onnx", nargs="+")
    a = ap.parse_args()

    from nn_meter import load_latency_predictor
    predictor = load_latency_predictor(a.predictor, a.version)    # downloads the predictor once
    for path in a.onnx:
        rec = dict(model=Path(path).stem, predictor=a.predictor, ok=False)
        t0 = time.time()
        try:
            rec.update(ok=True, latency_ms=round(float(predictor.predict(path, model_type="onnx")), 3))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[-1500:]}"
        rec["s"] = round(time.time() - t0, 1)
        print("JSON:" + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
