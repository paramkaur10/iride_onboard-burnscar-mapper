"""Myriad latency from nn-Meter's kernel predictors, for networks given as kernel units.

nn-Meter's own route (ONNX file -> its graph converter -> kernel detection -> kernel predictors)
stops on networks with resizing (KeyError 'type' on U-Nets). PyNAS lists the kernels of a network
itself (pynas.hardware.latency.kernel_units), so this script skips the first two steps and gives the
kernel units straight to nn-Meter's nn_predict, which is what nn-Meter's predict() does after
detecting the kernels. Runs in the nn-Meter environment (~/envs/nnmeter).

    ~/envs/nnmeter/bin/python nnmeter_kernels.py units.json [--breakdown] [--hw-max 224]

units.json is {"model name": [kernel unit, ...], ...}. Prints one line per model, starting with JSON:.
The predictors were fitted on feature maps up to 224 x 224, and random forests don't extrapolate, so a
kernel on a larger map is predicted at 224 x 224 and scaled by the ratio of the areas (--hw-max 0: off).
"""
import argparse
import copy
import json
import time

PREDICTOR = "myriadvpu_openvino2019r2"


def clamp(unit, hw_max):
    """The unit on at most hw_max x hw_max, and the area ratio to scale its time by."""
    h = unit.get("inputh")
    if h is None and unit.get("input_tensors"):
        h = unit["input_tensors"][0][1] if len(unit["input_tensors"][0]) == 4 else None
    if not hw_max or h is None or h <= hw_max:
        return unit, 1.0
    u = copy.deepcopy(unit)
    for k in ("inputh", "inputw"):
        if k in u:
            u[k] = hw_max
    for t in u.get("input_tensors", []):
        if len(t) == 4:
            t[1] = t[2] = hw_max
    return u, (h / hw_max) ** 2


def predict(nn_predict, kp, units, hw_max):
    """Sum of the kernels' predicted times (ms): units grouped by scale, nn_predict on each group."""
    groups = {}
    for u in units:
        cu, sc = clamp(u, hw_max)
        groups.setdefault(round(sc, 6), []).append(cu)
    return float(sum(sc * nn_predict(kp, us) for sc, us in groups.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("units")
    ap.add_argument("--predictor", default=PREDICTOR)
    ap.add_argument("--version", type=float, default=1.0)
    ap.add_argument("--hw-max", type=int, default=224)
    ap.add_argument("--breakdown", action="store_true", help="also the time per kind of kernel")
    a = ap.parse_args()

    from nn_meter import load_latency_predictor
    from nn_meter.predictor.prediction.predict_by_kernel import merge_conv_kernels, nn_predict
    from nn_meter.predictor.prediction.utils import get_kernel_name
    kp = load_latency_predictor(a.predictor, a.version).kernel_predictors
    models = json.load(open(a.units))
    for name, units in models.items():
        rec = dict(model=name, predictor=a.predictor, ok=False)
        t0 = time.time()
        try:
            rec.update(ok=True, latency_ms=round(predict(nn_predict, kp, units, a.hw_max), 3), kernels=len(units),
                       missing=sorted({u["op"] for u in units if get_kernel_name(merge_conv_kernels(u["op"])) not in kp}))
            if a.breakdown:
                rec["by_op"] = {op: round(predict(nn_predict, kp, [u for u in units if u["op"] == op], a.hw_max), 3)
                                for op in sorted({u["op"] for u in units})}
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[-1500:]}"
        rec["s"] = round(time.time() - t0, 2)
        print("JSON:" + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
