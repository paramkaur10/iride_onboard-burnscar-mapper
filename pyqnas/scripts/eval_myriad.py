# scripts/eval_myriad.py
from openvino.runtime import Core
import numpy as np, argparse, glob, json, time

def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x, dtype=np.float32)
    return e / e.sum(axis=axis, keepdims=True)

def mask_to_labels(m):
    """
    Accepts:
      - HxW integer labels
      - 1xHxW (squeeze)
      - HxWxC one-hot
      - CxHxW one-hot
    Returns HxW int labels.
    """
    if m.ndim == 2:
        # HxW labels already
        return m.astype(np.int64)
    if m.ndim == 3:
        # 1xHxW -> squeeze
        if m.shape[0] == 1:
            return m[0].astype(np.int64)
        # CxHxW one-hot
        if m.shape[0] in (3, 4, 5, 6, 7, 8) and m.max() <= 1.0:
            return np.argmax(m, axis=0).astype(np.int64)
        # HxWxC one-hot
        if m.shape[-1] in (3, 4, 5, 6, 7, 8) and m.max() <= 1.0:
            return np.argmax(m, axis=-1).astype(np.int64)
    raise ValueError(f"Unsupported mask shape {m.shape} / dtype {m.dtype}")

def compute_miou(pred, gt, num_classes, ignore_index=None):
    ious = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        p = (pred == c)
        g = (gt == c)
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union == 0:
            continue
        ious.append(inter / (union + 1e-8))
    return float(np.mean(ious)) if ious else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to IR .xml")
    ap.add_argument("--device", default="MYRIAD")
    ap.add_argument("--images", required=True, help="Glob of .npy inputs (CHW or HxWxC with C=7)")
    ap.add_argument("--masks",  required=False, help="Glob of .npy masks (HxW, 1xHxW, HxWx4, or 4xHxW)")
    ap.add_argument("--classes", type=int, required=True)
    ap.add_argument("--max_batches", type=int, default=32)
    ap.add_argument("--ignore_index", type=int, default=-1,
                    help="Set to 0 to ignore Background; -1 uses all classes")
    args = ap.parse_args()

    ie = Core()
    model = ie.read_model(args.xml)
    cfg = {"MYRIAD_ENABLE_FORCE_RESET":"YES", "MYRIAD_THROUGHPUT_STREAMS":"1"}
    compiled = ie.compile_model(model, args.device, cfg)
    infer = compiled.create_infer_request()

    # Expect single input/output
    assert len(model.inputs) == 1, f"Expected 1 input, got {len(model.inputs)}"
    assert len(model.outputs) == 1, f"Expected 1 output, got {len(model.outputs)}"
    in_name = model.inputs[0].get_any_name()
    in_shape = [int(d) for d in model.inputs[0].shape]   # [1,7,H,W]
    out_port = compiled.outputs[0]

    img_paths = sorted(glob.glob(args.images))[:args.max_batches]
    if not img_paths:
        raise RuntimeError(f"No inputs matched: {args.images}")

    msk_paths = sorted(glob.glob(args.masks))[:len(img_paths)] if args.masks else []
    use_masks = bool(msk_paths) and (len(msk_paths) == len(img_paths))
    ignore = args.ignore_index if args.ignore_index >= 0 else None

    # Warmup with correct input shape
    warm = np.random.rand(*in_shape).astype("float32")
    infer.infer({in_name: warm})

    miou_accum = []
    t0 = time.time()
    for i, p in enumerate(img_paths):
        x = np.load(p)
        # Accept CHW; if HxWxC, transpose to CHW; ensure NCHW
        if x.ndim == 3:
            if x.shape[-1] == in_shape[1]:      # HxWxC -> CHW
                x = np.transpose(x, (2, 0, 1))
            x = x[None, ...]
        x = x.astype("float32")
        infer.infer({in_name: x})
        y = infer.get_tensor(out_port).data[:]   # [1,C,H,W] logits/probs
        # softmax -> argmax
        y = softmax(y, axis=1)
        pred = np.argmax(y, axis=1)[0].astype(np.int64)   # HxW

        if use_masks:
            gt = np.load(msk_paths[i])
            gt = mask_to_labels(gt)
            # size check
            if gt.shape != pred.shape:
                raise ValueError(f"Mask shape {gt.shape} != pred shape {pred.shape} for {msk_paths[i]}")
            miou_accum.append(compute_miou(pred, gt, args.classes, ignore_index=ignore))
    t1 = time.time()

    fps = len(img_paths) / (t1 - t0)
    avg_ms = (t1 - t0) / len(img_paths) * 1000.0
    mean_iou = float(np.mean(miou_accum)) if miou_accum else 0.0

    print(json.dumps({
        "mean_iou": mean_iou,
        "fps": float(fps),
        "avg_ms": float(avg_ms),
        "num_samples": len(img_paths)
    }))

if __name__ == "__main__":
    main()
