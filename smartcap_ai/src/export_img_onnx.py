"""
Export a trained image model (models/img_cnn.pt) to ONNX for the Raspberry Pi (onnxruntime only, no torch needed there).

    python src/export_img_onnx.py --model models/img_cnn.pt        # -> models/img_cnn.onnx + img_cnn.json

Keep the .onnx and the .json together. live_img.py accepts either the .pt or the .onnx.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgmodels as im  # noqa: E402


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model, ck = im.load_checkpoint(args.model, torch.device("cpu"))
    ch = 1 if ck["gray"] else 3
    out = Path(args.out) if args.out else Path(args.model).with_suffix(".onnx")
    dummy = torch.zeros(1, ch, ck["img"], ck["img"])
    kw = dict(input_names=["image"], output_names=["logits"], opset_version=17)
    try:
        torch.onnx.export(model, (dummy,), str(out), **kw)
    except Exception as err:
        print(f"default exporter failed ({type(err).__name__}), retrying with the legacy exporter")
        torch.onnx.export(model, (dummy,), str(out), dynamo=False, **kw)
    side = Path(str(out) + ".data")
    if side.exists():                                   # fold weights back into ONE file
        import onnx
        onnx.save(onnx.load(str(out)), str(out), save_as_external_data=False)
        side.unlink()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for _ in range(4):
        x = rng.random((1, ch, ck["img"], ck["img"]), dtype=np.float32)
        with torch.no_grad():
            want = model(torch.from_numpy(x)).numpy()
        got = sess.run(None, {"image": x})[0]
        assert np.allclose(got, want, atol=1e-3), "ONNX output differs from PyTorch!"
    t0 = time.perf_counter()
    for _ in range(50):
        sess.run(None, {"image": x})
    ms = (time.perf_counter() - t0) / 50 * 1000
    meta = {"arch": ck["arch"], "img": ck["img"], "gray": ck["gray"], "classes": ck["classes"]}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Saved {out} + {out.with_suffix('.json').name} | matches PyTorch | {ms:.2f} ms per image on this CPU")


if __name__ == "__main__":
    main()
