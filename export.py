import torch
from pathlib import Path
from config import Paths, TrainCfg, CLASSES
from models import build

def main():
    paths = Paths()
    cfg = TrainCfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = build(cfg.model, len(CLASSES), pretrained=False).to(device)
    ckpt = torch.load(Path(paths.weights)/"best.pt", map_location=device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    x = torch.randn(1,3,cfg.image_size,cfg.image_size, device=device)
    traced = torch.jit.trace(m, x, strict=False)
    out_script = Path(paths.exports)/"model_script.pt"
    out_script.parent.mkdir(parents=True, exist_ok=True)
    traced.save(out_script)
    out_onnx = Path(paths.exports)/"model.onnx"
    torch.onnx.export(m, x, out_onnx, input_names=["input"], output_names=["logits"], opset_version=12, dynamic_axes={"input":{0:"N"},"logits":{0:"N"}})

if __name__ == "__main__":
    main()