from dataclasses import dataclass, asdict
from pathlib import Path
import os, json

@dataclass
class Paths:
    project: str = str(Path(__file__).resolve().parent)
    data: str = str(Path(__file__).resolve().parent / "data")
    work: str = str(Path(__file__).resolve().parent / "work")
    weights: str = str(Path(__file__).resolve().parent / "work" / "weights")
    logs: str = str(Path(__file__).resolve().parent / "work" / "logs")
    exports: str = str(Path(__file__).resolve().parent / "work" / "exports")

@dataclass
class TrainCfg:
    image_size: int = 256
    batch_size: int = 32
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 5
    label_smoothing: float = 0.05
    num_workers: int = 2
    model: str = "efficientnet_b0"
    seed: int = 42

@dataclass
class SplitCfg:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

@dataclass
class CalibCfg:
    enable: bool = True
    init_temperature: float = 1.0
    max_iter: int = 1000
    lr: float = 1e-2
    tol: float = 1e-6

CLASSES = ["fire","flood","unknown"]

def ensure_dirs(p: Paths):
    Path(p.work).mkdir(parents=True, exist_ok=True)
    Path(p.weights).mkdir(parents=True, exist_ok=True)
    Path(p.logs).mkdir(parents=True, exist_ok=True)
    Path(p.exports).mkdir(parents=True, exist_ok=True)

def env_or(name, default):
    v = os.getenv(name)
    if v is None:
        return default
    t = type(default)
    if t is bool:
        return v.lower() in {"1","true","yes","y","on"}
    if t is int:
        try:
            return int(v)
        except:
            return default
    if t is float:
        try:
            return float(v)
        except:
            return default
    return v

def override_train(cfg: TrainCfg):
    cfg.image_size = env_or("IMG_SIZE", cfg.image_size)
    cfg.batch_size = env_or("BATCH_SIZE", cfg.batch_size)
    cfg.epochs = env_or("EPOCHS", cfg.epochs)
    cfg.lr = env_or("LR", cfg.lr)
    cfg.weight_decay = env_or("WD", cfg.weight_decay)
    cfg.patience = env_or("PATIENCE", cfg.patience)
    cfg.label_smoothing = env_or("LS", cfg.label_smoothing)
    cfg.num_workers = env_or("NUM_WORKERS", cfg.num_workers)
    cfg.model = env_or("MODEL", cfg.model)
    cfg.seed = env_or("SEED", cfg.seed)
    return cfg

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)