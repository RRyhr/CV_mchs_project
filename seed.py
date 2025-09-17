import os, random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def from_env():
    v = os.getenv("SEED")
    if v is None:
        set_seed(42)
        return 42
    try:
        s = int(v)
    except:
        s = 42
    set_seed(s)
    return s