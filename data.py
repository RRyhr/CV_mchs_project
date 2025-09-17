from pathlib import Path
import random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import CLASSES

def is_img(p):
    e = p.suffix.lower()
    return e in {".jpg",".jpeg",".png",".bmp",".webp"}

def index_dir(root):
    root = Path(root)
    items = []
    for i, c in enumerate(CLASSES):
        d = root / c
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and is_img(p):
                items.append((str(p), i))
    return items

def stratified_split(pairs, a=0.7, b=0.15, c=0.15, seed=42):
    rng = random.Random(seed)
    by = {i:[] for i in range(len(CLASSES))}
    for p,y in pairs:
        by[y].append((p,y))
    for k in by:
        rng.shuffle(by[k])
    tr, va, te = [], [], []
    for k in by:
        n = len(by[k])
        i1 = int(n*a)
        i2 = int(n*(a+b))
        tr.extend(by[k][:i1])
        va.extend(by[k][i1:i2])
        te.extend(by[k][i2:])
    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te

def build_transforms(size):
    t_tr = A.Compose([
        A.RandomResizedCrop(size,size,scale=(0.7,1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10,p=0.5),
        A.ColorJitter(0.1,0.1,0.1,0.05,p=0.5),
        A.GaussNoise(var_limit=(5.0,20.0),p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])
    t_ev = A.Compose([A.Resize(size,size),A.Normalize(),ToTensorV2()])
    return t_tr, t_ev

class ImgDS(Dataset):
    def __init__(self, root, items, transform):
        self.root = Path(root)
        self.items = items
        self.t = transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        x = self.t(image=np.array(img))["image"]
        return x, y, p

def make_loaders(root, size, bs, nw, seed=42):
    pairs = index_dir(root)
    tr, va, te = stratified_split(pairs, 0.7, 0.15, 0.15, seed)
    ttr, tev = build_transforms(size)
    ds_tr = ImgDS(root, tr, ttr)
    ds_va = ImgDS(root, va, tev)
    ds_te = ImgDS(root, te, tev)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return dl_tr, dl_va, dl_te