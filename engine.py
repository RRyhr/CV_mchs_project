import torch
from torch.cuda.amp import autocast, GradScaler

class Avg:
    def __init__(self):
        self.s = 0.0
        self.n = 0
    def upd(self, v, k=1):
        self.s += float(v)*k
        self.n += k
    def val(self):
        if self.n == 0:
            return 0.0
        return self.s / self.n

class EarlyStop:
    def __init__(self, patience=5, minimize=True, min_delta=0.0):
        self.patience = patience
        self.minimize = minimize
        self.min_delta = min_delta
        self.best = None
        self.bad = 0
        self.stop = False
    def step(self, value):
        if self.best is None:
            self.best = value
            self.bad = 0
            return False
        imp = value < (self.best - self.min_delta) if self.minimize else value > (self.best + self.min_delta)
        if imp:
            self.best = value
            self.bad = 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.stop = True
        return self.stop

def train_epoch(model, loader, device, loss_fn, optimizer, scaler=None):
    model.train()
    meter = Avg()
    all_logits = []
    all_labels = []
    for x,y,_ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        else:
            with autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        meter.upd(loss.item(), x.size(0))
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
    return meter.val(), torch.cat(all_logits,0), torch.cat(all_labels,0)

@torch.no_grad()
def eval_epoch(model, loader, device, loss_fn):
    model.eval()
    meter = Avg()
    all_logits = []
    all_labels = []
    for x,y,_ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        meter.upd(loss.item(), x.size(0))
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
    return meter.val(), torch.cat(all_logits,0), torch.cat(all_labels,0)