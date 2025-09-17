import json, argparse
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from config import Paths, TrainCfg, SplitCfg, CalibCfg, ensure_dirs, override_train, CLASSES
from seed import set_seed
from data import make_loaders
from models import build, LabelSmoothingCE, freeze_backbone
from engine import train_epoch, eval_epoch, EarlyStop
from metrics import prf_per_class, accuracy, confusion, ece, brier, macro_auc_ovr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    args = ap.parse_args()
    paths = Paths()
    ensure_dirs(paths)
    cfg = override_train(TrainCfg())
    set_seed(cfg.seed)
    data_root = Path(args.data) if args.data else Path(paths.data) / "train"
    dl_tr, dl_va, dl_te = make_loaders(data_root, cfg.image_size, cfg.batch_size, cfg.num_workers, cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(cfg.model, len(CLASSES), pretrained=True).to(device)
    freeze_backbone(model, True)
    loss_fn = LabelSmoothingCE(cfg.label_smoothing)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    es = EarlyStop(cfg.patience, minimize=True)
    best_path = Path(paths.weights) / "best.pt"
    history = []
    for epoch in range(cfg.epochs):
        tr_loss, tr_logits, tr_labels = train_epoch(model, dl_tr, device, loss_fn, opt, torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available()))
        va_loss, va_logits, va_labels = eval_epoch(model, dl_va, device, loss_fn)
        p_tr = torch.softmax(tr_logits, dim=1)
        p_va = torch.softmax(va_logits, dim=1)
        yhat_tr = p_tr.argmax(1)
        yhat_va = p_va.argmax(1)
        _, mp_tr, mr_tr, mf_tr = prf_per_class(tr_labels, yhat_tr, len(CLASSES))
        _, mp_va, mr_va, mf_va = prf_per_class(va_labels, yhat_va, len(CLASSES))
        acc_va = accuracy(va_labels, yhat_va)
        ece_va = ece(p_va, va_labels, 15)
        brier_va = brier(p_va, va_labels, len(CLASSES))
        auc_va = macro_auc_ovr(p_va, va_labels, len(CLASSES))
        history.append({"epoch":epoch+1,"train_loss":float(tr_loss),"val_loss":float(va_loss),"val_accuracy":float(acc_va),"val_macro_precision":float(mp_va),"val_macro_recall":float(mr_va),"val_macro_f1":float(mf_va),"val_ece":float(ece_va),"val_brier":float(brier_va),"val_auc":float(auc_va)})
        if va_loss <= (es.best if es.best is not None else va_loss):
            torch.save({"model":model.state_dict(),"cfg":cfg.__dict__}, best_path)
        if es.step(va_loss):
            break
    te_loss, te_logits, te_labels = eval_epoch(model, dl_te, device, loss_fn)
    p_te = torch.softmax(te_logits, dim=1)
    yhat_te = p_te.argmax(1)
    _, mp_te, mr_te, mf_te = prf_per_class(te_labels, yhat_te, len(CLASSES))
    acc_te = accuracy(te_labels, yhat_te)
    ece_te = ece(p_te, te_labels, 15)
    brier_te = brier(p_te, te_labels, len(CLASSES))
    auc_te = macro_auc_ovr(p_te, te_labels, len(CLASSES))
    cm = confusion(te_labels, yhat_te, len(CLASSES)).tolist()
    report = {"history":history,"test":{"loss":float(te_loss),"accuracy":float(acc_te),"macro_precision":float(mp_te),"macro_recall":float(mr_te),"macro_f1":float(mf_te),"ece":float(ece_te),"brier":float(brier_te),"auc":float(auc_te),"cm":cm}}
    Path(Paths().logs).mkdir(parents=True, exist_ok=True)
    with open(Path(Paths().logs)/"train_report.json","w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)

if __name__ == "__main__":
    main()