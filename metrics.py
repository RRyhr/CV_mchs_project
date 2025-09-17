import torch, numpy as np
from sklearn.metrics import roc_auc_score

def one_hot(y, n):
    e = torch.zeros((y.size(0), n), device=y.device, dtype=torch.float32)
    e.scatter_(1, y.view(-1,1), 1.0)
    return e

def prf_per_class(y_true, y_pred, n):
    s = {}
    eps = 1e-9
    for k in range(n):
        tp = ((y_true==k) & (y_pred==k)).sum().item()
        fp = ((y_true!=k) & (y_pred==k)).sum().item()
        fn = ((y_true==k) & (y_pred!=k)).sum().item()
        p = tp / (tp+fp+eps)
        r = tp / (tp+fn+eps)
        f1 = 2*p*r/(p+r+eps)
        s[k] = (p,r,f1)
    ps = np.array([s[k][0] for k in range(n)])
    rs = np.array([s[k][1] for k in range(n)])
    fs = np.array([s[k][2] for k in range(n)])
    return s, ps.mean(), rs.mean(), fs.mean()

def accuracy(y_true, y_pred):
    return (y_true==y_pred).float().mean().item()

def confusion(y_true, y_pred, n):
    m = np.zeros((n,n), dtype=np.int64)
    a = y_true.cpu().numpy()
    b = y_pred.cpu().numpy()
    for i in range(a.shape[0]):
        m[a[i], b[i]] += 1
    return m

def ece(probs, labels, n_bins=15):
    probs = probs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds==labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i+1])
        if m.sum() == 0:
            continue
        e += np.abs(acc[m].mean() - conf[m].mean()) * (m.sum()/len(conf))
    return float(e)

def brier(probs, labels, n_classes):
    y = np.eye(n_classes)[labels.detach().cpu().numpy()]
    p = probs.detach().cpu().numpy()
    return float(np.mean(np.sum((p - y) ** 2, axis=1)))

def macro_auc_ovr(probs, labels, n_classes):
    y = np.eye(n_classes)[labels.detach().cpu().numpy()]
    p = probs.detach().cpu().numpy()
    try:
        return float(roc_auc_score(y, p, average="macro", multi_class="ovr"))
    except:
        return float("nan")