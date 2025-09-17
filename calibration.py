import torch, torch.nn as nn, torch.optim as optim

class Temperature(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()
        self.t = nn.Parameter(torch.ones(1)*init)
    def forward(self, logits):
        return logits / self.t.clamp_min(1e-3)

def fit_temperature(logits, labels, init=1.0, max_iter=1000, lr=1e-2, tol=1e-6, device="cpu"):
    m = Temperature(init).to(device)
    logits = logits.to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.LBFGS(m.parameters(), lr=lr, max_iter=max_iter, tolerance_grad=tol, tolerance_change=tol, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(m(logits), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return m