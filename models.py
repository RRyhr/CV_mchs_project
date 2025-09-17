import torch, torch.nn as nn, torchvision.models as M

class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, logits, target):
        n = logits.size(1)
        log_probs = self.log_softmax(logits)
        loss = -(log_probs.gather(1, target.view(-1,1))).squeeze(1)
        if self.eps > 0:
            loss = (1 - self.eps) * loss - self.eps * log_probs.mean(dim=1)
        return loss.mean()

def build(name, num_classes=3, pretrained=True):
    if name == "efficientnet_b0":
        m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m
    if name == "resnet50":
        m = M.resnet50(weights=M.ResNet50_Weights.DEFAULT if pretrained else None)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    raise ValueError(name)

def freeze_backbone(m, freeze=True):
    for n,p in m.named_parameters():
        if ("classifier" in n) or n.startswith("fc"):
            p.requires_grad = True
        else:
            p.requires_grad = not (not freeze)