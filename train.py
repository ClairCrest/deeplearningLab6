"""Lab 6: train vanilla CNN (Q1), improved CNN (Q2) or fine-tune ResNet-18 (Q4) on RealWaste.

Usage: python train.py --model {vanilla,improved,resnet18} --data path/to/RealWaste
       python train.py --model vanilla --data fake --epochs 1   # smoke test, no dataset needed
"""
import argparse, json, os, time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats (needed by pretrained ResNet)
NUM_CLASSES = 9


# ---------------------------------------------------------------- data
def make_dataset(root, tf):
    if root == "fake":
        return datasets.FakeData(size=96, image_size=(3, 256, 256), num_classes=NUM_CLASSES, transform=tf)
    return datasets.ImageFolder(root, transform=tf)


def build_data(root, seed=0, bs=64, workers=4, test_extra=None):
    """70/30 split by fixed seed. Two dataset objects share indices so test images get no augmentation.
    `test_extra` is an optional transform inserted before normalisation (used by compare.py for robustness)."""
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]
                                 + ([test_extra] if test_extra else [])
                                 + [transforms.Normalize(MEAN, STD)])
    train_full, test_full = make_dataset(root, train_tf), make_dataset(root, test_tf)
    n = len(train_full)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    cut = int(0.7 * n)
    pin = torch.cuda.is_available()
    train_dl = DataLoader(Subset(train_full, idx[:cut]), bs, shuffle=True, num_workers=workers, pin_memory=pin)
    test_dl = DataLoader(Subset(test_full, idx[cut:]), bs, shuffle=False, num_workers=workers, pin_memory=pin)
    classes = getattr(train_full, "classes", [str(i) for i in range(NUM_CLASSES)])
    return train_dl, test_dl, classes


# ---------------------------------------------------------------- models
class VanillaCNN(nn.Module):
    """Q1: only conv layers + ReLU. Downsampling via stride 2, no pooling/normalisation/skip.
    Last 1x1 conv outputs 9 class maps, averaged spatially -> logits (softmax lives in CrossEntropyLoss)."""
    def __init__(self, n=NUM_CLASSES):
        super().__init__()
        chans = [3, 32, 64, 128, 256, 256]
        layers = []
        for cin, cout in zip(chans, chans[1:]):
            layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(chans[-1], n, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).mean((2, 3))


class ResBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN + identity (1x1 projection when channels change), then ReLU."""
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        )
        self.skip = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class ImprovedCNN(nn.Module):
    """Q2: vanilla + BatchNorm (stable activations, higher usable LR), residual shortcuts (direct gradient path),
    max pooling (local invariance, cheaper), global avg pool + dropout (few params, less overfitting)."""
    def __init__(self, n=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        chans = [32, 64, 128, 256, 256]
        stages = []
        for cin, cout in zip(chans, chans[1:]):
            stages += [ResBlock(cin, cout), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(chans[-1], n))

    def forward(self, x):
        return self.head(self.features(self.stem(x)))


def build_model(name, pretrained=True):
    if name == "vanilla":
        return VanillaCNN()
    if name == "improved":
        return ImprovedCNN()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        return m
    raise ValueError(name)


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, dl, device, n_cls=NUM_CLASSES):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, conf = 0.0, torch.zeros(n_cls, n_cls, dtype=torch.long)
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x)
        total_loss += loss_fn(out.float(), y).item()
        pred = out.argmax(1)
        conf += torch.bincount((y * n_cls + pred).cpu(), minlength=n_cls * n_cls).view(n_cls, n_cls)
    n = conf.sum().item()
    return total_loss / n, conf.diag().sum().item() / n, conf


def make_opt(model, lr, epochs, steps):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps))  # stepped per batch
    return opt, sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vanilla", "improved", "resnet18"])
    ap.add_argument("--data", default="realwaste-main/RealWaste")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="model init / shuffling seed (data split is always seed 0)")
    ap.add_argument("--freeze_epochs", type=int, default=3, help="resnet18 only: head-only epochs before full fine-tune")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out, f"{args.model}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    train_dl, test_dl, classes = build_data(args.data, seed=0, bs=args.bs, workers=args.workers)
    model = build_model(args.model).to(device)
    print(f"device={device} model={args.model} params={sum(p.numel() for p in model.parameters()):,} "
          f"train={len(train_dl.dataset)} test={len(test_dl.dataset)}", flush=True)

    # Q4 phase 1: freeze backbone, train only the new head
    finetune = args.model == "resnet18"
    if finetune:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
        opt, sched = make_opt(model, args.lr, args.freeze_epochs, len(train_dl))
    else:
        opt, sched = make_opt(model, args.lr, args.epochs, len(train_dl))

    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    hist, best = [], 0.0
    for ep in range(args.epochs):
        if finetune and ep == args.freeze_epochs:  # phase 2: unfreeze everything, 10x smaller LR
            for p in model.parameters():
                p.requires_grad = True
            opt, sched = make_opt(model, args.lr / 10, args.epochs - ep, len(train_dl))
            print("-- unfroze backbone", flush=True)

        model.train()
        t0, batch_losses, correct, seen = time.time(), [], 0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            batch_losses.append(loss.item())
            correct += (out.argmax(1) == y).sum().item()
            seen += y.numel()

        bl = torch.tensor(batch_losses)
        te_loss, te_acc, _ = evaluate(model, test_dl, device)
        rec = dict(epoch=ep + 1, train_loss=bl.mean().item(), train_loss_std=bl.std().item() if len(bl) > 1 else 0.0,
                   train_acc=correct / seen, test_loss=te_loss, test_acc=te_acc, time=time.time() - t0)
        hist.append(rec)
        print(f"ep {rec['epoch']:3d} | train loss {rec['train_loss']:.4f} (std {rec['train_loss_std']:.3f}) "
              f"acc {rec['train_acc']:.3f} | test loss {te_loss:.4f} acc {te_acc:.3f} | {rec['time']:.0f}s", flush=True)
        if te_acc > best:
            best = te_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)

    # per-class report of the best checkpoint
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    _, acc, conf = evaluate(model, test_dl, device)
    print(f"\nBest test acc: {acc:.4f}\nPer-class accuracy:")
    for i, c in enumerate(classes):
        n = conf[i].sum().item()
        print(f"  {c:22s} {conf[i, i].item() / max(1, n):.3f}  (n={n})")
    torch.save(conf, os.path.join(out_dir, "confusion.pt"))


if __name__ == "__main__":
    main()
