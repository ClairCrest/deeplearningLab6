"""Lab 6 Q3: compare vanilla vs improved CNN (convergence, stability, robustness).

Reads runs/<model>_s<seed>/history.json + best.pt written by train.py.
Usage: python compare.py --data path/to/RealWaste [--models vanilla improved resnet18]
"""
import argparse, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from train import build_data, build_model, evaluate

PERTURB = {
    "clean": None,
    "gauss_noise": transforms.Lambda(lambda x: (x + 0.1 * torch.randn_like(x)).clamp(0, 1)),
    "brightness_contrast": transforms.ColorJitter(brightness=0.5, contrast=0.5),
    "rotation_15": transforms.RandomRotation(15),
    "blur": transforms.GaussianBlur(9, sigma=2.0),
}


def load_runs(runs_dir, model):
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir, f"{model}_s*"))):
        with open(os.path.join(d, "history.json")) as f:
            out.append((d, json.load(f)))
    return out


def epochs_to(hist, frac=0.9):
    """Convergence: first epoch whose test acc reaches `frac` of the run's best."""
    best = max(h["test_acc"] for h in hist)
    return next(h["epoch"] for h in hist if h["test_acc"] >= frac * best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="realwaste-main/RealWaste")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--models", nargs="+", default=["vanilla", "improved"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = {m: load_runs(args.runs, m) for m in args.models}
    runs = {m: r for m, r in runs.items() if r}
    if not runs:
        raise SystemExit(f"no runs found in {args.runs}/")

    # ---- curves (convergence + stability), every seed drawn
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for (m, rs), color in zip(runs.items(), ["tab:red", "tab:blue", "tab:green"]):
        for i, (_, h) in enumerate(rs):
            ep = [r["epoch"] for r in h]
            lab = m if i == 0 else None
            ax[0].plot(ep, [r["train_loss"] for r in h], color=color, label=lab)
            ax[0].plot(ep, [r["test_loss"] for r in h], color=color, ls="--")
            ax[1].plot(ep, [r["test_acc"] for r in h], color=color, label=lab)
            ax[2].plot(ep, [r["train_loss_std"] for r in h], color=color, label=lab)
    for a, t in zip(ax, ["loss (solid=train, dashed=test)", "test accuracy", "within-epoch batch-loss std"]):
        a.set_title(t); a.set_xlabel("epoch"); a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.runs, "curves.png"), dpi=120)
    print(f"saved {args.runs}/curves.png\n")

    # ---- summary table
    print("| model | seeds | best test acc (mean ± std) | final test acc | epochs to 90% of best | mean batch-loss std |")
    print("|---|---|---|---|---|---|")
    for m, rs in runs.items():
        best = torch.tensor([max(r["test_acc"] for r in h) for _, h in rs])
        final = torch.tensor([h[-1]["test_acc"] for _, h in rs])
        conv = torch.tensor([float(epochs_to(h)) for _, h in rs])
        lstd = torch.tensor([sum(r["train_loss_std"] for r in h) / len(h) for _, h in rs])
        sd = lambda t: t.std().item() if len(t) > 1 else 0.0
        print(f"| {m} | {len(rs)} | {best.mean():.4f} ± {sd(best):.4f} | {final.mean():.4f} | "
              f"{conv.mean():.1f} | {lstd.mean():.4f} |")

    # ---- robustness: best checkpoint of first seed, test set under perturbations
    print("\n| model | " + " | ".join(PERTURB) + " |")
    print("|---|" + "---|" * len(PERTURB))
    for m, rs in runs.items():
        model = build_model(m, pretrained=False).to(device)
        model.load_state_dict(torch.load(os.path.join(rs[0][0], "best.pt"), map_location=device))
        accs = []
        for name, tf in PERTURB.items():
            torch.manual_seed(0)  # same random perturbations for every model
            _, test_dl, _ = build_data(args.data, seed=0, workers=args.workers, test_extra=tf)
            accs.append(evaluate(model, test_dl, device)[1])
        cells = [f"{accs[0]:.3f}"] + [f"{a:.3f} ({a - accs[0]:+.3f})" for a in accs[1:]]
        print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
