"""
MNIST - MLP ile el yazısı rakam tanıma (CNN YOK, hazır model YOK, sıfırdan eğitim).

Kullanım:
    python train.py                 # Deney 1: temel MLP
    python train.py --all           # Deney 1 + hiperparametre karşılaştırmaları

Veri ayrımı:
    60.000 eğitim görüntüsü -> 55.000 train / 5.000 validation
    10.000 test görüntüsü   -> SADECE en sonda, bir kez kullanılır.
Model seçimi / hiperparametre karşılaştırması sadece validation ile yapılır.
"""
import argparse, json, os, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
MEAN, STD = 0.1307, 0.3081          # MNIST eğitim seti istatistikleri (normalizasyon)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Deney yapılandırmaları (rapordaki hiperparametre tablosu) ----
CONFIGS = {
    "baseline":   dict(hidden=[256, 128], optimizer="adam", lr=1e-3, dropout=0.0, batch_size=128, epochs=15),
    "small":      dict(hidden=[64, 32],   optimizer="adam", lr=1e-3, dropout=0.0, batch_size=128, epochs=15),
    "wide":       dict(hidden=[512, 256], optimizer="adam", lr=1e-3, dropout=0.0, batch_size=128, epochs=15),
    "deep":       dict(hidden=[256, 128, 64], optimizer="adam", lr=1e-3, dropout=0.0, batch_size=128, epochs=15),
    "sgd":        dict(hidden=[256, 128], optimizer="sgd",  lr=1e-2, dropout=0.0, batch_size=128, epochs=15),
    "dropout":    dict(hidden=[256, 128], optimizer="adam", lr=1e-3, dropout=0.3, batch_size=128, epochs=15),
    "high_lr":    dict(hidden=[256, 128], optimizer="adam", lr=1e-2, dropout=0.0, batch_size=128, epochs=15),
}


class MLP(nn.Module):
    def __init__(self, hidden, dropout=0.0):
        super().__init__()
        layers, in_dim = [nn.Flatten()], 28 * 28
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 10))   # softmax, CrossEntropyLoss içinde
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_loaders(batch_size):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    full_train = datasets.MNIST("data", train=True, download=True, transform=tf)
    test = datasets.MNIST("data", train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(SEED)
    train, val = random_split(full_train, [55000, 5000], generator=g)
    print(f"Train: {len(train)}  Validation: {len(val)}  Test: {len(test)}")
    return (DataLoader(train, batch_size, shuffle=True),
            DataLoader(val, 1000),
            DataLoader(test, 1000))


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss_sum += criterion(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return loss_sum / n, correct / n


def run(name, cfg, loaders):
    torch.manual_seed(SEED)
    train_l, val_l, _ = loaders
    model = MLP(cfg["hidden"], cfg["dropout"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = (torch.optim.Adam(model.parameters(), lr=cfg["lr"]) if cfg["optimizer"] == "adam"
           else torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9))
    hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
    best_val, best_state = -1, None
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for x, y in train_l:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            criterion(model(x), y).backward()
            opt.step()
        # her epoch sonunda train ve val metrikleri (eval modunda)
        tl, ta = evaluate(model, train_l, criterion)
        vl, va = evaluate(model, val_l, criterion)
        for k, v in zip(hist, [tl, vl, ta, va]):
            hist[k].append(v)
        if va > best_val:
            best_val, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"[{name}] ep {ep:02d} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
    train_time = time.time() - t0
    model.load_state_dict(best_state)          # validation'a göre en iyi epoch
    return model, hist, train_time, best_val


def plot_history(hist, name):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["train_loss"], label="Train"); ax[0].plot(ep, hist["val_loss"], label="Validation")
    ax[0].set(title=f"Loss ({name})", xlabel="Epoch", ylabel="Loss"); ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="Train"); ax[1].plot(ep, hist["val_acc"], label="Validation")
    ax[1].set(title=f"Accuracy ({name})", xlabel="Epoch", ylabel="Accuracy"); ax[1].legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/curves_{name}.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="tüm yapılandırmaları karşılaştır")
    args = ap.parse_args()
    names = list(CONFIGS) if args.all else ["baseline"]

    loaders = get_loaders(128)
    results, models = {}, {}
    for name in names:
        cfg = CONFIGS[name]
        if cfg["batch_size"] != 128:
            loaders = get_loaders(cfg["batch_size"])
        model, hist, t, best_val = run(name, cfg, loaders)
        plot_history(hist, name)
        models[name] = model
        results[name] = dict(config=cfg, best_val_acc=best_val, train_time_sec=round(t, 1),
                             final_train_loss=hist["train_loss"][-1], final_val_loss=hist["val_loss"][-1],
                             final_train_acc=hist["train_acc"][-1], final_val_acc=hist["val_acc"][-1])

    # ---- Test seti: model seçimi BİTTİKTEN sonra, sadece bir kez ----
    criterion = nn.CrossEntropyLoss()
    best_name = max(results, key=lambda n: results[n]["best_val_acc"])
    for name in {"baseline", best_name}:
        _, acc = evaluate(models[name], loaders[2], criterion)
        results[name]["test_acc"] = acc
        print(f"TEST accuracy [{name}]: {acc:.4f}")
    results["_selected_by_validation"] = best_name

    torch.save({"state_dict": models[best_name].state_dict(), "config": CONFIGS[best_name]}, "model.pt")
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSeçilen model: {best_name}  -> model.pt  |  özet: results/results.json")


if __name__ == "__main__":
    main()
