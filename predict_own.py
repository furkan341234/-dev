"""
Kendi el yazısı rakamlarını (öğrenci numarası) eğitilmiş modele verir.

Dosya adı biçimi:  my_digits/<sıra>_<gerçek_rakam>.png   örn: 01_2.png, 02_1.png ...
Her dosyada TEK rakam olsun (beyaz kâğıt, koyu kalem).

Kullanım:  python predict_own.py
"""
import glob, os
import numpy as np, torch
from PIL import Image, ImageOps
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import MLP, MEAN, STD


def preprocess(path):
    """MNIST biçimi: siyah arka plan, beyaz rakam, 20x20 kutuya sığdır, 28x28 ortala."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("L")     # gri tonlama
    a = np.array(img, dtype=np.float32)
    if a.mean() > 127:            # beyaz kağıt / koyu kalem -> renkleri ters çevir
        a = 255 - a
    a = (a - a.min()) / (a.max() - a.min() + 1e-8) * 255             # kontrast
    a[a < 0.2 * 255] = 0                                              # kâğıt gürültüsünü temizle
    ys, xs = np.where(a > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]               # rakamı kırp
    h, w = a.shape
    s = 20.0 / max(h, w)
    a = np.array(Image.fromarray(a.astype(np.uint8)).resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS), dtype=np.float32)
    canvas = np.zeros((28, 28), dtype=np.float32)
    y0, x0 = (28 - a.shape[0]) // 2, (28 - a.shape[1]) // 2
    canvas[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    cy, cx = ndimage.center_of_mass(canvas)                           # kütle merkezini ortala
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1)
    return canvas / 255.0


def main():
    ckpt = torch.load("model.pt", map_location="cpu")
    cfg = ckpt["config"]
    model = MLP(cfg["hidden"], cfg["dropout"]); model.load_state_dict(ckpt["state_dict"]); model.eval()

    files = sorted(glob.glob("my_digits/*.png") + glob.glob("my_digits/*.jpg") + glob.glob("my_digits/*.jpeg"))
    assert files, "my_digits/ klasörüne görselleri koy (örn. 01_2.png)"
    fig, axes = plt.subplots(1, len(files), figsize=(2 * len(files), 2.6), squeeze=False)
    preds, truths, correct = "", "", 0
    for ax, f in zip(axes[0], files):
        x = preprocess(f)
        t = torch.tensor((x - MEAN) / STD, dtype=torch.float32).view(1, 1, 28, 28)   # normalizasyon
        with torch.no_grad():
            p = torch.softmax(model(t), 1)[0]
        pred, conf = int(p.argmax()), float(p.max())
        truth = os.path.splitext(os.path.basename(f))[0].split("_")[-1]
        preds += str(pred); truths += truth; correct += int(str(pred) == truth)
        ax.imshow(x, cmap="gray"); ax.axis("off")
        ax.set_title(f"gerçek {truth}\ntahmin {pred} (%{conf*100:.0f})", color="green" if str(pred) == truth else "red", fontsize=9)
    plt.tight_layout(); os.makedirs("results", exist_ok=True)
    plt.savefig("results/own_digits_predictions.png", dpi=150)
    print(f"Gerçek numara : {truths}\nTahmin edilen : {preds}\nDoğru: {correct}/{len(files)}")


if __name__ == "__main__":
    main()
