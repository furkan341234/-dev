"""
Tek görselde yan yana yazılmış rakamları ayırır, her birini MNIST biçimine (28x28,
siyah arka plan, beyaz rakam, ortalanmış) çevirir ve my_digits/ klasörüne kaydeder.

Kullanım:
    python split_digits.py numara.png --number 2021123456
    python split_digits.py tek_rakam.png --number 7        # tek rakam da olur
Seçenekler:
    --gap 0.15   rakamlar arası boşluk eşiği (rakam yüksekliğine oranı). Rakamlar
                 birleşik çıkıyorsa küçült (0.08), tek rakam ikiye bölünüyorsa büyüt (0.25).
Çıktı:
    my_digits/01_2.png, 02_0.png, ...   ve   results/split_preview.png (kontrol için)
"""
import argparse, os
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def otsu(a):
    hist, _ = np.histogram(a, bins=256, range=(0, 255))
    p = hist / hist.sum(); w = np.cumsum(p); m = np.cumsum(p * np.arange(256))
    sb = (m[-1] * w - m) ** 2 / (w * (1 - w) + 1e-12)
    return int(np.argmax(sb))


def load_gray_inverted(path):
    a = np.array(ImageOps.exif_transpose(Image.open(path)).convert("L"), dtype=np.float32)
    bg = ndimage.gaussian_filter(a, max(a.shape) / 20)       # gölge / ışık farkını azalt
    if a.mean() > 127:                                       # beyaz kâğıt -> rakam beyaz olsun
        a = np.clip(bg - a, 0, None)
    else:
        a = np.clip(a - bg, 0, None)
    return a / (a.max() + 1e-8) * 255


def to_mnist(crop):
    """crop: rakam beyaz, arka plan siyah (0-255). 20x20'ye sığdır, 28x28'e ortala."""
    h, w = crop.shape
    s = 20.0 / max(h, w)
    img = Image.fromarray(crop.astype(np.uint8)).resize(
        (max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    a = np.array(img, dtype=np.float32)
    canvas = np.zeros((28, 28), np.float32)
    y0, x0 = (28 - a.shape[0]) // 2, (28 - a.shape[1]) // 2
    canvas[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    cy, cx = ndimage.center_of_mass(canvas)
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def segment(ink, gap_ratio):
    cols = ink.any(axis=0)
    xs = np.where(cols)[0]
    ys = np.where(ink.any(axis=1))[0]
    height = ys.max() - ys.min() + 1
    gap = max(2, int(gap_ratio * height))
    runs, start, prev = [], xs[0], xs[0]
    for x in xs[1:]:
        if x - prev > gap:
            runs.append((start, prev)); start = x
        prev = x
    runs.append((start, prev))
    # çok küçük lekeleri (gürültü) at
    total = ink.sum()
    return [(a, b) for a, b in runs if ink[:, a:b + 1].sum() > 0.01 * total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--number", default="", help="gerçek rakamlar, örn. 2021123456")
    ap.add_argument("--gap", type=float, default=0.15)
    ap.add_argument("--out", default="my_digits")
    args = ap.parse_args()

    gray = load_gray_inverted(args.image)
    ink = gray > otsu(gray) * 0.8
    ink = ndimage.binary_opening(ink, iterations=1)          # tek piksellik gürültüyü sil
    runs = segment(ink, args.gap)
    print(f"{len(runs)} rakam bulundu.")
    if args.number and len(args.number) != len(runs):
        print(f"UYARI: --number {len(args.number)} rakam, görselde {len(runs)} bulundu. "
              f"results/split_preview.png'ye bak, --gap değerini değiştir.")

    os.makedirs(args.out, exist_ok=True); os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(2, max(len(runs), 1), figsize=(2 * max(len(runs), 1), 4.4), squeeze=False)
    for i, (a, b) in enumerate(runs):
        rows = np.where(ink[:, a:b + 1].any(axis=1))[0]
        crop = gray[rows.min():rows.max() + 1, a:b + 1]
        # kırpma sırasında silinmiş zayıf pikselleri temizle
        crop = np.where(crop > otsu(gray) * 0.4, crop, 0)
        img28 = to_mnist(crop)
        label = args.number[i] if i < len(args.number) else "x"
        path = f"{args.out}/{i+1:02d}_{label}.png"
        Image.fromarray(img28).save(path)
        axes[0][i].imshow(gray[:, a:b + 1], cmap="gray"); axes[0][i].set_title("kesilen", fontsize=8)
        axes[1][i].imshow(img28, cmap="gray"); axes[1][i].set_title(f"28x28 → {label}", fontsize=8)
        for r in (0, 1): axes[r][i].axis("off")
        print("kaydedildi:", path)
    plt.tight_layout(); plt.savefig("results/split_preview.png", dpi=130)
    print("Kontrol görseli: results/split_preview.png")


if __name__ == "__main__":
    main()
