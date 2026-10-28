# MNIST ile MLP Tabanlı El Yazısı Rakam Tanıma

Ad Soyad / Öğrenci No: ...

## Çalıştırma
```bash
pip install -r requirements.txt
python train.py --all        # deneyler + model.pt + results/ grafikleri
python predict_own.py        # kendi el yazım (my_digits/ klasörü)
```

## Veri ayrımı
55.000 train / 5.000 validation (MNIST eğitim setinden, seed=42) / 10.000 test.
Test seti yalnızca model seçimi bittikten sonra bir kez kullanıldı.

## Model (Deney 1)
784 → 256 → ReLU → 128 → ReLU → 10 | CrossEntropyLoss | Adam lr=1e-3 | batch 128 | 15 epoch | normalizasyon (0.1307, 0.3081)

## Deney 1 sonuçları (results/results.json'dan doldur)
| Train loss | Val loss | Train acc | Val acc | Test acc | Süre (sn) |
|---|---|---|---|---|---|
| | | | | | |

![curves](results/curves_baseline.png)

## Hiperparametre karşılaştırması
| Config | Hidden | Optimizer | LR | Dropout | Val acc | Süre |
|---|---|---|---|---|---|---|
| baseline | 256-128 | Adam | 1e-3 | 0 | | |
| small | 64-32 | Adam | 1e-3 | 0 | | |
| wide | 512-256 | Adam | 1e-3 | 0 | | |
| deep | 256-128-64 | Adam | 1e-3 | 0 | | |
| sgd | 256-128 | SGD+mom. | 1e-2 | 0 | | |
| dropout | 256-128 | Adam | 1e-3 | 0.3 | | |
| high_lr | 256-128 | Adam | 1e-2 | 0 | | |

Yorum: (hangi ayar neden daha iyi/kötü, overfitting var mı, train-val farkı...)

## Kendi el yazım
Öğrenci numaram: ...  → `my_digits/` ve `results/own_digits_predictions.png`
Doğru tahmin: x/y

## Genel değerlendirme
- MNIST test accuracy ile kendi el yazımdaki başarı arasındaki fark ve nedeni (distribution shift: kalem kalınlığı, kâğıt/ışık, ortalama, eğim...)
- Yanlış tahmin edilen rakamlar ve olası sebepleri
- MLP'nin sınırları (piksel konumuna duyarlılık, CNN ile karşılaştırma yorumu)
