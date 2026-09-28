# Prediksi Klaim Asuransi Kesehatan Individu — AXA Financial Indonesia

Proyek data science untuk menganalisis dan memproyeksikan **frekuensi, severity, dan total
klaim** asuransi kesehatan individu. Studi kasus dari **Data Science Competition ITB 2026**
(AXA Financial Indonesia).

> **Latar belakang.** Klaim asuransi kesehatan individu naik ~25,5% (Jan–Jun 2025 vs periode
> sama 2024). Kenaikan ini mendorong penyesuaian premi yang membuat produk kurang terjangkau.
> Perusahaan membutuhkan analisa berbasis data untuk memahami pendorong klaim dan
> memproyeksikan trennya, agar bisa mengambil langkah di sisi seleksi risiko, pencegahan, dan
> penetapan harga.

---

## Tujuan

1. Eksplorasi & pembersihan data klaim.
2. Mengidentifikasi faktor yang paling memengaruhi **severity** (besar klaim).
3. Mengidentifikasi faktor yang paling memengaruhi **frequency** (sering klaim).
4. Membangun & membandingkan model untuk memproyeksikan tren frekuensi, severity, dan total klaim.
5. Evaluasi model dengan metrik yang sesuai (**MAPE**) dan memberikan rekomendasi bisnis.

---

## Dataset

Data periode **Januari 2024 – Juli 2025** (19 bulan).

| File | Baris | Deskripsi |
|---|---|---|
| `Data_Polis.csv` | 4.096 | 1 baris per polis: plan code, gender, tanggal lahir, tanggal efektif polis, domisili |
| `Data_Klaim.csv` | 4.627 | 1 baris per klaim: nominal, jenis rawat, diagnosis (ICD), tanggal masuk/keluar RS, lokasi RS, dll |

---

## Alur Pengerjaan

```
Load data
  → Cleaning (tanggal, missing, duplikat, konsistensi)
    → Feature engineering (umur, lama rawat/LOS, tenure, bab ICD, bulan)
      → EDA (distribusi, tren bulanan, severity per kategori)
        → Analisis faktor severity & frequency (feature importance)
          → Agregasi deret waktu bulanan
            → Forecasting (2 model dibanding pakai MAPE) → prediksi 2025–2026
              → Submission Kaggle + rekomendasi
```

Prinsip desain: **satu alur data yang lurus**, **tanpa kebocoran target (data leakage)**, dan
memilih **model sesederhana mungkin yang bisa dipertanggungjawabkan** (data bulanan hanya 19
titik dan relatif datar, sehingga model kompleks justru overfitting).

---

## Ringkasan Hasil

**Faktor pendorong severity** (dari feature importance):
lama rawat (LOS), lokasi RS (Singapura), tenure polis, umur, dan penyakit bab-C ICD
(neoplasma/kanker). → Kasus rawat inap panjang dan penyakit berat adalah pendorong biaya utama.

**Faktor pendorong frequency:** umur dan tenure polis sangat dominan.

**Dampak feature engineering:** R² model severity naik dari **0,32 → 0,51**.

**Perbandingan model (MAPE pada data uji):**

| Target | Linear Regression | Random Forest | Terpilih |
|---|---|---|---|
| Frequency | **8,1%** | 8,1% | Linear Regression |
| Severity | **5,4%** | 7,8% | Linear Regression |

**Proyeksi 2026:** total klaim ~Rp147,6 miliar/tahun, ~226 klaim/bulan, severity ~Rp54,5 juta/klaim.

> Catatan: total klaim = frekuensi × severity (identitas), sehingga cukup memproyeksikan
> frekuensi & severity, lalu total diturunkan agar konsisten.

---

## Cara Menjalankan

**Kebutuhan:** Python 3.9+ dengan `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Lalu buka `Code_Portfolio.ipynb`, pastikan `Data_Klaim.csv` dan `Data_Polis.csv` ada di folder
yang sama, dan jalankan **Run All**.

> **Tips (Windows):** jika muncul error `DLL load failed ... An Application Control policy has
> blocked this file` saat `import sklearn`, itu kebijakan keamanan Windows (mis. Smart App
> Control) yang memblokir file scikit-learn — bukan error kode. Solusi tercepat: jalankan di
> **Google Colab**.

Output yang dihasilkan: `submission_kaggle.csv` (prediksi Agu–Des 2025, format `Id,value`).

---

## Struktur Repo

```
├── Code.ipynb   # notebook utama (analisis + model + forecast)
├── Data_Polis.csv         # data polis
├── Data_Klaim.csv         # data klaim
└── README.md
```

---

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`

Model: Linear Regression & Random Forest · Evaluasi: MAPE
