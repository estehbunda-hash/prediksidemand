# 🚖 Prediksi Demand Taksi Online — NYC Taxi Dataset

**Mata Kuliah:** Ujian Tengah Semester — Machine Learning  
**Nama Mahasiswa:** Putri Najwa Syahrus Shiam  
**NIM:** 301240013  

---

## Deskripsi Singkat

Proyek ini membangun sistem prediksi jumlah permintaan taksi online (demand) per jam per area di kota New York menggunakan dataset NYC.csv. Pipeline mencakup preprocessing data, feature engineering spasial-temporal, pelatihan 5 model, dan aplikasi web interaktif untuk simulasi prediksi secara real-time.

---

## Algoritma yang Digunakan

| Model | Deskripsi |
|---|---|
| **LSTM** | Long Short-Term Memory — model deep learning berbasis sekuens waktu |
| **ANN** | Artificial Neural Network — jaringan syaraf tiruan feedforward |
| **Backpropagation** | Jaringan syaraf dengan pelatihan manual backprop |
| **K-Means** | Clustering berbasis centroid untuk segmentasi area demand |
| **Linear Regression** | Regresi linier sebagai baseline model |

---

## Struktur Proyek

```
├── templates/
│   └── index.html               # Tampilan antarmuka web 
├── 01_preprocessing.py          # Pipeline preprocessing & feature engineering
├── 02_training_evaluation.py    # Pelatihan & evaluasi model
├── app.py                       # Aplikasi web Flask 
├── model_result.json           # Model Grafik Chart.js
├── nyc_preprocessed.csv     # Data bersih (belum dinormalisasi)
├── nyc_scaled.csv           # Data siap model (ternormalisasi)
├── NYC.csv                  # Dataset mentah
├── procfile
├── README.md
└── requirements.txt
```

---

## Cara Instalasi & Menjalankan Aplikasi

### 1. Clone Repository

```bash
git clone https://github.com/[username]/[nama-repo].git
cd [nama-repo]
```

### 2. Buat Virtual Environment (opsional tapi dianjurkan)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

> Dependensi utama: `flask`, `numpy`, `pandas`, `scikit-learn`, `tensorflow` / `keras`

### 4. Jalankan Preprocessing

```bash
python 01_preprocessing.py
```

### 5. Jalankan Pelatihan Model

```bash
python 02_training_evaluation.py
```

### 6. Jalankan Aplikasi Web

```bash
python app.py
```

Buka browser dan akses: `http://127.0.0.1:5000`

---

## Hasil Performa Model

| Model | MAE | RMSE | R² | MAPE (%) |
|---|---|---|---|---|
| ANN | 0.1900 | 0.2462 | 0.0572 | 20.25 |
| LSTM | 0.2030 | 0.2437 | 0.0766 | 22.78 |
| K-Means | 0.2170 | 0.2503 | 0.0258 | 24.44 |
| Linear Regression | 0.2172 | 0.2503 | 0.0256 | 24.45 |
| Backpropagation | 0.3069 | 0.3892 | -1.3556 | 35.85 |

---

## Tautan Penting

| Sumber | Link |
|---|---|
| 🌐 Demo Aplikasi | [Link Demo](https://your-demo-url.com) |
| 📄 Laporan | [Link Laporan](https://your-report-url.com) |
| 🎥 Video YouTube | [Link Video](https://youtube.com/watch?v=your-video-id) |
| 🌐 Dataset | [Link Dataset](https://www.kaggle.com/code/asifsaad/nyc-taxi-trip-duration) |
---

## Lisensi

Proyek ini dibuat untuk keperluan akademik.
