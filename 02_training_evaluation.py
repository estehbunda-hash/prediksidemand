# =============================================================
# TAHAP 2 — PELATIHAN & EVALUASI MODEL (5 ALGORITMA)
# Topik: Prediksi Demand Taksi Online (NYC Taxi Dataset)
# =============================================================
# Modifikasi yang bisa dilakukan:
#   - Ganti hyperparameter setiap model di bagian CONFIG
#   - Aktifkan/nonaktifkan model tertentu di RUN_MODELS
#   - Ganti target antara 'log_order_count' atau 'log_trip_duration'
#   - Sesuaikan SAMPLE_SIZE untuk keseimbangan kecepatan/akurasi
# =============================================================

import os, warnings, json, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ─────────────────────────────────────────────
# CONFIG — sesuaikan di sini
# ─────────────────────────────────────────────
INPUT_PATH   = 'data/nyc_scaled.csv'
OUTPUT_JSON  = 'models/model_results.json'
TARGET_COL   = 'log_order_count'    # ganti ke 'log_trip_duration' jika perlu
SAMPLE_SIZE  = 20000               # None = pakai semua (~1 juta, butuh RAM besar)
TEST_SIZE    = 0.2                  # proporsi data test
RANDOM_STATE = 42

# Pilih model yang ingin dijalankan (True/False)
RUN_MODELS = {
    'linear_regression': True,
    'ann':               True,
    'lstm':              True,
    'kmeans':            True,
    'backprop':          True,
}

# Hyperparameter ANN
ANN_LAYERS      = [256, 128, 64]   # jumlah neuron per hidden layer
ANN_DROPOUT     = [0.2, 0.1]      # dropout setelah layer 1 dan 2
ANN_EPOCHS      = 50
ANN_BATCH_SIZE  = 256
ANN_LR          = 0.0005           # learning rate Adam

# Hyperparameter LSTM
LSTM_UNITS      = [128, 64]        # unit per LSTM layer
LSTM_EPOCHS     = 30
LSTM_BATCH_SIZE = 512
LSTM_LR         = 0.001

# Hyperparameter K-Means
KMEANS_K_RANGE  = [2, 4, 6, 8, 10, 12]   # nilai k yang diuji
KMEANS_FEATURES = ['pickup_latitude', 'pickup_longitude',
                   'pickup_hour', 'pickup_dayofweek']  # fitur untuk clustering

# Hyperparameter Backpropagation manual
BP_HIDDEN_LAYERS = [64, 32]   # neuron per hidden layer
BP_LR            = 0.001
BP_EPOCHS        = 20
BP_BATCH_SIZE    = 256
BP_SAMPLE        = 30000      # sample untuk manual NN (batasan memori NumPy)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("LOAD DATA")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)
print(f"Shape dataset  : {df.shape}")
print(f"Target         : {TARGET_COL}")

# Deteksi kolom fitur secara otomatis (exclude target dan non-fitur)
EXCLUDE = ['log_trip_duration', 'log_order_count', 'trip_duration', 'order_count']
FEATURES = [c for c in df.columns if c not in EXCLUDE]
print(f"Jumlah fitur   : {len(FEATURES)}")
print(f"Fitur          : {FEATURES}")

if SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Sampling ke    : {SAMPLE_SIZE:,} baris")
else:
    print(f"Menggunakan seluruh data ({len(df):,} baris) karena total populasi < SAMPLE_SIZE")
    
X = df[FEATURES].values
y = df[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\nTrain set : {X_train.shape}")
print(f"Test set  : {X_test.shape}")

results = {}


# ─────────────────────────────────────────────
# HELPER: HITUNG METRIK
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return {
        'mae' : round(float(mae),  4),
        'rmse': round(float(rmse), 4),
        'r2'  : round(float(r2),   4),
        'mape': round(float(mape), 2),
    }

def print_metrics(name, m, t):
    print(f"  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
          f"R²={m['r2']:.4f}  MAPE={m['mape']:.2f}%  "
          f"waktu={t:.1f}s")


# ══════════════════════════════════════════════
# MODEL 1: LINEAR REGRESSION (Baseline)
# ══════════════════════════════════════════════
if RUN_MODELS['linear_regression']:
    print("\n" + "=" * 60)
    print("MODEL 1: Linear Regression")
    print("=" * 60)
    print("Library : scikit-learn LinearRegression()")
    print("Metrik  : MAE, RMSE, R²")

    t0 = time.time()

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    t_lr = time.time() - t0
    m = compute_metrics(y_test, y_pred_lr)
    m['train_time'] = round(t_lr, 2)
    results['Linear Regression'] = m
    print_metrics('Linear Regression', m, t_lr)

    # Simpan koefisien terpenting
    coefs = sorted(zip(FEATURES, lr.coef_), key=lambda x: abs(x[1]), reverse=True)
    print("  Top 5 koefisien:")
    for feat, coef in coefs[:5]:
        print(f"    {feat:25s}: {coef:+.4f}")


# ══════════════════════════════════════════════
# MODEL 2: ANN (Artificial Neural Network)
# ══════════════════════════════════════════════
if RUN_MODELS['ann']:
    print("\n" + "=" * 60)
    print("MODEL 2: ANN (Dense layers)")
    print("=" * 60)
    print("Library : TensorFlow / Keras Dense layers")
    print("Metrik  : MAE, RMSE, Loss Curve")

    import tensorflow as tf
    try:
        tf.random.set_seed(RANDOM_STATE)
    except AttributeError:
        print("  [Warning] Gagal mengatur random seed TensorFlow, melanjutkan training...")
    
    
    t0 = time.time()

    # Bangun arsitektur ANN
    # ── MODIFIKASI: ganti ANN_LAYERS untuk eksperimen arsitektur ──
    ann_layers_list = [
        tf.keras.layers.Dense(ANN_LAYERS[0], activation='relu',
                              input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(ANN_DROPOUT[0]),
    ]
    for i, units in enumerate(ANN_LAYERS[1:]):
        ann_layers_list.append(tf.keras.layers.Dense(units, activation='relu'))
        if i < len(ANN_DROPOUT) - 1:
            ann_layers_list.append(tf.keras.layers.Dropout(ANN_DROPOUT[i + 1]))
    ann_layers_list.append(tf.keras.layers.Dense(1))  # output

    ann = tf.keras.Sequential(ann_layers_list)
    ann.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=ANN_LR),
        loss='mse',
        metrics=['mae']
    )
    ann.summary()

    history_ann = ann.fit(
        X_train, y_train,
        epochs=ANN_EPOCHS,
        batch_size=ANN_BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    y_pred_ann = ann.predict(X_test, verbose=0).flatten()
    t_ann = time.time() - t0

    m = compute_metrics(y_test, y_pred_ann)
    m['train_time']    = round(t_ann, 2)
    m['loss_curve']    = [round(float(v), 4) for v in history_ann.history['loss']]
    m['val_loss_curve']= [round(float(v), 4) for v in history_ann.history['val_loss']]
    results['ANN'] = m
    print_metrics('ANN', m, t_ann)


# ══════════════════════════════════════════════
# MODEL 3: RNN / LSTM
# ══════════════════════════════════════════════
if RUN_MODELS['lstm']:
    print("\n" + "=" * 60)
    print("MODEL 3: RNN / LSTM")
    print("=" * 60)
    print("Library : TensorFlow / Keras LSTM layers")
    print("Metrik  : MAE, RMSE, MAPE")

    import tensorflow as tf
    try:
        tf.random.set_seed(RANDOM_STATE)
    except AttributeError:
        pass

    t0 = time.time()

    # Reshape input: (samples, timesteps=1, features)
    # ── MODIFIKASI: ubah timesteps > 1 jika data time-series berurutan ──
    X_train_rnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_rnn  = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])

    lstm = tf.keras.Sequential([
        tf.keras.layers.LSTM(LSTM_UNITS[0], return_sequences=True,
                             input_shape=(1, X_train.shape[1])),
        tf.keras.layers.LSTM(LSTM_UNITS[1]),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    lstm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_LR),
        loss='mse',
        metrics=['mae']
    )
    lstm.summary()

    history_lstm = lstm.fit(
        X_train_rnn, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    y_pred_lstm = lstm.predict(X_test_rnn, verbose=0).flatten()
    t_lstm = time.time() - t0

    m = compute_metrics(y_test, y_pred_lstm)
    m['train_time']    = round(t_lstm, 2)
    m['loss_curve']    = [round(float(v), 4) for v in history_lstm.history['loss']]
    m['val_loss_curve']= [round(float(v), 4) for v in history_lstm.history['val_loss']]
    results['LSTM'] = m
    print_metrics('LSTM', m, t_lstm)


# ══════════════════════════════════════════════
# MODEL 4: K-MEANS CLUSTERING
# ══════════════════════════════════════════════
if RUN_MODELS['kmeans']:
    print("\n" + "=" * 60)
    print("MODEL 4: K-Means Clustering")
    print("=" * 60)
    print("Library : scikit-learn KMeans()")
    print("Metrik  : Inertia, Silhouette Score, Elbow Method")

    t0 = time.time()

    # Ambil fitur spasio-temporal untuk clustering
    feat_idx = [FEATURES.index(f) for f in KMEANS_FEATURES if f in FEATURES]
    X_cluster = X[:, feat_idx]
    X_cluster_small = X_cluster[:20000]  # subset untuk silhouette (mahal komputasinya)

    inertias, silhouettes = [], []
    print(f"\n  Pengujian k = {KMEANS_K_RANGE}")
    for k in KMEANS_K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_cluster)
        inertia = km.inertia_
        sil = silhouette_score(X_cluster_small, labels[:20000])
        inertias.append(round(float(inertia), 2))
        silhouettes.append(round(float(sil), 4))
        print(f"    k={k:2d}: inertia={inertia:,.0f}  silhouette={sil:.4f}")

    # Pilih k terbaik berdasarkan silhouette tertinggi
    best_k_idx = int(np.argmax(silhouettes))
    best_k     = KMEANS_K_RANGE[best_k_idx]
    print(f"\n  Best k = {best_k} (silhouette={silhouettes[best_k_idx]:.4f})")

    km_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = km_best.fit_predict(X_cluster)

    # Gunakan label cluster sebagai fitur tambahan → prediksi dengan LR
    X_with_cluster = np.hstack([X, cluster_labels.reshape(-1, 1)])
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X_with_cluster, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    lr_c = LinearRegression()
    lr_c.fit(Xc_tr, yc_tr)
    y_pred_km = lr_c.predict(Xc_te)

    t_km = time.time() - t0
    m = compute_metrics(yc_te, y_pred_km)
    m['train_time']  = round(t_km, 2)
    m['best_k']      = int(best_k)
    m['k_range']     = KMEANS_K_RANGE
    m['inertias']    = inertias
    m['silhouettes'] = silhouettes
    results['KMeans'] = m
    print_metrics('KMeans+LR', m, t_km)


# ══════════════════════════════════════════════
# MODEL 5: BACKPROPAGATION MANUAL (NumPy)
# ══════════════════════════════════════════════
if RUN_MODELS['backprop']:
    print("\n" + "=" * 60)
    print("MODEL 5: Backpropagation (Manual NumPy)")
    print("=" * 60)
    print("Library : NumPy (manual) — custom training loop")
    print("Metrik  : Loss, Convergence Rate")

    t0 = time.time()

    class ManualNN:
        """
        Jaringan saraf tiruan dengan backpropagation manual.
        Aktivasi: ReLU untuk hidden layer, linear untuk output.
        Optimizer: Mini-batch Gradient Descent.

        Args:
            layer_sizes : list jumlah neuron [input, hidden..., output]
            lr          : learning rate
        """

        def __init__(self, layer_sizes, lr=0.001):
            self.lr = lr
            self.W, self.b = [], []
            np.random.seed(RANDOM_STATE)
            for i in range(len(layer_sizes) - 1):
                # He initialization untuk ReLU
                scale = np.sqrt(2.0 / layer_sizes[i])
                self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale)
                self.b.append(np.zeros((1, layer_sizes[i+1])))

        def relu(self, z):   return np.maximum(0, z)
        def relu_d(self, z): return (z > 0).astype(float)  # turunan ReLU

        def forward(self, X):
            """Forward pass: simpan aktivasi setiap layer untuk backprop."""
            self.a = [X]
            self.z = []
            for i in range(len(self.W) - 1):
                z = self.a[-1] @ self.W[i] + self.b[i]
                self.z.append(z)
                self.a.append(self.relu(z))
            # Layer output: linear (tanpa aktivasi)
            z = self.a[-1] @ self.W[-1] + self.b[-1]
            self.z.append(z)
            self.a.append(z)
            return self.a[-1]

        def backward(self, X, y):
            """Backpropagation: hitung gradien & update bobot."""
            m  = X.shape[0]
            dz = (self.a[-1] - y.reshape(-1, 1)) * 2 / m  # dL/dz output

            dW_list, db_list = [], []
            for i in range(len(self.W) - 1, -1, -1):
                dW = self.a[i].T @ dz
                db = np.sum(dz, axis=0, keepdims=True)
                dW_list.insert(0, dW)
                db_list.insert(0, db)
                if i > 0:
                    da = dz @ self.W[i].T
                    dz = da * self.relu_d(self.z[i - 1])

            for i in range(len(self.W)):
                self.W[i] -= self.lr * dW_list[i]
                self.b[i] -= self.lr * db_list[i]

        def train(self, X, y, epochs, batch_size):
            """Latih dengan mini-batch, kembalikan history loss per epoch."""
            losses = []
            for ep in range(epochs):
                idx  = np.random.permutation(len(X))
                X_sh = X[idx]
                y_sh = y[idx]
                ep_loss = 0.0
                n_batch = 0
                for s in range(0, len(X), batch_size):
                    Xb   = X_sh[s:s + batch_size]
                    yb   = y_sh[s:s + batch_size]
                    pred = self.forward(Xb)
                    ep_loss += float(np.mean((pred.flatten() - yb) ** 2))
                    self.backward(Xb, yb)
                    n_batch += 1
                avg_loss = ep_loss / n_batch
                losses.append(round(avg_loss, 4))
                print(f"  Epoch {ep+1:2d}/{epochs}: loss={avg_loss:.4f}")
            return losses

        def predict(self, X):
            return self.forward(X).flatten()

    # Subsample karena manual NN lebih lambat dari framework
    X_bp      = X_train[:BP_SAMPLE]
    y_bp      = y_train[:BP_SAMPLE]
    X_bp_test = X_test[:10000]
    y_bp_test = y_test[:10000]

    # Arsitektur: [n_features] → hidden layers → [1]
    # ── MODIFIKASI: ubah BP_HIDDEN_LAYERS ──
    layer_sizes = [X_bp.shape[1]] + BP_HIDDEN_LAYERS + [1]
    print(f"\n  Arsitektur: {layer_sizes}")

    model_bp = ManualNN(layer_sizes, lr=BP_LR)
    bp_losses = model_bp.train(X_bp, y_bp, epochs=BP_EPOCHS, batch_size=BP_BATCH_SIZE)

    y_pred_bp = model_bp.predict(X_bp_test)
    t_bp = time.time() - t0

    m = compute_metrics(y_bp_test, y_pred_bp)
    m['train_time'] = round(t_bp, 2)
    m['loss_curve'] = bp_losses
    results['Backprop'] = m
    print_metrics('Backprop', m, t_bp)


# ─────────────────────────────────────────────
# RINGKASAN PERBANDINGAN
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("RINGKASAN PERBANDINGAN SEMUA MODEL")
print("=" * 60)
header = f"{'Algoritma':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'MAPE':>8} {'Waktu':>8}"
print(header)
print("-" * len(header))
for name, m in sorted(results.items(), key=lambda x: -x[1]['r2']):
    print(f"{name:<22} {m['mae']:>7.4f} {m['rmse']:>7.4f} "
          f"{m['r2']:>7.4f} {m['mape']:>7.2f}% {m['train_time']:>7.2f}s")

# Simpan hasil ke JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nHasil disimpan ke: {OUTPUT_JSON}")
print("✓ Pelatihan & evaluasi selesai.")
