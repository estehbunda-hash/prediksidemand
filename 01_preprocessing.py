# =============================================================
# TAHAP 1 — PREPROCESSING & FEATURE ENGINEERING
# Topik: Prediksi Demand Taksi Online (NYC Taxi Dataset)
# Target: order_count (jumlah order per jam per area)
# =============================================================
# Modifikasi yang bisa dilakukan:
#   - Ganti path dataset di bagian CONFIG
#   - Sesuaikan batas outlier sesuai domain knowledge
#   - Tambah/hapus fitur di bagian FEATURE ENGINEERING
#   - Ganti ukuran grid spasial (GRID_SIZE) untuk demand target
# =============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG — sesuaikan di sini
# ─────────────────────────────────────────────
INPUT_PATH  = 'data/NYC.csv'              # path ke dataset asli
OUTPUT_RAW  = 'data/nyc_preprocessed.csv' # output data clean (belum dinormalisasi)
OUTPUT_SCALED = 'data/nyc_scaled.csv'     # output data ternormalisasi (siap model)
SAMPLE_SIZE = 50000                   # None = pakai semua data, atau set angka misal 200000
GRID_SIZE   = 0.01                   # ukuran grid spasial ~1 km (derajat lat/lon)
SCALER_TYPE = 'standard'             # 'standard' (Z-score) atau 'minmax'
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Load Data")
print("=" * 55)

df = pd.read_csv(INPUT_PATH)
print(f"Shape awal      : {df.shape}")
print(f"Kolom           : {df.columns.tolist()}")
print(f"\nMissing values  :\n{df.isnull().sum()}")
print(f"\nTipe data       :\n{df.dtypes}")
print(f"\nStatistik dasar :\n{df.describe()}")

if SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"\nSampling ke {SAMPLE_SIZE:,} baris")


# ─────────────────────────────────────────────
# STEP 2: HANDLE MISSING VALUES
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Handle Missing Values")
print("=" * 55)

missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Tidak ada missing values.")

# ── Strategi per kolom (modifikasi sesuai kebutuhan) ──
# Numerik  → isi dengan median
# Kategori → isi dengan mode
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  {col}: diisi median = {df[col].median():.4f}")
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  {col}: diisi mode  = {df[col].mode()[0]}")

print(f"Missing setelah handling: {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────
# STEP 3: PARSING DATETIME & FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Feature Engineering — Datetime")
print("=" * 55)

df['pickup_datetime']  = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

# Fitur waktu
df['pickup_hour']       = df['pickup_datetime'].dt.hour
df['pickup_day']        = df['pickup_datetime'].dt.day
df['pickup_month']      = df['pickup_datetime'].dt.month
df['pickup_dayofweek']  = df['pickup_datetime'].dt.dayofweek   # 0=Senin
df['pickup_weekofyear'] = df['pickup_datetime'].dt.isocalendar().week.astype(int)

# Fitur biner turunan waktu
df['is_weekend']   = (df['pickup_dayofweek'] >= 5).astype(int)
df['is_rush_hour'] = df['pickup_hour'].apply(
    lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
)

# Segmen waktu hari (ordinal)
# 0=pagi, 1=siang, 2=sore, 3=malam
def time_segment(h):
    if 5 <= h < 12:  return 0   # pagi
    elif 12 <= h < 17: return 1  # siang
    elif 17 <= h < 21: return 2  # sore/malam awal
    else:              return 3  # malam

df['time_segment'] = df['pickup_hour'].apply(time_segment)

print("Fitur datetime yang dibuat:")
print("  pickup_hour, pickup_day, pickup_month, pickup_dayofweek,")
print("  pickup_weekofyear, is_weekend, is_rush_hour, time_segment")
print(df[['pickup_hour','pickup_dayofweek','is_weekend',
          'is_rush_hour','time_segment']].head(3).to_string())


# ─────────────────────────────────────────────
# STEP 4: FEATURE ENGINEERING — SPASIAL & FISIK
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Feature Engineering — Spasial & Fisik")
print("=" * 55)

def haversine(lat1, lon1, lat2, lon2):
    """Hitung jarak geodesik (km) antara dua titik koordinat GPS."""
    R = 6371  # radius bumi (km)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df['trip_distance_km'] = haversine(
    df['pickup_latitude'],  df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)

# Kecepatan rata-rata (km/jam)
df['speed_kmh'] = df['trip_distance_km'] / (df['trip_duration'] / 3600 + 1e-5)

# Grid spasial untuk agregasi demand
df['pickup_lat_bin'] = (df['pickup_latitude']  / GRID_SIZE).round() * GRID_SIZE
df['pickup_lon_bin'] = (df['pickup_longitude'] / GRID_SIZE).round() * GRID_SIZE

print(f"trip_distance_km : mean={df['trip_distance_km'].mean():.2f} km, "
      f"max={df['trip_distance_km'].max():.2f} km")
print(f"speed_kmh        : mean={df['speed_kmh'].mean():.2f}, "
      f"max={df['speed_kmh'].max():.2f}")


# ─────────────────────────────────────────────
# STEP 5: MEMBUAT TARGET — DEMAND PER JAM/AREA
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Membuat Target Variable (order_count)")
print("=" * 55)

# Slot waktu per jam
df['pickup_hour_slot'] = df['pickup_datetime'].dt.floor('h')

# Hitung jumlah order per kombinasi (jam, lat_bin, lon_bin)
demand = (df.groupby(['pickup_hour_slot', 'pickup_lat_bin', 'pickup_lon_bin'])
            .size()
            .reset_index(name='order_count'))

df = df.merge(demand, on=['pickup_hour_slot', 'pickup_lat_bin', 'pickup_lon_bin'], how='left')

print(f"order_count: min={df['order_count'].min()}, "
      f"max={df['order_count'].max()}, "
      f"mean={df['order_count'].mean():.2f}")
print(f"Distribusi order_count:\n{df['order_count'].describe()}")


# ─────────────────────────────────────────────
# STEP 6: OUTLIER DETECTION & REMOVAL
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Outlier Removal")
print("=" * 55)

before = len(df)

# ── Batas outlier (MODIFIKASI sesuai domain) ──
DURATION_MIN  = 60       # detik (1 menit)
DURATION_MAX  = 10800    # detik (3 jam)
DISTANCE_MIN  = 0.1      # km
DISTANCE_MAX  = 100      # km
PASSENGER_MIN = 1
PASSENGER_MAX = 6
SPEED_MAX     = 150      # km/jam
# Bounding box NYC
LAT_MIN, LAT_MAX = 40.5, 41.0
LON_MIN, LON_MAX = -74.3, -73.7

df = df[
    (df['trip_duration'].between(DURATION_MIN, DURATION_MAX)) &
    (df['trip_distance_km'].between(DISTANCE_MIN, DISTANCE_MAX)) &
    (df['passenger_count'].between(PASSENGER_MIN, PASSENGER_MAX)) &
    (df['speed_kmh'].between(0, SPEED_MAX)) &
    (df['pickup_latitude'].between(LAT_MIN, LAT_MAX)) &
    (df['pickup_longitude'].between(LON_MIN, LON_MAX)) &
    (df['dropoff_latitude'].between(LAT_MIN, LAT_MAX)) &
    (df['dropoff_longitude'].between(LON_MIN, LON_MAX))
]

after = len(df)
print(f"Baris sebelum : {before:,}")
print(f"Baris sesudah : {after:,}")
print(f"Dihapus       : {before - after:,} ({(before-after)/before*100:.2f}%)")


# ─────────────────────────────────────────────
# STEP 7: ENCODING FITUR KATEGORIKAL
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 7: Encoding")
print("=" * 55)

# Binary encoding: Y/N → 1/0
df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
print("store_and_fwd_flag : Y/N → 1/0")

# Label encoding: vendor_id 1,2 → 0,1
df['vendor_id'] = df['vendor_id'] - 1
print("vendor_id          : 1,2 → 0,1")

# time_segment sudah ordinal (0-3) dari step 3
print("time_segment       : morning/afternoon/evening/night → 0/1/2/3")


# ─────────────────────────────────────────────
# STEP 8: FEATURE SELECTION
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 8: Feature Selection")
print("=" * 55)

# ── Fitur final (MODIFIKASI: tambah/hapus sesuai eksperimen) ──
FEATURES = [
    # metadata trip
    'vendor_id',
    'passenger_count',
    # fitur fisik
    'trip_distance_km',
    'speed_kmh',
    # fitur temporal
    'pickup_hour',
    'pickup_dayofweek',
    'pickup_month',
    'pickup_weekofyear',
    'is_weekend',
    'is_rush_hour',
    'time_segment',
    # flag
    'store_and_fwd_flag',
    # fitur spasial
    'pickup_latitude',
    'pickup_longitude',
    'dropoff_latitude',
    'dropoff_longitude',
]

# Korelasi fitur vs target
df_clean = df[FEATURES + ['trip_duration', 'order_count']].copy()
df_clean['log_trip_duration'] = np.log1p(df_clean['trip_duration'])
df_clean['log_order_count']   = np.log1p(df_clean['order_count'])

corr = (df_clean[FEATURES + ['log_order_count']]
        .corr()['log_order_count']
        .drop('log_order_count')
        .sort_values(ascending=False))
print("Korelasi fitur vs log_order_count:")
print(corr.to_string())


# ─────────────────────────────────────────────
# STEP 9: NORMALISASI
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 9: Normalisasi")
print("=" * 55)

X = df_clean[FEATURES].copy()
y_demand   = df_clean['log_order_count'].copy()
y_duration = df_clean['log_trip_duration'].copy()

if SCALER_TYPE == 'standard':
    scaler = StandardScaler()
    print("Menggunakan StandardScaler (Z-score)")
else:
    scaler = MinMaxScaler()
    print("Menggunakan MinMaxScaler (0-1)")

X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES, index=X.index)

# Simpan output
df_clean.to_csv(OUTPUT_RAW, index=False)
X_out = X_scaled.copy()
X_out['log_trip_duration'] = y_duration.values
X_out['log_order_count']   = y_demand.values
X_out.to_csv(OUTPUT_SCALED, index=False)

print(f"\nOutput disimpan:")
print(f"  {OUTPUT_RAW}    → data clean, belum dinormalisasi")
print(f"  {OUTPUT_SCALED} → data siap model (ternormalisasi)")
print(f"\nShape final: {X_scaled.shape}")
print(f"Fitur: {FEATURES}")
print("\n✓ Preprocessing selesai.")
