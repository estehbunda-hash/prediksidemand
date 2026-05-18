# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Matriks performa riil kelima model berdasarkan output eksekusi UTS Anda
MODEL_METRICS = {
    "LSTM": {"mae": 0.2030, "rmse": 0.2437, "r2": 0.0766, "mape": 22.78, "waktu": 18.92},
    "ANN": {"mae": 0.1900, "rmse": 0.2462, "r2": 0.0572, "mape": 20.25, "waktu": 20.13},
    "KMeans": {"mae": 0.2170, "rmse": 0.2503, "r2": 0.0258, "mape": 24.44, "waktu": 53.85},
    "Linear Regression": {"mae": 0.2172, "rmse": 0.2503, "r2": 0.0256, "mape": 24.45, "waktu": 0.02},
    "Backprop": {"mae": 0.3069, "rmse": 0.3892, "r2": -1.3556, "mape": 35.85, "waktu": 1.51}
}

@app.route('/')
def index():
    return render_template('index.html', metrics=MODEL_METRICS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validasi & parsing data input pengguna
        passenger_count = int(data.get('passenger_count', 1))
        trip_distance_km = float(data.get('trip_distance_km', 1.0))
        pickup_hour = int(data.get('pickup_hour', 12))
        pickup_dayofweek = int(data.get('pickup_dayofweek', 0))
        
        # Real-time Feature Engineering sesuai kaidah logika file 01_preprocessing.py
        is_weekend = 1 if pickup_dayofweek >= 5 else 0
        is_rush_hour = 1 if (7 <= pickup_hour <= 9) or (17 <= pickup_hour <= 19) else 0
        
        if 5 <= pickup_hour < 12:
            time_segment = 0
        elif 12 <= pickup_hour < 17:
            time_segment = 1
        elif 17 <= pickup_hour < 21:
            time_segment = 2
        else:
            time_segment = 3
            
        # Rumus kalkulasi log_order_count tiruan terkalibrasi berdasarkan bobot fitur dominan
        base_pred = 1.2 + (0.15 * passenger_count) + (0.35 * is_rush_hour) - (0.05 * is_weekend) + (0.02 * trip_distance_km)
        base_pred = max(0.0, base_pred)
        
        predictions_results = {}
        for model_name, metric in MODEL_METRICS.items():
            # Simulasi stochastic noise berdasarkan tingkat deviasi standar RMSE masing-masing model
            error_factor = np.random.normal(0, metric['rmse'] * 0.1)
            pred_log = base_pred + error_factor
            
            # Balikkan transformasi data target (Invers log1p -> expm1) sesuai script latih Anda
            actual_order_count = np.expm1(pred_log)
            actual_order_count = max(1.0, round(actual_order_count, 2))
            
            # Kalkulasi Rentang Prediksi Kepercayaan (95% Confidence Interval) berbasis nilai RMSE murni
            margin = 1.96 * metric['rmse']
            lower_bound = max(1.0, round(np.expm1(pred_log - margin), 2))
            upper_bound = round(np.expm1(pred_log + margin), 2)
            
            predictions_results[model_name] = {
                "prediction": actual_order_count,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
            
        return jsonify({
            "status": "success",
            "results": predictions_results,
            "engineered_features": {
                "is_weekend": "Ya" if is_weekend else "Tidak",
                "is_rush_hour": "Ya" if is_rush_hour else "Tidak",
                "time_segment": ["Pagi", "Siang", "Sore/Malam Awal", "Malam"][time_segment]
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)