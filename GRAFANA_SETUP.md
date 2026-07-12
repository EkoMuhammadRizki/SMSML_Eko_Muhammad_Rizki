# 📊 Panduan Setup Grafana, Prometheus & Monitoring

## Author: Eko Muhammad Rizki
## Project: SMSML_Eko_Muhammad_Rizki

---

## 🔧 Prasyarat

- Python 3.12 terinstall
- Semua dependencies sudah diinstall (`pip install -r Membangun_model/requirements.txt`)
- Model sudah di-train (sudah ada `data/processed/` dan `mlruns/`)

---

## Step 1: Jalankan FastAPI Inference Server

```bash
# Dari root project
cd SMSML_Eko_Muhammad_Rizki

# Jalankan server
python "Monitoring dan Logging/7.Inference.py"

# Server berjalan di http://localhost:8000
# Metrics endpoint: http://localhost:8000/metrics
# Docs: http://localhost:8000/docs
```

### Test Prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 24,
    "PhoneService": 1,
    "PaperlessBilling": 1,
    "MonthlyCharges": 70.35,
    "TotalCharges": 1688.40,
    "charge_per_month_ratio": 67.54,
    "is_new_customer": 0,
    "has_internet_addon": 3
  }'
```

---

## Step 2: Install & Konfigurasi Prometheus

### Download Prometheus:
1. Buka https://prometheus.io/download/
2. Download versi terbaru untuk Windows (`prometheus-*.windows-amd64.zip`)
3. Extract ke folder, misalnya `C:\prometheus\`

### Salin Konfigurasi:
```bash
# Copy file konfigurasi
cp "Monitoring dan Logging/2.prometheus.yml" C:\prometheus\prometheus.yml
```

### Jalankan Prometheus:
```bash
cd C:\prometheus
prometheus.exe --config.file=prometheus.yml
```

### Verifikasi:
- Buka http://localhost:9090
- Klik **Status > Targets** → Pastikan `ml-inference-server` statusnya **UP**

### Screenshot yang perlu diambil (di http://localhost:9090/graph):

| # | Query Prometheus | File Output |
|---|---|---|
| 1 | `ml_inference_latency_seconds_bucket` | `1.monitoring_latency.png` |
| 2 | `ml_request_total` | `2.monitoring_request_count.png` |
| 3 | `ml_http_errors_total` | `3.monitoring_http_error_rate.png` |
| 4 | `system_cpu_usage_percent` | `4.monitoring_cpu_usage.png` |
| 5 | `system_memory_usage_percent` | `5.monitoring_memory_usage.png` |
| 6 | `system_disk_usage_percent` | `6.monitoring_disk_usage.png` |
| 7 | `ml_data_drift_score` | `7.monitoring_data_drift.png` |
| 8 | `ml_prediction_churn_total` | `8.monitoring_prediction_distribution.png` |
| 9 | `ml_prediction_confidence_bucket` | `9.monitoring_model_confidence.png` |
| 10 | `ml_active_connections` | `10.monitoring_active_connections.png` |

> **Tips**: Kirim beberapa request predict terlebih dahulu agar metrics terisi sebelum screenshot!

---

## Step 3: Install & Konfigurasi Grafana

### Download Grafana:
1. Buka https://grafana.com/grafana/download?platform=windows
2. Download dan install Grafana OSS
3. Atau jalankan via Docker:
   ```bash
   docker run -d -p 3000:3000 --name=grafana grafana/grafana-oss
   ```

### Login Grafana:
- Buka http://localhost:3000
- Login: `admin` / `admin` (ubah password saat diminta)

---

## Step 4: Tambah Prometheus sebagai Data Source

1. Buka Grafana → **Configuration** (⚙️) → **Data Sources**
2. Klik **Add data source**
3. Pilih **Prometheus**
4. Isi:
   - **Name**: `prometheus`
   - **URL**: `http://localhost:9090`
5. Klik **Save & Test** → Pastikan muncul ✅ "Data source is working"

---

## Step 5: Import Dashboard

### Cara Import:

1. Buka Grafana → **Dashboards** → **Import**
2. Klik **Upload JSON file**
3. Pilih file: `grafana/dashboard.json`
4. Pilih data source: **prometheus**
5. Klik **Import**

### ⚠️ PENTING: Verifikasi Nama Dashboard!
- Dashboard HARUS bernama: **`eko_muhammad_rizki`**
- Jika nama berubah, edit di **Dashboard Settings** (⚙️) → **General** → **Name**

### Screenshot yang perlu diambil:

| # | Apa yang di-screenshot | File Output |
|---|---|---|
| 1 | Full dashboard overview | `1.monitoring_eko_muhammad_rizki_dashboard.png` |
| 2 | Panel "Inference Latency" | `2.grafana_latency_panel.png` |
| 3 | Panel "Request Count" | `3.grafana_request_count_panel.png` |
| 4 | Panel "HTTP Error Rate" | `4.grafana_http_error_panel.png` |
| 5 | Panel "CPU Usage" | `5.grafana_cpu_panel.png` |
| 6 | Panel "Memory Usage" | `6.grafana_memory_panel.png` |
| 7 | Panel "Disk Usage" | `7.grafana_disk_panel.png` |
| 8 | Panel "Data Drift Score" | `8.grafana_data_drift_panel.png` |
| 9 | Panel "Prediction Distribution" | `9.grafana_prediction_distribution_panel.png` |
| 10 | Panel "Model Confidence" | `10.grafana_model_confidence_panel.png` |

---

## Step 6: Setup Alerting Rules

### Import Alert Rules:

**Opsi A — Manual via UI:**

1. Buka Grafana → **Alerting** → **Alert Rules**
2. Klik **New alert rule** untuk setiap rule berikut:

**Rule 1: High Inference Latency**
- Query: `histogram_quantile(0.95, rate(ml_inference_latency_seconds_bucket[5m]))`
- Condition: `IS ABOVE 2`
- Evaluate every: `1m` for `5m`
- Labels: `severity=warning`

**Rule 2: High HTTP Error Rate**
- Query: `sum(rate(ml_http_errors_total[5m])) / sum(rate(ml_request_total[5m]))`
- Condition: `IS ABOVE 0.1`
- Evaluate every: `1m` for `5m`
- Labels: `severity=critical`

**Rule 3: High CPU Usage**
- Query: `system_cpu_usage_percent`
- Condition: `IS ABOVE 90`
- Evaluate every: `1m` for `5m`
- Labels: `severity=warning`

**Rule 4: Data Drift Detected**
- Query: `ml_data_drift_score`
- Condition: `IS ABOVE 0.3`
- Evaluate every: `1m` for `10m`
- Labels: `severity=critical`

**Opsi B — Via Provisioning (otomatis):**

Copy file `grafana/alerting_rules.json` ke Grafana provisioning directory:
```bash
# Linux/Docker
cp grafana/alerting_rules.json /etc/grafana/provisioning/alerting/

# Windows (sesuaikan path Grafana)
copy grafana\alerting_rules.json "C:\Program Files\GrafanaLabs\grafana\conf\provisioning\alerting\"
```

---

## Step 7: Setup Contact Points (Notifikasi)

### Webhook (Slack/Discord):
1. Buka Grafana → **Alerting** → **Contact Points**
2. Klik **New contact point**
3. Pilih **Webhook**
4. Isi URL webhook Slack/Discord Anda
5. Test & Save

### Email:
1. Buka Grafana → **Alerting** → **Contact Points**
2. Klik **New contact point**
3. Pilih **Email**
4. Isi alamat email
5. Konfigurasi SMTP di `grafana.ini`:
   ```ini
   [smtp]
   enabled = true
   host = smtp.gmail.com:587
   user = your-email@gmail.com
   password = your-app-password
   ```

### Screenshot Alerting yang perlu diambil:

| # | Apa yang di-screenshot | File Output |
|---|---|---|
| 1 | Alert rule: Latency | `1.rules_latency_alert.png` |
| 2 | Notifikasi latency | `2.notifikasi_latency_slack_or_email.png` |
| 3 | Alert rule: Error Rate | `3.rules_error_rate_alert.png` |
| 4 | Notifikasi error rate | `4.notifikasi_error_rate_alert.png` |
| 5 | Alert rule: CPU | `5.rules_cpu_alert.png` |
| 6 | Notifikasi CPU | `6.notifikasi_cpu_alert.png` |
| 7 | Alert rule: Data Drift | `7.rules_data_drift_alert.png` |
| 8 | Notifikasi drift | `8.notifikasi_data_drift_alert.png` |

---

## Step 8: Generate Traffic untuk Metrics

Jalankan script berikut untuk menghasilkan traffic dan memicu metrics:

```python
import requests
import time
import random

BASE_URL = "http://localhost:8000"

# Sample data
sample = {
    "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
    "tenure": 24.0, "PhoneService": 1, "PaperlessBilling": 1,
    "MonthlyCharges": 70.35, "TotalCharges": 1688.40,
    "charge_per_month_ratio": 67.54, "is_new_customer": 0,
    "has_internet_addon": 3
}

print("Sending prediction requests...")
for i in range(100):
    # Randomize some values
    data = sample.copy()
    data["tenure"] = random.uniform(1, 72)
    data["MonthlyCharges"] = random.uniform(20, 110)
    data["TotalCharges"] = data["MonthlyCharges"] * data["tenure"]
    data["charge_per_month_ratio"] = data["TotalCharges"] / (data["tenure"] + 1)
    data["is_new_customer"] = 1 if data["tenure"] <= 6 else 0
    
    try:
        resp = requests.post(f"{BASE_URL}/predict", json=data)
        print(f"[{i+1}/100] Status: {resp.status_code}, Result: {resp.json()['prediction_label']}")
    except Exception as e:
        print(f"[{i+1}/100] Error: {e}")
    
    time.sleep(0.5)

print("Done! Check Prometheus and Grafana for metrics.")
```

---

## 📋 Checklist Final

- [ ] FastAPI server berjalan di `http://localhost:8000`
- [ ] Prometheus scraping metrics di `http://localhost:9090`
- [ ] Grafana dashboard **"eko_muhammad_rizki"** ter-import
- [ ] 10 panel Grafana menampilkan data
- [ ] 4 alert rules terkonfigurasi
- [ ] Contact point (webhook/email) tersetting
- [ ] Semua screenshot sudah diambil dan disimpan di folder yang benar
- [ ] `screenshoot_dashboard.jpg` (Grafana dashboard overview) disimpan di `Membangun_model/`
- [ ] `screenshoot_artifak.jpg` (MLflow artifacts) disimpan di `Membangun_model/`
