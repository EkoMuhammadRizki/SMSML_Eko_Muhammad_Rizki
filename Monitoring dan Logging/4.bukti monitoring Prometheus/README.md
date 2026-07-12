Folder ini berisi bukti screenshot monitoring Prometheus.

File yang perlu di-capture:
1. 1.monitoring_latency.png          — ml_inference_latency_seconds
2. 2.monitoring_request_count.png    — ml_request_total
3. 3.monitoring_http_error_rate.png  — ml_http_errors_total
4. 4.monitoring_cpu_usage.png        — system_cpu_usage_percent
5. 5.monitoring_memory_usage.png     — system_memory_usage_percent
6. 6.monitoring_disk_usage.png       — system_disk_usage_percent
7. 7.monitoring_data_drift.png       — ml_data_drift_score
8. 8.monitoring_prediction_distribution.png — ml_prediction_churn/no_churn_total
9. 9.monitoring_model_confidence.png — ml_prediction_confidence
10. 10.monitoring_active_connections.png — ml_active_connections

Cara capture: Buka http://localhost:9090/graph, query masing-masing metric, lalu screenshot.
