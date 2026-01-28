# Security Hardening

- Use [OSQuery](https://github.com/DemodLLC/osquery) for runtime checks
- Implement rate limiting with Redis
- Add API gateway (Fastly/Cloudflare)

---

# Performance Monitoring

- Use Prometheus + Grafana (see monitoring/ directory)
- Track GPU utilization with DCGM exporter
- Set up alert thresholds for latency >500ms

---

# Model Management

- Store models in S3 with versioning
- Use model hash verification before loading
- Implement blue-green deployments for updates