# TSN v2 Deployment Guide

## Server Deployment (Central Processing)

### Prerequisites

- Ubuntu 22.04+ or Debian 12+
- MySQL 8.0+ or MariaDB 10.6+ (remote or local)
- NVIDIA GPU with CUDA 11.8+ (for transcription)
- 16GB+ RAM
- 500GB+ storage

### 1. Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    default-mysql-client \
    nginx \
    git \
    ffmpeg \
    libsndfile1 \
    cuda-toolkit-11-8

# Create TSN user
sudo useradd -r -m -d /opt/tsn -s /bin/bash tsn
```

### 2. Configure MySQL

Provision a MySQL (or MariaDB) schema that TSN can use. If you manage the
database locally, install `mysql-server` and create the schema; if you use a
hosted instance, run equivalent commands from any MySQL client:

```bash
mysql -u root -p
```

```sql
CREATE DATABASE tsn CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'tsn_user'@'%' IDENTIFIED BY 'your_secure_password';
GRANT ALL PRIVILEGES ON tsn.* TO 'tsn_user'@'%';
FLUSH PRIVILEGES;
```

Update firewalls/security groups so the TSN server can reach the database host
on port 3306.

### 3. Install TSN

```bash
# Switch to TSN user
sudo su - tsn

# Clone repository
git clone https://github.com/yourorg/TSN_V2.git /opt/tsn
cd /opt/tsn

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
pip install -e ".[gpu]"

# Configure
cp .env.example .env
nano .env
```

### 4. Configure Environment

Edit `/opt/tsn/.env`:

```bash
# Database
TSN_DB_HOST=localhost
TSN_DB_PASSWORD=your_secure_password

# Server
TSN_SERVER_ENABLED=true
TSN_NODE_ENABLED=false
TSN_SERVER_INCOMING_DIR=/opt/tsn/incoming
TSN_STORAGE_BASE_PATH=/opt/tsn/storage

# Transcription
TSN_WHISPER_BACKEND=faster-whisper
TSN_WHISPER_MODEL=medium.en
TSN_WHISPER_DEVICE=cuda
TSN_WHISPER_MAX_CONCURRENT=4

# vLLM
TSN_VLLM_BASE_URL=http://localhost:8001/v1
TSN_VLLM_API_KEY=sk-no-auth

# Monitoring
TSN_MONITORING_ENABLED=true
TSN_MONITORING_HOST=0.0.0.0
TSN_MONITORING_PORT=8080
```

### 5. Initialize Database

```bash
source venv/bin/activate
tsn-init-db
```

### 6. Install systemd Service

```bash
# Copy service file
sudo cp deployment/tsn-server.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start
sudo systemctl enable tsn-server
sudo systemctl start tsn-server

# Check status
sudo systemctl status tsn-server

# View logs
sudo journalctl -u tsn-server -f
```

### 7. Setup Nginx Reverse Proxy (Optional)

```bash
# Create nginx config
sudo nano /etc/nginx/sites-available/tsn
```

```nginx
server {
    listen 80;
    server_name tsn.example.com;

    location /health {
        proxy_pass http://localhost:8080/health;
        proxy_set_header Host $host;
    }

    location /metrics {
        proxy_pass http://localhost:8080/metrics;
        proxy_set_header Host $host;
        
        # Optional: Restrict access
        allow 192.168.0.0/16;
        deny all;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/tsn /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Node Deployment (Repeater Site)

### Prerequisites

- Ubuntu 22.04+ or Debian 12+
- 4GB+ RAM
- 50GB+ storage
- Network access to central server

### 1. Install System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Create TSN user
sudo useradd -r -m -d /opt/tsn-node -s /bin/bash tsn
```

### 2. Install TSN Node

```bash
sudo su - tsn
git clone https://github.com/yourorg/TSN_V2.git /opt/tsn-node
cd /opt/tsn-node

python3.11 -m venv venv
source venv/bin/activate
pip install -e .

cp .env.example .env
nano .env
```

### 3. Configure Environment

Edit `/opt/tsn-node/.env`:

```bash
# Database (connect to central server)
TSN_DB_HOST=server.example.com
TSN_DB_PASSWORD=your_secure_password

# Node
TSN_NODE_ENABLED=true
TSN_SERVER_ENABLED=false
TSN_NODE_NODE_ID=node001
TSN_NODE_AUDIO_INCOMING_DIR=/opt/tsn-node/incoming
TSN_NODE_AUDIO_ARCHIVE_DIR=/opt/tsn-node/archive

# SFTP
TSN_NODE_SFTP_HOST=server.example.com
TSN_NODE_SFTP_PORT=22
TSN_NODE_SFTP_USERNAME=tsn_upload
TSN_NODE_SFTP_PASSWORD=secure_password
TSN_NODE_SFTP_REMOTE_DIR=/opt/tsn/incoming
TSN_NODE_TRANSFER_WORKERS=2

# Logging
TSN_LOG_LEVEL=INFO
```

### 4. Install systemd Service

```bash
sudo cp deployment/tsn-node.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tsn-node
sudo systemctl start tsn-node
sudo systemctl status tsn-node
```

### 5. Setup Audio Capture

Integrate with AllStar/Asterisk to save audio files to `/opt/tsn-node/incoming/`

Example cron job:
```cron
*/5 * * * * /usr/local/bin/record_repeater.sh > /opt/tsn-node/incoming/$(date +\%Y\%m\%d_\%H\%M\%S).wav
```

## Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit (for GPU)

### 1. Clone and Configure

```bash
git clone https://github.com/yourorg/TSN_V2.git
cd TSN_V2

cp .env.example .env
nano .env
```

### 2. Start Services

```bash
# Server only
docker compose up -d

# Server + Monitoring
docker compose --profile monitoring up -d

# Server + Node (testing)
docker compose --profile node up -d

# View logs
docker compose logs -f tsn_server
```

### 3. Initialize Database

```bash
docker compose exec tsn_server tsn-init-db
```

### 4. Monitor

```bash
# Health check
curl http://localhost:8080/health

# Metrics
curl http://localhost:8080/metrics

# Grafana (if monitoring profile enabled)
open http://localhost:3000
# Default login: admin / admin
```

## Monitoring & Maintenance

### Health Checks

```bash
# CLI status
tsn status

# HTTP health check
curl http://localhost:8080/health | jq

# Prometheus metrics
curl http://localhost:8080/metrics/json | jq
```

### Log Management

```bash
# systemd logs
sudo journalctl -u tsn-server -f --since "1 hour ago"

# Docker logs
docker compose logs -f --tail=100 tsn_server
```

### Database Maintenance

```bash
# Backup
mysqldump -h localhost -u tsn_user -p tsn > tsn_backup_$(date +%Y%m%d).sql

# Optimize / analyze tables
mysqlcheck -h localhost -u tsn_user -p --analyze --optimize tsn

# Check size
mysql -h localhost -u tsn_user -p -e "SELECT table_schema AS db, ROUND(SUM(data_length + index_length)/1024/1024, 2) AS size_mb FROM information_schema.TABLES WHERE table_schema='tsn';"
```

### Performance Tuning

**MySQL/MariaDB** (`/etc/mysql/mysql.conf.d/mysqld.cnf`):
```ini
innodb_buffer_pool_size = 4G
innodb_log_file_size = 1G
innodb_flush_log_at_trx_commit = 2
max_connections = 200
table_open_cache = 4000
```

**TSN Worker Counts** (`.env`):
```bash
# Adjust based on hardware
TSN_WHISPER_MAX_CONCURRENT=4      # 1 per GPU
TSN_VLLM_MAX_CONCURRENT=10        # 5-10 per vLLM instance
TSN_NODE_TRANSFER_WORKERS=2       # 2-4 typical
```

## Troubleshooting

### Issue: Transcription Workers Stuck

```bash
# Check GPU
nvidia-smi

# Check model loading
docker compose exec tsn_server ls -lh ~/.cache/huggingface/hub/

# Restart workers
sudo systemctl restart tsn-server
```

### Issue: High Error Rate

```bash
# Check failed files
tsn status

# View specific failure
mysql -h localhost -u tsn_user -p -e "SELECT * FROM audio_files WHERE state LIKE 'FAILED%' LIMIT 10;" tsn

# Clean up
tsn clean-failed --no-dry-run
```

### Issue: Database Connection Errors

```bash
# Check MySQL/MariaDB
sudo systemctl status mysql || sudo systemctl status mariadb

# Check connections
mysql -h localhost -u tsn_user -p -e "SHOW STATUS LIKE 'Threads_connected';"

# Restart pooler
sudo systemctl restart tsn-server
```

## Security Considerations

1. **Database**: Use strong passwords, enable SSL connections
2. **SFTP**: Use SSH keys instead of passwords
3. **vLLM API**: Use authentication tokens
4. **Metrics**: Restrict access with firewall or nginx auth
5. **Logs**: Rotate regularly, no sensitive data in logs

## Backup Strategy

```bash
# Daily database backup
0 2 * * * /usr/local/bin/backup_tsn.sh

# Weekly audio archive
0 3 * * 0 tar czf /backups/tsn_audio_$(date +\%Y\%m\%d).tar.gz /opt/tsn/storage
```

## Scaling

### Horizontal Scaling

- Run multiple transcription workers on different GPUs
- Run extraction/analysis workers on separate CPU-only machines
- Use MySQL connection pooling (ProxySQL or HAProxy)

### Vertical Scaling

- Increase worker counts based on CPU/GPU availability
- Increase MySQL `innodb_buffer_pool_size`
- Add more storage for audio archives

---

For detailed architecture, see `docs/ARCHITECTURE.md`
