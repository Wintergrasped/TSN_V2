# TSN Node Deployment Guide - From Scratch

Complete step-by-step guide to deploy a TSN Node on a Raspberry Pi (or any Linux system).

## Prerequisites

- Raspberry Pi 4 (4GB+ RAM recommended) or similar Linux system
- Fresh Raspberry Pi OS (64-bit recommended) or Ubuntu
- Network connectivity
- SSH access enabled

## Step 1: Initial System Setup

```bash
# SSH into your Raspberry Pi
ssh pi@192.168.0.129

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    openssh-client

# Create TSN user (optional but recommended)
sudo useradd -r -m -d /opt/tsn-node -s /bin/bash tsn
sudo usermod -aG sudo tsn  # If you need sudo access
```

## Step 2: Clone and Install TSN

```bash
# Switch to TSN user (or stay as pi user)
sudo su - tsn
# OR: cd /opt

# Clone repository
git clone https://github.com/Wintergrasped/TSN_V2.git /opt/tsn-node
cd /opt/tsn-node

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install TSN (node only, no GPU dependencies)
pip install --upgrade pip
pip install -e .
```

## Step 3: Configure Node Settings

```bash
# Copy example config
cp .env.example .env

# Edit configuration
nano .env
```

**Node Configuration** (`.env`):
```bash
# ============================================
# NODE CONFIGURATION
# ============================================

# Enable node services ONLY
TSN_NODE_ENABLED=true
TSN_SERVER_ENABLED=false

# Database connection (your remote database)
TSN_DB_HOST=51.81.202.9
TSN_DB_PORT=5432
TSN_DB_NAME=tsn
TSN_DB_USER=tsn_user
TSN_DB_PASSWORD=your_secure_database_password

# Node identification
TSN_NODE_NODE_ID=rpi_node_129
# Or use: TSN_NODE_NODE_ID=w1abc_repeater

# Audio directories
TSN_NODE_AUDIO_INCOMING_DIR=/opt/tsn-node/incoming
TSN_NODE_AUDIO_ARCHIVE_DIR=/opt/tsn-node/archive

# File watcher settings
TSN_NODE_WATCH_INTERVAL_SEC=1.0
TSN_NODE_MIN_FILE_AGE_SEC=2.0
TSN_NODE_MIN_FILE_SIZE=1000

# SFTP transfer to TSN Server
TSN_NODE_SFTP_HOST=192.168.0.104
TSN_NODE_SFTP_PORT=22
TSN_NODE_SFTP_USERNAME=tsn_upload
TSN_NODE_SFTP_PASSWORD=your_sftp_password
# OR use SSH key:
# TSN_NODE_SFTP_KEY_PATH=/opt/tsn-node/.ssh/id_rsa
TSN_NODE_SFTP_REMOTE_DIR=/opt/tsn/incoming

# Transfer workers
TSN_NODE_TRANSFER_WORKERS=2

# Logging
TSN_LOG_LEVEL=INFO
TSN_LOG_FORMAT=json
TSN_LOG_OUTPUT=stdout
```

Save and exit (Ctrl+X, Y, Enter)

## Step 4: Create Directories

```bash
# Create required directories
mkdir -p /opt/tsn-node/incoming
mkdir -p /opt/tsn-node/archive
mkdir -p /opt/tsn-node/logs

# Set permissions (if using tsn user)
sudo chown -R tsn:tsn /opt/tsn-node
```

## Step 5: Setup SSH Key for SFTP (Recommended)

```bash
# Generate SSH key (if not using password)
ssh-keygen -t rsa -b 4096 -f /opt/tsn-node/.ssh/id_rsa -N ""

# Copy public key to TSN Server
ssh-copy-id -i /opt/tsn-node/.ssh/id_rsa.pub tsn_upload@192.168.0.104

# Update .env to use key instead of password
# TSN_NODE_SFTP_KEY_PATH=/opt/tsn-node/.ssh/id_rsa
# TSN_NODE_SFTP_PASSWORD=  # Remove or comment out
```

## Step 6: Test Connection

```bash
# Test database connection
python -c "
from tsn_common.config import get_settings
from tsn_common.db import get_engine
import asyncio

async def test():
    settings = get_settings()
    engine = get_engine(settings.database)
    async with engine.begin() as conn:
        print('Database connection: OK')
    await engine.dispose()

asyncio.run(test())
"

# Test SFTP connection
sftp tsn_upload@192.168.0.104
# Type 'exit' to quit if successful
```

## Step 7: Create systemd Service

```bash
# Create service file
sudo nano /etc/systemd/system/tsn-node.service
```

**Service File**:
```ini
[Unit]
Description=TSN Node - The Spoken Network Node Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=tsn
Group=tsn
WorkingDirectory=/opt/tsn-node
Environment="PATH=/opt/tsn-node/venv/bin"
EnvironmentFile=/opt/tsn-node/.env

ExecStart=/opt/tsn-node/venv/bin/python tsn_orchestrator.py

# Restart on failure
Restart=always
RestartSec=10s

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tsn-node

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/tsn-node/incoming /opt/tsn-node/archive

[Install]
WantedBy=multi-user.target
```

Save and exit.

## Step 8: Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable tsn-node

# Start service
sudo systemctl start tsn-node

# Check status
sudo systemctl status tsn-node

# View logs
sudo journalctl -u tsn-node -f
```

## Step 9: Setup Audio Capture Integration

### For AllStar/Asterisk:

```bash
# Edit your recording script or add to dialplan
sudo nano /usr/local/bin/record_repeater.sh
```

```bash
#!/bin/bash
# Record repeater audio and save to TSN incoming directory

OUTPUT_DIR="/opt/tsn-node/incoming"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/${TIMESTAMP}.wav"

# Record audio (adjust for your setup)
# Example using sox:
# rec -r 8000 -c 1 -b 16 "$OUTPUT_FILE" trim 0 300

# Example using Asterisk Monitor:
# Monitor files automatically go to /var/spool/asterisk/monitor/
# Set up a cron job to move them:
# */5 * * * * find /var/spool/asterisk/monitor -name "*.wav" -mmin +1 -exec mv {} /opt/tsn-node/incoming/ \;

echo "Audio saved to $OUTPUT_FILE"
```

### Or use a cron job:

```bash
# Edit crontab
crontab -e

# Add line to check for new recordings every 5 minutes
*/5 * * * * find /path/to/asterisk/recordings -name "*.wav" -mmin +1 -exec mv {} /opt/tsn-node/incoming/ \;
```

## Step 10: Verify Operation

```bash
# Check service is running
sudo systemctl status tsn-node

# View live logs
sudo journalctl -u tsn-node -f

# Test with a sample file
cp /path/to/test.wav /opt/tsn-node/incoming/test_$(date +%s).wav

# Watch it get processed
tail -f /var/log/syslog | grep tsn-node
```

## Troubleshooting

### Service won't start

```bash
# Check logs
sudo journalctl -u tsn-node -n 50

# Run manually to see errors
cd /opt/tsn-node
source venv/bin/activate
python tsn_orchestrator.py
```

### SFTP connection fails

```bash
# Test SFTP manually
sftp -v tsn_upload@192.168.0.104

# Check firewall on server
# Server should allow port 22 from node IP
```

### Database connection fails

```bash
# Test with psql
psql -h 51.81.202.9 -U tsn_user -d tsn

# Check firewall allows MySQL port 3306
# Check database accepts remote connections
```

### Files not being detected

```bash
# Check permissions
ls -la /opt/tsn-node/incoming/

# Check service user
sudo systemctl status tsn-node | grep "User:"

# Check logs for watcher activity
sudo journalctl -u tsn-node | grep file_watcher
```

## Monitoring

### Check Node Status

```bash
# Service status
sudo systemctl status tsn-node

# Recent logs (last 100 lines)
sudo journalctl -u tsn-node -n 100

# Follow logs in real-time
sudo journalctl -u tsn-node -f

# Check queue status (on server via CLI)
# ssh to server and run:
tsn status
```

### Performance Monitoring

```bash
# CPU/Memory usage
top -p $(pgrep -f tsn_orchestrator)

# Disk usage
df -h /opt/tsn-node/archive

# Network usage
iftop  # Install with: sudo apt install iftop
```

## Maintenance

### Update TSN Software

```bash
cd /opt/tsn-node
git pull
source venv/bin/activate
pip install -e .
sudo systemctl restart tsn-node
```

### Clean Archive

```bash
# Delete files older than 30 days
find /opt/tsn-node/archive -name "*.wav" -mtime +30 -delete

# Or add to crontab:
0 2 * * * find /opt/tsn-node/archive -name "*.wav" -mtime +30 -delete
```

### Backup Configuration

```bash
# Backup .env file
sudo cp /opt/tsn-node/.env /opt/tsn-node/.env.backup.$(date +%Y%m%d)
```

## Quick Command Reference

```bash
# Start/Stop/Restart
sudo systemctl start tsn-node
sudo systemctl stop tsn-node
sudo systemctl restart tsn-node

# Enable/Disable
sudo systemctl enable tsn-node
sudo systemctl disable tsn-node

# View logs
sudo journalctl -u tsn-node -f          # Follow
sudo journalctl -u tsn-node -n 100      # Last 100 lines
sudo journalctl -u tsn-node --since "1 hour ago"

# Check status
sudo systemctl status tsn-node
ps aux | grep tsn_orchestrator
```

## Summary

Your TSN Node is now:
- ✅ Monitoring `/opt/tsn-node/incoming/` for WAV files
- ✅ Uploading files to TSN Server at `192.168.0.104` via SFTP
- ✅ Archiving files locally after successful upload
- ✅ Automatically starting on boot
- ✅ Logging to systemd journal

Files flow: `Audio Capture → Incoming Dir → SFTP Upload → Archive → Server Processing`
