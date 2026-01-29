# TSN RTL-SDR Node

A Docker-based RTL-SDR monitoring node for the TSN (Transcription Service Network) platform. This module uses a Software Defined Radio (RTL-SDR) USB dongle to monitor amateur radio frequencies, detect transmissions via squelch breaking, record audio, and automatically upload recordings to the TSN server for transcription and analysis.

## Features

- **RTL-SDR Integration**: Direct frequency monitoring using RTL-SDR USB dongles
- **Squelch Detection**: Automatic recording triggered when signal exceeds threshold
- **FM Demodulation**: Built-in FM demodulation for voice signals
- **Automatic Upload**: SFTP-based file transfer to TSN server
- **Configurable Parameters**: Frequency, gain, squelch, sample rates via environment variables
- **Docker Deployment**: Containerized with all dependencies included

## Hardware Requirements

- RTL-SDR USB dongle (RTL2832U-based)
  - Examples: RTL-SDR Blog V3, NooElec NESDR SMArt
- USB 2.0 or 3.0 port
- Linux host with USB device access

## Quick Start

### 1. Configure Environment Variables

Create or update your `.env` file:

```bash
# RTL-SDR Node Configuration
RTLSDR_NODE_ID=rtlsdr_main
RTLSDR_FREQUENCY=146.72e6    # 146.72 MHz (2m repeater)
RTLSDR_SAMPLE_RATE=240000    # 240 kHz
RTLSDR_GAIN=20.0             # 20 dB gain
RTLSDR_SQUELCH=-40.0         # -40 dBFS squelch threshold
RTLSDR_SQUELCH_DELAY=2.0     # 2 seconds of silence before stopping

# SFTP Server Connection
SFTP_HOST=192.168.1.100      # TSN server IP
SFTP_PORT=22
SFTP_USERNAME=tsn
SFTP_PASSWORD=your_password
SFTP_REMOTE_DIR=/incoming
SFTP_DELETE_AFTER_UPLOAD=true
SFTP_SCAN_INTERVAL=10        # Scan for new files every 10 seconds
```

### 2. Start the RTL-SDR Node

```bash
# Build and start the RTL-SDR service
docker-compose --profile rtlsdr up -d tsn_rtlsdr

# View logs
docker-compose logs -f tsn_rtlsdr
```

### 3. Verify Operation

Check logs for successful initialization:
```
RTL-SDR initialized: freq=146.7200 MHz, sample_rate=240 kHz, gain=20.0 dB
Monitoring 146.7200 MHz (squelch=-40.0 dBFS)
Connected to SFTP server: 192.168.1.100:22
Monitoring directory: /recordings
```

When a transmission is detected:
```
Recording started (power=-32.5 dBFS)
Recording stopped (power=-45.2 dBFS)
Saved recording: rtlsdr_main_20260128143055123456.wav (12.3s)
Uploading: rtlsdr_main_20260128143055123456.wav -> /incoming/rtlsdr_main_20260128143055123456.wav
Upload successful: rtlsdr_main_20260128143055123456.wav (394240 bytes)
```

## Configuration

### Frequency Settings

Common amateur radio frequencies:

```bash
# 2 meter band (VHF)
RTLSDR_FREQUENCY=146.52e6    # National simplex calling frequency
RTLSDR_FREQUENCY=146.72e6    # Example repeater output
RTLSDR_FREQUENCY=145.50e6    # Repeater input

# 70 cm band (UHF)
RTLSDR_FREQUENCY=446.00e6    # UHF repeater
RTLSDR_FREQUENCY=433.50e6    # UHF simplex
```

### Gain Settings

RTL-SDR gain can be set manually or to auto:

```bash
RTLSDR_GAIN=auto    # Automatic gain control
RTLSDR_GAIN=0.0     # Minimum gain
RTLSDR_GAIN=20.0    # Medium gain (recommended)
RTLSDR_GAIN=49.6    # Maximum gain
```

**Recommendation**: Start with `20.0` and adjust based on signal strength and noise.

### Squelch Threshold

Squelch threshold determines when recording starts:

```bash
RTLSDR_SQUELCH=-50.0    # Sensitive (weak signals)
RTLSDR_SQUELCH=-40.0    # Balanced (recommended)
RTLSDR_SQUELCH=-30.0    # Less sensitive (strong signals only)
```

**Tuning Tips**:
- Monitor logs to see typical signal power levels
- Set threshold ~5-10 dB below average signal strength
- Too sensitive = noise recordings
- Not sensitive enough = missed transmissions

### Sample Rates

```bash
# Higher sample rates = better quality, more CPU
RTLSDR_SAMPLE_RATE=240000    # 240 kHz (recommended for FM voice)
RTLSDR_SAMPLE_RATE=960000    # 960 kHz (wideband FM)
RTLSDR_SAMPLE_RATE=1200000   # 1.2 MHz (maximum)
```

## Deployment

### Development/Testing

```bash
# Start with console output
docker-compose --profile rtlsdr up tsn_rtlsdr
```

### Production

```bash
# Background service
docker-compose --profile rtlsdr up -d tsn_rtlsdr

# Enable automatic restart
docker update --restart=unless-stopped tsn_rtlsdr
```

### SSH Key Authentication (Recommended)

For production deployments, use SSH keys instead of passwords:

1. Generate SSH key pair:
```bash
mkdir -p ssh_keys
ssh-keygen -t ed25519 -f ssh_keys/rtlsdr_key -N ""
```

2. Copy public key to TSN server:
```bash
ssh-copy-id -i ssh_keys/rtlsdr_key.pub tsn@192.168.1.100
```

3. Update environment:
```bash
SFTP_KEY_FILE=/root/.ssh/rtlsdr_key
SFTP_PASSWORD=  # Leave empty when using key
```

4. Restart service:
```bash
docker-compose --profile rtlsdr restart tsn_rtlsdr
```

## Troubleshooting

### RTL-SDR Not Detected

```bash
# Check if device is visible to host
lsusb | grep RTL

# Expected output:
# Bus 001 Device 004: ID 0bda:2838 Realtek Semiconductor Corp. RTL2838 DVB-T

# Check container USB access
docker exec tsn_rtlsdr lsusb
```

### Permission Errors

Ensure container has USB access:
```yaml
# In docker-compose.yml
privileged: true
devices:
  - /dev/bus/usb:/dev/bus/usb
```

### Poor Signal Quality

1. Check antenna connection
2. Adjust gain: `RTLSDR_GAIN=auto` or try different values
3. Verify frequency is correct
4. Check for interference (move away from computers, power supplies)

### SFTP Upload Failures

```bash
# Test SFTP connection from container
docker exec -it tsn_rtlsdr bash
sftp tsn@192.168.1.100

# Check server logs
docker-compose logs tsn_server | grep incoming
```

### High CPU Usage

- Reduce sample rate: `RTLSDR_SAMPLE_RATE=240000`
- Increase squelch threshold (record less)
- Check for continuous noise triggering recordings

## File Format

Recordings are saved with this naming pattern:
```
{NODE_ID}_{TIMESTAMP}.wav
```

Example: `rtlsdr_main_20260128143055123456.wav`

- **Format**: 16-bit PCM WAV
- **Sample Rate**: 16 kHz (output)
- **Channels**: 1 (mono)
- **Duration**: Variable (stops after silence)

## Architecture

```
RTL-SDR Dongle
      ↓
  Recorder Module
  - Frequency monitoring
  - FM demodulation
  - Squelch detection
  - WAV file creation
      ↓
  Local Storage (/recordings)
      ↓
  Uploader Module
  - SFTP transfer
  - Retry logic
  - File cleanup
      ↓
  TSN Server (/incoming)
      ↓
  Ingestion → Transcription → Analysis
```

## Integration with TSN

The RTL-SDR node integrates seamlessly with TSN:

1. **Recordings** are uploaded to `/incoming` on the TSN server
2. **Ingestion service** detects new files automatically
3. **Filename parsing** extracts `node_id` from filename (e.g., `rtlsdr_main`)
4. **Transcription** processes audio using Whisper
5. **Analysis** extracts callsigns, topics, and net structure
6. **Web portal** displays recordings with node badge

## Multiple RTL-SDR Nodes

Deploy multiple nodes for different frequencies:

```yaml
# docker-compose.yml
tsn_rtlsdr_2m:
  extends: tsn_rtlsdr
  container_name: tsn_rtlsdr_2m
  environment:
    NODE_ID: rtlsdr_2m
    RTLSDR_FREQUENCY: 146.52e6

tsn_rtlsdr_70cm:
  extends: tsn_rtlsdr
  container_name: tsn_rtlsdr_70cm
  environment:
    NODE_ID: rtlsdr_70cm
    RTLSDR_FREQUENCY: 446.00e6
  devices:
    - /dev/bus/usb/001/005:/dev/bus/usb/001/005  # Specific device
```

## Performance

Typical resource usage:
- **CPU**: 5-15% (single core)
- **Memory**: 100-200 MB
- **Disk**: ~1 MB per minute of recording
- **Network**: ~100 KB per minute of recording (upload)

## License

Part of the TSN V2 project. See LICENSE file for details.

## Support

For issues or questions:
- Check container logs: `docker-compose logs tsn_rtlsdr`
- Verify RTL-SDR hardware: `rtl_test`
- Review configuration in `.env` file
- Consult TSN documentation: `/docs`

## Related Documentation

- [TSN Deployment Guide](../deployment/DEPLOYMENT.md)
- [Node Deployment](../deployment/NODE_DEPLOYMENT.md)
- [Architecture Overview](../docs/ARCHITECTURE.md)
