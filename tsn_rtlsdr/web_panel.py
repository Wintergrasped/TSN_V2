"""Web control panel for RTL-SDR node configuration and monitoring."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)


class RTLSDRWebPanel:
    """Web-based control panel for RTL-SDR node."""

    def __init__(self, recorder, port: int = 8585):
        """
        Initialize web control panel.

        Args:
            recorder: RTLSDRRecorder instance to monitor and control
            port: Web server port (default 8585)
        """
        self.recorder = recorder
        self.port = port
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        """Configure HTTP routes."""
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/status", self.handle_status)
        self.app.router.add_post("/update", self.handle_update)
        self.app.router.add_get("/recordings", self.handle_recordings)

    async def handle_index(self, request):
        """Serve main control panel HTML."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>RTL-SDR Control Panel</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 28px;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        .status-card {
            padding: 25px;
            border-bottom: 1px solid #e0e0e0;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        .status-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .status-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .status-value {
            font-size: 24px;
            font-weight: 600;
            color: #333;
        }
        .status-value.active {
            color: #22c55e;
        }
        .status-value.inactive {
            color: #94a3b8;
        }
        .control-card {
            padding: 25px;
        }
        .control-group {
            margin-bottom: 25px;
        }
        .control-group:last-child {
            margin-bottom: 0;
        }
        label {
            display: block;
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
            font-size: 14px;
        }
        input[type="number"], input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.2s;
        }
        input[type="number"]:focus, input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        .input-hint {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: transform 0.1s, box-shadow 0.2s;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .btn:active {
            transform: translateY(0);
        }
        .recordings-card {
            padding: 25px;
            background: #f8f9fa;
        }
        .recordings-list {
            max-height: 200px;
            overflow-y: auto;
            margin-top: 15px;
        }
        .recording-item {
            background: white;
            padding: 12px;
            margin-bottom: 8px;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 14px;
        }
        .recording-item:last-child {
            margin-bottom: 0;
        }
        .recording-name {
            font-weight: 500;
            color: #333;
        }
        .recording-time {
            color: #666;
            font-size: 12px;
        }
        .alert {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        .alert.success {
            background: #d1fae5;
            color: #065f46;
            border: 1px solid #10b981;
        }
        .alert.error {
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #ef4444;
        }
        .footer {
            padding: 20px;
            text-align: center;
            font-size: 12px;
            color: #666;
            background: #f8f9fa;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .recording-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #ef4444;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 1.5s ease-in-out infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎙️ RTL-SDR Control Panel</h1>
            <p>Real-time monitoring and configuration</p>
        </div>

        <div class="status-card">
            <h2 style="margin-bottom: 15px; color: #333;">Current Status</h2>
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-label">Recording</div>
                    <div class="status-value" id="recording-status">...</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Frequency</div>
                    <div class="status-value" id="frequency">...</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Squelch</div>
                    <div class="status-value" id="squelch">...</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Signal Power</div>
                    <div class="status-value" id="signal-power">...</div>
                </div>
            </div>
        </div>

        <div class="control-card">
            <h2 style="margin-bottom: 20px; color: #333;">Configuration</h2>
            <div id="alert" class="alert"></div>
            <form id="control-form">
                <div class="control-group">
                    <label for="freq">Frequency (MHz)</label>
                    <input type="number" id="freq" name="frequency" step="0.001" min="24" max="1766">
                    <div class="input-hint">Valid range: 24-1766 MHz (e.g., 146.720)</div>
                </div>
                <div class="control-group">
                    <label for="squelch">Squelch Threshold (dBFS)</label>
                    <input type="number" id="squelch" name="squelch_threshold" step="0.5" min="-100" max="0">
                    <div class="input-hint">More negative = less sensitive (e.g., -40.0)</div>
                </div>
                <div class="control-group">
                    <label for="gain">Gain (dB)</label>
                    <input type="number" id="gain" name="gain" step="0.1" min="0" max="50">
                    <div class="input-hint">Receiver gain: 0-49.6 dB</div>
                </div>
                <button type="submit" class="btn">Update Settings</button>
            </form>
        </div>

        <div class="recordings-card">
            <h2 style="margin-bottom: 15px; color: #333;">Recent Recordings</h2>
            <div class="recordings-list" id="recordings-list">
                <p style="text-align: center; color: #666;">Loading...</p>
            </div>
        </div>

        <div class="footer">
            TSN RTL-SDR Node | Refreshing every 2 seconds
        </div>
    </div>

    <script>
        let lastPower = null;
        let isRecording = false;

        async function updateStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                // Update recording status
                const recordingEl = document.getElementById('recording-status');
                isRecording = data.is_recording;
                if (data.is_recording) {
                    recordingEl.innerHTML = '<span class="recording-indicator"></span>ACTIVE';
                    recordingEl.className = 'status-value active';
                } else {
                    recordingEl.textContent = 'IDLE';
                    recordingEl.className = 'status-value inactive';
                }
                
                // Update frequency
                document.getElementById('frequency').textContent = (data.frequency / 1e6).toFixed(4) + ' MHz';
                
                // Update squelch
                document.getElementById('squelch').textContent = data.squelch_threshold.toFixed(1) + ' dBFS';
                
                // Update signal power
                const powerEl = document.getElementById('signal-power');
                if (data.current_power !== null) {
                    powerEl.textContent = data.current_power.toFixed(1) + ' dBFS';
                    if (data.current_power > data.squelch_threshold) {
                        powerEl.className = 'status-value active';
                    } else {
                        powerEl.className = 'status-value inactive';
                    }
                } else {
                    powerEl.textContent = '-- dBFS';
                    powerEl.className = 'status-value inactive';
                }
                
                // Update form values
                document.getElementById('freq').value = (data.frequency / 1e6).toFixed(6);
                document.getElementById('squelch').value = data.squelch_threshold;
                document.getElementById('gain').value = data.gain;
                
            } catch (error) {
                console.error('Failed to fetch status:', error);
            }
        }

        async function updateRecordings() {
            try {
                const response = await fetch('/recordings');
                const data = await response.json();
                
                const listEl = document.getElementById('recordings-list');
                if (data.recordings.length === 0) {
                    listEl.innerHTML = '<p style="text-align: center; color: #666;">No recordings yet</p>';
                } else {
                    listEl.innerHTML = data.recordings.map(rec => `
                        <div class="recording-item">
                            <span class="recording-name">${rec.filename}</span>
                            <span class="recording-time">${rec.duration}s</span>
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error('Failed to fetch recordings:', error);
            }
        }

        document.getElementById('control-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {
                frequency: parseFloat(formData.get('frequency')) * 1e6,
                squelch_threshold: parseFloat(formData.get('squelch_threshold')),
                gain: parseFloat(formData.get('gain'))
            };
            
            try {
                const response = await fetch('/update', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                const alertEl = document.getElementById('alert');
                
                if (result.success) {
                    alertEl.textContent = 'Settings updated successfully!';
                    alertEl.className = 'alert success';
                    alertEl.style.display = 'block';
                    setTimeout(() => { alertEl.style.display = 'none'; }, 3000);
                } else {
                    alertEl.textContent = 'Error: ' + result.error;
                    alertEl.className = 'alert error';
                    alertEl.style.display = 'block';
                }
            } catch (error) {
                const alertEl = document.getElementById('alert');
                alertEl.textContent = 'Failed to update settings: ' + error.message;
                alertEl.className = 'alert error';
                alertEl.style.display = 'block';
            }
        });

        // Update status every 2 seconds
        updateStatus();
        updateRecordings();
        setInterval(updateStatus, 2000);
        setInterval(updateRecordings, 5000);
    </script>
</body>
</html>
        """
        return web.Response(text=html, content_type="text/html")

    async def handle_status(self, request):
        """Return current RTL-SDR status as JSON."""
        status = {
            "frequency": self.recorder.frequency,
            "squelch_threshold": self.recorder.squelch_threshold,
            "gain": self.recorder.gain,
            "is_recording": self.recorder.is_recording,
            "current_power": self.recorder.last_power if hasattr(self.recorder, 'last_power') else None,
            "node_id": self.recorder.node_id,
        }
        return web.json_response(status)

    async def handle_update(self, request):
        """Update RTL-SDR configuration."""
        try:
            data = await request.json()
            
            # Validate inputs
            frequency = data.get("frequency")
            squelch = data.get("squelch_threshold")
            gain = data.get("gain")
            
            if frequency is not None:
                if not (24e6 <= frequency <= 1766e6):
                    return web.json_response(
                        {"success": False, "error": "Frequency must be between 24-1766 MHz"},
                        status=400
                    )
                self.recorder.frequency = frequency
                if self.recorder.sdr:
                    self.recorder.sdr.center_freq = frequency
                logger.info(f"Frequency updated to {frequency/1e6:.4f} MHz")
            
            if squelch is not None:
                if not (-100 <= squelch <= 0):
                    return web.json_response(
                        {"success": False, "error": "Squelch must be between -100 and 0 dBFS"},
                        status=400
                    )
                self.recorder.squelch_threshold = squelch
                logger.info(f"Squelch updated to {squelch} dBFS")
            
            if gain is not None:
                if not (0 <= gain <= 50):
                    return web.json_response(
                        {"success": False, "error": "Gain must be between 0 and 50 dB"},
                        status=400
                    )
                self.recorder.gain = gain
                if self.recorder.sdr:
                    self.recorder.sdr.gain = gain
                logger.info(f"Gain updated to {gain} dB")
            
            return web.json_response({"success": True})
            
        except Exception as e:
            logger.error(f"Failed to update settings: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500
            )

    async def handle_recordings(self, request):
        """Return list of recent recordings."""
        try:
            recordings = []
            output_dir = Path(self.recorder.output_dir)
            
            if output_dir.exists():
                wav_files = sorted(
                    output_dir.glob("*.wav"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )[:10]  # Last 10 recordings
                
                for wav_file in wav_files:
                    stat = wav_file.stat()
                    # Estimate duration from file size (rough approximation)
                    duration = stat.st_size / (self.recorder.audio_sample_rate * 2)
                    recordings.append({
                        "filename": wav_file.name,
                        "duration": round(duration, 1),
                        "size": stat.st_size,
                        "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            
            return web.json_response({"recordings": recordings})
            
        except Exception as e:
            logger.error(f"Failed to list recordings: {e}")
            return web.json_response({"recordings": []})

    async def start(self):
        """Start web server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        logger.info(f"Web control panel started on http://0.0.0.0:{self.port}")

    async def run(self):
        """Run web server indefinitely."""
        await self.start()
        # Keep running forever
        while True:
            await asyncio.sleep(3600)
