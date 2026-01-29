"""RTL-SDR frequency monitoring and audio recording."""

import asyncio
import logging
import os
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from rtlsdr import RtlSdr

logger = logging.getLogger(__name__)


class RTLSDRRecorder:
    """Records audio from RTL-SDR when squelch threshold is exceeded."""

    def __init__(
        self,
        frequency: float = 146.72e6,  # 146.72 MHz default
        sample_rate: int = 240000,  # 240 kHz
        gain: float = 20.0,  # dB
        squelch_threshold: float = -40.0,  # dBFS
        squelch_delay: float = 2.0,  # seconds of silence before stopping
        output_dir: str = "/recordings",
        node_id: str = "rtlsdr",
        audio_sample_rate: int = 16000,  # Output audio sample rate
    ):
        """
        Initialize RTL-SDR recorder.

        Args:
            frequency: Center frequency in Hz (default 146.72 MHz)
            sample_rate: SDR sample rate in Hz
            gain: Receiver gain in dB
            squelch_threshold: Signal threshold in dBFS to trigger recording
            squelch_delay: Seconds of silence before stopping recording
            output_dir: Directory to save recordings
            node_id: Unique identifier for this RTL-SDR node
            audio_sample_rate: Output WAV file sample rate
        """
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.gain = gain
        self.squelch_threshold = squelch_threshold
        self.squelch_delay = squelch_delay
        self.output_dir = Path(output_dir)
        self.node_id = node_id
        self.audio_sample_rate = audio_sample_rate

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sdr: Optional[RtlSdr] = None
        self.is_recording = False
        self.current_recording: Optional[list] = None
        self.silence_start: Optional[float] = None
        self.recording_start: Optional[datetime] = None
        self.last_power: Optional[float] = None  # Track last signal power for web UI

    async def initialize_sdr(self):
        """Initialize RTL-SDR device."""
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.frequency
            self.sdr.gain = self.gain
            logger.info(
                f"RTL-SDR initialized: freq={self.frequency/1e6:.4f} MHz, "
                f"sample_rate={self.sample_rate/1e3:.0f} kHz, gain={self.gain} dB"
            )
        except Exception as e:
            logger.error(f"Failed to initialize RTL-SDR: {e}")
            raise

    async def cleanup_sdr(self):
        """Clean up RTL-SDR resources."""
        if self.sdr:
            self.sdr.close()
            logger.info("RTL-SDR closed")

    def calculate_power(self, samples: np.ndarray) -> float:
        """
        Calculate signal power in dBFS.

        Args:
            samples: Complex IQ samples

        Returns:
            Power level in dBFS
        """
        # Calculate power from IQ samples
        power = np.mean(np.abs(samples) ** 2)
        if power > 0:
            return 10 * np.log10(power)
        return -100.0  # Very low power

    def demodulate_fm(self, samples: np.ndarray) -> np.ndarray:
        """
        Demodulate FM signal to audio.

        Args:
            samples: Complex IQ samples

        Returns:
            Audio samples (real values)
        """
        # Simple FM demodulation using phase difference
        phase = np.angle(samples)
        phase_diff = np.diff(phase)
        
        # Unwrap phase to handle discontinuities
        phase_diff = np.unwrap(phase_diff)
        
        # Normalize to audio range
        audio = phase_diff / np.pi
        
        return audio

    def resample_audio(self, audio: np.ndarray, input_rate: int, output_rate: int) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio: Input audio samples
            input_rate: Input sample rate
            output_rate: Output sample rate

        Returns:
            Resampled audio
        """
        # Simple linear interpolation resampling
        duration = len(audio) / input_rate
        num_samples = int(duration * output_rate)
        
        input_indices = np.arange(len(audio))
        output_indices = np.linspace(0, len(audio) - 1, num_samples)
        
        resampled = np.interp(output_indices, input_indices, audio)
        return resampled

    def save_recording(self) -> Optional[Path]:
        """
        Save current recording to WAV file.

        Returns:
            Path to saved file, or None if no recording
        """
        if not self.current_recording or not self.recording_start:
            return None

        try:
            # Concatenate all audio chunks
            audio_data = np.concatenate(self.current_recording)
            
            # Normalize to 16-bit PCM range
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)

            # Generate filename with timestamp
            timestamp = self.recording_start.strftime("%Y%m%d%H%M%S%f")[:16]
            filename = f"{self.node_id}_{timestamp}.wav"
            filepath = self.output_dir / filename

            # Write WAV file
            with wave.open(str(filepath), "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.audio_sample_rate)
                wf.writeframes(audio_data.tobytes())

            duration = len(audio_data) / self.audio_sample_rate
            logger.info(f"Saved recording: {filename} ({duration:.1f}s)")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            return None

    async def process_samples(self, samples: np.ndarray, context=None):
        """
        Process IQ samples and handle recording logic.

        Args:
            samples: Complex IQ samples from RTL-SDR
            context: Unused, for compatibility with rtlsdr callback
        """
        current_time = asyncio.get_event_loop().time()
        
        # Calculate signal power
        power = self.calculate_power(samples)
        self.last_power = power  # Update for web UI
        
        # Check if signal exceeds squelch threshold
        signal_present = power > self.squelch_threshold

        if signal_present:
            if not self.is_recording:
                # Start new recording
                self.is_recording = True
                self.current_recording = []
                self.recording_start = datetime.now()
                self.silence_start = None
                logger.info(f"Recording started (power={power:.1f} dBFS)")

            # Demodulate FM to audio
            audio = self.demodulate_fm(samples)
            
            # Resample to output audio rate
            audio_resampled = self.resample_audio(
                audio, self.sample_rate, self.audio_sample_rate
            )
            
            self.current_recording.append(audio_resampled)
            self.silence_start = None  # Reset silence timer

        else:
            if self.is_recording:
                if self.silence_start is None:
                    self.silence_start = current_time
                elif current_time - self.silence_start >= self.squelch_delay:
                    # Silence duration exceeded, stop recording
                    logger.info(f"Recording stopped (power={power:.1f} dBFS)")
                    filepath = self.save_recording()
                    
                    # Reset recording state
                    self.is_recording = False
                    self.current_recording = None
                    self.recording_start = None
                    self.silence_start = None
                    
                    return filepath  # Return path for uploader

        return None

    async def run(self):
        """Main monitoring loop."""
        try:
            await self.initialize_sdr()
            
            logger.info(
                f"Monitoring {self.frequency/1e6:.4f} MHz "
                f"(squelch={self.squelch_threshold} dBFS)"
            )

            # Read samples continuously
            while True:
                try:
                    samples = await asyncio.to_thread(self.sdr.read_samples, 256 * 1024)
                    await self.process_samples(samples)
                    await asyncio.sleep(0.01)  # Small delay between reads
                except Exception as e:
                    logger.error(f"Error reading samples: {e}")
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Recording task cancelled")
        finally:
            # Save any in-progress recording
            if self.is_recording:
                self.save_recording()
            await self.cleanup_sdr()


async def main():
    """Entry point for RTL-SDR recorder."""
    # Load configuration from environment variables
    frequency = float(os.getenv("RTLSDR_FREQUENCY", "146.72e6"))
    sample_rate = int(os.getenv("RTLSDR_SAMPLE_RATE", "240000"))
    gain = float(os.getenv("RTLSDR_GAIN", "20.0"))
    squelch = float(os.getenv("RTLSDR_SQUELCH", "-40.0"))
    squelch_delay = float(os.getenv("RTLSDR_SQUELCH_DELAY", "2.0"))
    output_dir = os.getenv("RTLSDR_OUTPUT_DIR", "/recordings")
    node_id = os.getenv("NODE_ID", "rtlsdr")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    recorder = RTLSDRRecorder(
        frequency=frequency,
        sample_rate=sample_rate,
        gain=gain,
        squelch_threshold=squelch,
        squelch_delay=squelch_delay,
        output_dir=output_dir,
        node_id=node_id,
    )

    await recorder.run()


if __name__ == "__main__":
    asyncio.run(main())
