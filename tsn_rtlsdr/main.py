"""Main orchestrator for RTL-SDR node - runs recorder and uploader concurrently."""

import asyncio
import logging
import signal
import sys

from tsn_rtlsdr.recorder import RTLSDRRecorder
from tsn_rtlsdr.uploader import SFTPUploader


logger = logging.getLogger(__name__)


class RTLSDRNode:
    """Orchestrates RTL-SDR recording and SFTP uploading."""

    def __init__(self, recorder: RTLSDRRecorder, uploader: SFTPUploader):
        """
        Initialize node orchestrator.

        Args:
            recorder: RTL-SDR recorder instance
            uploader: SFTP uploader instance
        """
        self.recorder = recorder
        self.uploader = uploader
        self.tasks = []

    async def run(self):
        """Run recorder and uploader concurrently."""
        logger.info("Starting RTL-SDR node...")

        # Create concurrent tasks
        recorder_task = asyncio.create_task(self.recorder.run())
        uploader_task = asyncio.create_task(self.uploader.run())

        self.tasks = [recorder_task, uploader_task]

        try:
            # Wait for both tasks
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("Node shutdown initiated")
        except Exception as e:
            logger.error(f"Node error: {e}")
        finally:
            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            logger.info("RTL-SDR node stopped")

    def shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        for task in self.tasks:
            task.cancel()


async def main():
    """Entry point for RTL-SDR node."""
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Recorder configuration
    recorder = RTLSDRRecorder(
        frequency=float(os.getenv("RTLSDR_FREQUENCY", "146.72e6")),
        sample_rate=int(os.getenv("RTLSDR_SAMPLE_RATE", "240000")),
        gain=float(os.getenv("RTLSDR_GAIN", "20.0")),
        squelch_threshold=float(os.getenv("RTLSDR_SQUELCH", "-40.0")),
        squelch_delay=float(os.getenv("RTLSDR_SQUELCH_DELAY", "2.0")),
        output_dir=os.getenv("RTLSDR_OUTPUT_DIR", "/recordings"),
        node_id=os.getenv("NODE_ID", "rtlsdr"),
    )

    # Uploader configuration
    uploader = SFTPUploader(
        host=os.getenv("SFTP_HOST", "localhost"),
        port=int(os.getenv("SFTP_PORT", "22")),
        username=os.getenv("SFTP_USERNAME", "tsn"),
        password=os.getenv("SFTP_PASSWORD"),
        key_file=os.getenv("SFTP_KEY_FILE"),
        remote_dir=os.getenv("SFTP_REMOTE_DIR", "/data/incoming"),
        local_dir=os.getenv("RTLSDR_OUTPUT_DIR", "/recordings"),
        delete_after_upload=os.getenv("SFTP_DELETE_AFTER_UPLOAD", "true").lower() == "true",
    )

    # Create and run node
    node = RTLSDRNode(recorder, uploader)

    # Register signal handlers
    signal.signal(signal.SIGINT, node.shutdown)
    signal.signal(signal.SIGTERM, node.shutdown)

    await node.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
