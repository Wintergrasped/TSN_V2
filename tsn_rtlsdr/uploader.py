"""SFTP uploader for transferring recordings to TSN server."""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

import paramiko

logger = logging.getLogger(__name__)


class SFTPUploader:
    """Uploads audio recordings to TSN server via SFTP."""

    def __init__(
        self,
        host: str,
        port: int = 22,
        username: str = "tsn",
        password: Optional[str] = None,
        key_file: Optional[str] = None,
        remote_dir: str = "/data/incoming",
        local_dir: str = "/recordings",
        delete_after_upload: bool = True,
        retry_delay: int = 30,
        max_retries: int = 5,
    ):
        """
        Initialize SFTP uploader.

        Args:
            host: Server hostname or IP address
            port: SSH port (default 22)
            username: SSH username
            password: SSH password (optional if using key)
            key_file: Path to SSH private key file (optional)
            remote_dir: Remote directory on server for uploads
            local_dir: Local directory to monitor for recordings
            delete_after_upload: Delete local file after successful upload
            retry_delay: Seconds to wait between retry attempts
            max_retries: Maximum number of upload retry attempts
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_file = key_file
        self.remote_dir = remote_dir
        self.local_dir = Path(local_dir)
        self.delete_after_upload = delete_after_upload
        self.retry_delay = retry_delay
        self.max_retries = max_retries

        self.ssh_client: Optional[paramiko.SSHClient] = None
        self.sftp_client: Optional[paramiko.SFTPClient] = None
        self.uploaded_files = set()  # Track uploaded files

    async def connect(self):
        """Establish SFTP connection to server."""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect with password or key
            connect_kwargs = {
                "hostname": self.host,
                "port": self.port,
                "username": self.username,
            }

            if self.key_file:
                connect_kwargs["key_filename"] = self.key_file
            elif self.password:
                connect_kwargs["password"] = self.password
            else:
                raise ValueError("Must provide either password or key_file")

            await asyncio.to_thread(self.ssh_client.connect, **connect_kwargs)
            self.sftp_client = self.ssh_client.open_sftp()

            # Ensure remote directory exists
            await self.ensure_remote_dir()

            logger.info(f"Connected to SFTP server: {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to connect to SFTP server: {e}")
            raise

    async def disconnect(self):
        """Close SFTP connection."""
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
        logger.info("Disconnected from SFTP server")

    async def ensure_remote_dir(self):
        """Create remote directory if it doesn't exist."""
        try:
            await asyncio.to_thread(self.sftp_client.stat, self.remote_dir)
        except FileNotFoundError:
            # Directory doesn't exist, create it
            try:
                await asyncio.to_thread(self.sftp_client.mkdir, self.remote_dir)
                logger.info(f"Created remote directory: {self.remote_dir}")
            except Exception as e:
                logger.error(f"Failed to create remote directory: {e}")

    async def upload_file(self, local_path: Path, retries: int = 0) -> bool:
        """
        Upload a single file to server.

        Args:
            local_path: Local file path
            retries: Current retry attempt count

        Returns:
            True if upload successful, False otherwise
        """
        if not local_path.exists():
            logger.warning(f"File not found: {local_path}")
            return False

        remote_path = f"{self.remote_dir}/{local_path.name}"

        try:
            # Ensure connection is active
            if not self.sftp_client:
                await self.connect()

            # Upload file
            logger.info(f"Uploading: {local_path.name} -> {remote_path}")
            await asyncio.to_thread(
                self.sftp_client.put, str(local_path), remote_path
            )

            # Verify upload by checking remote file size
            remote_stat = await asyncio.to_thread(self.sftp_client.stat, remote_path)
            local_size = local_path.stat().st_size

            if remote_stat.st_size == local_size:
                logger.info(
                    f"Upload successful: {local_path.name} ({local_size} bytes)"
                )
                self.uploaded_files.add(str(local_path))

                # Delete local file if configured
                if self.delete_after_upload:
                    local_path.unlink()
                    logger.info(f"Deleted local file: {local_path.name}")

                return True
            else:
                logger.error(
                    f"Upload size mismatch: local={local_size}, remote={remote_stat.st_size}"
                )
                return False

        except Exception as e:
            logger.error(f"Upload failed: {e}")

            # Retry logic
            if retries < self.max_retries:
                logger.info(
                    f"Retrying upload in {self.retry_delay}s (attempt {retries + 1}/{self.max_retries})"
                )
                await asyncio.sleep(self.retry_delay)

                # Reconnect and retry
                try:
                    await self.disconnect()
                    await self.connect()
                    return await self.upload_file(local_path, retries + 1)
                except Exception as reconnect_error:
                    logger.error(f"Reconnection failed: {reconnect_error}")

            logger.error(f"Upload failed after {self.max_retries} retries: {local_path.name}")
            return False

    async def scan_and_upload(self):
        """Scan local directory for recordings and upload them."""
        try:
            # Find all WAV files
            wav_files = list(self.local_dir.glob("*.wav")) + list(self.local_dir.glob("*.WAV"))

            # Filter out already uploaded files
            pending_files = [
                f for f in wav_files if str(f) not in self.uploaded_files
            ]

            if pending_files:
                logger.info(f"Found {len(pending_files)} file(s) to upload")

                for file_path in pending_files:
                    # Skip files that are still being written (modified in last 5 seconds)
                    file_age = time.time() - file_path.stat().st_mtime
                    if file_age < 5:
                        logger.debug(f"Skipping recent file: {file_path.name}")
                        continue

                    await self.upload_file(file_path)

        except Exception as e:
            logger.error(f"Error scanning directory: {e}")

    async def run(self, scan_interval: int = 10):
        """
        Main upload loop.

        Args:
            scan_interval: Seconds between directory scans
        """
        try:
            await self.connect()

            logger.info(f"Monitoring directory: {self.local_dir}")
            logger.info(f"Uploading to: {self.host}:{self.remote_dir}")

            while True:
                try:
                    await self.scan_and_upload()
                    await asyncio.sleep(scan_interval)
                except Exception as e:
                    logger.error(f"Error in upload loop: {e}")
                    await asyncio.sleep(scan_interval)

        except asyncio.CancelledError:
            logger.info("Upload task cancelled")
        finally:
            await self.disconnect()


async def main():
    """Entry point for SFTP uploader."""
    # Load configuration from environment variables
    host = os.getenv("SFTP_HOST", "localhost")
    port = int(os.getenv("SFTP_PORT", "22"))
    username = os.getenv("SFTP_USERNAME", "tsn")
    password = os.getenv("SFTP_PASSWORD")
    key_file = os.getenv("SFTP_KEY_FILE")
    remote_dir = os.getenv("SFTP_REMOTE_DIR", "/data/incoming")
    local_dir = os.getenv("RTLSDR_OUTPUT_DIR", "/recordings")
    delete_after = os.getenv("SFTP_DELETE_AFTER_UPLOAD", "true").lower() == "true"
    scan_interval = int(os.getenv("SFTP_SCAN_INTERVAL", "10"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uploader = SFTPUploader(
        host=host,
        port=port,
        username=username,
        password=password,
        key_file=key_file,
        remote_dir=remote_dir,
        local_dir=local_dir,
        delete_after_upload=delete_after,
    )

    await uploader.run(scan_interval=scan_interval)


if __name__ == "__main__":
    asyncio.run(main())
