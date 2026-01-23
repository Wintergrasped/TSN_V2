"""
Transfer agent - uploads files to server via SFTP with retry logic.
"""

import asyncio
from pathlib import Path
from typing import Optional

import paramiko
from paramiko import SSHClient, SFTPClient

from tsn_common.config import NodeSettings, get_settings
from tsn_common.logging import get_logger
from tsn_common.utils import compute_sha256

logger = get_logger(__name__)


class TransferAgent:
    """
    Handles file transfers to the server via SFTP.
    Implements retry logic with exponential backoff.
    """

    def __init__(self, settings: NodeSettings):
        self.settings = settings
        self.ssh_client: Optional[SSHClient] = None
        self.sftp_client: Optional[SFTPClient] = None
        self.connected = False
        
        logger.info(
            "transfer_agent_initialized",
            sftp_host=settings.sftp_host,
            sftp_port=settings.sftp_port,
            remote_dir=settings.sftp_remote_dir,
        )

    async def connect(self) -> bool:
        """
        Establish SFTP connection to server.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            # Run SSH connection in executor (it's blocking)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._connect_sync)
            
            self.connected = True
            logger.info("sftp_connection_established", host=self.settings.sftp_host)
            return True
            
        except Exception as e:
            logger.error(
                "sftp_connection_failed",
                host=self.settings.sftp_host,
                error=str(e),
            )
            self.connected = False
            return False

    def _connect_sync(self) -> None:
        """Synchronous SSH/SFTP connection (runs in executor)."""
        self.ssh_client = SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect with key or password
        if self.settings.sftp_key_path:
            self.ssh_client.connect(
                hostname=self.settings.sftp_host,
                port=self.settings.sftp_port,
                username=self.settings.sftp_username,
                key_filename=str(self.settings.sftp_key_path),
                timeout=30,
            )
        else:
            password = (
                self.settings.sftp_password.get_secret_value()
                if self.settings.sftp_password
                else None
            )
            self.ssh_client.connect(
                hostname=self.settings.sftp_host,
                port=self.settings.sftp_port,
                username=self.settings.sftp_username,
                password=password,
                timeout=30,
            )
        
        self.sftp_client = self.ssh_client.open_sftp()

    async def disconnect(self) -> None:
        """Close SFTP connection."""
        try:
            if self.sftp_client:
                self.sftp_client.close()
            if self.ssh_client:
                self.ssh_client.close()
            
            self.connected = False
            logger.info("sftp_connection_closed")
        except Exception as e:
            logger.error("sftp_disconnect_error", error=str(e))

    async def upload_file(
        self,
        local_path: Path,
        retry_count: int = 0,
        max_retries: int = 3,
    ) -> bool:
        """
        Upload a file to the server with retry logic.
        
        Args:
            local_path: Path to local file
            retry_count: Current retry attempt
            max_retries: Maximum retry attempts
            
        Returns:
            True if uploaded successfully, False otherwise
        """
        if not local_path.exists():
            logger.error("upload_file_not_found", path=str(local_path))
            return False
        
        # Ensure connection
        if not self.connected:
            if not await self.connect():
                logger.error("upload_failed_no_connection", filename=local_path.name)
                return False
        
        remote_path = f"{self.settings.sftp_remote_dir}/{local_path.name}"
        
        try:
            # Compute SHA256 before upload
            sha256 = compute_sha256(local_path)
            file_size = local_path.stat().st_size
            
            logger.info(
                "upload_starting",
                filename=local_path.name,
                size_bytes=file_size,
                sha256=sha256[:16] + "...",
            )
            
            # Upload in executor (blocking I/O)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._upload_sync,
                str(local_path),
                remote_path,
            )
            
            logger.info(
                "upload_completed",
                filename=local_path.name,
                remote_path=remote_path,
                sha256=sha256,
            )
            
            # Move to archive
            await self._archive_file(local_path)
            
            return True
            
        except Exception as e:
            logger.error(
                "upload_failed",
                filename=local_path.name,
                retry_count=retry_count,
                error=str(e),
            )
            
            # Retry with exponential backoff
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # 1s, 2s, 4s, 8s
                logger.info(
                    "upload_retrying",
                    filename=local_path.name,
                    wait_sec=wait_time,
                )
                await asyncio.sleep(wait_time)
                
                # Reconnect before retry
                await self.disconnect()
                
                return await self.upload_file(local_path, retry_count + 1, max_retries)
            else:
                logger.error(
                    "upload_failed_max_retries",
                    filename=local_path.name,
                    max_retries=max_retries,
                )
                return False

    def _upload_sync(self, local_path: str, remote_path: str) -> None:
        """Synchronous file upload (runs in executor)."""
        if self.sftp_client is None:
            raise RuntimeError("SFTP client not connected")
        
        self.sftp_client.put(local_path, remote_path)

    async def _archive_file(self, file_path: Path) -> None:
        """Move uploaded file to archive directory."""
        try:
            archive_dir = self.settings.audio_archive_dir
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            archive_path = archive_dir / file_path.name
            
            # If file already exists in archive, append timestamp
            if archive_path.exists():
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_path = archive_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            
            file_path.rename(archive_path)
            
            logger.info(
                "file_archived",
                filename=file_path.name,
                archive_path=str(archive_path),
            )
        except Exception as e:
            logger.error(
                "file_archive_failed",
                filename=file_path.name,
                error=str(e),
            )


async def transfer_worker(
    queue: asyncio.Queue[Path],
    settings: NodeSettings,
) -> None:
    """
    Transfer worker - consumes files from queue and uploads them.
    
    Args:
        queue: Queue of files to upload
        settings: Node settings
    """
    agent = TransferAgent(settings)
    
    try:
        while True:
            file_path = await queue.get()
            
            try:
                success = await agent.upload_file(file_path)
                
                if not success:
                    logger.error("transfer_failed", filename=file_path.name)
                    
            except Exception as e:
                logger.error(
                    "transfer_worker_error",
                    filename=file_path.name,
                    error=str(e),
                )
            finally:
                queue.task_done()
                
    finally:
        await agent.disconnect()


async def main() -> None:
    """Test transfer agent."""
    from tsn_common import setup_logging
    
    settings = get_settings()
    if settings.node is None:
        logger.error("node_settings_not_configured")
        return
    
    setup_logging(settings.logging)
    
    # Create test queue
    queue: asyncio.Queue[Path] = asyncio.Queue()
    
    # Start worker
    worker_task = asyncio.create_task(transfer_worker(queue, settings.node))
    
    # Wait for worker (would run forever in production)
    await worker_task


if __name__ == "__main__":
    asyncio.run(main())
