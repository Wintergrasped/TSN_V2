"""
Health check and metrics endpoint for monitoring.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel
from starlette.responses import Response

from tsn_common.config import get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger
from tsn_common.models import AiRunLog, AudioFile, AudioFileState, GpuUtilizationSample, SystemHealth

logger = get_logger(__name__)

app = FastAPI(title="TSN Health & Metrics")

# Prometheus metrics
files_processed = Counter(
    "tsn_files_processed_total",
    "Total files processed",
    ["state"],
)

files_in_state = Gauge(
    "tsn_files_in_state",
    "Number of files in each state",
    ["state"],
)

processing_duration = Histogram(
    "tsn_processing_duration_seconds",
    "Time spent processing files",
    ["stage"],
)

callsigns_total = Gauge(
    "tsn_callsigns_total",
    "Total unique callsigns",
)

nets_total = Gauge(
    "tsn_nets_total",
    "Total net sessions",
)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    database: str
    queue_depth: dict[str, int]
    uptime_seconds: Optional[float] = None


class MetricsResponse(BaseModel):
    """Metrics response."""
    files_by_state: dict[str, int]
    total_callsigns: int
    total_nets: int
    processing_rates: dict[str, float]
    error_rate: float
    ai_runs_total: int
    ai_tokens_total: int
    gpu_samples_last_hour: int
    gpu_saturation_percent_last_hour: float


start_time = datetime.now(timezone.utc)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        System health status
    """
    try:
        # Check database
        async with get_session() as session:
            from sqlalchemy import func, select
            
            # Count files by state
            result = await session.execute(
                select(
                    AudioFile.state,
                    func.count(AudioFile.id).label("count"),
                )
                .group_by(AudioFile.state)
            )
            
            queue_depth = {row.state.value: row.count for row in result.all()}
        
        # Calculate uptime
        uptime = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            database="connected",
            queue_depth=queue_depth,
            uptime_seconds=uptime,
        )
        
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/metrics/json", response_model=MetricsResponse)
async def metrics_json():
    """
    Metrics endpoint (JSON format).
    
    Returns:
        System metrics
    """
    try:
        async with get_session() as session:
            from sqlalchemy import case, func, select
            from tsn_common.models import Callsign, NetSession
            
            # Files by state
            result = await session.execute(
                select(
                    AudioFile.state,
                    func.count(AudioFile.id).label("count"),
                )
                .group_by(AudioFile.state)
            )
            
            files_by_state = {row.state.value: row.count for row in result.all()}
            
            # Update Prometheus gauges
            for state, count in files_by_state.items():
                files_in_state.labels(state=state).set(count)
            
            # Total callsigns
            result = await session.execute(select(func.count(Callsign.id)))
            total_callsigns = result.scalar()
            callsigns_total.set(total_callsigns)
            
            # Total nets
            result = await session.execute(select(func.count(NetSession.id)))
            total_nets = result.scalar()
            nets_total.set(total_nets)
            
            # Calculate error rate
            total_files = sum(files_by_state.values())
            failed_files = sum(
                files_by_state.get(state, 0)
                for state in [
                    AudioFileState.FAILED_TRANSCRIPTION.value,
                    AudioFileState.FAILED_EXTRACTION.value,
                    AudioFileState.FAILED_ANALYSIS.value,
                ]
            )

            error_rate = (failed_files / total_files * 100) if total_files > 0 else 0.0

            # AI run stats
            ai_stmt = await session.execute(
                select(
                    func.count(AiRunLog.id),
                    func.coalesce(func.sum(AiRunLog.total_tokens), 0),
                )
            )
            ai_runs_total, ai_tokens_total = ai_stmt.one()
            ai_runs_total = int(ai_runs_total or 0)
            ai_tokens_total = int(ai_tokens_total or 0)

            # GPU utilization lookback (last hour)
            window_start = datetime.now(timezone.utc) - timedelta(hours=1)
            gpu_stmt = await session.execute(
                select(
                    func.count(GpuUtilizationSample.id),
                    func.coalesce(
                        func.sum(
                            case((GpuUtilizationSample.is_saturated.is_(True), 1), else_=0)
                        ),
                        0,
                    ),
                ).where(GpuUtilizationSample.created_at >= window_start)
            )
            gpu_sample_count, gpu_saturated_count = gpu_stmt.one()
            gpu_sample_count = int(gpu_sample_count or 0)
            gpu_saturated_count = int(gpu_saturated_count or 0)
            gpu_saturation_percent = (
                (gpu_saturated_count / gpu_sample_count) * 100
                if gpu_sample_count
                else 0.0
            )
        
        return MetricsResponse(
            files_by_state=files_by_state,
            total_callsigns=total_callsigns,
            total_nets=total_nets,
            processing_rates={},  # TODO: Calculate from processing metrics
            error_rate=error_rate,
            ai_runs_total=ai_runs_total,
            ai_tokens_total=ai_tokens_total,
            gpu_samples_last_hour=gpu_sample_count,
            gpu_saturation_percent_last_hour=gpu_saturation_percent,
        )
        
    except Exception as e:
        logger.error("metrics_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Metrics unavailable")


@app.get("/metrics")
async def metrics_prometheus():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Prometheus text format metrics
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/metrics/record")
async def record_metric(
    stage: str,
    duration_seconds: float,
    success: bool = True,
):
    """
    Record a processing metric.
    
    Args:
        stage: Processing stage name
        duration_seconds: Duration in seconds
        success: Whether processing succeeded
    """
    try:
        # Record to Prometheus
        processing_duration.labels(stage=stage).observe(duration_seconds)
        
        if success:
            files_processed.labels(state=f"{stage}_success").inc()
        else:
            files_processed.labels(state=f"{stage}_failed").inc()
        
        # Record to database
        async with get_session() as session:
            from tsn_common.models.support import ProcessingMetric
            
            metric = ProcessingMetric(
                stage=stage,
                duration_ms=int(duration_seconds * 1000),
                success=success,
            )
            session.add(metric)
        
        return {"status": "recorded"}
        
    except Exception as e:
        logger.error("metric_recording_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to record metric")


@app.post("/health/update")
async def update_system_health(
    component: str,
    status: str,
    details: Optional[str] = None,
):
    """
    Update system health status.
    
    Args:
        component: Component name (e.g., "transcriber", "extractor")
        status: Health status ("healthy", "degraded", "unhealthy")
        details: Optional details
    """
    try:
        async with get_session() as session:
            health = SystemHealth(
                component=component,
                status=status,
                details=details,
            )
            session.add(health)
        
        logger.info(
            "system_health_updated",
            component=component,
            status=status,
        )
        
        return {"status": "updated"}
        
    except Exception as e:
        logger.error("health_update_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update health")


async def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the health check server."""
    import uvicorn
    
    logger.info("starting_health_server", host=host, port=port)
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=None,  # Use our logging
    )
    
    server = uvicorn.Server(config)
    await server.serve()


def main():
    """Main entry point."""
    from tsn_common import setup_logging
    
    settings = get_settings()
    setup_logging(settings.logging)
    
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
