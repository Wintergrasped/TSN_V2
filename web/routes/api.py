"""JSON endpoints for the portal."""

from fastapi import APIRouter, Depends

from web.dependencies import get_db_session
from web.services.dashboard import (
    get_club_profiles,
    get_dashboard_payload,
    get_recent_callsigns,
    get_recent_nets,
    get_recent_transcriptions,
    get_system_health,
    get_trend_highlights,
)

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/dashboard")
async def api_dashboard(session=Depends(get_db_session)):
    return await get_dashboard_payload(session)


@router.get("/callsigns")
async def api_callsigns(session=Depends(get_db_session), limit: int = 200):
    return await get_recent_callsigns(session, limit=limit)


@router.get("/nets")
async def api_nets(session=Depends(get_db_session), limit: int = 100):
    return await get_recent_nets(session, limit=limit)


@router.get("/transcriptions")
async def api_transcriptions(session=Depends(get_db_session), limit: int = 50):
    return await get_recent_transcriptions(session, limit=limit)


@router.get("/clubs")
async def api_clubs(session=Depends(get_db_session), limit: int = 100):
    return await get_club_profiles(session, limit=limit)


@router.get("/trends")
async def api_trends(session=Depends(get_db_session), limit: int = 10):
    return await get_trend_highlights(session, limit=limit)


@router.get("/health")
async def api_health(session=Depends(get_db_session)):
    return await get_system_health(session)
