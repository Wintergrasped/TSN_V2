"""JSON endpoints for the portal."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from web.dependencies import get_db_session
from web.services import net_control, profiles
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


@router.get("/callsigns/{callsign}")
async def api_callsign_detail(callsign: str, session=Depends(get_db_session)):
    payload = await profiles.fetch_callsign_profile(session, callsign)
    if payload is None:
        raise HTTPException(status_code=404, detail="Callsign not found")
    return payload


@router.get("/nets")
async def api_nets(session=Depends(get_db_session), limit: int = 100):
    return await get_recent_nets(session, limit=limit)


@router.get("/transcriptions")
async def api_transcriptions(session=Depends(get_db_session), limit: int = 50):
    return await get_recent_transcriptions(session, limit=limit)


@router.get("/clubs")
async def api_clubs(session=Depends(get_db_session), limit: int = 100):
    return await get_club_profiles(session, limit=limit)


@router.get("/clubs/{club_name}")
async def api_club_detail(club_name: str, session=Depends(get_db_session)):
    payload = await profiles.fetch_club_profile(session, club_name)
    if payload is None:
        raise HTTPException(status_code=404, detail="Club not found")
    return payload


@router.get("/trends")
async def api_trends(session=Depends(get_db_session), limit: int = 10):
    return await get_trend_highlights(session, limit=limit)


@router.get("/health")
async def api_health(session=Depends(get_db_session)):
    return await get_system_health(session)


@router.get("/net-control/feed")
async def api_net_control_feed(
    session=Depends(get_db_session),
    limit: int = 50,
    format: str = "json",
):
    if format == "csv":
        csv_body = await net_control.export_feed_csv(session, limit=limit)
        return PlainTextResponse(csv_body, media_type="text/csv")
    return await net_control.fetch_checkin_feed(session, limit=limit)
