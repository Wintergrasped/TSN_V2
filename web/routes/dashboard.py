"""Server-rendered pages for the KK7NQN-inspired dashboard."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from web.dependencies import get_current_user, get_db_session, maybe_current_user, templates
from web.services.dashboard import (
    get_club_profiles,
    get_dashboard_payload,
    get_recent_callsigns,
    get_recent_nets,
    get_recent_transcriptions,
    get_system_health,
    get_trend_highlights,
)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def landing_page(
    request: Request,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    payload = await get_dashboard_payload(session)
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "payload": payload,
            "current_user": current_user,
        },
    )


@router.get("/callsigns", response_class=HTMLResponse)
async def callsign_page(
    request: Request,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    callsigns = await get_recent_callsigns(session, limit=200)
    trends = await get_trend_highlights(session)
    return templates.TemplateResponse(
        "callsigns.html",
        {
            "request": request,
            "callsigns": callsigns,
            "trends": trends,
            "current_user": current_user,
        },
    )


@router.get("/nets", response_class=HTMLResponse)
async def nets_page(
    request: Request,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    nets = await get_recent_nets(session, limit=100)
    return templates.TemplateResponse(
        "nets.html",
        {
            "request": request,
            "nets": nets,
            "current_user": current_user,
        },
    )


@router.get("/transcriptions", response_class=HTMLResponse)
async def transcriptions_page(
    request: Request,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    transcriptions = await get_recent_transcriptions(session, limit=50)
    return templates.TemplateResponse(
        "transcriptions.html",
        {
            "request": request,
            "transcriptions": transcriptions,
            "current_user": current_user,
        },
    )


@router.get("/clubs", response_class=HTMLResponse)
async def clubs_page(
    request: Request,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    clubs = await get_club_profiles(session, limit=100)
    return templates.TemplateResponse(
        "clubs.html",
        {
            "request": request,
            "clubs": clubs,
            "current_user": current_user,
        },
    )


@router.get("/health", response_class=HTMLResponse)
async def health_page(
    request: Request,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    health = await get_system_health(session)
    return templates.TemplateResponse(
        "health.html",
        {
            "request": request,
            "health": health,
            "current_user": current_user,
        },
    )


@router.get("/user/dashboard", response_class=HTMLResponse)
async def user_dashboard(
    request: Request,
    current_user=Depends(get_current_user),
):
    return templates.TemplateResponse(
        "user_dashboard.html",
        {
            "request": request,
            "current_user": current_user,
        },
    )
