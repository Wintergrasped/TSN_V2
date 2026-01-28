"""Server-rendered pages for the KK7NQN-inspired dashboard."""

from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from web.dependencies import get_current_user, get_db_session, maybe_current_user, templates
from web.services.dashboard import (
    get_club_profiles,
    get_dashboard_payload,
    get_recent_callsigns,
    get_recent_nets,
    get_recent_transcriptions,
    get_system_health,
    get_trend_highlights,
    normalize_node_scope,
)
from web.services import nets
from web.services import user_dashboard as user_dash_service

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def landing_page(
    request: Request,
    node: str | None = None,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    payload = await get_dashboard_payload(session, node_scope=node)
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
    node: str | None = None,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    node_scope = normalize_node_scope(node)
    initial_callsigns = await get_recent_callsigns(
        session,
        limit=100,
        node_scope=node_scope,
        order_by="mentions",
    )
    query = {
        "limit": 400,
        "order": "mentions",
    }
    if node_scope != "all":
        query["node"] = node_scope
    feed_url = f"/api/callsigns?{urlencode(query)}"
    trends = await get_trend_highlights(session)
    return templates.TemplateResponse(
        "callsigns.html",
        {
            "request": request,
            "trends": trends,
            "current_user": current_user,
            "callsign_feed_url": feed_url,
            "node_scope": node_scope,
            "initial_callsigns": initial_callsigns,
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


@router.get("/nets/{net_id}", response_class=HTMLResponse)
async def net_summary_page(
    request: Request,
    net_id: str,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    payload = await nets.fetch_net_summary(session, net_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Net not found")
    return templates.TemplateResponse(
        "net_summary.html",
        {
            "request": request,
            "net": payload,
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
    clubs = await get_club_profiles(session, limit=100, order_by="mentions")
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
    session=Depends(get_db_session),
    current_user=Depends(get_current_user),
):
    payload = await user_dash_service.build_user_dashboard_payload(session, current_user)
    return templates.TemplateResponse(
        "user_dashboard.html",
        {
            "request": request,
            "current_user": current_user,
            "payload": payload,
        },
    )


@router.post("/user/dashboard")
async def update_user_dashboard(
    bio: str = Form(""),
    photo_url: str = Form(""),
    club_memberships: str = Form(""),
    session=Depends(get_db_session),
    current_user=Depends(get_current_user),
):
    await user_dash_service.update_profile_preferences(
        session,
        current_user,
        bio=bio,
        photo_url=photo_url,
        club_memberships=club_memberships,
    )
    return RedirectResponse("/user/dashboard", status_code=status.HTTP_303_SEE_OTHER)
