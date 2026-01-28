"""Detail pages for callsigns, clubs, and the Net Control cockpit."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload

from tsn_common.models import Callsign, CallsignProfile, ClubMembership, ClubProfile
from tsn_common.utils import normalize_callsign
from web.dependencies import get_current_user, get_db_session, maybe_current_user, templates
from web.models.user import PortalUser
from web.services import net_control, profiles

from tsn_common.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/callsigns/{callsign}", response_class=HTMLResponse)
async def callsign_profile_page(
    request: Request,
    callsign: str,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    payload = await profiles.fetch_callsign_profile(session, callsign)
    if payload is None:
        raise HTTPException(status_code=404, detail="Callsign not found")
    return templates.TemplateResponse(
        "callsign_profile.html",
        {
            "request": request,
            "profile": payload,
            "current_user": current_user,
        },
    )


@router.post("/callsigns/{callsign}/note")
async def update_callsign_note(
    callsign: str,
    note: str = Form(""),
    session=Depends(get_db_session),
    current_user: PortalUser = Depends(get_current_user),
):
    normalized = normalize_callsign(callsign)
    if normalize_callsign(current_user.callsign or "") != normalized:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your callsign")

    result = await session.execute(
        select(Callsign)
        .options(joinedload(Callsign.profile))
        .where(Callsign.callsign == normalized)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Callsign not found")

    profile = record.profile
    if profile is None:
        profile = CallsignProfile(callsign_id=record.id)
        session.add(profile)
        await session.flush()

    metadata = dict(profile.metadata_ or {})
    metadata["custom_note"] = note.strip()
    profile.metadata_ = metadata
    return RedirectResponse(f"/callsigns/{normalized}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/ncs/{callsign}", response_class=HTMLResponse)
async def ncs_profile_page(
    request: Request,
    callsign: str,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    payload = await profiles.fetch_ncs_profile(session, callsign)
    if payload is None:
        raise HTTPException(status_code=404, detail="Callsign not found")
    return templates.TemplateResponse(
        "ncs_profile.html",
        {
            "request": request,
            "payload": payload,
            "current_user": current_user,
        },
    )


@router.get("/clubs/{club_name}", response_class=HTMLResponse)
async def club_profile_page(
    request: Request,
    club_name: str,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    payload = await profiles.fetch_club_profile(session, club_name)
    if payload is None:
        raise HTTPException(status_code=404, detail="Club not found")
    can_customize = False
    if current_user and current_user.callsign:
        normalized = normalize_callsign(current_user.callsign)
        can_customize = current_user.is_admin or any(
            member["callsign"] == normalized for member in payload["members"]
        )
    return templates.TemplateResponse(
        "club_detail.html",
        {
            "request": request,
            "club": payload,
            "can_customize": can_customize,
            "current_user": current_user,
        },
    )


@router.post("/clubs/{club_name}/note")
async def update_club_note(
    club_name: str,
    note: str = Form(""),
    session=Depends(get_db_session),
    current_user: PortalUser = Depends(get_current_user),
):
    if not current_user.callsign:
        raise HTTPException(status_code=403, detail="Calls sign required")

    membership = await session.execute(
        select(ClubMembership)
        .join(ClubProfile, ClubMembership.club_id == ClubProfile.id)
        .join(Callsign, ClubMembership.callsign_id == Callsign.id)
        .where(
            ClubProfile.name == club_name.strip(),
            Callsign.callsign == normalize_callsign(current_user.callsign),
        )
    )
    if membership.scalar_one_or_none() is None and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Membership required")

    club_result = await session.execute(select(ClubProfile).where(ClubProfile.name == club_name.strip()))
    club = club_result.scalar_one_or_none()
    if club is None:
        raise HTTPException(status_code=404, detail="Club not found")

    metadata = dict(club.metadata_ or {})
    metadata["custom_note"] = note.strip()
    club.metadata_ = metadata
    return RedirectResponse(f"/clubs/{club_name}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/net-control", response_class=HTMLResponse)
async def net_control_page(
    request: Request,
    node_id: str | None = None,
    session=Depends(get_db_session),
    current_user=Depends(maybe_current_user),
):
    active = None
    sessions: list[dict] = []
    feed: list[dict] = []
    live_callsigns: list[dict] = []
    available_nodes: list[str] = []
    load_error = None
    try:
        active = await net_control.get_active_session(session)
        sessions = await net_control.list_sessions(session)
        feed = await net_control.fetch_checkin_feed(session, limit=25, node_id=node_id)
        available_nodes = await net_control.get_available_nodes(session)
        
        # If there's an active net, fetch live callsigns
        if active:
            live_callsigns = await net_control.fetch_live_callsigns(session, limit=5, node_id=node_id)
    except SQLAlchemyError as exc:
        logger.error("net_control_page_load_failed", error=str(exc))
        load_error = "Unable to load live net data. Please try again in a moment."
    return templates.TemplateResponse(
        "net_control.html",
        {
            "request": request,
            "current_user": current_user,
            "active_session": active,
            "sessions": sessions,
            "feed": feed,
            "live_callsigns": live_callsigns,
            "available_nodes": available_nodes,
            "selected_node": node_id,
            "requires_login": current_user is None,
            "load_error": load_error,
        },
    )


@router.post("/net-control/start")
async def start_net_control(
    name: str = Form(""),
    notes: str = Form(""),
    node_id: str = Form(""),
    session=Depends(get_db_session),
    current_user: PortalUser = Depends(get_current_user),
):
    payload = await net_control.start_session(
        session,
        name=name or f"Manual Net {uuid.uuid4().hex[:4]}",
        started_by=current_user.display_name,
        started_by_callsign=current_user.callsign,
        notes=notes or None,
        node_id=node_id or None,
    )
    return RedirectResponse("/net-control", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/net-control/{session_id}/stop")
async def stop_net_control(
    session_id: uuid.UUID,
    session=Depends(get_db_session),
    current_user: PortalUser = Depends(get_current_user),
):
    if current_user is None:
        raise HTTPException(status_code=403, detail="Login required")
    payload = await net_control.stop_session(session, session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return RedirectResponse("/net-control", status_code=status.HTTP_303_SEE_OTHER)
