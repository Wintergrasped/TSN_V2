"""Authentication + account management routes."""

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.exc import IntegrityError

from web.config import get_web_settings
from web.dependencies import get_current_user, get_db_session, maybe_current_user, templates
from web.services.users import authenticate_user, create_user, record_login

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_page(
    request: Request,
    current_user=Depends(maybe_current_user),
):
    if current_user:
        return RedirectResponse(url="/user/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "current_user": current_user,
            "error": None,
        },
    )


@router.post("/login")
async def login_action(
    request: Request,
    session=Depends(get_db_session),
    email: str = Form(...),
    password: str = Form(...),
):
    user = await authenticate_user(session, email=email, password=password)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "current_user": None,
                "error": "Invalid email or password",
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    request.session["user_id"] = str(user.id)
    await record_login(session, user)
    return RedirectResponse(url="/user/dashboard", status_code=status.HTTP_302_FOUND)


@router.post("/logout")
async def logout_action(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)


@router.get("/register", response_class=HTMLResponse)
async def register_page(
    request: Request,
    current_user=Depends(maybe_current_user),
):
    settings = get_web_settings()
    if current_user:
        return RedirectResponse(url="/user/dashboard", status_code=status.HTTP_302_FOUND)
    form_state = {"display_name": "", "email": "", "callsign": ""}
    return templates.TemplateResponse(
        "register.html",
        {
            "request": request,
            "current_user": current_user,
            "error": None,
            "registration_open": settings.allow_registration,
            "form_data": form_state,
        },
        status_code=status.HTTP_200_OK,
    )


@router.post("/register")
async def register_action(
    request: Request,
    session=Depends(get_db_session),
    email: str = Form(...),
    password: str = Form(...),
    display_name: str = Form(...),
    callsign: str | None = Form(None),
):
    settings = get_web_settings()
    form_state = {
        "display_name": display_name,
        "email": email,
        "callsign": callsign or "",
    }
    if not settings.allow_registration:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "current_user": None,
                "error": "Registration is currently disabled.",
                "registration_open": False,
                "form_data": form_state,
            },
            status_code=status.HTTP_403_FORBIDDEN,
        )

    try:
        user = await create_user(
            session,
            email=email,
            password=password,
            display_name=display_name,
            callsign=callsign,
        )
    except IntegrityError:
        await session.rollback()
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "current_user": None,
                "error": "Email already registered",
                "registration_open": settings.allow_registration,
                "form_data": form_state,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    except ValueError as exc:
        await session.rollback()
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "current_user": None,
                "error": str(exc),
                "registration_open": settings.allow_registration,
                "form_data": form_state,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    request.session["user_id"] = str(user.id)
    return RedirectResponse(url="/user/dashboard", status_code=status.HTTP_302_FOUND)


@router.get("/me")
async def me(current_user=Depends(get_current_user)):
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "display_name": current_user.display_name,
        "callsign": current_user.callsign,
        "is_admin": current_user.is_admin,
        "last_login_at": current_user.last_login_at.isoformat() if current_user.last_login_at else None,
    }
