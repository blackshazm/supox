from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from ...database import get_db
from ...models import User
from ...auth_headers import get_signed_user_id, USER_ID_HEADER
from ...config import get_config

logger = logging.getLogger(__name__)
router = APIRouter(tags=["preferences"])
config = get_config()

def _get_authenticated_user_id(request: Request) -> str:
    if config.monetization_enabled:
        return get_signed_user_id(request, config)

    user_id = request.headers.get("user_id") or request.headers.get(USER_ID_HEADER)
    if not user_id:
        raise HTTPException(status_code=401, detail="User authentication required")
    return user_id

@router.get("/preferences")
async def get_preferences(request: Request, db: AsyncSession = Depends(get_db)):
    """Get user font preferences."""
    user_id = _get_authenticated_user_id(request)
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    return {
        "fontFamily": user.default_font_family,
        "fontSize": user.default_font_size,
        "fontColor": user.default_font_color
    }

@router.patch("/preferences")
async def update_preferences(request: Request, db: AsyncSession = Depends(get_db)):
    """Update user font preferences."""
    user_id = _get_authenticated_user_id(request)
    data = await request.json()
    
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if "fontFamily" in data:
        user.default_font_family = data["fontFamily"]
    if "fontSize" in data:
        user.default_font_size = data["fontSize"]
    if "fontColor" in data:
        user.default_font_color = data["fontColor"]
        
    await db.commit()
    return {"message": "Preferences updated successfully"}
