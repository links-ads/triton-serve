from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Any

from triton_serve.api.operations import domain
from triton_serve.config import AppSettings, get_settings
from triton_serve.extensions import get_db
from triton_serve.security import require_admin
from triton_serve.database.schema import QueueMessageDeleteResponseSchema

router = APIRouter()


@router.delete(
    "/queue/messages",
    response_model=QueueMessageDeleteResponseSchema,
    status_code=200,
    tags=["operations"],
)
def delete_queue_messages(
    db: Session = Depends(get_db),
    settings: AppSettings = Depends(get_settings),
    _: Any = Depends(require_admin),
):
    """
    Delete messages from the queue within a specified time window.

    Parameters:
    - db (Session): Database session
    - settings (AppSettings): Application settings

    Returns:
    - dict: Information about the deleted messages

    """
    return domain.delete_queue_messages(db=db, older_than_hours=settings.older_than_hours_to_purge)
