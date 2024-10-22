import logging
from fastapi import HTTPException
from sqlalchemy.orm import Session

from sqlalchemy import delete
from datetime import timedelta

from triton_serve.database.model import KombuMessage
from triton_serve.database.schema import timezone_aware_now, QueueMessageDeleteResponseSchema

LOG = logging.getLogger("uvicorn")


def delete_queue_messages(db: Session, older_than_hours: int) -> None:
    """
    Deletes queue messages older than the specified window.

    Args:
        db (Session): The database session.
        storage (ModelStorage): The storage implementation to use.
        older_than_hours (int): Delete messages that are older than this many hours.

    Returns:
        dict: Information about the deleted messages

    Raises:
        HTTPException: If an error occurs while deleting the messages
    """
    try:
        LOG.debug("Deleting queue messages older than %d hours", older_than_hours)
        query = delete(KombuMessage).where(
            KombuMessage.timestamp < (timezone_aware_now() - timedelta(hours=older_than_hours))
        )

        result = db.execute(query)
        db.commit()
        return QueueMessageDeleteResponseSchema(deleted_messages=result.rowcount)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting queue messages: {e}")
