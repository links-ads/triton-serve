import logging
from datetime import timedelta

from sqlalchemy import delete
from sqlalchemy.orm import Session

from triton_serve.database.model import KombuMessage, timezone_aware_now
from triton_serve.database.schema import QueueMessageDeleteResponseSchema

LOG = logging.getLogger("uvicorn")


def delete_queue_messages(db: Session, older_than_hours: int) -> QueueMessageDeleteResponseSchema:
    """
    Deletes queue messages older than the specified window.

    Args:
        db (Session): The database session.
        older_than_hours (int): Delete messages that are older than this many hours.

    Returns:
        QueueMessageDeleteResponseSchema: The number of messages deleted.
    """
    LOG.debug("Deleting queue messages older than %d hours", older_than_hours)
    query = delete(KombuMessage).where(
        KombuMessage.timestamp < (timezone_aware_now() - timedelta(hours=older_than_hours))
    )

    result = db.execute(query)
    db.commit()
    return QueueMessageDeleteResponseSchema(deleted_messages=result.rowcount)  # type: ignore
