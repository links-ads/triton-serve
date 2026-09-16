"""drop kombu broker tables

Revision ID: e2f1a7c93b45
Revises: 5ac6ec66eb4c
Create Date: 2026-09-16

"""

from alembic import op

revision: str = "e2f1a7c93b45"
down_revision: str | None = "5ac6ec66eb4c"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # the child first: kombu_message.queue_id references kombu_queue.id
    op.execute("DROP TABLE IF EXISTS kombu_message")
    op.execute("DROP TABLE IF EXISTS kombu_queue")
    op.execute("DROP SEQUENCE IF EXISTS queue_id_sequence")


def downgrade() -> None:
    # deliberately empty: the schema belonged to the kombu sqlalchemy transport, which recreates it
    # on first use. rebuilding a foreign schema here would mean owning it in our migration history
    pass
