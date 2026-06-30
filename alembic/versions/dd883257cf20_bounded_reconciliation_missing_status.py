"""bounded reconciliation missing status

Revision ID: dd883257cf20
Revises: 48ee7ea316c5
Create Date: 2026-06-29 20:03:20.128143

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dd883257cf20"
down_revision: str | None = "48ee7ea316c5"
branch_labels: str | (Sequence[str] | None) = None
depends_on: str | (Sequence[str] | None) = None


def upgrade() -> None:
    # ADD VALUE cannot run inside the surrounding migration transaction
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE servicestatus ADD VALUE IF NOT EXISTS 'MISSING'")
    op.add_column("services", sa.Column("restart_attempts", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("services", sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("services", "last_attempt_at")
    op.drop_column("services", "restart_attempts")
    # NOTE: PostgreSQL cannot remove an enum value; 'MISSING' stays on the servicestatus type.
