"""image status changed at

Revision ID: 5ac6ec66eb4c
Revises: f1a2b3c4d5e6
Create Date: 2026-09-15

"""

import sqlalchemy as sa
from alembic import op

revision: str = "5ac6ec66eb4c"
down_revision: str | None = "f1a2b3c4d5e6"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "service_images",
        sa.Column("status_changed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    # rows already orphaned in BUILDING must reap on the first sweep, not be grandfathered in as fresh
    op.execute("UPDATE service_images SET status_changed_at = created_at")


def downgrade() -> None:
    op.drop_column("service_images", "status_changed_at")
