"""lifecycle contract

Revision ID: b4cbad202c5e
Revises: c37d863f3180
Create Date: 2026-07-21 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b4cbad202c5e"
down_revision: str | None = "c37d863f3180"
branch_labels: str | (Sequence[str] | None) = None
depends_on: str | (Sequence[str] | None) = None


def upgrade() -> None:
    op.drop_column("services", "container_status")
    op.execute("DROP TYPE servicestatus")


def downgrade() -> None:
    old = sa.Enum("starting", "active", "error", "stopped", "deleted", "missing", name="servicestatus")
    old.create(op.get_bind(), checkfirst=True)
    op.add_column("services", sa.Column("container_status", old, nullable=False, server_default="starting"))
