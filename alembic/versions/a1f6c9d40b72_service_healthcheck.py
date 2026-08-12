"""service healthcheck

Revision ID: a1f6c9d40b72
Revises: b4cbad202c5e
Create Date: 2026-08-11 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a1f6c9d40b72"
down_revision: str | None = "b4cbad202c5e"
branch_labels: str | (Sequence[str] | None) = None
depends_on: str | (Sequence[str] | None) = None


def upgrade() -> None:
    # nullable with no default: existing services keep the boot-grace uptime timer
    op.add_column("service_resources", sa.Column("healthcheck", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("service_resources", "healthcheck")
