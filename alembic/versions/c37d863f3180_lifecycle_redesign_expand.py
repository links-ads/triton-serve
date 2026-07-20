"""lifecycle redesign expand

Revision ID: c37d863f3180
Revises: dd883257cf20
Create Date: 2026-07-20 12:49:08.223067

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c37d863f3180"
down_revision: str | None = "dd883257cf20"
branch_labels: str | (Sequence[str] | None) = None
depends_on: str | (Sequence[str] | None) = None

desired = sa.Enum("available", "suspended", "retired", name="desiredstate")
runtime = sa.Enum("ready", "warming", "idle", "recovering", "failed", "suspended", "retired", name="runtimestatus")


def upgrade() -> None:
    bind = op.get_bind()
    desired.create(bind, checkfirst=True)
    runtime.create(bind, checkfirst=True)

    op.add_column("services", sa.Column("desired_state", desired, nullable=True))
    op.add_column("services", sa.Column("runtime_status", runtime, nullable=True))

    # backfill: intent from deleted_at; runtime projected from the old container_status
    op.execute(
        """
        UPDATE services SET
          desired_state = CASE WHEN deleted_at IS NOT NULL THEN 'retired' ELSE 'available' END::desiredstate,
          runtime_status = CASE container_status
            WHEN 'ACTIVE'   THEN 'ready'
            WHEN 'STARTING' THEN 'warming'
            WHEN 'STOPPED'  THEN 'idle'
            WHEN 'MISSING'  THEN 'recovering'
            WHEN 'ERROR'    THEN 'failed'
            WHEN 'DELETED'  THEN 'retired'
            ELSE 'warming'
          END::runtimestatus
        """
    )
    op.alter_column("services", "desired_state", nullable=False)
    op.alter_column("services", "runtime_status", nullable=False)


def downgrade() -> None:
    op.drop_column("services", "runtime_status")
    op.drop_column("services", "desired_state")
    op.execute("DROP TYPE runtimestatus")
    op.execute("DROP TYPE desiredstate")
