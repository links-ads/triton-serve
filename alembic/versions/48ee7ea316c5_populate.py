"""populate

Revision ID: 48ee7ea316c5
Revises: 915f97395c0d
Create Date: 2023-12-22 10:58:21.040444

"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from triton_serve.api.services.resources import get_gpu_info, get_machine_info
from triton_serve.database.model import Device, Machine

# revision identifiers, used by Alembic.
revision: str = "48ee7ea316c5"
down_revision: str | None = "915f97395c0d"
branch_labels: str | (Sequence[str] | None) = None
depends_on: str | (Sequence[str] | None) = None
log = logging.getLogger("alembic")


def upgrade() -> None:
    hostname, num_cpus, total_mem = get_machine_info()
    op.execute(
        sa.insert(Machine).values(
            host_name=hostname,
            num_cpus=num_cpus,
            total_memory=total_mem,
        )
    )
    try:
        gpus = get_gpu_info()
    except Exception as e:
        log.warning(f"Failed to get GPU info: {e}")
        gpus = []
    for gpu in gpus:
        op.execute(sa.insert(Device).values(host_id=1, **gpu.model_dump()))


def downgrade() -> None:
    pass
