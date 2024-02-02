"""populate

Revision ID: 48ee7ea316c5
Revises: 3373b9da9862
Create Date: 2023-12-22 10:58:21.040444

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from triton_serve.api.services.resources import get_gpu_info, get_machine_info
from triton_serve.database.model import Device, Machine

# revision identifiers, used by Alembic.
revision: str = "48ee7ea316c5"
down_revision: str | None = "3373b9da9862"
branch_labels: str | (Sequence[str] | None) = None
depends_on: str | (Sequence[str] | None) = None


def upgrade() -> None:
    hostname, num_cpus, total_mem = get_machine_info()
    op.execute(
        sa.insert(Machine).values(
            host_name=hostname,
            num_cpus=num_cpus,
            total_memory=total_mem,
        )
    )
    gpus = get_gpu_info()
    for gpu in gpus:
        op.execute(sa.insert(Device).values(host_id=1, **gpu))


def downgrade() -> None:
    pass
