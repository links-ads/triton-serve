"""populate

Revision ID: 48ee7ea316c5
Revises: af7db8f0dc88
Create Date: 2023-12-22 10:58:21.040444

"""

import logging
from collections.abc import Sequence

from alembic import op
from sqlalchemy.orm import Session

from triton_serve.api.services.resources import get_gpu_info, get_machine_info
from triton_serve.database.model import Device, Machine

# revision identifiers, used by Alembic.
revision: str = "48ee7ea316c5"
down_revision: str | None = "af7db8f0dc88"
branch_labels: str | (Sequence[str] | None) = None
depends_on: str | (Sequence[str] | None) = None
log = logging.getLogger("uvicorn")


def upgrade() -> None:
    bind = op.get_bind()
    session = Session(bind=bind)

    try:
        # Get machine info
        hostname, num_cpus, total_mem = get_machine_info()

        # Insert machine info
        machine = Machine(
            host_name=hostname,
            num_cpus=num_cpus,
            total_memory=total_mem,
        )
        session.add(machine)
        session.flush()  # Flush to get the machine ID

        # Get GPU info
        try:
            gpus = get_gpu_info()
        except Exception as e:
            log.warning(f"Failed to get GPU info: {e}")
            log.warning("Continuing without GPU")
            gpus = []

        # Insert GPU info
        for gpu in gpus:
            device = Device(host_id=machine.host_id, **gpu.model_dump())
            session.add(device)

        session.commit()
    except Exception as e:
        log.error(f"Error during population: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def downgrade() -> None:
    pass
