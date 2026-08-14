"""service images

Revision ID: f1a2b3c4d5e6
Revises: a1f6c9d40b72
Create Date: 2026-08-13

"""

import hashlib
import json

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f1a2b3c4d5e6"
down_revision: str | None = "a1f6c9d40b72"
branch_labels: str | None = None
depends_on: str | None = None

# create_type=False: the type is created explicitly below, otherwise create_table emits a second
# CREATE TYPE and the migration fails on the duplicate
image_status = postgresql.ENUM("pending", "building", "ready", "failed", name="imagestatus", create_type=False)


def _empty_spec_hash(ref: str) -> str:
    """The hash BuildSpec computes for a dependency-free service on `ref`.

    The canonical form is inlined rather than imported so a later change to BuildSpec cannot rewrite
    what this revision backfilled. If the two ever diverge, the first write to an upgraded service
    inserts a duplicate row for content that is already in the table.
    """
    canonical = json.dumps(
        {"apt_packages": [], "base_image": ref, "pip_packages": []}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def upgrade() -> None:
    image_status.create(op.get_bind(), checkfirst=True)
    op.create_table(
        "service_images",
        sa.Column("image_hash", sa.String(), nullable=False),
        sa.Column("image_ref", sa.String(), nullable=False),
        sa.Column("status", image_status, server_default="pending", nullable=False),
        sa.Column("managed", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("base_image", sa.String(), nullable=True),
        sa.Column("apt_packages", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("pip_packages", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("build_log", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("built_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("image_hash"),
    )
    op.add_column(
        "models",
        sa.Column("system_dependencies", postgresql.ARRAY(sa.String()), server_default="{}", nullable=False),
    )
    op.add_column("services", sa.Column("image_hash", sa.String(), nullable=True))
    op.create_foreign_key("services_image_hash_fkey", "services", "service_images", ["image_hash"], ["image_hash"])

    # existing services keep the exact image they run today, as unmanaged rows that never build
    bind = op.get_bind()
    refs = [row[0] for row in bind.execute(sa.text("SELECT DISTINCT service_image FROM services"))]
    for ref in refs:
        digest = _empty_spec_hash(ref)
        bind.execute(
            sa.text(
                "INSERT INTO service_images "
                "(image_hash, image_ref, status, managed, apt_packages, pip_packages, created_at, built_at) "
                "VALUES (:h, :r, 'ready', false, '{}', '{}', now(), now()) "
                "ON CONFLICT (image_hash) DO NOTHING"
            ),
            {"h": digest, "r": ref},
        )
        bind.execute(
            sa.text("UPDATE services SET image_hash = :h WHERE service_image = :r"),
            {"h": digest, "r": ref},
        )


def downgrade() -> None:
    op.drop_constraint("services_image_hash_fkey", "services", type_="foreignkey")
    op.drop_column("services", "image_hash")
    op.drop_column("models", "system_dependencies")
    op.drop_table("service_images")
    image_status.drop(op.get_bind(), checkfirst=True)
