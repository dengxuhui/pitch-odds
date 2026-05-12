"""add model_params table

Revision ID: 20260512_0002
Revises: 20260512_0001
Create Date: 2026-05-12 00:30:00

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260512_0002"
down_revision = "20260512_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_params",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("league_id", sa.String(length=10), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("train_until", sa.Date(), nullable=False),
        sa.Column("params", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("brier_score", sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column("n_matches", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["league_id"], ["leagues.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("league_id", "model_version", "train_until", name="uq_model_params_scope"),
    )


def downgrade() -> None:
    op.drop_table("model_params")
