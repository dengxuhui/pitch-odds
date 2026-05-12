"""init schema

Revision ID: 20260512_0001
Revises:
Create Date: 2026-05-12 00:00:00

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260512_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "leagues",
        sa.Column("id", sa.String(length=10), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("country", sa.String(length=50), nullable=False),
        sa.Column("avg_goals", sa.Numeric(precision=4, scale=2), nullable=False, server_default="2.70"),
        sa.Column("home_adv", sa.Numeric(precision=4, scale=3), nullable=False, server_default="1.080"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("league_id", sa.String(length=10), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("short_name", sa.String(length=20), nullable=True),
        sa.ForeignKeyConstraint(["league_id"], ["leagues.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("league_id", "name", name="uq_team_league_name"),
    )

    op.create_table(
        "matches",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("league_id", sa.String(length=10), nullable=False),
        sa.Column("season", sa.String(length=10), nullable=False),
        sa.Column("match_date", sa.Date(), nullable=False),
        sa.Column("match_week", sa.SmallInteger(), nullable=True),
        sa.Column("home_team_id", sa.Integer(), nullable=False),
        sa.Column("away_team_id", sa.Integer(), nullable=False),
        sa.Column("home_goals", sa.SmallInteger(), nullable=True),
        sa.Column("away_goals", sa.SmallInteger(), nullable=True),
        sa.Column("result", sa.String(length=1), nullable=True),
        sa.Column("home_shots", sa.SmallInteger(), nullable=True),
        sa.Column("away_shots", sa.SmallInteger(), nullable=True),
        sa.Column("home_shots_on", sa.SmallInteger(), nullable=True),
        sa.Column("away_shots_on", sa.SmallInteger(), nullable=True),
        sa.Column("home_possession", sa.Numeric(precision=4, scale=1), nullable=True),
        sa.Column("away_possession", sa.Numeric(precision=4, scale=1), nullable=True),
        sa.Column("home_corners", sa.SmallInteger(), nullable=True),
        sa.Column("away_corners", sa.SmallInteger(), nullable=True),
        sa.Column("home_yellow", sa.SmallInteger(), nullable=True),
        sa.Column("away_yellow", sa.SmallInteger(), nullable=True),
        sa.Column("home_red", sa.SmallInteger(), nullable=True),
        sa.Column("away_red", sa.SmallInteger(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["away_team_id"], ["teams.id"]),
        sa.ForeignKeyConstraint(["home_team_id"], ["teams.id"]),
        sa.ForeignKeyConstraint(["league_id"], ["leagues.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("league_id", "season", "match_date", "home_team_id", "away_team_id", name="uq_match_identity"),
    )
    op.create_index("idx_matches_date", "matches", ["match_date"], unique=False)
    op.create_index("idx_matches_home", "matches", ["home_team_id", "match_date"], unique=False)
    op.create_index("idx_matches_away", "matches", ["away_team_id", "match_date"], unique=False)
    op.create_index("idx_matches_league", "matches", ["league_id", "season"], unique=False)

    op.create_table(
        "odds_opening",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("bookmaker", sa.String(length=50), nullable=False),
        sa.Column("odds_home", sa.Numeric(precision=6, scale=3), nullable=False),
        sa.Column("odds_draw", sa.Numeric(precision=6, scale=3), nullable=False),
        sa.Column("odds_away", sa.Numeric(precision=6, scale=3), nullable=False),
        sa.Column("overround", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("match_id", "bookmaker", name="uq_opening_match_bookmaker"),
    )

    op.create_table(
        "odds_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("bookmaker", sa.String(length=50), nullable=False),
        sa.Column("odds_home", sa.Numeric(precision=6, scale=3), nullable=False),
        sa.Column("odds_draw", sa.Numeric(precision=6, scale=3), nullable=False),
        sa.Column("odds_away", sa.Numeric(precision=6, scale=3), nullable=False),
        sa.Column("overround", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hours_to_kick", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_odds_match", "odds_snapshots", ["match_id", "snapshot_at"], unique=False)
    op.create_index("idx_odds_hours", "odds_snapshots", ["match_id", "hours_to_kick"], unique=False)

    op.create_table(
        "odds_anomalies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("alert_level", sa.String(length=10), nullable=False),
        sa.Column("anomaly_type", sa.String(length=30), nullable=True),
        sa.Column("max_step_change", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("total_drift_pct", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("exclude_flag", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "team_status",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("form_score_5", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("form_score_10", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("fatigue_index", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("injury_impact", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("momentum_score", sa.Numeric(precision=4, scale=3), nullable=True),
        sa.Column("matches_last_30d", sa.SmallInteger(), nullable=True),
        sa.Column("travel_km", sa.Integer(), nullable=True),
        sa.Column("missing_players", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("team_id", "as_of_date", name="uq_team_status_date"),
    )

    op.create_table(
        "player_injuries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("player_name", sa.String(length=100), nullable=False),
        sa.Column("injury_type", sa.String(length=100), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("importance", sa.Numeric(precision=3, scale=2), nullable=True),
        sa.Column("position_multiplier", sa.Numeric(precision=3, scale=2), nullable=True),
        sa.Column("reported_at", sa.Date(), nullable=False),
        sa.Column("expected_return", sa.Date(), nullable=True),
        sa.Column("source", sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "model_predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("p_home", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("p_draw", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("p_away", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("ev_home", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column("ev_draw", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column("ev_away", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column("edge_home", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("edge_draw", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("edge_away", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("is_calibrated", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("match_id", "model_version", "predicted_at", name="uq_model_prediction"),
    )

    op.create_table(
        "parlay_plans",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("plan_date", sa.Date(), nullable=False),
        sa.Column("tier", sa.String(length=20), nullable=False),
        sa.Column("legs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("total_odds", sa.Numeric(precision=8, scale=3), nullable=True),
        sa.Column("win_rate", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("expected_ev", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column("kelly_pct", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("stake", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("is_simulation", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "bet_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("plan_id", sa.Integer(), nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("won", sa.Boolean(), nullable=True),
        sa.Column("payout", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("profit", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.ForeignKeyConstraint(["plan_id"], ["parlay_plans.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("bet_results")
    op.drop_table("parlay_plans")
    op.drop_table("model_predictions")
    op.drop_table("player_injuries")
    op.drop_table("team_status")
    op.drop_table("odds_anomalies")
    op.drop_index("idx_odds_hours", table_name="odds_snapshots")
    op.drop_index("idx_odds_match", table_name="odds_snapshots")
    op.drop_table("odds_snapshots")
    op.drop_table("odds_opening")
    op.drop_index("idx_matches_league", table_name="matches")
    op.drop_index("idx_matches_away", table_name="matches")
    op.drop_index("idx_matches_home", table_name="matches")
    op.drop_index("idx_matches_date", table_name="matches")
    op.drop_table("matches")
    op.drop_table("teams")
    op.drop_table("leagues")
