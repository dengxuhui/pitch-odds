from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, Numeric, SmallInteger, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    country: Mapped[str] = mapped_column(String(50), nullable=False)
    avg_goals: Mapped[Decimal] = mapped_column(Numeric(4, 2), default=Decimal("2.70"), nullable=False)
    home_adv: Mapped[Decimal] = mapped_column(Numeric(4, 3), default=Decimal("1.080"), nullable=False)


class Team(Base):
    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("league_id", "name", name="uq_team_league_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[str] = mapped_column(String(10), ForeignKey("leagues.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    short_name: Mapped[str | None] = mapped_column(String(20), nullable=True)


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint(
            "league_id",
            "season",
            "match_date",
            "home_team_id",
            "away_team_id",
            name="uq_match_identity",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[str] = mapped_column(String(10), ForeignKey("leagues.id"), nullable=False)
    season: Mapped[str] = mapped_column(String(10), nullable=False)
    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    match_week: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    home_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)

    home_goals: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    away_goals: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    result: Mapped[str | None] = mapped_column(String(1), nullable=True)

    home_shots: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    away_shots: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    home_shots_on: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    away_shots_on: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    home_possession: Mapped[Decimal | None] = mapped_column(Numeric(4, 1), nullable=True)
    away_possession: Mapped[Decimal | None] = mapped_column(Numeric(4, 1), nullable=True)
    home_corners: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    away_corners: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    home_yellow: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    away_yellow: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    home_red: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    away_red: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class OddsOpening(Base):
    __tablename__ = "odds_opening"
    __table_args__ = (UniqueConstraint("match_id", "bookmaker", name="uq_opening_match_bookmaker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)
    odds_home: Mapped[Decimal] = mapped_column(Numeric(6, 3), nullable=False)
    odds_draw: Mapped[Decimal] = mapped_column(Numeric(6, 3), nullable=False)
    odds_away: Mapped[Decimal] = mapped_column(Numeric(6, 3), nullable=False)
    overround: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)
    odds_home: Mapped[Decimal] = mapped_column(Numeric(6, 3), nullable=False)
    odds_draw: Mapped[Decimal] = mapped_column(Numeric(6, 3), nullable=False)
    odds_away: Mapped[Decimal] = mapped_column(Numeric(6, 3), nullable=False)
    overround: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    snapshot_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    hours_to_kick: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)


class OddsAnomaly(Base):
    __tablename__ = "odds_anomalies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    alert_level: Mapped[str] = mapped_column(String(10), nullable=False)
    anomaly_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    max_step_change: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    total_drift_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    exclude_flag: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class TeamStatus(Base):
    __tablename__ = "team_status"
    __table_args__ = (UniqueConstraint("team_id", "as_of_date", name="uq_team_status_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)

    form_score_5: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    form_score_10: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    fatigue_index: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    injury_impact: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    momentum_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    matches_last_30d: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    travel_km: Mapped[int | None] = mapped_column(Integer, nullable=True)
    missing_players: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PlayerInjury(Base):
    __tablename__ = "player_injuries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id"), nullable=False)
    player_name: Mapped[str] = mapped_column(String(100), nullable=False)
    injury_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    importance: Mapped[Decimal | None] = mapped_column(Numeric(3, 2), nullable=True)
    position_multiplier: Mapped[Decimal | None] = mapped_column(Numeric(3, 2), nullable=True)
    reported_at: Mapped[date] = mapped_column(Date, nullable=False)
    expected_return: Mapped[date | None] = mapped_column(Date, nullable=True)
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    __table_args__ = (UniqueConstraint("match_id", "model_version", "predicted_at", name="uq_model_prediction"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(Integer, ForeignKey("matches.id"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    p_home: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    p_draw: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    p_away: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    ev_home: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    ev_draw: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    ev_away: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    edge_home: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    edge_draw: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    edge_away: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    is_calibrated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class ModelParams(Base):
    __tablename__ = "model_params"
    __table_args__ = (
        UniqueConstraint("league_id", "model_version", "train_until", name="uq_model_params_scope"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[str] = mapped_column(String(10), ForeignKey("leagues.id"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    train_until: Mapped[date] = mapped_column(Date, nullable=False)
    params: Mapped[dict] = mapped_column(JSONB, nullable=False)
    brier_score: Mapped[Decimal | None] = mapped_column(Numeric(6, 5), nullable=True)
    n_matches: Mapped[int | None] = mapped_column(Integer, nullable=True)


class ParlayPlanModel(Base):
    __tablename__ = "parlay_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_date: Mapped[date] = mapped_column(Date, nullable=False)
    tier: Mapped[str] = mapped_column(String(20), nullable=False)
    legs: Mapped[list[dict]] = mapped_column(JSONB, nullable=False)
    total_odds: Mapped[Decimal | None] = mapped_column(Numeric(8, 3), nullable=True)
    win_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    expected_ev: Mapped[Decimal | None] = mapped_column(Numeric(6, 4), nullable=True)
    kelly_pct: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    stake: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    is_simulation: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class BetResult(Base):
    __tablename__ = "bet_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[int] = mapped_column(Integer, ForeignKey("parlay_plans.id"), nullable=False)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    won: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    payout: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    profit: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
