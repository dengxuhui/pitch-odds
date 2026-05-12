CREATE TABLE IF NOT EXISTS leagues (
    id          VARCHAR(10) PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    country     VARCHAR(50) NOT NULL,
    avg_goals   NUMERIC(4,2) NOT NULL DEFAULT 2.70,
    home_adv    NUMERIC(4,3) NOT NULL DEFAULT 1.080
);

CREATE TABLE IF NOT EXISTS teams (
    id          SERIAL PRIMARY KEY,
    league_id   VARCHAR(10) NOT NULL REFERENCES leagues(id),
    name        VARCHAR(100) NOT NULL,
    short_name  VARCHAR(20),
    CONSTRAINT uq_team_league_name UNIQUE (league_id, name)
);

CREATE TABLE IF NOT EXISTS matches (
    id              SERIAL PRIMARY KEY,
    league_id       VARCHAR(10) NOT NULL REFERENCES leagues(id),
    season          VARCHAR(10) NOT NULL,
    match_date      DATE NOT NULL,
    match_week      SMALLINT,
    home_team_id    INT NOT NULL REFERENCES teams(id),
    away_team_id    INT NOT NULL REFERENCES teams(id),
    home_goals      SMALLINT,
    away_goals      SMALLINT,
    result          CHAR(1),
    home_shots      SMALLINT,
    away_shots      SMALLINT,
    home_shots_on   SMALLINT,
    away_shots_on   SMALLINT,
    home_possession NUMERIC(4,1),
    away_possession NUMERIC(4,1),
    home_corners    SMALLINT,
    away_corners    SMALLINT,
    home_yellow     SMALLINT,
    away_yellow     SMALLINT,
    home_red        SMALLINT,
    away_red        SMALLINT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_match_identity UNIQUE (league_id, season, match_date, home_team_id, away_team_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_home ON matches(home_team_id, match_date);
CREATE INDEX IF NOT EXISTS idx_matches_away ON matches(away_team_id, match_date);
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_id, season);

CREATE TABLE IF NOT EXISTS odds_opening (
    id          SERIAL PRIMARY KEY,
    match_id    INT NOT NULL REFERENCES matches(id),
    bookmaker   VARCHAR(50) NOT NULL,
    odds_home   NUMERIC(6,3) NOT NULL,
    odds_draw   NUMERIC(6,3) NOT NULL,
    odds_away   NUMERIC(6,3) NOT NULL,
    overround   NUMERIC(5,4),
    recorded_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_opening_match_bookmaker UNIQUE (match_id, bookmaker)
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id              SERIAL PRIMARY KEY,
    match_id        INT NOT NULL REFERENCES matches(id),
    bookmaker       VARCHAR(50) NOT NULL,
    odds_home       NUMERIC(6,3) NOT NULL,
    odds_draw       NUMERIC(6,3) NOT NULL,
    odds_away       NUMERIC(6,3) NOT NULL,
    overround       NUMERIC(5,4),
    snapshot_at     TIMESTAMPTZ NOT NULL,
    hours_to_kick   NUMERIC(5,2)
);

CREATE INDEX IF NOT EXISTS idx_odds_match ON odds_snapshots(match_id, snapshot_at);
CREATE INDEX IF NOT EXISTS idx_odds_hours ON odds_snapshots(match_id, hours_to_kick);

CREATE TABLE IF NOT EXISTS odds_anomalies (
    id              SERIAL PRIMARY KEY,
    match_id        INT NOT NULL REFERENCES matches(id),
    alert_level     VARCHAR(10) NOT NULL,
    anomaly_type    VARCHAR(30),
    max_step_change NUMERIC(5,4),
    total_drift_pct NUMERIC(5,2),
    exclude_flag    BOOLEAN NOT NULL DEFAULT FALSE,
    detected_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS team_status (
    id               SERIAL PRIMARY KEY,
    team_id          INT NOT NULL REFERENCES teams(id),
    as_of_date       DATE NOT NULL,
    form_score_5     NUMERIC(4,3),
    form_score_10    NUMERIC(4,3),
    fatigue_index    NUMERIC(4,3),
    injury_impact    NUMERIC(4,3),
    momentum_score   NUMERIC(4,3),
    matches_last_30d SMALLINT,
    travel_km        INT,
    missing_players  JSONB,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_team_status_date UNIQUE (team_id, as_of_date)
);

CREATE TABLE IF NOT EXISTS player_injuries (
    id                  SERIAL PRIMARY KEY,
    team_id             INT NOT NULL REFERENCES teams(id),
    player_name         VARCHAR(100) NOT NULL,
    injury_type         VARCHAR(100),
    status              VARCHAR(20) NOT NULL,
    importance          NUMERIC(3,2),
    position_multiplier NUMERIC(3,2),
    reported_at         DATE NOT NULL,
    expected_return     DATE,
    source              VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id            SERIAL PRIMARY KEY,
    match_id      INT NOT NULL REFERENCES matches(id),
    model_version VARCHAR(50) NOT NULL,
    predicted_at  TIMESTAMPTZ NOT NULL,
    p_home        NUMERIC(5,4) NOT NULL,
    p_draw        NUMERIC(5,4) NOT NULL,
    p_away        NUMERIC(5,4) NOT NULL,
    ev_home       NUMERIC(6,4),
    ev_draw       NUMERIC(6,4),
    ev_away       NUMERIC(6,4),
    edge_home     NUMERIC(5,4),
    edge_draw     NUMERIC(5,4),
    edge_away     NUMERIC(5,4),
    is_calibrated BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT uq_model_prediction UNIQUE (match_id, model_version, predicted_at)
);

CREATE TABLE IF NOT EXISTS parlay_plans (
    id            SERIAL PRIMARY KEY,
    plan_date     DATE NOT NULL,
    tier          VARCHAR(20) NOT NULL,
    legs          JSONB NOT NULL,
    total_odds    NUMERIC(8,3),
    win_rate      NUMERIC(5,4),
    expected_ev   NUMERIC(6,4),
    kelly_pct     NUMERIC(5,4),
    stake         NUMERIC(10,2),
    is_simulation BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bet_results (
    id         SERIAL PRIMARY KEY,
    plan_id    INT NOT NULL REFERENCES parlay_plans(id),
    settled_at TIMESTAMPTZ,
    won        BOOLEAN,
    payout     NUMERIC(10,2),
    profit     NUMERIC(10,2)
);
