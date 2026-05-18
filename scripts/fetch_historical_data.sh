#!/usr/bin/env bash
# fetch_historical_data.sh — 从 football-data.co.uk 下载五大联赛历史 CSV
#
# 用法：
#   bash scripts/fetch_historical_data.sh                     # 下载所有联赛
#   bash scripts/fetch_historical_data.sh --leagues E0 SP1    # 只下载指定联赛
#   bash scripts/fetch_historical_data.sh --import            # 下载并自动导入数据库
#
# 数据来源：https://www.football-data.co.uk/data.php

set -euo pipefail

BASE_URL="https://www.football-data.co.uk/mmz4281"
OUTPUT_BASE="data/samples"
SEASONS=(1819 1920 2021 2122 2223 2324 2425)

# 联赛 ID → 子目录名
declare -A LEAGUE_DIR=(
  [E0]="e0"
  [SP1]="sp1"
  [D1]="d1"
  [I1]="i1"
  [F1]="f1"
)

# 联赛 ID → 赛季编码对应的 season 参数（格式 20XX-XX）
declare -A SEASON_MAP=(
  [1819]="2018-19"
  [1920]="2019-20"
  [2021]="2020-21"
  [2122]="2021-22"
  [2223]="2022-23"
  [2324]="2023-24"
  [2425]="2024-25"
)

# ── 参数解析 ───────────────────────────────────────────────
SELECTED_LEAGUES=()
DO_IMPORT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --leagues)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        SELECTED_LEAGUES+=("$1")
        shift
      done
      ;;
    --import)
      DO_IMPORT=true
      shift
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# 默认下载全部五大联赛
if [[ ${#SELECTED_LEAGUES[@]} -eq 0 ]]; then
  SELECTED_LEAGUES=(E0 SP1 D1 I1 F1)
fi

# ── 下载 ───────────────────────────────────────────────────
TOTAL_FILES=0
FAILED_FILES=0

for league in "${SELECTED_LEAGUES[@]}"; do
  if [[ -z "${LEAGUE_DIR[$league]+_}" ]]; then
    echo "⚠️  unknown league: $league (supported: E0 SP1 D1 I1 F1)" >&2
    continue
  fi

  dir="${OUTPUT_BASE}/${LEAGUE_DIR[$league]}"
  mkdir -p "$dir"

  for season in "${SEASONS[@]}"; do
    target="${dir}/${league}_${season}.csv"
    url="${BASE_URL}/${season}/${league}.csv"

    if [[ -f "$target" ]]; then
      echo "skip (exists): $target"
      continue
    fi

    echo "downloading ${url} -> ${target}"
    if curl -k --retry 5 --retry-delay 2 -fsSL "$url" -o "$target"; then
      TOTAL_FILES=$((TOTAL_FILES + 1))
    else
      echo "⚠️  failed: $url" >&2
      FAILED_FILES=$((FAILED_FILES + 1))
      rm -f "$target"
    fi
  done
done

echo ""
echo "download done: ${TOTAL_FILES} files fetched, ${FAILED_FILES} failed"

# ── 可选：导入数据库 ───────────────────────────────────────
if [[ "$DO_IMPORT" == true ]]; then
  echo ""
  echo "importing into database..."
  IMPORT_OK=0
  IMPORT_FAIL=0

  for league in "${SELECTED_LEAGUES[@]}"; do
    if [[ -z "${LEAGUE_DIR[$league]+_}" ]]; then continue; fi
    dir="${OUTPUT_BASE}/${LEAGUE_DIR[$league]}"

    for season in "${SEASONS[@]}"; do
      csv="${dir}/${league}_${season}.csv"
      season_str="${SEASON_MAP[$season]}"

      if [[ ! -f "$csv" ]]; then continue; fi

      echo "importing $csv (league=${league} season=${season_str})"
      if PYTHONPATH="$(pwd)" python3 -m data.collectors.historical \
          "$csv" --league-id "$league" --season "$season_str"; then
        IMPORT_OK=$((IMPORT_OK + 1))
      else
        echo "⚠️  import failed: $csv" >&2
        IMPORT_FAIL=$((IMPORT_FAIL + 1))
      fi

      echo "importing odds $csv (league=${league} season=${season_str})"
      if PYTHONPATH="$(pwd)" python3 -m data.collectors.odds \
          "$csv" --league-id "$league" --season "$season_str" --bookmaker bet365; then
        IMPORT_OK=$((IMPORT_OK + 1))
      else
        echo "⚠️  odds import failed: $csv" >&2
        IMPORT_FAIL=$((IMPORT_FAIL + 1))
      fi
    done
  done

  echo ""
  echo "import done: ${IMPORT_OK} succeeded, ${IMPORT_FAIL} failed"
fi
