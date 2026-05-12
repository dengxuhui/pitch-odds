#!/usr/bin/env bash

set -euo pipefail

COLIMA_IMAGE_URL="https://github.com/abiosoft/colima-core/releases/download/v0.10.1/ubuntu-24.04-minimal-cloudimg-arm64-docker.qcow2"

start_colima_or_exit() {
  if colima start --cpu 2 --memory 4 --disk 20; then
    return 0
  fi

  echo "Colima 启动失败。常见原因是网络无法访问 Colima 镜像下载地址。"
  if command -v curl >/dev/null 2>&1; then
    if ! curl -I --max-time 15 "$COLIMA_IMAGE_URL" >/dev/null 2>&1; then
      echo "网络连通性检查失败：无法访问 $COLIMA_IMAGE_URL"
      echo "请尝试："
      echo "1) 切换 VPN 节点或临时关闭 VPN 后重试"
      echo "2) 使用可访问 GitHub 的代理（确保终端已继承 HTTP(S)_PROXY）"
      echo "3) 先手动启动 Docker Desktop，再重跑本脚本"
    fi
  fi
  exit 1
}

ensure_docker_ready() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v colima >/dev/null 2>&1; then
    echo "Docker daemon 未就绪，且未检测到 colima。请先启动 Docker Desktop 或安装并启动 Colima。"
    exit 1
  fi

  echo "检测到 Docker daemon 未启动，尝试启动 Colima..."
  if colima status >/dev/null 2>&1; then
    echo "Colima 已运行，但 Docker daemon 仍不可用，继续等待..."
  else
    start_colima_or_exit
  fi

  for _ in {1..30}; do
    if docker info >/dev/null 2>&1; then
      echo "Docker daemon 已就绪。"
      return 0
    fi
    sleep 1
  done

  echo "Docker daemon 启动超时，请检查 colima status 或 Docker Desktop 状态。"
  exit 1
}

if [[ $# -lt 1 ]]; then
  echo "用法: bash scripts/run_phase1_e2e.sh <csv_path> [--league-id E0] [--season 2024-25] [--bookmaker bet365]"
  exit 1
fi

CSV_PATH="$1"
shift

LEAGUE_ID="E0"
SEASON=""
BOOKMAKER="bet365"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --league-id)
      LEAGUE_ID="$2"
      shift 2
      ;;
    --season)
      SEASON="$2"
      shift 2
      ;;
    --bookmaker)
      BOOKMAKER="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

SEASON_ARGS=()
if [[ -n "$SEASON" ]]; then
  SEASON_ARGS=(--season "$SEASON")
fi

echo "[1/6] 启动基础服务"
ensure_docker_ready
docker compose up -d

echo "[2/6] 初始化数据库"
python3 scripts/init_db.py --mode alembic

echo "[3/6] 导入历史比赛数据"
python3 -m data.collectors.historical "$CSV_PATH" --league-id "$LEAGUE_ID" "${SEASON_ARGS[@]}"

echo "[4/6] 导入离线赔率数据"
python3 -m data.collectors.odds "$CSV_PATH" --league-id "$LEAGUE_ID" --bookmaker "$BOOKMAKER" "${SEASON_ARGS[@]}"

echo "[5/6] 执行 Phase 1 严格验收"
PYTHONPATH="$(pwd)" python3 scripts/verify_phase1.py --league-id "$LEAGUE_ID" --strict

echo "[6/6] 运行测试"
python3 -m pytest

echo "Phase 1 E2E 流程执行完成"
