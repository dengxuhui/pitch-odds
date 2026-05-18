"""config.py — 统一配置与密钥入口

所有环境变量从此处读取，不在其他模块中直接调用 os.getenv。
密钥永远不写入代码，只存在于本地 .env 文件（已被 .gitignore 排除）。

使用方法：
    from config import get_odds_api_key, get_database_url

首次使用前将 .env.example 复制为 .env，填入真实密钥：
    cp .env.example .env
"""
from __future__ import annotations

import os
from pathlib import Path

# 自动加载项目根目录的 .env 文件（仅本地开发使用，不影响 CI/生产）
try:
    from dotenv import load_dotenv

    _root = Path(__file__).resolve().parent
    load_dotenv(_root / ".env", override=False)
except ImportError:
    pass  # python-dotenv 未安装时跳过，依赖系统环境变量


# ──────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────

def _require(key: str) -> str:
    """读取必须存在的环境变量，缺失时给出明确提示。"""
    val = os.getenv(key, "").strip()
    if not val:
        raise RuntimeError(
            f"\n[配置错误] 缺少环境变量 {key}\n"
            f"请将 .env.example 复制为 .env，并填入真实值：\n"
            f"    cp .env.example .env\n"
            f"然后编辑 .env，设置 {key}=<你的密钥>\n"
        )
    return val


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ──────────────────────────────────────────────
# 数据库 / 缓存
# ──────────────────────────────────────────────

def get_database_url() -> str:
    return _optional(
        "DATABASE_URL",
        "postgresql+psycopg://pitch_odds:pitch_odds@localhost:5432/pitch_odds",
    )


def get_redis_url() -> str:
    return _optional("REDIS_URL", "redis://localhost:6379/0")


# ──────────────────────────────────────────────
# 外部 API 密钥
# ──────────────────────────────────────────────

def get_odds_api_key() -> str:
    """The Odds API 密钥（必填）。
    注册地址：https://the-odds-api.com/
    免费额度：500 credits/月，每月 1 日重置。
    """
    return _require("ODDS_API_KEY")


def get_football_data_api_key() -> str | None:
    """football-data.co.uk API key（可选）。
    该网站 CSV 下载无需认证，此字段仅供未来扩展使用。
    """
    val = _optional("FOOTBALL_DATA_API_KEY")
    return val if val else None


# ──────────────────────────────────────────────
# 应用环境
# ──────────────────────────────────────────────

def is_development() -> bool:
    return _optional("APP_ENV", "development").lower() == "development"
