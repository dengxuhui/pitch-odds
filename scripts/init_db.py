from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]


def run_alembic_upgrade() -> None:
    cmd = [sys.executable, "-m", "alembic", "upgrade", "head"]
    subprocess.run(cmd, check=True, cwd=ROOT_DIR)


def run_schema_sql() -> None:
    schema_path = ROOT_DIR / "data" / "storage" / "schema.sql"
    print(f"已生成 schema 文件: {schema_path}")
    print("如需手动执行，请在 PostgreSQL 中运行该 SQL 文件。")


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化数据库结构")
    parser.add_argument(
        "--mode",
        choices=["alembic", "schema"],
        default="alembic",
        help="初始化方式：alembic(默认) 或 schema(仅提示 SQL 文件)",
    )
    args = parser.parse_args()

    if args.mode == "alembic":
        run_alembic_upgrade()
        print("数据库迁移已完成。")
    else:
        run_schema_sql()


if __name__ == "__main__":
    main()
