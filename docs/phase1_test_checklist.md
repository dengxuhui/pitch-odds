# Phase 1 全流程测试清单

> 适用范围：`Phase 1（项目基础 + 数据管道）`  
> 目标：验证从环境到数据入库的端到端链路可复现、可回归、可幂等。

## 1. 前置条件

- 已安装 Docker / Docker Compose
- 已安装 Python 3.11+
- 已安装依赖：`pip install -r requirements.txt`
- 已准备 football-data CSV（示例：`E0_2024_25.csv`）

## 2. 一次完整执行（E2E）

### 2.1 启动服务

```bash
docker compose up -d
```

预期结果：
- PostgreSQL 15 与 Redis 7 容器处于 `running` 状态

### 2.2 初始化数据库结构

```bash
python3 scripts/init_db.py --mode alembic
```

预期结果：
- Alembic 迁移执行成功
- 不出现连接失败或 SQL 执行错误

### 2.3 导入历史比赛数据

```bash
python3 -m data.collectors.historical "/path/to/E0_2024_25.csv" --league-id E0 --season 2024-25
```

预期结果：
- 输出 `新增球队`、`新增比赛`、`跳过比赛`
- 首次导入 `新增比赛 > 0`

### 2.4 导入离线赔率数据

```bash
python3 -m data.collectors.odds "/path/to/E0_2024_25.csv" --league-id E0 --season 2024-25 --bookmaker bet365
```

预期结果：
- 输出 `新增赔率`、`缺失比赛`、`缺失赔率`、`已存在跳过`
- 首次导入 `新增赔率 > 0`

### 2.5 执行严格验收

```bash
python3 scripts/verify_phase1.py --league-id E0 --strict
```

预期结果：
- 表结构检查通过
- 显示球队数、比赛数
- 输出 `Phase 1 验收通过`

### 2.6 执行单元测试

```bash
python3 -m pytest
```

预期结果：
- 全量测试通过（当前基线：`12 passed`）

## 3. 幂等性回归测试

重复执行 `2.3` 和 `2.4`，然后再执行 `2.5`、`2.6`。

预期结果：
- 历史导入：`跳过比赛` 增加，不重复插入
- 赔率导入：`已存在跳过` 增加，不重复插入
- 验收与测试仍通过

## 4. 一键执行（推荐）

```bash
bash scripts/run_phase1_e2e.sh "/path/to/E0_2024_25.csv" --league-id E0 --season 2024-25 --bookmaker bet365
```

## 5. 常见失败与排查

- 数据库连接失败：检查 `docker compose ps` 与 `DATABASE_URL`
- 缺少 CSV 必要列：确认存在 `Date/HomeTeam/AwayTeam` 与赔率列（优先 `B365H/B365D/B365A`）
- `缺失比赛`过高：确认先执行历史导入，再执行赔率导入，且 `league-id/season` 一致
- 依赖模块缺失：重新执行 `pip install -r requirements.txt`
