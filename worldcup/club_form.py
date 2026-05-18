from __future__ import annotations

from collections import defaultdict


_MAX_ELO_ADJ = 50.0   # 俱乐部状态最多调整 ±50 Elo 点


class ClubFormMapper:
    """将国家队球员的俱乐部近况映射为国家队 Elo 调整值。

    工作流程：
    1. 赛前调用 add_player(team_id, form_score) 登记每名球员的俱乐部近况分
       （form_score 来自 data.processors.form_score，范围约 [-1, +1]）。
    2. 调用 get_adjustment(team_id) 获取该国家队的 Elo 调整量（[-50, +50]）。
    3. 在 WorldCupModel.predict() 中将调整量叠加到原始 Elo 评分。

    调整公式：
        avg_form = mean(club_form_scores for all registered players)
        elo_adj = avg_form × MAX_ELO_ADJ  （截断至 [-50, +50]）
    """

    def __init__(self) -> None:
        self._team_forms: dict[int, list[float]] = defaultdict(list)

    def add_player(self, team_id: int, club_form_score: float) -> None:
        """登记一名球员的俱乐部近况分数。

        Args:
            team_id:         国家队 ID（与 EloRating 使用同一命名空间）。
            club_form_score: 俱乐部近况分数，范围约 [-1, +1]。
        """
        self._team_forms[team_id].append(float(club_form_score))

    def get_adjustment(self, team_id: int) -> float:
        """返回该国家队的 Elo 调整值（范围 [-50, +50]）。

        未登记球员的球队返回 0.0。
        """
        forms = self._team_forms.get(team_id, [])
        if not forms:
            return 0.0
        avg = sum(forms) / len(forms)
        adj = avg * _MAX_ELO_ADJ
        return max(-_MAX_ELO_ADJ, min(_MAX_ELO_ADJ, adj))

    def clear(self) -> None:
        """清空所有登记记录。"""
        self._team_forms.clear()

    def team_ids(self) -> list[int]:
        """返回已登记球员的国家队 ID 列表。"""
        return list(self._team_forms)
