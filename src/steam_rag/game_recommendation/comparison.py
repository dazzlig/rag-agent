"""Same-axis comparison of two or three candidates.

기획안 §4.3 / §4.5: "비교 축은 장르명보다 전투 조작, 화면 시점, 스토리 진행,
반복 요소, 난도, 협동 방식처럼 실제 경험을 중심으로 한다."

축마다 게임별 값을 같은 형식으로 뽑고, 근거가 없는 축은 값을 지어내지 않고
``미확인``으로 남긴다. 비교 결과는 화면과 답변 Agent가 함께 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from steam_rag.game_metadata.playstyle import coerce_list
from steam_rag.game_recommendation.constraints import VALUE_LABELS


@dataclass(frozen=True, slots=True)
class ComparisonAxis:
    """One experience-level axis and the profile fields that can answer it."""

    axis_id: str
    label: str
    facet_field: str
    values: tuple[str, ...]
    tags: tuple[str, ...] = ()
    question: str = ""


COMPARISON_AXES: tuple[ComparisonAxis, ...] = (
    ComparisonAxis(
        axis_id="combat_control",
        label="전투 조작",
        facet_field="combat_facets",
        values=(
            "real_time",
            "turn_based",
            "direct_control",
            "command_based",
            "auto_combat",
            "tactical",
            "party_based",
            "shooter",
            "stealth",
        ),
        question="직접 조작하는 전투인지, 명령·턴 기반인지",
    ),
    ComparisonAxis(
        axis_id="perspective",
        label="화면 시점",
        facet_field="perspective_facets",
        values=("first_person", "third_person", "side_view", "top_down", "isometric"),
        question="어떤 시점으로 플레이하는지",
    ),
    ComparisonAxis(
        axis_id="dimension",
        label="표현 방식",
        facet_field="dimension_facets",
        values=("2d", "2_5d", "3d", "vr"),
        question="2D인지 3D인지",
    ),
    ComparisonAxis(
        axis_id="story",
        label="스토리 진행",
        facet_field="playstyle_facets",
        values=("story_rich", "choices_matter"),
        tags=("story_rich", "choices_matter", "visual_novel"),
        question="이야기 중심 진행인지",
    ),
    ComparisonAxis(
        axis_id="repetition",
        label="반복 요소",
        facet_field="playstyle_facets",
        values=("roguelike", "hunting", "survival", "crafting", "character_progression"),
        tags=("roguelike", "roguelite", "grinding", "hunting", "farming_sim"),
        question="반복 플레이 비중이 큰지",
    ),
    ComparisonAxis(
        axis_id="difficulty",
        label="난도 성향",
        facet_field="playstyle_facets",
        values=("souls_like",),
        tags=("difficult", "souls_like", "bullet_hell", "precision_platformer"),
        question="높은 난도를 전제로 하는지",
    ),
    ComparisonAxis(
        axis_id="co_op",
        label="협동 방식",
        facet_field="playstyle_facets",
        values=("co_op", "multiplayer", "online_multiplayer", "pvp"),
        tags=("co_op", "online_co_op", "local_co_op", "multiplayer", "singleplayer"),
        question="혼자 하는 게임인지, 함께 하는 게임인지",
    ),
)

UNVERIFIED_LABEL = "미확인"


@dataclass(slots=True)
class AxisCell:
    appid: int
    name: str
    values: list[str] = field(default_factory=list)
    verified: bool = False
    evidence: list[str] = field(default_factory=list)

    @property
    def display(self) -> str:
        return " · ".join(self.values) if self.values else UNVERIFIED_LABEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "appid": self.appid,
            "name": self.name,
            "values": self.values,
            "display": self.display,
            "verified": self.verified,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class AxisComparison:
    axis_id: str
    label: str
    question: str
    cells: list[AxisCell] = field(default_factory=list)

    @property
    def differs(self) -> bool:
        """True when the verified games do not share the same value set."""

        verified = [tuple(sorted(cell.values)) for cell in self.cells if cell.verified]
        return len(set(verified)) > 1

    @property
    def unverified_games(self) -> list[str]:
        return [cell.name for cell in self.cells if not cell.verified]

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "label": self.label,
            "question": self.question,
            "differs": self.differs,
            "unverified_games": self.unverified_games,
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass(slots=True)
class ComparisonTable:
    games: list[dict[str, Any]]
    axes: list[AxisComparison]

    @property
    def differing_axes(self) -> list[AxisComparison]:
        return [axis for axis in self.axes if axis.differs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "axes": [axis.to_dict() for axis in self.axes],
            "differing_axis_ids": [axis.axis_id for axis in self.differing_axes],
            "unverified_axis_ids": [
                axis.axis_id for axis in self.axes if axis.unverified_games
            ],
        }


def compare_profiles(
    profiles: Sequence[dict[str, Any]],
    *,
    axes: Iterable[ComparisonAxis] = COMPARISON_AXES,
) -> ComparisonTable:
    """Build the fixed-axis comparison used by the 비교 화면."""

    selected = [profile for profile in profiles if isinstance(profile, dict)][:3]
    games = [
        {
            "appid": _int(profile.get("appid")),
            "name": str(profile.get("name") or profile.get("game_key") or ""),
            "header_image": profile.get("header_image") or "",
        }
        for profile in selected
    ]
    table_axes: list[AxisComparison] = []
    for axis in axes:
        comparison = AxisComparison(axis.axis_id, axis.label, axis.question)
        for profile, game in zip(selected, games):
            comparison.cells.append(_axis_cell(axis, profile, game))
        table_axes.append(comparison)
    return ComparisonTable(games=games, axes=table_axes)


def _axis_cell(axis: ComparisonAxis, profile: dict[str, Any], game: dict[str, Any]) -> AxisCell:
    facets = set(coerce_list(profile.get(axis.facet_field)))
    tags = _profile_tags(profile)
    matched_facets = [value for value in axis.values if value in facets]
    matched_tags = [tag for tag in axis.tags if tag in tags and tag not in matched_facets]
    values = [VALUE_LABELS.get(item, item.replace("_", " ")) for item in matched_facets]
    values.extend(VALUE_LABELS.get(item, item.replace("_", " ")) for item in matched_tags)

    evidence: list[str] = []
    if matched_facets:
        evidence.append(f"{axis.facet_field}={','.join(matched_facets)}")
    if matched_tags:
        evidence.append(f"steam_popular_tags={','.join(matched_tags)}")

    # 축에 해당하는 원본 항목이 아예 비어 있으면 "값 없음"이 아니라 "확인하지 못함"이다.
    field_populated = bool(facets) or bool(tags)
    return AxisCell(
        appid=game["appid"],
        name=game["name"],
        values=_dedupe(values),
        verified=bool(values) or (field_populated and axis.axis_id in _CLOSED_AXES),
        evidence=evidence,
    )


#: 프로필에 분류 정보가 있으면 "해당 없음"도 확인된 사실로 볼 수 있는 축.
_CLOSED_AXES = frozenset({"combat_control", "perspective", "dimension"})


def comparison_markdown(table: ComparisonTable) -> str:
    """Deterministic fallback rendering used when the answer Agent is unavailable."""

    if len(table.games) < 2:
        return "비교하려면 게임을 2개 이상 선택해 주세요."
    header = " | ".join(["비교 축", *(game["name"] for game in table.games)])
    divider = " | ".join(["---"] * (len(table.games) + 1))
    lines = ["## 같은 기준으로 비교", "", f"| {header} |", f"| {divider} |"]
    for axis in table.axes:
        cells = " | ".join(cell.display for cell in axis.cells)
        lines.append(f"| {axis.label} | {cells} |")
    lines.append("")
    differing = [axis.label for axis in table.differing_axes]
    if differing:
        lines.append(f"선택을 바꿀 수 있는 차이: {', '.join(differing)}.")
    unverified = sorted({axis.label for axis in table.axes if axis.unverified_games})
    if unverified:
        lines.append(f"아직 확인하지 못한 축: {', '.join(unverified)}. 이 축은 추측하지 않았습니다.")
    return "\n".join(lines)


def _profile_tags(profile: dict[str, Any]) -> set[str]:
    ranked = {
        str(item.get("normalized"))
        for item in profile.get("popular_user_tags") or []
        if isinstance(item, dict) and item.get("normalized")
    }
    return ranked or set(coerce_list(profile.get("steam_tags_normalized")))


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
