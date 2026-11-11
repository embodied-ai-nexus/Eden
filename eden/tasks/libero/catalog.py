"""Immutable benchmark metadata parsed from upstream LIBERO task files.

These dataclasses are intentionally not Eden ``Options``. They represent the
static upstream task catalog and parsed BDDL structure that the Eden config
layer consumes when it builds runnable LIBERO environments.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from eden.options.objects.libero import LiberoObjectOptions

# Official continual-learning permutations for each 10-task suite. These are catalog data, not tunable task physics.
_TASK_ORDERS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (4, 6, 8, 7, 3, 1, 2, 0, 9, 5),
    (6, 3, 5, 0, 4, 2, 9, 1, 8, 7),
    (7, 4, 3, 0, 8, 1, 2, 5, 9, 6),
    (4, 5, 6, 3, 8, 0, 2, 7, 1, 9),
    (1, 2, 3, 0, 6, 9, 5, 7, 4, 8),
    (3, 7, 8, 1, 6, 2, 9, 4, 0, 5),
    (4, 2, 9, 7, 6, 8, 5, 1, 3, 0),
    (1, 8, 5, 4, 0, 9, 6, 7, 2, 3),
    (8, 3, 6, 4, 9, 5, 1, 2, 0, 7),
    (6, 9, 0, 5, 7, 1, 2, 8, 3, 4),
    (6, 8, 3, 1, 0, 2, 5, 9, 7, 4),
    (8, 0, 6, 9, 4, 1, 7, 3, 2, 5),
    (3, 8, 6, 4, 2, 5, 0, 7, 1, 9),
    (7, 1, 5, 6, 3, 2, 8, 9, 4, 0),
    (2, 0, 9, 5, 3, 6, 8, 7, 1, 4),
    (3, 5, 9, 6, 2, 4, 8, 7, 1, 0),
    (7, 6, 5, 9, 0, 3, 4, 2, 8, 1),
    (2, 5, 0, 9, 3, 1, 6, 4, 8, 7),
    (3, 5, 1, 2, 7, 8, 6, 0, 4, 9),
    (3, 4, 1, 9, 7, 6, 8, 2, 0, 5),
)

_TASKS_BY_SUITE: dict[str, tuple[str, ...]] = {
    "libero_spatial": (
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ),
    "libero_object": (
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
    ),
    "libero_goal": (
        "open_the_middle_drawer_of_the_cabinet",
        "put_the_bowl_on_the_stove",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_bowl_on_the_plate",
        "put_the_wine_bottle_on_the_rack",
    ),
    "libero_10": (
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ),
    "libero_90": (
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
        "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it",
        "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
        "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
        "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
        "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
        "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
        "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
        "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
        "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
        "KITCHEN_SCENE3_turn_on_the_stove",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it",
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
        "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
        "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
        "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
        "KITCHEN_SCENE6_close_the_microwave",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
        "KITCHEN_SCENE7_open_the_microwave",
        "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
        "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate",
        "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
        "KITCHEN_SCENE8_turn_off_the_stove",
        "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
        "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
        "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
        "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
        "KITCHEN_SCENE9_turn_on_the_stove",
        "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it",
        "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
        "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
        "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
        "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
        "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
        "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf",
    ),
}


@dataclass(frozen=True)
class LiberoRegionSpec:
    """Parsed BDDL region definition for a single task."""

    local_name: str
    target: str
    ranges: tuple[tuple[float, float, float, float], ...] = ()


@dataclass(frozen=True)
class LiberoPredicate:
    """Single logical predicate from a LIBERO initial or goal state."""

    relation: str
    entity_name: str
    target_name: str | None = None

    @property
    def key(self) -> str:
        """Return a stable diagnostic key for this predicate."""
        target = "" if self.target_name is None else self.target_name
        return f"{self.relation}:{self.entity_name}:{target}"


@dataclass(frozen=True)
class LiberoTaskSpec:
    """Parsed upstream LIBERO task definition before conversion to Eden config.

    ``initial_state`` preserves the symbolic BDDL predicates for catalog use.
    Runtime reset uses the separate numeric fixed-state bundle.
    """

    name: str
    suite_name: str
    problem_name: str
    language_instruction: str
    entity_categories: Mapping[str, str]
    fixture_names: frozenset[str]
    regions: Mapping[str, LiberoRegionSpec]
    initial_state: tuple[LiberoPredicate, ...]
    goal_state: tuple[LiberoPredicate, ...]


def _resolve_bddl_path(
    suite_name: str,
    task_name: str,
) -> Path:
    relative_path = Path("bddl_files") / suite_name / f"{task_name}.bddl"
    return Path(LiberoObjectOptions.fetch_family_asset(str(relative_path)))


def _parse_sexpr(tokens: list[str]):
    if not tokens:
        raise ValueError("Unexpected end of BDDL expression")
    token = tokens.pop(0)
    if token == ")":
        raise ValueError("Unexpected closing parenthesis in BDDL expression")
    if token != "(":
        return token
    expr = []
    while True:
        if not tokens:
            raise ValueError("Unclosed parenthesis in BDDL expression")
        if tokens[0] == ")":
            break
        expr.append(_parse_sexpr(tokens))
    tokens.pop(0)
    return expr


def _parse_typed_instances(items: list) -> dict[str, tuple[str, ...]]:
    typed_instances: dict[str, list[str]] = {}
    buffer: list[str] = []
    index = 0
    while index < len(items):
        token = items[index]
        if token == "-":
            category_name = items[index + 1]
            typed_instances.setdefault(category_name, []).extend(buffer)
            buffer = []
            index += 2
            continue
        buffer.append(token)
        index += 1
    if buffer:
        typed_instances.setdefault("object", []).extend(buffer)
    return {category_name: tuple(instances) for category_name, instances in typed_instances.items()}


def _parse_regions(items: list) -> dict[str, LiberoRegionSpec]:
    regions: dict[str, LiberoRegionSpec] = {}
    for region_def in items:
        local_name = region_def[0]
        target = ""
        ranges: tuple[tuple[float, float, float, float], ...] = ()
        for attr in region_def[1:]:
            tag = attr[0]
            if tag == ":target":
                target = attr[1]
            elif tag == ":ranges":
                ranges = tuple(tuple(float(value) for value in rect) for rect in attr[1])
        full_name = f"{target}_{local_name}" if target else local_name
        regions[full_name] = LiberoRegionSpec(
            local_name=local_name,
            target=target,
            ranges=ranges,
        )
    return regions


def _make_predicate(pred: list) -> LiberoPredicate:
    if len(pred) in (2, 3) and all(isinstance(token, str) for token in pred):
        return LiberoPredicate(*pred)
    raise ValueError(f"Unsupported predicate: {pred}")


def _flatten_goal(expr) -> tuple[LiberoPredicate, ...]:
    if not isinstance(expr, list) or not expr:
        raise ValueError(f"Expected a non-empty goal expression, got {expr!r}")
    if expr[0] == "And":
        if len(expr) == 1:
            raise ValueError("Goal conjunction must contain at least one predicate")
        predicates: list[LiberoPredicate] = []
        for subexpr in expr[1:]:
            predicates.extend(_flatten_goal(subexpr))
        return tuple(predicates)
    return (_make_predicate(expr),)


def _parse_bddl_file(bddl_path: Path, suite_name: str) -> LiberoTaskSpec:
    tokens = bddl_path.read_text(encoding="utf-8").replace("(", " ( ").replace(")", " ) ").split()
    sexpr = _parse_sexpr(tokens)
    if tokens:
        raise ValueError(f"Unexpected trailing BDDL expression in {bddl_path}")
    if not isinstance(sexpr, list) or not sexpr or sexpr[0] != "define":
        raise ValueError(f"Unexpected BDDL root in {bddl_path}")

    problem_name: str | None = None
    language_instruction: str | None = None
    fixtures: dict[str, tuple[str, ...]] = {}
    objects: dict[str, tuple[str, ...]] = {}
    regions: dict[str, LiberoRegionSpec] = {}
    initial_state: tuple[LiberoPredicate, ...] = ()
    goal_state: tuple[LiberoPredicate, ...] | None = None

    for section in sexpr[1:]:
        if not section:
            continue
        head = section[0]
        if head == "problem":
            if len(section) != 2 or not isinstance(section[1], str):
                raise ValueError(f"Invalid problem declaration in {bddl_path}: {section!r}")
            problem_name = section[1]
        elif head == ":language":
            if len(section) == 1 or not all(isinstance(token, str) for token in section[1:]):
                raise ValueError(f"Invalid language declaration in {bddl_path}: {section!r}")
            language_instruction = " ".join(section[1:])
        elif head == ":fixtures":
            fixtures = _parse_typed_instances(section[1:])
        elif head == ":objects":
            objects = _parse_typed_instances(section[1:])
        elif head == ":regions":
            regions = _parse_regions(section[1:])
        elif head == ":init":
            initial_state = tuple(_make_predicate(predicate) for predicate in section[1:])
        elif head == ":goal":
            if len(section) != 2:
                raise ValueError(f"Invalid goal declaration in {bddl_path}: {section!r}")
            try:
                goal_state = _flatten_goal(section[1])
            except ValueError as exc:
                raise ValueError(f"Invalid goal declaration in {bddl_path}: {exc}") from exc

    if problem_name is None or language_instruction is None or goal_state is None:
        missing_sections = [
            section_name
            for section_name, section_value in (
                ("problem", problem_name),
                (":language", language_instruction),
                (":goal", goal_state),
            )
            if section_value is None
        ]
        raise ValueError(f"BDDL file {bddl_path} is missing required sections: {', '.join(missing_sections)}")

    entity_categories = {
        instance_name: category_name
        for typed_instances in (fixtures, objects)
        for category_name, instances in typed_instances.items()
        for instance_name in instances
    }
    return LiberoTaskSpec(
        name=bddl_path.stem,
        suite_name=suite_name,
        problem_name=problem_name,
        language_instruction=language_instruction,
        entity_categories=MappingProxyType(entity_categories),
        fixture_names=frozenset(instance_name for instances in fixtures.values() for instance_name in instances),
        regions=MappingProxyType(regions),
        initial_state=initial_state,
        goal_state=goal_state,
    )


def get_task_names() -> tuple[str, ...]:
    """Return all 130 catalog task names in canonical suite order."""
    return tuple(task_name for task_names in _TASKS_BY_SUITE.values() for task_name in task_names)


def get_suite_names() -> tuple[str, ...]:
    """Return the discoverable suite names, including the derived LIBERO-100 union."""
    return (*_TASKS_BY_SUITE, "libero_100")


@functools.cache
def get_task_spec(task_name: str) -> LiberoTaskSpec:
    """Resolve, parse, and memoize one task without loading unrelated BDDL files."""
    suite_name = next((name for name, task_names in _TASKS_BY_SUITE.items() if task_name in task_names), None)
    if suite_name is None:
        raise KeyError(f"Unknown LIBERO task: {task_name!r}")
    bddl_path = _resolve_bddl_path(suite_name, task_name)
    return _parse_bddl_file(bddl_path, suite_name=suite_name)


def get_suite_task_names(suite_name: str, task_order_index: int = 0) -> tuple[str, ...]:
    if suite_name == "libero_100":
        return get_suite_task_names("libero_90", task_order_index=0) + get_suite_task_names(
            "libero_10",
            task_order_index=task_order_index,
        )
    try:
        task_names = _TASKS_BY_SUITE[suite_name]
    except KeyError as exc:
        raise KeyError(f"Unknown LIBERO suite: {suite_name!r}") from exc
    if suite_name == "libero_90":
        if task_order_index != 0:
            raise IndexError("libero_90 supports only task_order_index=0")
        return task_names
    if not 0 <= task_order_index < len(_TASK_ORDERS):
        raise IndexError(
            f"task_order_index={task_order_index} out of range [0, {len(_TASK_ORDERS)}) for suite {suite_name!r}"
        )
    order = _TASK_ORDERS[task_order_index]
    return tuple(task_names[index] for index in order)


__all__ = [
    "LiberoPredicate",
    "LiberoRegionSpec",
    "LiberoTaskSpec",
    "get_suite_names",
    "get_suite_task_names",
    "get_task_names",
    "get_task_spec",
]
