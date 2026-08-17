#!/usr/bin/env python3
"""Maximal Voxel Pruning solver for three-view cube puzzles.

Coordinates are centralized here and deliberately follow a physical convention:
``x`` increases left-to-right in the front view, ``y`` increases from the front
edge to the back edge, and ``z`` increases from the ground (z=1).  A front ray
therefore visits increasing y, and a left ray increasing x.  In a top matrix the
top row is the back edge; in a left matrix the left column is the back edge.
No input matrix is mutated, transposed, or mirrored behind the view adapters.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations, product
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

Coord = Tuple[int, int, int]
Base = Tuple[int, int]
Cell = str
VALID_CELLS = {"W", "B", "X"}
VALID_VIEWS = {"front", "left", "top"}


class PuzzleError(ValueError):
    pass


class SolverInconsistency(RuntimeError):
    pass


@dataclass(frozen=True)
class View:
    view: str
    cells: Tuple[Tuple[Cell, ...], ...]


@dataclass(frozen=True)
class Option:
    label: str
    cells: Tuple[Tuple[Cell, ...], ...]


@dataclass(frozen=True)
class Puzzle:
    white_count: int
    black_count: int
    views: Mapping[str, View]
    options: Tuple[Option, ...]
    # Kept only to make the non-use of fixture metadata explicit.
    answer: Optional[str] = None


@dataclass
class BlackRequirement:
    id: int
    view: str
    row: int
    col: int
    z: Optional[int]
    candidates: Set[Coord]
    representative: Coord


@dataclass
class BlackIdentity:
    """A relationship forced in every retained interpretation."""

    requirement_ids: Set[int]
    candidates: Set[Coord]
    representative: Optional[Coord] = None


@dataclass(frozen=True)
class BlackScenario:
    """A Step-5 identity/location choice with Step-6 blocker caps pending."""

    black_positions: FrozenSet[Coord]
    identity_groups: Tuple[FrozenSet[int], ...]
    requirement_positions: Tuple[Coord, ...]
    required_height_caps: Tuple[int, ...]


@dataclass(frozen=True)
class BlackVisibilityInterpretation:
    """A Step-6 branch after visibility-driven blocker removals."""

    heights: Tuple[int, ...]
    black_positions: FrozenSet[Coord]
    identity_groups: Tuple[FrozenSet[int], ...]
    requirement_positions: Tuple[Coord, ...]
    blocker_removals: FrozenSet[Coord]


@dataclass(frozen=True)
class GlobalInterpretation:
    """A correlated retained state, generated once from the shared solid."""

    heights: Tuple[int, ...]
    black_positions: FrozenSet[Coord]
    identity_groups: Tuple[FrozenSet[int], ...]


@dataclass(frozen=True)
class RemovalGroup:
    voxels: FrozenSet[Coord]
    remove_count: int
    reason: str
    # Per-column height alternatives retain stacking-aware multi-cube
    # reductions. The cardinality applies across the whole shared group.
    height_alternatives: Mapping[Base, FrozenSet[int]] = field(
        default_factory=dict)
    # Complete correlated height vectors; entries are in canonical base order.
    height_states: Tuple[Tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class StageSnapshot:
    stage: str
    heights: Mapping[Base, int]
    compatible_options: Tuple[str, ...]
    notes: Tuple[str, ...] = ()
    black_candidates: Tuple[Tuple[int, Tuple[Coord, ...]], ...] = ()
    black_scenario_count: int = 0
    black_visibility_interpretation_count: int = 0
    global_interpretation_count: int = 0
    black_identity_alternatives: Tuple[Tuple[FrozenSet[int], ...], ...] = ()
    forced_black: FrozenSet[Coord] = frozenset()
    possible_black: FrozenSet[Coord] = frozenset()
    forced_removals: FrozenSet[Coord] = frozenset()
    conditional_removals: FrozenSet[Coord] = frozenset()
    remaining_cube_count_range: Tuple[int, int] = (0, 0)
    front_heights: Tuple[int, ...] = ()
    left_heights: Tuple[int, ...] = ()
    footprint: Mapping[Base, bool] = field(default_factory=dict)


@dataclass
class SolveState:
    puzzle: Puzzle
    width: int
    depth: int
    height: int
    missing_view: str
    front_heights: List[int] = field(default_factory=list)
    left_heights: List[int] = field(default_factory=list)
    footprint: Dict[Base, bool] = field(default_factory=dict)
    # Literal Step-2 min(x, y) table, retained for explanation.
    initial_max_heights: Dict[Base, int] = field(default_factory=dict)
    # Current union envelope across branches; it need not itself be realizable.
    max_heights: Dict[Base, int] = field(default_factory=dict)
    option_consensus: Dict[Tuple[int, int],
                           FrozenSet[Cell]] = field(default_factory=dict)
    compatible_labels: List[str] = field(default_factory=list)
    black_requirements: List[BlackRequirement] = field(default_factory=list)
    black_identities: List[BlackIdentity] = field(default_factory=list)
    black_identity_alternatives: List[Tuple[FrozenSet[int], ...]] = field(
        default_factory=list)
    black_scenarios: List[BlackScenario] = field(default_factory=list)
    black_visibility_interpretations: List[BlackVisibilityInterpretation] = field(
        default_factory=list)
    global_interpretations: List[GlobalInterpretation] = field(
        default_factory=list)
    # Representatives are display-only; logical code uses the alternatives above.
    black_positions: Set[Coord] = field(default_factory=set)
    black_position_candidates: Set[Coord] = field(default_factory=set)
    definitely_black: Set[Coord] = field(default_factory=set)
    definitely_occupied: Set[Coord] = field(default_factory=set)
    possibly_occupied: Set[Coord] = field(default_factory=set)
    definitely_empty: Set[Coord] = field(default_factory=set)
    forced_removals: Set[Coord] = field(default_factory=set)
    conditional_removals: Set[Coord] = field(default_factory=set)
    maximal_voxels: Set[Coord] = field(default_factory=set)
    removal_groups: List[RemovalGroup] = field(default_factory=list)
    snapshots: List[StageSnapshot] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def occupied_voxels(self) -> Set[Coord]:
        return {
            (x, y, z)
            for (x, y), h in self.max_heights.items()
            for z in range(1, h + 1)
        }

    def white_count_range(self) -> Tuple[int, int]:
        if self.global_interpretations:
            # Step 7 retains exact-count interpretations only.
            return (self.puzzle.white_count, self.puzzle.white_count)
        total = sum(self.max_heights.values())
        uncertain_removals = sum(g.remove_count for g in self.removal_groups)
        black = self.puzzle.black_count
        return (total - uncertain_removals - black, total - black)


def _matrix(value: object, name: str) -> Tuple[Tuple[str, ...], ...]:
    if not isinstance(value, list) or not value or not all(isinstance(r, list) and r for r in value):
        raise PuzzleError("%s must be a non-empty matrix" % name)
    width = len(value[0])
    if any(len(r) != width for r in value):
        raise PuzzleError("%s must be rectangular" % name)
    result = tuple(tuple(c for c in row) for row in value)
    if any(not isinstance(c, str) or c not in VALID_CELLS for row in result for c in row):
        raise PuzzleError("%s contains a cell other than W, B, or X" % name)
    return result


def parse_puzzle(data: object) -> Puzzle:
    """Parse validated production fields; deliberately discard fixture answer."""
    if not isinstance(data, dict):
        raise PuzzleError("puzzle root must be an object")
    try:
        white, black = data["W"], data["B"]
        raw_views, raw_options = data["question"], data["options"]
    except KeyError as exc:
        raise PuzzleError("missing field %s" % exc.args[0])
    if not isinstance(white, int) or isinstance(white, bool) or white < 0:
        raise PuzzleError("W must be a non-negative integer")
    if not isinstance(black, int) or isinstance(black, bool) or black < 0:
        raise PuzzleError("B must be a non-negative integer")
    if not isinstance(raw_views, list) or len(raw_views) != 2:
        raise PuzzleError("question must contain exactly two views")
    views: Dict[str, View] = {}
    for i, raw in enumerate(raw_views):
        if not isinstance(raw, dict) or raw.get("view") not in VALID_VIEWS:
            raise PuzzleError("invalid question view")
        name = raw["view"]
        if name in views:
            raise PuzzleError("question views must be distinct")
        views[name] = View(name, _matrix(
            raw.get("cells"), "question[%d].cells" % i))
    if not isinstance(raw_options, list) or not raw_options:
        raise PuzzleError("options must be a non-empty list")
    options: List[Option] = []
    labels: Set[str] = set()
    for i, raw in enumerate(raw_options):
        if not isinstance(raw, dict) or not isinstance(raw.get("option"), str) or not raw["option"]:
            raise PuzzleError("invalid option label")
        label = raw["option"]
        if label in labels:
            raise PuzzleError("option labels must be unique")
        labels.add(label)
        options.append(Option(label, _matrix(
            raw.get("cells"), "options[%d].cells" % i)))
    return Puzzle(white, black, views, tuple(options), answer=None)


def load_puzzle(path: Path) -> Puzzle:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return parse_puzzle(json.load(handle))
    except OSError as exc:
        raise PuzzleError(str(exc))
    except json.JSONDecodeError as exc:
        raise PuzzleError("malformed JSON: %s" % exc)


def extract_side_heights(cells: Sequence[Sequence[Cell]]) -> List[int]:
    rows = len(cells)
    cols = len(cells[0])
    return [max((rows - r for r in range(rows) if cells[r][c] != "X"), default=0) for c in range(cols)]


def _dimensions(puzzle: Puzzle) -> Tuple[int, int, int]:
    front, left, top = puzzle.views.get(
        "front"), puzzle.views.get("left"), puzzle.views.get("top")
    width = len(front.cells[0]) if front else len(top.cells[0]) if top else 0
    depth = len(left.cells[0]) if left else len(top.cells) if top else 0
    height = len(front.cells) if front else len(left.cells) if left else 0
    if front and len(front.cells) != height or left and len(left.cells) != height:
        raise PuzzleError("side-view heights disagree")
    if top and (len(top.cells) != depth or len(top.cells[0]) != width):
        raise PuzzleError("top-view dimensions disagree")
    missing = ({"front", "left", "top"} - set(puzzle.views)).pop()
    expected = {"front": (height, width), "left": (
        height, depth), "top": (depth, width)}[missing]
    if any((len(o.cells), len(o.cells[0])) != expected for o in puzzle.options):
        raise PuzzleError("option dimensions do not match the missing view")
    return width, depth, height


def new_state(puzzle: Puzzle) -> SolveState:
    width, depth, height = _dimensions(puzzle)
    missing = ({"front", "left", "top"} - set(puzzle.views)).pop()
    return SolveState(puzzle, width, depth, height, missing, compatible_labels=[o.label for o in puzzle.options])


def matrix_cell_to_ray(view: str, row: int, col: int, width: int, depth: int, height: int) -> List[Coord]:
    if view == "front":
        z = height - row
        return [(col, y, z) for y in range(depth)]
    if view == "left":
        z, y = height - row, depth - 1 - col
        return [(x, y, z) for x in range(width)]
    if view == "top":
        y, x = depth - 1 - row, col
        return [(x, y, z) for z in range(height, 0, -1)]
    raise PuzzleError("unsupported view %s" % view)


def ray_coordinates(view: str, row: int, col: int, width: int, depth: int, height: int) -> List[Coord]:
    return matrix_cell_to_ray(view, row, col, width, depth, height)


def project_exact(voxels: Mapping[Coord, Cell], view: str, width: int, depth: int, height: int) -> List[List[Cell]]:
    if view not in VALID_VIEWS:
        raise PuzzleError("unsupported view %s" % view)
    shape = {"front": (height, width), "left": (
        height, depth), "top": (depth, width)}[view]
    result: List[List[Cell]] = []
    for row in range(shape[0]):
        out: List[Cell] = []
        for col in range(shape[1]):
            visible = next((voxels[p] for p in matrix_cell_to_ray(
                view, row, col, width, depth, height) if p in voxels), "X")
            out.append(visible)
        result.append(out)
    return result


def is_legally_stacked(voxels: Iterable[Coord]) -> bool:
    occupied = set(voxels)
    return all(z == 1 or (x, y, z - 1) in occupied for x, y, z in occupied)


def _snapshot(state: SolveState, stage: str, *notes: str) -> None:
    blacks = tuple((r.id, tuple(sorted(r.candidates)))
                   for r in state.black_requirements)
    if state.global_interpretations:
        counts = [sum(item.heights) for item in state.global_interpretations]
    elif state.black_visibility_interpretations:
        counts = [sum(item.heights)
                  for item in state.black_visibility_interpretations]
    else:
        counts = [sum(state.max_heights.values())]
    state.snapshots.append(StageSnapshot(
        stage=stage,
        heights=dict(state.max_heights),
        compatible_options=tuple(state.compatible_labels),
        notes=tuple(notes),
        black_candidates=blacks,
        black_scenario_count=len(state.black_scenarios),
        black_visibility_interpretation_count=len(
            state.black_visibility_interpretations),
        global_interpretation_count=len(state.global_interpretations),
        black_identity_alternatives=tuple(state.black_identity_alternatives),
        forced_black=frozenset(state.definitely_black),
        possible_black=frozenset(state.black_position_candidates),
        forced_removals=frozenset(state.forced_removals),
        conditional_removals=frozenset(state.conditional_removals),
        remaining_cube_count_range=(min(counts), max(counts)),
        front_heights=tuple(state.front_heights),
        left_heights=tuple(state.left_heights),
        footprint=dict(state.footprint),
    ))


# Step 1
def build_base_table(state: SolveState) -> SolveState:
    front = state.puzzle.views.get("front")
    left = state.puzzle.views.get("left")
    top = state.puzzle.views.get("top")
    state.front_heights = extract_side_heights(front.cells) if front else [
        state.height] * state.width
    if left:
        visual = extract_side_heights(left.cells)
        # internal y is front-to-back
        state.left_heights = list(reversed(visual))
    else:
        state.left_heights = [state.height] * state.depth
    if top:
        state.footprint = {
            (x, state.depth - 1 - row): top.cells[row][x] != "X"
            for row in range(state.depth) for x in range(state.width)
        }
    else:
        state.footprint = {
            (x, y): state.front_heights[x] > 0 and state.left_heights[y] > 0
            for y in range(state.depth) for x in range(state.width)
        }
    _snapshot(state, "build_base_table",
              "top-plane table and perpendicular silhouettes")
    return state


# Step 2
def fill_maximal_voxels(state: SolveState) -> SolveState:
    if not state.footprint:
        raise SolverInconsistency("Step 1 must run before Step 2")
    state.max_heights = {
        (x, y): min(state.front_heights[x], state.left_heights[y]) if state.footprint[(x, y)] else 0
        for y in range(state.depth) for x in range(state.width)
    }
    state.initial_max_heights = dict(state.max_heights)
    _snapshot(state, "fill_maximal_voxels",
              "each table entry is min(front, left)")
    return state


# Step 3
def prune_using_options(state: SolveState) -> SolveState:
    """Conservatively prune rays whose contribution no active option uses.

    With only silhouette information, unanimous ``X`` is the strongest local
    contribution test that is always safe: a non-X option may rely on any
    voxel along that ray after later reductions. Stronger color/position facts
    are therefore deferred rather than guessed.
    """
    active_options = [option for option in state.puzzle.options
                      if option.label in state.compatible_labels]
    rows, cols = len(active_options[0].cells), len(active_options[0].cells[0])
    for row in range(rows):
        for col in range(cols):
            values = frozenset(o.cells[row][col] for o in active_options)
            state.option_consensus[(row, col)] = values
            # No active option requires any occupancy contribution on this ray.
            if values == frozenset({"X"}):
                for x, y, z in matrix_cell_to_ray(state.missing_view, row, col, state.width, state.depth, state.height):
                    if state.max_heights[(x, y)] >= z:
                        state.max_heights[(x, y)] = z - 1
    state.maximal_voxels = state.occupied_voxels()
    _snapshot(state, "prune_using_options",
              "collective option support; unsupported missing-view rays removed")
    return state


def _known_presence_matches(state: SolveState, heights: Mapping[Base, int]) -> bool:
    occupied = {(x, y, z): "W" for (x, y), h in heights.items()
                for z in range(1, h + 1)}
    for name, view in state.puzzle.views.items():
        projected = project_exact(
            occupied, name, state.width, state.depth, state.height)
        for row in range(len(view.cells)):
            for col in range(len(view.cells[0])):
                if (projected[row][col] == "X") != (view.cells[row][col] == "X"):
                    return False
    return True


def _can_remove_level(state: SolveState, p: Coord) -> bool:
    x, y, z = p
    if state.max_heights[(x, y)] < z:
        return False
    # Lowering to z-1 removes this blocker and every cube above it. This also
    # permits a whole height-one column to disappear when both silhouettes are
    # still supplied elsewhere; stacking is preserved by construction.
    trial = dict(state.max_heights)
    trial[(x, y)] = z - 1
    return _known_presence_matches(state, trial)


def _black_candidates(state: SolveState, view: str, row: int, col: int) -> Tuple[Set[Coord], Coord, Optional[int]]:
    ray = matrix_cell_to_ray(
        view, row, col, state.width, state.depth, state.height)
    occupied = [p for p in ray if state.max_heights[(p[0], p[1])] >= p[2]]
    if not occupied:
        raise SolverInconsistency("black cell has no possible voxel")
    if view == "top":
        x, y, _ = ray[0]
        h = state.max_heights[(x, y)]
        candidates = {(x, y, z) for z in range(1, h + 1)}
        return candidates, (x, y, h), None
    candidates: Set[Coord] = set()
    for p in occupied:
        blockers = [q for q in ray[:ray.index(
            p)] if state.max_heights[(q[0], q[1])] >= q[2]]
        if all(_can_remove_level(state, q) for q in blockers):
            candidates.add(p)
    if not candidates:
        candidates.add(occupied[0])
    return candidates, occupied[0], state.height - row


# Step 4
def mark_black_voxels(state: SolveState) -> SolveState:
    requirements: List[BlackRequirement] = []
    for view_name, view in state.puzzle.views.items():
        for row, cells in enumerate(view.cells):
            for col, value in enumerate(cells):
                if value == "B":
                    candidates, representative, z = _black_candidates(
                        state, view_name, row, col)
                    requirements.append(BlackRequirement(
                        len(requirements), view_name, row, col, z, candidates, representative))
    state.black_requirements = requirements
    _snapshot(state, "mark_black_voxels",
              "black rays retain candidate sets plus nearest-edge representatives")
    return state


def _base_order(state: SolveState) -> Tuple[Base, ...]:
    return tuple((x, y) for y in range(state.depth) for x in range(state.width))


def _heights_tuple(state: SolveState, heights: Mapping[Base, int]) -> Tuple[int, ...]:
    return tuple(heights[b] for b in _base_order(state))


def _height_map(state: SolveState, heights: Sequence[int]) -> Dict[Base, int]:
    return dict(zip(_base_order(state), heights))


def _visible_coord(state: SolveState, heights: Mapping[Base, int], view: str,
                   row: int, col: int) -> Optional[Coord]:
    return next((p for p in matrix_cell_to_ray(view, row, col, state.width,
                                               state.depth, state.height)
                 if heights[(p[0], p[1])] >= p[2]), None)


def _voxel_model(heights: Mapping[Base, int], black_positions: Iterable[Coord]) -> Dict[Coord, Cell]:
    black = set(black_positions)
    return {(x, y, z): ("B" if (x, y, z) in black else "W")
            for (x, y), h in heights.items() for z in range(1, h + 1)}


def _model_matches_known(state: SolveState, heights: Mapping[Base, int],
                         black_positions: Iterable[Coord]) -> bool:
    voxels = _voxel_model(heights, black_positions)
    return all(project_exact(voxels, name, state.width, state.depth, state.height)
               == [list(row) for row in view.cells]
               for name, view in state.puzzle.views.items())


def _identity_groups(requirement_positions: Sequence[Coord]) -> Tuple[FrozenSet[int], ...]:
    by_position: Dict[Coord, Set[int]] = {}
    for rid, position in enumerate(requirement_positions):
        by_position.setdefault(position, set()).add(rid)
    return tuple(sorted((frozenset(ids) for ids in by_position.values()),
                        key=lambda group: tuple(sorted(group))))


def _required_blocker_caps(state: SolveState,
                           choices: Sequence[Coord]) -> Optional[Tuple[int, ...]]:
    """Compute pending Step-6 caps without applying or validating them."""
    caps = dict(state.max_heights)
    for requirement, position in zip(state.black_requirements, choices):
        ray = matrix_cell_to_ray(requirement.view, requirement.row,
                                 requirement.col, state.width,
                                 state.depth, state.height)
        if position not in ray:
            return None
        if requirement.view == "top":
            x, y, z = position
            caps[(x, y)] = min(caps[(x, y)], z)
        else:
            for x, y, z in ray[:ray.index(position)]:
                caps[(x, y)] = min(caps[(x, y)], z - 1)
    return _heights_tuple(state, caps)


def _step5_scenarios_from_choices(state: SolveState,
                                  choices: Sequence[Coord]) -> List[BlackScenario]:
    """Expand only identity, location, hidden-black, and pending-cap choices."""
    chosen = set(choices)
    if len(chosen) > state.puzzle.black_count:
        return []
    caps = _required_blocker_caps(state, choices)
    if caps is None:
        return []
    occupied = state.occupied_voxels()
    # Hidden black cubes must not already change a known visible cell. Whether
    # pending blocker removals expose one is deliberately left to Step 6.
    currently_visible = {
        p for name, view in state.puzzle.views.items()
        for row in range(len(view.cells)) for col in range(len(view.cells[0]))
        for p in [_visible_coord(state, state.max_heights, name, row, col)]
        if p is not None
    }
    hidden_candidates = sorted(occupied - currently_visible - chosen)
    hidden_needed = state.puzzle.black_count - len(chosen)
    if hidden_needed > len(hidden_candidates):
        return []
    groups = _identity_groups(choices)
    return [BlackScenario(
        frozenset(chosen | set(hidden)), groups, tuple(choices), caps)
        for hidden in combinations(hidden_candidates, hidden_needed)
    ]


def _forced_black_identities(state: SolveState) -> List[BlackIdentity]:
    count = len(state.black_requirements)
    if not state.black_scenarios or not count:
        return []
    always_equal = [[True] * count for _ in range(count)]
    for scenario in state.black_scenarios:
        group_for = {rid: group for group in scenario.identity_groups
                     for rid in group}
        for i in range(count):
            for j in range(i + 1, count):
                always_equal[i][j] &= group_for[i] == group_for[j]
                always_equal[j][i] = always_equal[i][j]
    remaining = set(range(count))
    identities: List[BlackIdentity] = []
    while remaining:
        first = min(remaining)
        group = {rid for rid in remaining if always_equal[first][rid]}
        remaining -= group
        candidates = {scenario.requirement_positions[first]
                      for scenario in state.black_scenarios}
        identities.append(BlackIdentity(
            group, candidates,
            state.black_scenarios[0].requirement_positions[first]))
    return identities


# Step 5
def merge_black_identities(state: SolveState) -> SolveState:
    """Use known geometry and B only; no W, white pruning, or option equality."""
    candidate_lists = [sorted(requirement.candidates)
                       for requirement in state.black_requirements]
    scenarios: Dict[Tuple[FrozenSet[Coord], Tuple[FrozenSet[int], ...],
                          Tuple[Coord, ...], Tuple[int, ...]], BlackScenario] = {}
    choice_product = product(*candidate_lists) if candidate_lists else [()]
    for choices in choice_product:
        for scenario in _step5_scenarios_from_choices(state, choices):
            key = (scenario.black_positions, scenario.identity_groups,
                   scenario.requirement_positions,
                   scenario.required_height_caps)
            scenarios[key] = scenario
    if not scenarios:
        raise SolverInconsistency(
            "black requirements cannot fit the specified black-cube count")
    state.black_scenarios = sorted(
        scenarios.values(),
        key=lambda item: (sorted(item.black_positions),
                          item.required_height_caps,
                          tuple(tuple(sorted(g))
                                for g in item.identity_groups)))
    state.black_identity_alternatives = sorted(
        {scenario.identity_groups for scenario in state.black_scenarios},
        key=lambda groups: tuple(tuple(sorted(g)) for g in groups))
    state.black_identities = _forced_black_identities(state)
    state.black_position_candidates = set().union(
        *(scenario.black_positions for scenario in state.black_scenarios))
    state.definitely_black = set.intersection(
        *(set(scenario.black_positions) for scenario in state.black_scenarios))
    state.black_positions = set(state.black_scenarios[0].black_positions)
    _snapshot(state, "merge_black_identities",
              "identity/location alternatives use known views and B only; blocker caps remain pending")
    return state


def _refresh_occupancy_facts(state: SolveState,
                             occupied_sets: Sequence[Set[Coord]]) -> None:
    state.possibly_occupied = set().union(*occupied_sets)
    state.definitely_occupied = set.intersection(
        *(set(value) for value in occupied_sets))
    universe = state.maximal_voxels or state.occupied_voxels()
    state.forced_removals = universe - state.possibly_occupied
    state.conditional_removals = (
        universe & state.possibly_occupied) - state.definitely_occupied
    state.definitely_empty = universe - state.possibly_occupied


# Step 6
def remove_black_occluders(state: SolveState) -> SolveState:
    """Apply pending black-visibility caps and retain every valid branch."""
    interpretations: Dict[
        Tuple[Tuple[int, ...], FrozenSet[Coord], Tuple[FrozenSet[int], ...]],
        BlackVisibilityInterpretation,
    ] = {}
    for scenario in state.black_scenarios:
        heights = _height_map(state, scenario.required_height_caps)
        if any(heights[(x, y)] < z
               for x, y, z in scenario.black_positions):
            continue
        if not _model_matches_known(state, heights,
                                    scenario.black_positions):
            continue
        actual_positions = tuple(
            _visible_coord(state, heights, requirement.view,
                           requirement.row, requirement.col)
            for requirement in state.black_requirements
        )
        if (any(position is None for position in actual_positions)
                or actual_positions != scenario.requirement_positions
                or any(position not in scenario.black_positions
                       for position in actual_positions)):
            continue
        occupied = {(x, y, z) for (x, y), h in heights.items()
                    for z in range(1, h + 1)}
        blocker_removals = frozenset(state.maximal_voxels - occupied)
        interpretation = BlackVisibilityInterpretation(
            scenario.required_height_caps, scenario.black_positions,
            scenario.identity_groups, scenario.requirement_positions,
            blocker_removals)
        key = (interpretation.heights, interpretation.black_positions,
               interpretation.identity_groups)
        interpretations[key] = interpretation
    if not interpretations:
        raise SolverInconsistency(
            "black visibility has no legal blocker-removal interpretation")
    state.black_visibility_interpretations = sorted(
        interpretations.values(),
        key=lambda item: (item.heights, sorted(item.black_positions)))
    occupied_sets = [
        {(x, y, z) for (x, y), h in _height_map(state, item.heights).items()
         for z in range(1, h + 1)}
        for item in state.black_visibility_interpretations
    ]
    _refresh_occupancy_facts(state, occupied_sets)
    order = _base_order(state)
    state.max_heights = {
        base: max(item.heights[index]
                  for item in state.black_visibility_interpretations)
        for index, base in enumerate(order)
    }
    state.black_position_candidates = set().union(
        *(item.black_positions
          for item in state.black_visibility_interpretations))
    state.definitely_black = set.intersection(
        *(set(item.black_positions)
          for item in state.black_visibility_interpretations))
    state.black_positions = set(
        state.black_visibility_interpretations[0].black_positions)
    _snapshot(state, "remove_black_occluders",
              "visibility blockers removed per branch; universal and conditional removals summarized")
    return state


def _representative_voxels(state: SolveState,
                           heights: Optional[Mapping[Base, int]] = None) -> Dict[Coord, Cell]:
    if heights is not None:
        return _voxel_model(heights, state.black_positions)
    if state.global_interpretations:
        representative = state.global_interpretations[0]
        return _voxel_model(_height_map(state, representative.heights),
                            representative.black_positions)
    if state.black_visibility_interpretations:
        representative = state.black_visibility_interpretations[0]
        return _voxel_model(_height_map(state, representative.heights),
                            representative.black_positions)
    return _voxel_model(state.max_heights, state.black_positions)


def _projection_pattern(state: SolveState, heights: Mapping[Base, int],
                        black_positions: Iterable[Coord]) -> Tuple[Tuple[Cell, ...], ...]:
    projected = project_exact(_voxel_model(heights, black_positions),
                              state.missing_view, state.width,
                              state.depth, state.height)
    return tuple(tuple(row) for row in projected)


def is_view_irrelevant_white_reduction(
        state: SolveState, heights: Mapping[Base, int], base: Base,
        new_height: int, black_positions: FrozenSet[Coord]) -> bool:
    """Whether lowering one column suffix removes only projection-irrelevant white."""
    current = heights[base]
    if not 0 <= new_height < current:
        return False
    x, y = base
    removed = {(x, y, z) for z in range(new_height + 1, current + 1)}
    if removed & black_positions:
        return False
    trial = dict(heights)
    trial[base] = new_height
    # The current branch already matches both known views. Requiring the trial
    # to match proves that this whole white suffix contributes to neither.
    return _model_matches_known(state, trial, black_positions)


def _correlated_white_reductions(
        state: SolveState,
        interpretation: BlackVisibilityInterpretation) -> Set[Tuple[int, ...]]:
    start = _height_map(state, interpretation.heights)
    target = state.puzzle.white_count + state.puzzle.black_count
    if sum(start.values()) < target:
        return set()
    finals: Set[Tuple[int, ...]] = set()
    visited: Set[Tuple[int, ...]] = set()

    def visit(heights: Dict[Base, int]) -> None:
        values = _heights_tuple(state, heights)
        if values in visited:
            return
        visited.add(values)
        current_total = sum(values)
        if current_total == target:
            finals.add(values)
            return
        if current_total < target:
            return
        excess = current_total - target
        for base in _base_order(state):
            current = heights[base]
            for reduction in range(1, min(current, excess) + 1):
                new_height = current - reduction
                if not is_view_irrelevant_white_reduction(
                        state, heights, base, new_height,
                        interpretation.black_positions):
                    continue
                trial = dict(heights)
                trial[base] = new_height
                visit(trial)

    visit(start)
    return finals


# Step 7
def prune_by_cube_counts(state: SolveState) -> SolveState:
    """Apply W+B for the first time and retain correlated legal white reductions."""
    interpretations: Dict[Tuple[Tuple[int, ...], FrozenSet[Coord]],
                          GlobalInterpretation] = {}
    for visibility in state.black_visibility_interpretations:
        for height_values in _correlated_white_reductions(state, visibility):
            key = (height_values, visibility.black_positions)
            interpretations[key] = GlobalInterpretation(
                height_values, visibility.black_positions,
                visibility.identity_groups)
    if not interpretations:
        raise SolverInconsistency(
            "exact white/black totals allow no view-irrelevant white reductions")
    state.global_interpretations = sorted(
        interpretations.values(),
        key=lambda item: (item.heights, sorted(item.black_positions)))
    order = _base_order(state)
    state.max_heights = {
        base: max(item.heights[index] for item in state.global_interpretations)
        for index, base in enumerate(order)
    }
    occupied_sets = [
        {(x, y, z) for (x, y), h in _height_map(state, item.heights).items()
         for z in range(1, h + 1)}
        for item in state.global_interpretations
    ]
    _refresh_occupancy_facts(state, occupied_sets)
    state.black_position_candidates = set().union(
        *(item.black_positions for item in state.global_interpretations))
    state.definitely_black = set.intersection(
        *(set(item.black_positions) for item in state.global_interpretations))
    state.black_positions = set(
        state.global_interpretations[0].black_positions)
    height_states = tuple(sorted(
        {item.heights for item in state.global_interpretations}))
    height_domains = {
        base: frozenset(values[index] for values in height_states)
        for index, base in enumerate(order)
    }
    if len(height_states) > 1:
        state.removal_groups = [RemovalGroup(
            frozenset(state.forced_removals | state.conditional_removals),
            sum(state.max_heights.values()) -
            (state.puzzle.white_count + state.puzzle.black_count),
            "complete correlated view-irrelevant white reductions",
            height_domains, height_states,
        )]
    _snapshot(state, "prune_by_cube_counts",
              "W+B applied; every retained suffix reduction preserves known views")
    return state


STAGES = (
    build_base_table,
    fill_maximal_voxels,
    prune_using_options,
    mark_black_voxels,
    merge_black_identities,
    remove_black_occluders,
    prune_by_cube_counts,
)


def run_through(puzzle: Puzzle, stage_number: int) -> SolveState:
    if not isinstance(stage_number, int) or not 0 <= stage_number <= len(STAGES):
        raise ValueError("stage_number must be between 0 and 7")
    state = new_state(puzzle)
    for stage in STAGES[:stage_number]:
        stage(state)
    return state


def run_all_stages(puzzle: Puzzle) -> SolveState:
    return run_through(puzzle, len(STAGES))


def project_state_representative(state: SolveState, view: str) -> List[List[Cell]]:
    return project_exact(_representative_voxels(state), view, state.width, state.depth, state.height)


def _retained_projection_patterns(state: SolveState) -> Set[Tuple[Tuple[Cell, ...], ...]]:
    if not state.global_interpretations:
        raise SolverInconsistency(
            "missing-view constraints require all seven pruning stages")
    return {
        _projection_pattern(state, _height_map(state, item.heights),
                            item.black_positions)
        for item in state.global_interpretations
    }


def derive_missing_view_constraints(state: SolveState) -> Dict[Tuple[int, int], FrozenSet[Cell]]:
    """Union cell values across complete retained global interpretations."""
    patterns = _retained_projection_patterns(state)
    rows, cols = len(next(iter(patterns))), len(next(iter(patterns))[0])
    return {
        (row, col): frozenset(pattern[row][col] for pattern in patterns)
        for row in range(rows) for col in range(cols)
    }


def compatible_options(state: SolveState, options: Sequence[Option]) -> List[str]:
    # Compare complete matrices, not independent marginal cell domains. This
    # prevents accepting a projection assembled from incompatible branches.
    patterns = _retained_projection_patterns(state)
    labels = [option.label for option in options if option.cells in patterns]
    state.compatible_labels = labels
    return labels


def solve(puzzle: Puzzle) -> List[str]:
    state = run_all_stages(puzzle)
    return compatible_options(state, puzzle.options)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python solver.py <puzzle.json>", file=sys.stderr)
        return 2
    try:
        puzzle = load_puzzle(Path(argv[0]))
        labels = solve(puzzle)
        if not labels:
            raise SolverInconsistency("no compatible options")
        if len(labels) != 1:
            raise SolverInconsistency(
                "multiple compatible options: %s" % ", ".join(labels))
        print(labels[0])
        return 0
    except (PuzzleError, SolverInconsistency) as exc:
        print("solver error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
