import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The pytest console script may put test/ rather than the repository root on
# sys.path (notably with Homebrew Python 3.14 / pytest 9). Make the project
# module import deterministic for both `pytest` and `python -m pytest`.
sys.path.insert(0, str(ROOT))

import solver


FIXTURE_DIR = ROOT / "test" / "tests"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))
if not FIXTURES:
    raise RuntimeError(f"no JSON puzzle fixtures found under {FIXTURE_DIR}")


def tiny_puzzle(**overrides):
    data = {
        "W": 2,
        "B": 0,
        "question": [
            {"view": "front", "cells": [["W", "X"], ["W", "W"]]},
            {"view": "left", "cells": [["X", "W"], ["W", "W"]]},
        ],
        "options": [{"option": "A", "cells": [["W", "W"], ["W", "W"]]}],
    }
    data.update(overrides)
    return solver.parse_puzzle(data)


def test_parse_json_and_ignore_answer(tmp_path):
    raw = json.loads(FIXTURES[0].read_text())
    puzzle = solver.parse_puzzle(raw)
    assert puzzle.white_count == 12
    assert puzzle.black_count == 3
    assert puzzle.answer is None
    raw["answer"] = "definitely-not-used"
    assert solver.parse_puzzle(raw).answer is None


@pytest.mark.parametrize("path", FIXTURES)
def test_fixture_answer(path):
    raw = json.loads(path.read_text())
    assert solver.solve(solver.parse_puzzle(raw)) == [raw["answer"]]


@pytest.mark.parametrize("path", FIXTURES)
def test_cli(path):
    expected = json.loads(path.read_text())["answer"]
    result = subprocess.run(
        [sys.executable, str(ROOT / "solver.py"), str(path)],
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == expected + "\n"


def test_coordinate_mapping_and_projection():
    # x is front-view left-to-right; y=0 is the front edge; z=1 is ground.
    voxels = {(0, 0, 1): "B", (1, 1, 1): "W", (1, 1, 2): "W"}
    assert solver.project_exact(voxels, "front", width=2, depth=2, height=2) == [
        ["X", "W"], ["B", "W"]
    ]
    # In left and top matrices, the visually left/top edge is the back (large y).
    assert solver.project_exact(voxels, "left", width=2, depth=2, height=2) == [
        ["W", "X"], ["W", "B"]
    ]
    assert solver.project_exact(voxels, "top", width=2, depth=2, height=2) == [
        ["X", "W"], ["B", "X"]
    ]
    assert solver.ray_coordinates("front", 1, 0, 2, 2, 2) == [
        (0, 0, 1), (0, 1, 1)]
    assert solver.ray_coordinates("left", 1, 0, 2, 2, 2) == [
        (0, 1, 1), (1, 1, 1)]
    assert solver.ray_coordinates("top", 0, 1, 2, 2, 2) == [
        (1, 1, 2), (1, 1, 1)]


def test_height_extraction():
    cells = [["X", "W", "X"], ["B", "W", "X"], ["W", "W", "B"]]
    assert solver.extract_side_heights(cells) == [2, 3, 1]


def test_legal_stacking():
    assert solver.is_legally_stacked({(0, 0, 1), (0, 0, 2)})
    assert not solver.is_legally_stacked({(0, 0, 2)})


def test_steps_1_and_2_minimum_table():
    puzzle = tiny_puzzle()
    state = solver.new_state(puzzle)
    solver.build_base_table(state)
    assert state.front_heights == [2, 1]
    # left matrix columns are back-to-front, internal y is front-to-back.
    assert state.left_heights == [2, 1]
    solver.fill_maximal_voxels(state)
    assert state.max_heights == {(0, 0): 2, (1, 0): 1, (0, 1): 1, (1, 1): 1}
    assert state.snapshots[1].stage == "fill_maximal_voxels"


def test_top_footprint_x_and_side_constraint():
    raw = json.loads(FIXTURES[1].read_text())
    state = solver.new_state(solver.parse_puzzle(raw))
    solver.build_base_table(state)
    solver.fill_maximal_voxels(state)
    # Bottom visual top row is back; X at visual bottom-left is front x=0.
    assert state.max_heights[(0, 0)] == 0
    assert state.max_heights[(1, 0)] == 1
    assert state.max_heights[(0, 2)] == 3


def test_collective_option_pruning_uses_consensus_not_per_option():
    raw = json.loads(FIXTURES[1].read_text())
    state = solver.run_through(solver.parse_puzzle(raw), 3)
    assert state.max_heights[(1, 1)] == 2
    assert state.max_heights[(1, 2)] == 2
    assert set(state.compatible_labels) == {"A", "B", "C", "D"}
    assert state.option_consensus[(0, 1)] == frozenset({"X"})


def test_black_visibility_and_uncertain_placement():
    raw = json.loads(FIXTURES[0].read_text())
    state = solver.run_through(solver.parse_puzzle(raw), 4)
    bottom_front = next(
        r for r in state.black_requirements if r.view == "front" and r.z == 1)
    assert bottom_front.representative == (1, 0, 1)
    assert (1, 0, 1) in bottom_front.candidates
    # Side-ray black locations remain uncertain when preceding columns can be
    # lowered without destroying either known silhouette.
    level_two = next(
        r for r in state.black_requirements if r.view == "front" and r.z == 2)
    assert len(level_two.candidates) > 1
    assert level_two.representative in level_two.candidates


def test_ground_column_can_be_removed_to_expose_black():
    raw = {
        "W": 1, "B": 1,
        "question": [
            {"view": "front", "cells": [["X", "X"], ["W", "B"]]},
            {"view": "left", "cells": [["X", "X"], ["W", "B"]]},
        ],
        "options": [{"option": "A", "cells": [["W", "X"], ["X", "B"]]}],
    }
    state = solver.run_through(solver.parse_puzzle(raw), 6)
    assert state.max_heights[(0, 0)] == 0
    assert solver.is_legally_stacked(state.occupied_voxels())
    assert solver.solve(solver.parse_puzzle(raw)) == ["A"]


def test_black_identity_merging_and_count():
    raw = json.loads(FIXTURES[0].read_text())
    state = solver.run_through(solver.parse_puzzle(raw), 5)
    assert len(state.black_identities) == state.puzzle.black_count
    assert any(len(identity.requirement_ids) >
               1 for identity in state.black_identities)
    assert all(identity.candidates for identity in state.black_identities)


def test_remove_white_blocker_without_floating():
    raw = json.loads(FIXTURES[0].read_text())
    state = solver.run_through(solver.parse_puzzle(raw), 6)
    # The front blocker of the shared level-two black is removed from the top of its column.
    assert state.max_heights[(0, 0)] == 1
    assert solver.is_legally_stacked(state.occupied_voxels())
    assert solver.project_state_representative(state, "front")[1][0] == "B"


def test_white_count_pruning_and_ambiguity_group():
    raw = {
        "W": 7, "B": 0,
        "question": [
            {"view": "front", "cells": [["W", "W"], ["W", "W"]]},
            {"view": "left", "cells": [["W", "W"], ["W", "W"]]},
        ],
        "options": [{"option": "A", "cells": [["W", "W"], ["W", "W"]]}],
    }
    state = solver.run_through(solver.parse_puzzle(raw), 7)
    assert state.removal_groups
    assert state.removal_groups[-1].remove_count == 1
    assert len(state.removal_groups[-1].voxels) > 1
    assert state.white_count_range()[0] <= 7 <= state.white_count_range()[1]


def test_forced_possible_impossible_and_option_matching():
    state = solver.run_all_stages(solver.parse_puzzle(
        json.loads(FIXTURES[0].read_text())))
    domains = solver.derive_missing_view_constraints(state)
    assert domains[(0, 0)] == frozenset({"B"})
    assert "X" not in domains[(1, 1)]
    # Step-7 snapshot precedes final option selection by design.
    assert state.snapshots[-1].compatible_options == ("A", "B", "C", "D")
    assert solver.compatible_options(state, state.puzzle.options) == ["A"]


def test_ambiguous_domains_and_multiple_compatible_options():
    raw = {
        "W": 0, "B": 1,
        "question": [
            {"view": "front", "cells": [["B"]]},
            {"view": "left", "cells": [["B"]]},
        ],
        "options": [
            {"option": "A", "cells": [["B"]]},
            {"option": "B", "cells": [["B"]]},
        ],
    }
    state = solver.run_all_stages(solver.parse_puzzle(raw))
    assert solver.compatible_options(state, state.puzzle.options) == ["A", "B"]


def diagonal_puzzle(color):
    return solver.parse_puzzle({
        "W": 2 if color == "W" else 0,
        "B": 2 if color == "B" else 0,
        "question": [
            {"view": "front", "cells": [[color, color]]},
            {"view": "left", "cells": [[color, color]]},
        ],
        "options": [
            {"option": "A", "cells": [[color, "X"], ["X", color]]},
            {"option": "B", "cells": [["X", color], [color, "X"]]},
        ],
    })


def test_multiple_black_identity_groupings_are_retained():
    state = solver.run_through(diagonal_puzzle("B"), 5)
    assert len(state.black_identity_alternatives) == 2
    assert {
        tuple(sorted(tuple(sorted(group)) for group in grouping))
        for grouping in state.black_identity_alternatives
    } == {
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    }
    # No cross-view pair is merged in every valid interpretation.
    assert all(len(identity.requirement_ids) == 1
               for identity in state.black_identities)


def test_edge_representative_is_non_authoritative():
    state = solver.run_all_stages(diagonal_puzzle("B"))
    representative = tuple(tuple(row) for row in
                           solver.project_state_representative(state, "top"))
    assert representative in {option.cells for option in state.puzzle.options}
    assert len(state.global_interpretations) == 2
    assert solver.compatible_options(state, state.puzzle.options) == ["A", "B"]
    assert state.black_positions != state.black_position_candidates


def test_conditional_black_occluders_are_not_forced():
    state = solver.run_through(diagonal_puzzle("B"), 6)
    assert not state.forced_removals
    assert state.conditional_removals
    assert state.max_heights == {
        (0, 0): 1, (1, 0): 1, (0, 1): 1, (1, 1): 1,
    }


def test_white_reductions_retain_only_jointly_valid_states():
    state = solver.run_all_stages(diagonal_puzzle("W"))
    assert len(state.global_interpretations) == 2
    height_maps = [solver._height_map(state, item.heights)
                   for item in state.global_interpretations]
    # Either x=0 column may disappear, but both cannot disappear together
    # because the known front x=0 silhouette must remain visible.
    assert any(heights[(0, 0)] == 0 for heights in height_maps)
    assert any(heights[(0, 1)] == 0 for heights in height_maps)
    assert not any(heights[(0, 0)] == heights[(0, 1)] == 0
                   for heights in height_maps)
    assert state.removal_groups[-1].height_states
    assert state.white_count_range() == (2, 2)


def test_global_ambiguity_drives_domains_and_non_unique_options():
    state = solver.run_all_stages(diagonal_puzzle("W"))
    domains = solver.derive_missing_view_constraints(state)
    assert all(values == frozenset({"W", "X"})
               for values in domains.values())
    assert solver.compatible_options(state, state.puzzle.options) == ["A", "B"]


def test_step5_is_independent_of_white_total():
    first = diagonal_puzzle("B")
    raw = {
        "W": 1, "B": first.black_count,
        "question": [
            {"view": name, "cells": [list(row) for row in view.cells]}
            for name, view in first.views.items()
        ],
        "options": [
            {"option": option.label,
             "cells": [list(row) for row in option.cells]}
            for option in first.options
        ],
    }
    second = solver.parse_puzzle(raw)
    state_a = solver.run_through(first, 5)
    state_b = solver.run_through(second, 5)
    assert state_a.black_scenarios == state_b.black_scenarios
    assert state_a.black_identity_alternatives == state_b.black_identity_alternatives
    assert state_a.snapshots[-1].global_interpretation_count == 0


def test_step5_is_independent_of_final_option_patterns():
    base = diagonal_puzzle("B")
    raw = {
        "W": 0, "B": 2,
        "question": [
            {"view": name, "cells": [list(row) for row in view.cells]}
            for name, view in base.views.items()
        ],
        "options": [
            *[{"option": option.label,
               "cells": [list(row) for row in option.cells]}
              for option in base.options],
            {"option": "C", "cells": [["B", "B"], ["B", "B"]]},
        ],
    }
    variant = solver.parse_puzzle(raw)
    state_a = solver.run_through(base, 5)
    state_b = solver.run_through(variant, 5)
    assert state_a.max_heights == state_b.max_heights
    assert state_a.black_scenarios == state_b.black_scenarios
    assert state_a.black_identity_alternatives == state_b.black_identity_alternatives


def test_step6_not_step5_applies_pending_black_blocker_removal():
    raw = {
        "W": 1, "B": 1,
        "question": [
            {"view": "front", "cells": [["X", "X"], ["W", "B"]]},
            {"view": "left", "cells": [["X", "X"], ["W", "B"]]},
        ],
        "options": [
            {"option": "A", "cells": [["W", "X"], ["X", "B"]]},
            {"option": "B", "cells": [["W", "W"], ["W", "W"]]},
        ],
    }
    puzzle = solver.parse_puzzle(raw)
    step5 = solver.run_through(puzzle, 5)
    assert step5.max_heights[(0, 0)] == 1
    assert any(s.required_height_caps[0] == 0 for s in step5.black_scenarios)
    step6 = solver.run_through(puzzle, 6)
    assert step6.max_heights[(0, 0)] == 0
    assert (0, 0, 1) in step6.forced_removals


def test_step7_is_first_stage_to_apply_white_total():
    raw = {
        "W": 3, "B": 0,
        "question": [
            {"view": "front", "cells": [["W", "W"]]},
            {"view": "left", "cells": [["W", "W"]]},
        ],
        "options": [
            {"option": "A", "cells": [["W", "W"], ["W", "X"]]},
            {"option": "B", "cells": [["W", "W"], ["X", "W"]]},
            {"option": "C", "cells": [["W", "X"], ["W", "W"]]},
            {"option": "D", "cells": [["X", "W"], ["W", "W"]]},
        ],
    }
    puzzle = solver.parse_puzzle(raw)
    step6 = solver.run_through(puzzle, 6)
    assert sum(step6.max_heights.values()) == 4
    assert not step6.global_interpretations
    step7 = solver.run_through(puzzle, 7)
    assert {sum(item.heights) for item in step7.global_interpretations} == {3}
    assert step7.snapshots[-1].remaining_cube_count_range == (3, 3)


def test_complete_patterns_reject_composite_of_cell_domains():
    state = solver.run_all_stages(diagonal_puzzle("W"))
    composite = solver.Option("C", (("W", "W"), ("W", "W")))
    domains = solver.derive_missing_view_constraints(state)
    assert all("W" in values for values in domains.values())
    assert solver.compatible_options(
        state, (*state.puzzle.options, composite)) == ["A", "B"]


def test_missing_left_view_is_supported():
    raw = {
        "W": 0, "B": 1,
        "question": [
            {"view": "front", "cells": [["B"]]},
            {"view": "top", "cells": [["B"]]},
        ],
        "options": [{"option": "L", "cells": [["B"]]}],
    }
    assert solver.solve(solver.parse_puzzle(raw)) == ["L"]


def test_cli_errors_are_nonzero(tmp_path):
    malformed = tmp_path / "bad.json"
    malformed.write_text("{not json")
    result = subprocess.run([sys.executable, str(ROOT / "solver.py"), str(malformed)],
                            text=True, capture_output=True, cwd=ROOT)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "malformed JSON" in result.stderr


def test_all_seven_stage_snapshots_exist():
    state = solver.run_all_stages(solver.parse_puzzle(
        json.loads(FIXTURES[0].read_text())))
    assert [s.stage for s in state.snapshots] == [
        "build_base_table", "fill_maximal_voxels", "prune_using_options",
        "mark_black_voxels", "merge_black_identities",
        "remove_black_occluders", "prune_by_cube_counts",
    ]
