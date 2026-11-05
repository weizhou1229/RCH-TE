"""EDC evaluation with exact polynomial-time triple alignment.

This module preserves the data conversion, pairwise scoring, aggregate metrics,
and command-line interface from evaluation_script.py. It replaces only the
factorial permutation search in calculateSystemScore with the Hungarian
algorithm. The selected assignment maximizes the same mean of ent_type,
partial, strict, and exact F1 scores as the legacy implementation.
"""

from argparse import ArgumentParser
import itertools
import math
import os
import random
import statistics
import sys

legacy = None


def _load_legacy():
    global legacy
    if legacy is not None:
        return legacy
    try:
        from . import evaluation_script as loaded_legacy
    except ImportError:
        import evaluation_script as loaded_legacy
    legacy = loaded_legacy
    return legacy


def _pair_score(result):
    """Return the exact objective used by the legacy permutation search."""
    return statistics.mean(
        [
            result["ent_type"]["f1"],
            result["partial"]["f1"],
            result["strict"]["f1"],
            result["exact"]["f1"],
        ]
    )


def _hungarian_maximize(score_matrix):
    """Return an exact maximum-weight assignment and its value in O(n^3).

    This is the primal-dual Hungarian algorithm. It minimizes negated scores
    internally and supports floating-point weights.
    """
    row_count = len(score_matrix)
    if row_count == 0:
        return [], 0.0
    column_count = len(score_matrix[0])
    if row_count > column_count or any(len(row) != column_count for row in score_matrix):
        raise ValueError("Hungarian solver requires a rectangular matrix with rows <= columns")

    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        minimum_reduced_cost = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0

        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = math.inf
            next_column = 0

            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = (
                    -float(score_matrix[current_row - 1][candidate_column - 1])
                    - row_potential[current_row]
                    - column_potential[candidate_column]
                )
                if reduced_cost < minimum_reduced_cost[candidate_column]:
                    minimum_reduced_cost[candidate_column] = reduced_cost
                    predecessor[candidate_column] = column
                if minimum_reduced_cost[candidate_column] < delta:
                    delta = minimum_reduced_cost[candidate_column]
                    next_column = candidate_column

            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum_reduced_cost[candidate_column] -= delta

            column = next_column
            if matched_row[column] == 0:
                break

        while True:
            previous_column = predecessor[column]
            matched_row[column] = matched_row[previous_column]
            column = previous_column
            if column == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column] != 0:
            assignment[matched_row[column] - 1] = column - 1

    value = sum(
        float(score_matrix[row][column])
        for row, column in enumerate(assignment)
    )
    return assignment, value


def _optimal_value(score_matrix, rows, columns):
    """Return the maximum assignment value for a square subproblem."""
    if not rows:
        return 0.0
    submatrix = [
        [score_matrix[row][column] for column in columns]
        for row in rows
    ]
    _, value = _hungarian_maximize(submatrix)
    return value


def _lexicographic_optimal_assignment(score_matrix):
    """Return the lexicographically first maximum-weight assignment.

    itertools.permutations emits assignments in lexicographic order and the
    legacy evaluator replaces its best result only on a strict improvement.
    Selecting the smallest feasible column at each row among assignments with
    the best attainable total reproduces that tie-breaking without enumerating
    every permutation.
    """
    score_matrix = [list(map(float, row)) for row in score_matrix]
    size = len(score_matrix)
    if any(len(row) != size for row in score_matrix):
        raise ValueError(
            "Hungarian alignment requires a square score matrix; "
            f"got {size} rows. Run calculateAllScores first so "
            "candidate/reference lists are padded equally."
        )

    available_columns = list(range(size))
    assignment = []

    for row in range(size):
        remaining_rows = list(range(row + 1, size))
        attainable = []
        for column in available_columns:
            remaining_columns = [c for c in available_columns if c != column]
            value = float(score_matrix[row][column]) + _optimal_value(
                score_matrix, remaining_rows, remaining_columns
            )
            attainable.append((column, value))

        best_value = max(value for _, value in attainable)
        tied_columns = [
            column
            for column, value in attainable
            if math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12)
        ]
        selected_column = min(tied_columns)
        assignment.append(selected_column)
        available_columns.remove(selected_column)

    return assignment


def _hungarian_assignments(totalsemevallist):
    assignments = []
    for sample_index, candidate_scores in enumerate(totalsemevallist):
        if not candidate_scores:
            raise ValueError(
                f"Sample {sample_index} has no candidate/reference entries after padding"
            )
        score_matrix = (
            [
                [_pair_score(pair_result) for pair_result in candidate_row]
                for candidate_row in candidate_scores
            ]
        )
        assignments.append(_lexicographic_optimal_assignment(score_matrix))
    return assignments


def calculateSystemScore(totalsemevallist, totalsemevallistpertag, newreflist, newcandlist):
    """Calculate legacy metrics using an exact Hungarian optimal alignment."""
    legacy_module = _load_legacy()
    assignments = _hungarian_assignments(totalsemevallist)

    reordered_scores = []
    reordered_scores_per_tag = []
    reordered_references = []
    for sample_index, assignment in enumerate(assignments):
        reordered_scores.append(
            [
                [candidate_row[column] for column in assignment]
                for candidate_row in totalsemevallist[sample_index]
            ]
        )
        reordered_scores_per_tag.append(
            [
                [candidate_row[column] for column in assignment]
                for candidate_row in totalsemevallistpertag[sample_index]
            ]
        )
        reordered_references.append(
            [newreflist[sample_index][column] for column in assignment]
        )

    # The optimal reference for candidate i is now at column i. Reuse the
    # original aggregation/reporting code with the permutation source reduced
    # to the identity, preserving every metric calculation and output label.
    original_permutations = legacy_module.itertools.permutations

    def identity_permutation(iterable, length=None):
        values = tuple(iterable)
        if length is None:
            length = len(values)
        yield values[:length]

    legacy_module.itertools.permutations = identity_permutation
    try:
        result = legacy_module.calculateSystemScore(
            reordered_scores,
            reordered_scores_per_tag,
            reordered_references,
            newcandlist,
        )
    finally:
        legacy_module.itertools.permutations = original_permutations

    selected_scores, selected_scores_per_tag, selected_alignment, sample_scores = result
    remapped_alignment = []
    for alignment, assignment in zip(selected_alignment, assignments):
        candidate_indices, reordered_reference_indices = alignment
        original_reference_indices = [assignment[i] for i in reordered_reference_indices]
        remapped_alignment.append((candidate_indices, original_reference_indices))

    return (
        selected_scores,
        selected_scores_per_tag,
        remapped_alignment,
        sample_scores,
    )


def main(reffile, candfile):
    legacy_module = _load_legacy()
    reflist, newreflist = legacy_module.getRefs(reffile)
    candlist, newcandlist = legacy_module.getCands(candfile)
    totalsemevallist, totalsemevallistpertag = legacy_module.calculateAllScores(
        newreflist, newcandlist
    )
    calculateSystemScore(
        totalsemevallist,
        totalsemevallistpertag,
        newreflist,
        newcandlist,
    )
    legacy_module.calculateExactTripleScore(reflist, candlist)


def verify_solver():
    """Exhaustively verify objective and tie-breaking for small matrices."""
    rng = random.Random(20260813)
    checked = 0
    for size in range(1, 8):
        matrices = [
            [
                [rng.randint(0, 8) / 8.0 for _ in range(size)]
                for _ in range(size)
            ]
            for _ in range(20)
        ]
        matrices.append([[1.0] * size for _ in range(size)])
        for matrix in matrices:
            expected_assignment = None
            expected_value = -math.inf
            for permutation in itertools.permutations(range(size)):
                value = sum(matrix[row][column] for row, column in enumerate(permutation))
                if value > expected_value:
                    expected_value = value
                    expected_assignment = list(permutation)

            actual_assignment = _lexicographic_optimal_assignment(matrix)
            actual_value = sum(
                matrix[row][column] for row, column in enumerate(actual_assignment)
            )
            if actual_assignment != expected_assignment or not math.isclose(
                actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise AssertionError(
                    "Hungarian verification failed: "
                    f"size={size}, expected={expected_assignment}/{expected_value}, "
                    f"actual={actual_assignment}/{actual_value}"
                )
            checked += 1
    print(f"Hungarian solver verification passed: {checked} matrices")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--edc_output")
    parser.add_argument("--reference")
    parser.add_argument("--max_length_diff", default=None)
    parser.add_argument(
        "--evaluation_output",
        default=None,
        help="Path to save evaluation metrics. Defaults to evaluation_scores.txt next to --edc_output.",
    )
    parser.add_argument(
        "--verify_solver",
        action="store_true",
        help="Compare the Hungarian solver with exhaustive search on small matrices and exit.",
    )
    args = parser.parse_args()

    if args.verify_solver:
        verify_solver()
        raise SystemExit(0)
    if not args.edc_output or not args.reference:
        parser.error("--edc_output and --reference are required unless --verify_solver is used")

    legacy_module = _load_legacy()
    evaluation_output_path = legacy_module.resolve_evaluation_output_path(
        args.edc_output, args.evaluation_output
    )
    os.makedirs(os.path.dirname(evaluation_output_path), exist_ok=True)
    result_xml_path, gold_xml_path = legacy_module.convert_to_xml(
        args.edc_output, args.reference, args.max_length_diff
    )

    with open(evaluation_output_path, "w", encoding="utf-8") as evaluation_log:
        original_stdout = sys.stdout
        sys.stdout = legacy_module._Tee(original_stdout, evaluation_log)
        try:
            main(gold_xml_path, result_xml_path)
        finally:
            sys.stdout = original_stdout

    print(f"Evaluation scores saved to: {evaluation_output_path}")
