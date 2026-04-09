import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _resolve_table_path() -> Path:
    env_path = os.environ.get("CHEMICAL_SCREENER_TEST_TABLE")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            REPO_ROOT / "tmp/pipeline_test/tables/table_128.csv",
            REPO_ROOT / "parser/data/tables/table_128.csv",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    pytest.skip(
        "No test manifest found. Generate tmp/pipeline_test/tables/table_128.csv "
        "via the README pipeline section, or set CHEMICAL_SCREENER_TEST_TABLE.",
        allow_module_level=True,
    )


TABLE_PATH = _resolve_table_path()

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beast.database import DataBase, ZINC
from beast.run import get_sim_compounds, get_sim_scores
from beast.tanimoto_cpp import calculate_tanimoto_score_packed_f16


def _manual_scores_from_packed(
    chunk: np.ndarray, query_bytes: np.ndarray, ones_a: int
) -> np.ndarray:
    query_bits = np.unpackbits(query_bytes, bitorder="big")
    chunk_bits = np.unpackbits(chunk, axis=1, bitorder="big")
    intersections = np.count_nonzero(chunk_bits & query_bits, axis=1)
    ones_q = int(query_bits.sum())
    denominators = ones_q + int(ones_a) - intersections
    scores = np.divide(
        intersections,
        denominators,
        out=np.ones_like(intersections, dtype=np.float32),
        where=denominators != 0,
    )
    return scores.astype(np.float32)


def _manual_hits_for_query(
    database: ZINC, query_bits: np.ndarray, lower_bound: float, upper_bound: float
) -> dict[int, dict[str, np.ndarray]]:
    query_bytes = np.packbits(query_bits, bitorder="big")
    expected = {}

    for chunk_idx, path, _, on_bits, _, _ in database.chunk_info:
        chunk = np.load(path, mmap_mode="r")
        scores = _manual_scores_from_packed(chunk, query_bytes, int(on_bits)).astype(
            np.float32
        )
        hit_mask = (scores >= lower_bound) & (scores <= upper_bound)
        if np.any(hit_mask):
            expected[chunk_idx] = {
                "positions": np.flatnonzero(hit_mask).astype(np.uint32),
                "score": scores[hit_mask],
            }

    return expected


def _manual_positions_for_queries(
    database: ZINC, queries_bits: np.ndarray, lower_bound: float, upper_bound: float
) -> dict[int, dict[int, np.ndarray]]:
    expected = {}

    for query_idx, query_bits in enumerate(queries_bits):
        query_hits = _manual_hits_for_query(database, query_bits, lower_bound, upper_bound)
        for chunk_idx, chunk_hits in query_hits.items():
            expected.setdefault(chunk_idx, {})[query_idx] = chunk_hits["positions"]

    return expected


@pytest.fixture(scope="module")
def zinc_manifest() -> pd.DataFrame:
    return pd.read_csv(TABLE_PATH)


@pytest.mark.parametrize(
    ("manifest_index", "query_row", "limit"),
    [
        (0, 0, 2),
        (10, 0, 64),
        (100, 5, 64),
    ],
)
def test_score_fp_matches_manual_scores_for_real_chunks(
    zinc_manifest: pd.DataFrame, manifest_index: int, query_row: int, limit: int
) -> None:
    manifest_row = zinc_manifest.iloc[manifest_index]
    chunk = np.load(manifest_row["npy"], mmap_mode="r")[:limit]
    chunk = np.ascontiguousarray(chunk, dtype=np.uint8)
    query_bytes = np.ascontiguousarray(chunk[query_row], dtype=np.uint8)

    scores = np.empty(chunk.shape[0], dtype=np.float16)
    ones_q = int(np.unpackbits(query_bytes, bitorder="big").sum())
    calculate_tanimoto_score_packed_f16(
        chunk,
        query_bytes,
        ones_q,
        int(manifest_row["onbits"]),
        scores,
        n_threads=1,
        fp_size=chunk.shape[1],
        n_rows=chunk.shape[0],
    )

    expected = _manual_scores_from_packed(
        chunk, query_bytes, int(manifest_row["onbits"])
    ).astype(np.float16)
    np.testing.assert_array_equal(scores, expected)
    assert scores[query_row] == np.float16(1.0)


def test_get_sim_scores_matches_manual_scores_from_zinc_table() -> None:
    database = ZINC(TABLE_PATH, max_files=3)
    first_chunk = np.load(database.fp_files[0], mmap_mode="r")
    query_bits = np.unpackbits(first_chunk[0], bitorder="big").astype(np.uint8)
    query_bytes = np.packbits(query_bits, bitorder="big")

    scores = get_sim_scores(
        database, query_bits, num_workers=1, threads_per_worker=1
    )

    expected_parts = []
    for _, path, _, on_bits, _, _ in database.chunk_info:
        chunk = np.load(path, mmap_mode="r")
        expected_parts.append(
            _manual_scores_from_packed(chunk, query_bytes, int(on_bits)).astype(
                np.float16
            )
        )
    expected = np.concatenate(expected_parts)

    assert scores.shape == (len(database),)
    assert scores.dtype == np.float16
    np.testing.assert_array_equal(scores, expected)


def test_database_alias_matches_zinc_shape_and_chunk_info() -> None:
    zinc = ZINC(TABLE_PATH, max_files=3)
    database = DataBase(TABLE_PATH, max_files=3)

    assert len(database) == len(zinc)
    assert database.fp_size == zinc.fp_size
    assert list(database.fp_files) == list(zinc.fp_files)
    assert database.chunk_info == zinc.chunk_info


def test_get_sim_compounds_single_query_matches_manual_hits() -> None:
    database = ZINC(TABLE_PATH, max_files=3)
    first_chunk = np.load(database.fp_files[0], mmap_mode="r")
    query_bits = np.unpackbits(first_chunk[0], bitorder="big").astype(np.uint8)
    lower_bound = 0.2
    upper_bound = 1.0

    hits = get_sim_compounds(
        database,
        query_bits,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_workers=1,
        threads_per_worker=1,
    )
    expected = _manual_hits_for_query(database, query_bits, lower_bound, upper_bound)

    assert set(hits.keys()) == set(expected.keys())
    for chunk_idx, chunk_hits in expected.items():
        np.testing.assert_array_equal(
            hits[chunk_idx]["positions"], chunk_hits["positions"]
        )
        np.testing.assert_allclose(hits[chunk_idx]["score"], chunk_hits["score"])


def test_get_sim_compounds_single_query_return_smiles_is_direct_payload() -> None:
    database = ZINC(TABLE_PATH, max_files=3)
    first_chunk = np.load(database.fp_files[0], mmap_mode="r")
    query_bits = np.unpackbits(first_chunk[0], bitorder="big").astype(np.uint8)
    lower_bound = 0.2
    upper_bound = 1.0

    hits = get_sim_compounds(
        database,
        query_bits,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        return_smiles=True,
        num_workers=1,
        threads_per_worker=1,
    )
    expected = _manual_hits_for_query(database, query_bits, lower_bound, upper_bound)

    assert set(hits.keys()) == set(expected.keys())
    for chunk_idx, chunk_hits in expected.items():
        expected_smiles = database.get_smiles_and_ids_by_ref_indices(
            {chunk_idx: chunk_hits["positions"]}
        )[chunk_idx]
        assert isinstance(hits[chunk_idx]["smiles"], list)
        assert hits[chunk_idx]["smiles"] == expected_smiles


def test_get_sim_compounds_multi_query_matches_manual_positions() -> None:
    database = ZINC(TABLE_PATH, max_files=3)
    first_chunk = np.load(database.fp_files[0], mmap_mode="r")
    queries_bits = np.vstack(
        [
            np.unpackbits(first_chunk[0], bitorder="big"),
            np.unpackbits(first_chunk[1], bitorder="big"),
        ]
    ).astype(np.uint8)
    lower_bound = 0.2
    upper_bound = 1.0

    hits = get_sim_compounds(
        database,
        queries_bits,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        num_workers=1,
        threads_per_worker=1,
    )
    expected = _manual_positions_for_queries(
        database, queries_bits, lower_bound, upper_bound
    )

    assert set(hits.keys()) == set(expected.keys())
    for chunk_idx, chunk_hits in expected.items():
        assert set(hits[chunk_idx].keys()) == set(chunk_hits.keys())
        for query_idx, positions in chunk_hits.items():
            np.testing.assert_array_equal(
                hits[chunk_idx][query_idx]["positions"], positions
            )


def test_get_sim_compounds_multi_query_return_smiles_is_direct_payload() -> None:
    database = ZINC(TABLE_PATH, max_files=3)
    first_chunk = np.load(database.fp_files[0], mmap_mode="r")
    queries_bits = np.vstack(
        [
            np.unpackbits(first_chunk[0], bitorder="big"),
            np.unpackbits(first_chunk[1], bitorder="big"),
        ]
    ).astype(np.uint8)
    lower_bound = 0.2
    upper_bound = 1.0

    hits = get_sim_compounds(
        database,
        queries_bits,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        return_smiles=True,
        num_workers=1,
        threads_per_worker=1,
    )
    expected = _manual_positions_for_queries(
        database, queries_bits, lower_bound, upper_bound
    )

    assert set(hits.keys()) == set(expected.keys())
    for chunk_idx, chunk_hits in expected.items():
        assert set(hits[chunk_idx].keys()) == set(chunk_hits.keys())
        for query_idx, positions in chunk_hits.items():
            expected_smiles = database.get_smiles_and_ids_by_ref_indices(
                {chunk_idx: positions}
            )[chunk_idx]
            assert isinstance(hits[chunk_idx][query_idx]["smiles"], list)
            assert hits[chunk_idx][query_idx]["smiles"] == expected_smiles
