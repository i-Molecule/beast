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
from beast.run import (
    _query_ids_allowed_for_on_bits,
    get_sim_compounds,
    get_sim_compounds_multi_query_with_scores,
    get_sim_scores,
)
from beast.tanimoto_cpp import (
    calculate_overlap_union_packed_with_scores,
    calculate_tanimoto_score_packed_f16,
    calculate_tanimoto_score_packed_u8,
)


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


def _quantize_scores_u8(scores: np.ndarray) -> np.ndarray:
    return np.floor(scores.astype(np.float32) * 100.0 + 0.5).clip(0, 100).astype(
        np.uint8
    )


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


def _manual_positions_and_scores_for_queries(
    database: ZINC,
    queries_bits: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    expected = {}

    for query_idx, query_bits in enumerate(queries_bits):
        query_bytes = np.packbits(query_bits, bitorder="big")
        for chunk_idx, path, _, on_bits, _, _ in database.chunk_info:
            chunk = np.load(path, mmap_mode="r")
            scores = _manual_scores_from_packed(
                chunk, query_bytes, int(on_bits)
            ).astype(np.float32)
            hit_mask = (scores >= lower_bound) & (scores <= upper_bound)
            if np.any(hit_mask):
                expected.setdefault(chunk_idx, {})[query_idx] = {
                    "positions": np.flatnonzero(hit_mask).astype(np.uint32),
                    "scores": _quantize_scores_u8(scores[hit_mask]),
                }

    return expected


def _expected_tsv_hit_tuples(expected):
    return {
        (int(chunk_idx), int(query_idx), int(position), int(score))
        for chunk_idx, chunk_hits in expected.items()
        for query_idx, expected_hit in chunk_hits.items()
        for position, score in zip(expected_hit["positions"], expected_hit["scores"])
    }


def _read_tsv_hit_tuples(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "chunk_idx\tquery_idx\tposition\tscore\tsmiles\tcompound_id"
    return {
        (int(chunk_idx), int(query_idx), int(position), int(score))
        for chunk_idx, query_idx, position, score, *_ in (
            line.split("\t") for line in lines[1:]
        )
    }


def test_query_ids_allowed_for_on_bits_filters_impossible_queries() -> None:
    ones_q = np.array([10, 50], dtype=np.float32)

    query_ids = _query_ids_allowed_for_on_bits(
        ones_q,
        on_bits=60,
        fp_size_bits=128,
        lower_bound=0.8,
        upper_bound=1.0,
    )

    np.testing.assert_array_equal(query_ids, np.array([1]))


def test_query_ids_allowed_for_on_bits_respects_upper_bound() -> None:
    ones_q = np.array([8], dtype=np.float32)

    query_ids = _query_ids_allowed_for_on_bits(
        ones_q,
        on_bits=8,
        fp_size_bits=10,
        lower_bound=0.0,
        upper_bound=0.5,
    )

    assert query_ids.size == 0


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


def test_score_packed_u8_matches_manual_quantized_scores(
    zinc_manifest: pd.DataFrame,
) -> None:
    manifest_row = zinc_manifest.iloc[0]
    chunk = np.load(manifest_row["npy"], mmap_mode="r")[:64]
    chunk = np.ascontiguousarray(chunk, dtype=np.uint8)
    query_bytes = np.ascontiguousarray(chunk[0], dtype=np.uint8)

    scores = np.empty(chunk.shape[0], dtype=np.uint8)
    ones_q = int(np.unpackbits(query_bytes, bitorder="big").sum())
    calculate_tanimoto_score_packed_u8(
        chunk,
        query_bytes,
        ones_q,
        int(manifest_row["onbits"]),
        scores,
        n_threads=1,
        fp_size=chunk.shape[1],
        n_rows=chunk.shape[0],
    )

    expected = _quantize_scores_u8(
        _manual_scores_from_packed(chunk, query_bytes, int(manifest_row["onbits"]))
    )
    np.testing.assert_array_equal(scores, expected)
    assert scores[0] == np.uint8(100)


def test_score_packed_u8_uses_half_up_integer_rounding() -> None:
    query_bytes = np.array([0b11111111], dtype=np.uint8)
    chunk = np.array(
        [
            [0b00000000],
            [0b10000000],
            [0b11110000],
            [0b11111111],
        ],
        dtype=np.uint8,
    )
    scores = np.empty(chunk.shape[0], dtype=np.uint8)

    calculate_tanimoto_score_packed_u8(
        chunk,
        query_bytes,
        onesQ=8,
        onesA=8,
        scores_out=scores,
        n_threads=1,
        fp_size=chunk.shape[1],
        n_rows=chunk.shape[0],
    )

    np.testing.assert_array_equal(scores, np.array([0, 7, 33, 100], dtype=np.uint8))


def test_score_packed_u8_handles_unaligned_rows() -> None:
    query_bytes = np.arange(16, dtype=np.uint8)
    base = np.zeros((4, 17), dtype=np.uint8)
    chunk = base[:, 1:]
    chunk[:] = np.array(
        [
            query_bytes,
            np.zeros(16, dtype=np.uint8),
            np.full(16, 0xFF, dtype=np.uint8),
            query_bytes ^ np.uint8(0x55),
        ],
        dtype=np.uint8,
    )
    scores = np.empty(chunk.shape[0], dtype=np.uint8)
    ones_q = int(np.unpackbits(query_bytes, bitorder="big").sum())

    calculate_tanimoto_score_packed_u8(
        chunk,
        query_bytes,
        onesQ=ones_q,
        onesA=ones_q,
        scores_out=scores,
        n_threads=1,
        fp_size=chunk.shape[1],
        n_rows=chunk.shape[0],
    )

    expected = _quantize_scores_u8(
        _manual_scores_from_packed(chunk, query_bytes, ones_q)
    )
    np.testing.assert_array_equal(scores, expected)


@pytest.mark.parametrize("width_bytes", [8, 32, 64])
def test_score_packed_u8_fast_word_widths_match_manual_quantized_scores(
    width_bytes: int,
) -> None:
    rows = 9
    storage = np.arange(rows * width_bytes + 1, dtype=np.uint8)
    chunk = np.ndarray(
        shape=(rows, width_bytes),
        dtype=np.uint8,
        buffer=storage,
        offset=1,
    )
    query_bytes = np.ascontiguousarray(chunk[0], dtype=np.uint8)
    scores = np.empty(rows, dtype=np.uint8)
    ones_q = int(np.unpackbits(query_bytes, bitorder="big").sum())
    ones_a = width_bytes * 8

    calculate_tanimoto_score_packed_u8(
        chunk,
        query_bytes,
        onesQ=ones_q,
        onesA=ones_a,
        scores_out=scores,
        n_threads=1,
        fp_size=chunk.shape[1],
        n_rows=chunk.shape[0],
    )

    expected = _quantize_scores_u8(_manual_scores_from_packed(chunk, query_bytes, ones_a))
    np.testing.assert_array_equal(scores, expected)


def test_overlap_union_packed_with_scores_returns_boundary_scores() -> None:
    chunk_bits = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    query_bits = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    chunk = np.packbits(chunk_bits, axis=1, bitorder="big")
    query_bytes = np.packbits(query_bits, axis=1, bitorder="big")
    ones_q = query_bits.sum(axis=1).astype(np.float32)
    hit_query_ids = np.empty(8, dtype=np.uint32)
    hit_positions = np.empty(8, dtype=np.uint32)
    hit_scores = np.empty(8, dtype=np.uint8)
    hit_count = np.zeros(1, dtype=np.uint64)
    overflow = np.zeros(1, dtype=np.uint8)

    n_hits = calculate_overlap_union_packed_with_scores(
        chunk,
        query_bytes,
        ones_q,
        onesA=4.0,
        lower_bound=0.5,
        upper_bound=1.0,
        hit_query_ids=hit_query_ids,
        hit_positions=hit_positions,
        hit_scores=hit_scores,
        hit_count=hit_count,
        overflow=overflow,
        n_threads=1,
    )

    assert n_hits == 5
    assert hit_count[0] == 5
    assert overflow[0] == 0
    actual = {
        (int(q), int(position), int(score))
        for q, position, score in zip(
            hit_query_ids[:n_hits],
            hit_positions[:n_hits],
            hit_scores[:n_hits],
        )
    }
    assert actual == {
        (0, 0, 100),
        (0, 1, 60),
        (1, 0, 50),
        (1, 1, 50),
        (1, 2, 50),
    }


def test_overlap_union_packed_with_scores_reports_overflow() -> None:
    chunk_bits = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    query_bits = np.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.uint8)
    chunk = np.packbits(chunk_bits, axis=1, bitorder="big")
    query_bytes = np.packbits(query_bits, axis=1, bitorder="big")
    ones_q = query_bits.sum(axis=1).astype(np.float32)
    hit_query_ids = np.empty(1, dtype=np.uint32)
    hit_positions = np.empty(1, dtype=np.uint32)
    hit_scores = np.empty(1, dtype=np.uint8)
    hit_count = np.zeros(1, dtype=np.uint64)
    overflow = np.zeros(1, dtype=np.uint8)

    n_hits = calculate_overlap_union_packed_with_scores(
        chunk,
        query_bytes,
        ones_q,
        onesA=4.0,
        lower_bound=0.5,
        upper_bound=1.0,
        hit_query_ids=hit_query_ids,
        hit_positions=hit_positions,
        hit_scores=hit_scores,
        hit_count=hit_count,
        overflow=overflow,
        n_threads=1,
    )

    assert n_hits == -1
    assert hit_count[0] == 2
    assert overflow[0] == 1


def test_get_sim_scores_matches_manual_quantized_scores() -> None:
    database = ZINC(TABLE_PATH, max_files=3)
    first_chunk = np.load(database.fp_files[0], mmap_mode="r")
    query_bits = np.unpackbits(first_chunk[0], bitorder="big").astype(np.uint8)
    query_bytes = np.packbits(query_bits, bitorder="big")

    scores = get_sim_scores(
        database,
        query_bits,
        num_workers=1,
        threads_per_worker=1,
    )

    expected_parts = []
    for _, path, _, on_bits, _, _ in database.chunk_info:
        chunk = np.load(path, mmap_mode="r")
        expected_parts.append(
            _quantize_scores_u8(
                _manual_scores_from_packed(chunk, query_bytes, int(on_bits))
            )
        )
    expected = np.concatenate(expected_parts)

    assert scores.shape == (len(database),)
    assert scores.dtype == np.uint8
    np.testing.assert_array_equal(scores, expected)


def test_get_sim_scores_memmap_uses_uint8_dtype(tmp_path: Path) -> None:
    database = ZINC(TABLE_PATH, max_files=3)
    first_chunk = np.load(database.fp_files[0], mmap_mode="r")
    query_bits = np.unpackbits(first_chunk[0], bitorder="big").astype(np.uint8)
    output_path = tmp_path / "scores.u8"

    scores = get_sim_scores(
        database,
        query_bits,
        output_memmap=output_path,
        num_workers=1,
        threads_per_worker=1,
    )
    reopened = np.memmap(output_path, dtype=np.uint8, mode="r", shape=(len(database),))

    assert scores.dtype == np.uint8
    assert output_path.stat().st_size == len(database)
    np.testing.assert_array_equal(reopened, scores)


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


def test_get_sim_compounds_multi_query_with_scores_writes_manual_hits(
    tmp_path: Path,
) -> None:
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
    output_path = tmp_path / "hits.tsv"

    written_path = get_sim_compounds_multi_query_with_scores(
        database,
        queries_bits,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        smiles_output_path=output_path,
        num_workers=1,
        threads_per_worker=1,
    )
    expected = _manual_positions_and_scores_for_queries(
        database, queries_bits, lower_bound, upper_bound
    )

    assert written_path == output_path
    assert _read_tsv_hit_tuples(output_path) == _expected_tsv_hit_tuples(expected)


def test_get_sim_compounds_multi_query_with_scores_streams_smiles_to_disk(
    tmp_path: Path,
) -> None:
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
    output_path = tmp_path / "hits.tsv"

    expected = _manual_positions_and_scores_for_queries(
        database, queries_bits, lower_bound, upper_bound
    )
    expected_rows = sum(
        len(expected_hit["positions"])
        for chunk_hits in expected.values()
        for expected_hit in chunk_hits.values()
    )

    original_get_smiles = database.get_smiles_and_ids_by_ref_indices
    smiles_calls = []

    def counting_get_smiles(chunk_to_indices, *args, **kwargs):
        smiles_calls.append(
            {
                int(chunk_idx): np.asarray(indices).copy()
                for chunk_idx, indices in chunk_to_indices.items()
            }
        )
        return original_get_smiles(chunk_to_indices, *args, **kwargs)

    database.get_smiles_and_ids_by_ref_indices = counting_get_smiles

    written_path = get_sim_compounds_multi_query_with_scores(
        database,
        queries_bits,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        smiles_output_path=output_path,
        num_workers=1,
        threads_per_worker=1,
    )

    assert written_path == output_path
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "chunk_idx\tquery_idx\tposition\tscore\tsmiles\tcompound_id"
    assert len(lines) == expected_rows + 1
    assert len(smiles_calls) == len(expected)
    assert {next(iter(call)) for call in smiles_calls} == set(expected)
    assert _read_tsv_hit_tuples(output_path) == _expected_tsv_hit_tuples(expected)


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

    expected = _manual_positions_for_queries(
        database, queries_bits, lower_bound, upper_bound
    )
    expected_smiles = {
        chunk_idx: {
            query_idx: database.get_smiles_and_ids_by_ref_indices(
                {chunk_idx: positions}
            )[chunk_idx]
            for query_idx, positions in chunk_hits.items()
        }
        for chunk_idx, chunk_hits in expected.items()
    }

    original_get_smiles = database.get_smiles_and_ids_by_ref_indices
    smiles_calls = []

    def counting_get_smiles(chunk_to_indices, *args, **kwargs):
        smiles_calls.append(
            {
                int(chunk_idx): np.asarray(indices).copy()
                for chunk_idx, indices in chunk_to_indices.items()
            }
        )
        return original_get_smiles(chunk_to_indices, *args, **kwargs)

    database.get_smiles_and_ids_by_ref_indices = counting_get_smiles

    hits = get_sim_compounds(
        database,
        queries_bits,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        return_smiles=True,
        num_workers=1,
        threads_per_worker=1,
    )

    assert set(hits.keys()) == set(expected.keys())
    for chunk_idx, chunk_hits in expected.items():
        assert set(hits[chunk_idx].keys()) == set(chunk_hits.keys())
        for query_idx, positions in chunk_hits.items():
            assert isinstance(hits[chunk_idx][query_idx]["smiles"], list)
            assert hits[chunk_idx][query_idx]["smiles"] == expected_smiles[chunk_idx][
                query_idx
            ]
    assert len(smiles_calls) == len(expected)
    assert {next(iter(call)) for call in smiles_calls} == set(expected)
