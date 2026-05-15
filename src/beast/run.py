from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
import numpy as np
from tqdm import tqdm

from beast.tanimoto_cpp import (
    calculate_tanimoto_score_packed_u8_unchecked,
    calculate_tanimoto_score_for_hits_packed,
    calculate_overlap_union_packed_with_scores,
)


def _make_thread_buffer_cache():
    tls = local()

    def get_buffer(name, shape, dtype):
        cache = getattr(tls, "buffers", None)
        if cache is None:
            cache = {}
            tls.buffers = cache
        key = (name, shape, np.dtype(dtype))
        buf = cache.get(key)
        if buf is None:
            buf = np.empty(shape, dtype=dtype)
            cache[key] = buf
        return buf

    return get_buffer


def prepare_query(x_q):

    x_q = np.asarray(x_q)
    if x_q.ndim == 1:
        x_q = x_q[None, :]
    elif x_q.ndim != 2:
        raise ValueError("x_q must be a 1D or 2D array.")

    x_q_packed = np.packbits(x_q, axis=1, bitorder="big")
    ones_q = x_q.sum(axis=1)

    return x_q_packed, ones_q


def _is_single_query_input(x_q) -> bool:
    x_q = np.asarray(x_q)
    return x_q.ndim == 1 or len(x_q) == 1


def _query_ids_allowed_for_on_bits(
    onesQ: np.ndarray,
    on_bits: int,
    fp_size_bits: int,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    onesQ = np.asarray(onesQ, dtype=np.float32)
    onesA = float(on_bits)

    if lower_bound > upper_bound:
        return np.array([], dtype=np.int64)

    allowed = onesQ > 0.0

    if lower_bound > 0.0:
        max_ab = np.maximum(onesA, onesQ)
        max_tanimoto = np.divide(
            np.minimum(onesA, onesQ),
            max_ab,
            out=np.zeros_like(onesQ, dtype=np.float32),
            where=max_ab > 0.0,
        )
        allowed &= max_tanimoto >= lower_bound

    if upper_bound < 1.0:
        min_inter = np.maximum(0.0, onesA + onesQ - float(fp_size_bits))
        denom = onesA + onesQ - min_inter
        min_tanimoto = np.divide(
            min_inter,
            denom,
            out=np.zeros_like(onesQ, dtype=np.float32),
            where=denom > 0.0,
        )
        allowed &= min_tanimoto <= upper_bound

    return np.flatnonzero(allowed)


def _get_smiles_for_chunk_query_hits(x_b, chunk_idx: int, query_hits):
    if not query_hits:
        return {}

    lengths = [len(positions) for _, positions in query_hits]
    if len(query_hits) == 1:
        query_idx, positions = query_hits[0]
        smiles = x_b.get_smiles_and_ids_by_ref_indices({chunk_idx: positions})[
            chunk_idx
        ]
        return {query_idx: smiles}

    all_positions = np.concatenate([positions for _, positions in query_hits])
    all_smiles = x_b.get_smiles_and_ids_by_ref_indices({chunk_idx: all_positions})[
        chunk_idx
    ]

    smiles_by_query = {}
    offset = 0
    for (query_idx, _), length in zip(query_hits, lengths):
        next_offset = offset + length
        smiles_by_query[query_idx] = all_smiles[offset:next_offset]
        offset = next_offset

    return smiles_by_query


def _format_tsv_cell(value) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _format_smiles_block_tsv_chunks(x_b, block, batch_size: int = 8192):
    chunk_idx, query_ids, positions_all, scores_all = block
    smiles_rows = x_b.get_smiles_and_ids_by_ref_indices({chunk_idx: positions_all})[
        chunk_idx
    ]

    line_buffer = []
    text_chunks = []

    def flush_lines():
        if line_buffer:
            text_chunks.append("".join(line_buffer))
            line_buffer.clear()

    for query_idx, position, score, (smiles, compound_id) in zip(
        query_ids,
        positions_all,
        scores_all,
        smiles_rows,
    ):
        line_buffer.append(
            f"{chunk_idx}\t{int(query_idx)}\t{int(position)}\t"
            f"{int(score)}\t{_format_tsv_cell(smiles)}\t"
            f"{_format_tsv_cell(compound_id)}\n"
        )
        if len(line_buffer) >= batch_size:
            flush_lines()

    flush_lines()
    return text_chunks


def get_sim_scores(
    x_b,
    x_q,
    row_width=None,
    output_memmap=None,
    num_workers=8,
    threads_per_worker: int = 6,
):
    """Compute similarity scores for all compounds in the database.

    Args:
        x_b: Database-like object with ``chunk_info`` metadata and ``__len__``.
            Each chunk entry is expected to contain the chunk path, chunk size,
            on-bit count, and output offsets.
        x_q: Query fingerprint array. Accepts a 1D fingerprint or a single-row
            2D array. Inputs are bit-packed internally.
        row_width: Reserved argument. Present in the API but not used by the
            current implementation.
        output_memmap: Optional filesystem path where the score vector should
            be written as a NumPy memmap. If omitted, scores are kept in memory.
        num_workers: Number of Python worker threads used to process chunks.
        threads_per_worker: Number of native threads passed to the packed
            Tanimoto kernel for each chunk.

    Returns:
        A uint8 NumPy array or memmap containing one centi-score per database
        row. Values are stored as ``floor(score * 100 + 0.5)`` in the range
        0..100.
    """
    x_q = np.asarray(x_q)
    if _is_single_query_input(x_q):
        x_q = x_q.reshape(-1)
    else:
        raise ValueError(
            "get_sim_scores currently supports only a single query fingerprint. "
            "Pass a 1D query or a single-row 2D query."
        )

    x_q_packed, onesQ = prepare_query(x_q)
    onesQ_scalar = int(onesQ[0])
    database_size = len(x_b)
    arguments_raw = x_b.chunk_info

    arguments = [
        (path, batch_size, on_bits, start, end)
        for i, path, batch_size, on_bits, start, end in arguments_raw
    ]

    # arguments = arguments[:2]
    def _worker(
        path: str,
        batch_size: int,
        on_bits: int,
        start: int,
        end: int,
    ):
        if batch_size == 0:
            return batch_size

        chunk = np.load(path, mmap_mode="r")

        if not chunk.flags.c_contiguous:
            chunk = np.ascontiguousarray(chunk, dtype=np.uint8)

        score_view = scores_flat[start:end]
        onesA = on_bits
        calculate_tanimoto_score_packed_u8_unchecked(
            chunk,
            x_q_packed,
            onesQ_scalar,
            onesA,
            score_view,
            n_threads=threads_per_worker,
            fp_size=chunk.shape[1],
            n_rows=chunk.shape[0],
        )
        return batch_size

    if output_memmap is not None:
        scores_flat = np.memmap(
            filename=output_memmap, dtype=np.uint8, shape=(database_size,), mode="w+"
        )
    else:
        scores_flat = np.empty(database_size, dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=num_workers) as ex, tqdm(
        total=database_size
    ) as pbar:
        futures = [ex.submit(_worker, *args) for args in arguments]
        for fut in as_completed(futures):
            batch_size = fut.result()
            pbar.update(batch_size)

    return scores_flat


def get_sim_compounds_single_query(
    x_b,
    x_q,
    lower_bound: float,
    upper_bound: float,
    return_smiles: bool = False,
    num_workers: int = 8,
    threads_per_worker: int = 6,
):
    """Find compounds similar to a single query fingerprint.

    Args:
        x_b: Database-like object with ``chunk_info``, ``fp_size``, ``__len__``,
            and ``get_smiles_and_ids_by_ref_indices``.
        x_q: Query fingerprint array. Expected to describe a single query,
            although 1D and 2D inputs are accepted and normalized internally.
        lower_bound: Inclusive minimum Tanimoto similarity threshold.
        upper_bound: Inclusive maximum Tanimoto similarity threshold.
        return_smiles: If ``True``, return SMILES and identifiers for hits.
            Otherwise, return hit positions within each chunk.
        num_workers: Number of Python worker threads used to process chunks.
        threads_per_worker: Number of native threads passed to the packed hit
            search kernel for each chunk.

    Returns:
        A mapping keyed by chunk index. Each value contains hit scores and
        either hit positions or SMILES payloads.
    """

    x_q = np.asarray(x_q)
    query_fp_size = x_q.shape[0] if x_q.ndim == 1 else x_q.shape[1]
    if query_fp_size != x_b.fp_size:
        print(
            f"Warning: length of query fp: {query_fp_size}, length of db fps: {x_b.fp_size}"
        )

    x_q_packed, onesQ = prepare_query(x_q)

    database_size = len(x_b)
    arguments_raw = x_b.chunk_info

    arguments = []
    max_batch = 0  # we need it for effective buffer caching in workers
    for i, path, batch_size, on_bits, start, end in arguments_raw:
        arguments.append((i, path, batch_size, on_bits, start, end))
        if batch_size > max_batch:
            max_batch = batch_size

    search_results = defaultdict(dict)
    buffer_cache = _make_thread_buffer_cache()

    bit_bound_range = [onesQ * lower_bound, onesQ / lower_bound]

    def _worker(idx: int, path: str, batch_size: int, on_bits: int, *args, **kwargs):
        if (
            batch_size == 0
            or on_bits < bit_bound_range[0]
            or on_bits > bit_bound_range[1]
        ):
            return batch_size

        chunk = np.load(path, mmap_mode="r")
        if not chunk.flags.c_contiguous:
            chunk = np.ascontiguousarray(chunk, dtype=np.uint8)

        n_rows = chunk.shape[0]
        # if we use n_rows, it can explode memory if there are a lot of different batch sizes
        score_view = buffer_cache("score", (max_batch,), np.float32)
        score_view = score_view[:n_rows]

        hit_positions = buffer_cache("hit_positions", (max_batch,), np.uint32)
        hit_positions = hit_positions[:n_rows]

        onesA = on_bits
        n_hits = calculate_tanimoto_score_for_hits_packed(
            chunk,
            x_q_packed,
            onesQ,
            onesA,
            score_view,
            n_threads=threads_per_worker,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            hit_positions=hit_positions,
            fp_size=chunk.shape[1],  # size of packed, e.g for 128 it should be 128 // 8
            n_rows=chunk.shape[0],
        )

        if n_hits:
            positions = hit_positions[:n_hits]
            search_results[idx]["score"] = score_view[positions].copy()
            if return_smiles:
                search_results[idx]["smiles"] = (
                    x_b.get_smiles_and_ids_by_ref_indices({idx: positions})[idx]
                )
            else:
                search_results[idx]["positions"] = positions.copy()

        return batch_size

    with ThreadPoolExecutor(max_workers=num_workers) as ex, tqdm(
        total=database_size
    ) as pbar:
        futures = [ex.submit(_worker, *args) for args in arguments]
        for fut in as_completed(futures):
            batch_size = fut.result()
            pbar.update(batch_size)

    return search_results


def get_sim_compounds(
    x_b,
    x_q,
    lower_bound: float,
    upper_bound: float,
    return_smiles: bool = False,
    num_workers: int = 8,
    threads_per_worker: int = 6,
):
    """Dispatch to the single-query or multi-query similarity search API."""

    if _is_single_query_input(x_q):
        return get_sim_compounds_single_query(
            x_b,
            x_q,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            return_smiles=return_smiles,
            num_workers=num_workers,
            threads_per_worker=threads_per_worker,
        )

    return get_sim_compounds_multi_query(
        x_b,
        x_q,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        return_smiles=return_smiles,
        num_workers=num_workers,
        threads_per_worker=threads_per_worker,
    )


def get_sim_compounds_multi_query(
    x_b,
    x_q,
    lower_bound: float,
    upper_bound: float,
    return_smiles: bool = False,
    hit_capacity_per_query: int = 10,
    min_hit_capacity_per_chunk: int = 10_000,
    max_hit_capacity_per_chunk: int = 1_000_000,
    num_workers: int = 8,
    threads_per_worker: int = 6,
):
    """Find compounds similar to multiple query fingerprints.

    Args:
        x_b: Database-like object with ``chunk_info``, ``__len__``, and
            ``get_smiles_and_ids_by_ref_indices``.
        x_q: 2D query fingerprint array where each row is a query fingerprint.
            Inputs are bit-packed internally before searching.
        lower_bound: Inclusive minimum Tanimoto similarity threshold.
        upper_bound: Inclusive maximum Tanimoto similarity threshold.
        return_smiles: If ``True``, return SMILES and identifiers for hits.
            Otherwise, return hit positions within each chunk for each query.
        num_workers: Number of Python worker threads used to process chunks.
        threads_per_worker: Number of native threads passed to the packed
            multi-query overlap kernel for each chunk.

    Returns:
        A mapping keyed by chunk index, then by query index, containing hit
        positions or SMILES payloads for each query that matched in that chunk.
    """

    x_q_arr = np.asarray(x_q)
    if hit_capacity_per_query < 1:
        raise ValueError("hit_capacity_per_query must be positive.")
    if min_hit_capacity_per_chunk < 1:
        raise ValueError("min_hit_capacity_per_chunk must be positive.")
    if max_hit_capacity_per_chunk < min_hit_capacity_per_chunk:
        raise ValueError(
            "max_hit_capacity_per_chunk must be greater than or equal to "
            "min_hit_capacity_per_chunk."
        )

    x_q_packed, onesQ = prepare_query(x_q_arr)
    onesQ_u32 = np.ascontiguousarray(onesQ, dtype=np.uint32)
    fp_size_bits = x_q_arr.shape[1] if x_q_arr.ndim == 2 else x_q_arr.shape[0]

    database_size = len(x_b)
    arguments_raw = x_b.chunk_info

    arguments = []
    query_payload_by_on_bits = {}
    for i, path, batch_size, on_bits, start, end in arguments_raw:
        arguments.append((i, path, batch_size, on_bits, start, end))
        if on_bits not in query_payload_by_on_bits:
            query_ids = _query_ids_allowed_for_on_bits(
                onesQ,
                on_bits,
                fp_size_bits,
                lower_bound,
                upper_bound,
            )
            query_payload_by_on_bits[on_bits] = (
                query_ids,
                np.ascontiguousarray(x_q_packed[query_ids], dtype=np.uint8),
                np.ascontiguousarray(onesQ_u32[query_ids], dtype=np.uint32),
            )

    search_results = defaultdict(lambda: defaultdict(dict))

    def _worker(idx: int, path: str, batch_size: int, on_bits: int, *args, **kwargs):
        query_ids, query_bytes, active_onesQ = query_payload_by_on_bits[on_bits]
        n_active_queries = len(query_ids)
        if batch_size == 0 or n_active_queries == 0:
            return batch_size

        chunk = np.load(path, mmap_mode="r")
        onesA = on_bits

        hit_capacity = min(
            max_hit_capacity_per_chunk,
            max(min_hit_capacity_per_chunk, n_active_queries * hit_capacity_per_query),
        )
        hit_query_ids = np.empty(hit_capacity, dtype=np.uint32)
        hit_positions = np.empty(hit_capacity, dtype=np.uint32)
        hit_scores = np.empty(hit_capacity, dtype=np.uint8)
        hit_count = np.zeros(1, dtype=np.uint64)
        overflow = np.zeros(1, dtype=np.uint8)

        n_hits = calculate_overlap_union_packed_with_scores(
            chunk,
            query_bytes,
            active_onesQ,
            onesA,
            lower_bound,
            upper_bound,
            hit_query_ids=hit_query_ids,
            hit_positions=hit_positions,
            hit_scores=hit_scores,
            hit_count=hit_count,
            overflow=overflow,
            n_threads=threads_per_worker,
        )

        if overflow[0]:
            actual_hits = int(hit_count[0])
            raise MemoryError(
                f"Chunk {idx} produced {actual_hits} hits, exceeding compact "
                f"hit_capacity_per_chunk={hit_capacity}. Increase "
                "hit_capacity_per_query or max_hit_capacity_per_chunk."
            )

        if n_hits:
            local_results = defaultdict(dict)
            smiles_hits = []
            n_hits = int(n_hits)
            local_hit_query_ids = hit_query_ids[:n_hits]
            local_hit_positions = hit_positions[:n_hits]
            for local_query_idx in np.unique(local_hit_query_ids):
                hit_mask = local_hit_query_ids == local_query_idx
                positions = np.sort(local_hit_positions[hit_mask]).astype(
                    np.uint32,
                    copy=False,
                )
                query_idx = int(query_ids[int(local_query_idx)])
                if return_smiles:
                    smiles_hits.append((query_idx, positions))
                else:
                    local_results[query_idx]["positions"] = positions

            if return_smiles:
                smiles_by_query = _get_smiles_for_chunk_query_hits(
                    x_b, idx, smiles_hits
                )
                for query_idx, smiles in smiles_by_query.items():
                    local_results[query_idx]["smiles"] = smiles

            search_results[idx] = local_results

        return batch_size

    with ThreadPoolExecutor(max_workers=num_workers) as ex, tqdm(
        total=database_size
    ) as pbar:
        futures = [ex.submit(_worker, *args) for args in arguments]
        for fut in as_completed(futures):
            batch_size = fut.result()
            pbar.update(batch_size)

    return search_results


def get_sim_compounds_multi_query_with_scores(
    x_b,
    x_q,
    lower_bound: float,
    upper_bound: float,
    smiles_output_path,
    hit_capacity_per_query: int = 10,
    min_hit_capacity_per_chunk: int = 10_000,
    max_hit_capacity_per_chunk: int = 1_000_000,
    num_workers: int = 8,
    threads_per_worker: int = 6,
):
    """Find multi-query similarity hits and stream them to a TSV file.

    Returns:
        Path to the written TSV file. Rows contain chunk index, query index,
        position within chunk, uint8 centi-score, SMILES, and compound id.
    """

    if smiles_output_path is None:
        raise ValueError("smiles_output_path is required for scored multi-query search.")

    x_q_arr = np.asarray(x_q)
    if _is_single_query_input(x_q_arr):
        raise ValueError(
            "get_sim_compounds_multi_query_with_scores expects multiple queries. "
            "Pass a 2D query array with at least two rows."
        )
    if lower_bound > upper_bound:
        raise ValueError("lower_bound must be less than or equal to upper_bound.")
    if hit_capacity_per_query < 1:
        raise ValueError("hit_capacity_per_query must be positive.")
    if min_hit_capacity_per_chunk < 1:
        raise ValueError("min_hit_capacity_per_chunk must be positive.")
    if max_hit_capacity_per_chunk < min_hit_capacity_per_chunk:
        raise ValueError(
            "max_hit_capacity_per_chunk must be greater than or equal to "
            "min_hit_capacity_per_chunk."
        )

    x_q_packed, onesQ = prepare_query(x_q_arr)
    onesQ_u32 = np.ascontiguousarray(onesQ, dtype=np.uint32)
    fp_size_bits = x_q_arr.shape[1]

    database_size = len(x_b)
    arguments_raw = x_b.chunk_info

    arguments = []
    query_payload_by_on_bits = {}
    for i, path, batch_size, on_bits, start, end in arguments_raw:
        arguments.append((i, path, batch_size, on_bits, start, end))
        if on_bits not in query_payload_by_on_bits:
            query_ids = _query_ids_allowed_for_on_bits(
                onesQ,
                on_bits,
                fp_size_bits,
                lower_bound,
                upper_bound,
            )
            query_payload_by_on_bits[on_bits] = (
                query_ids,
                np.ascontiguousarray(x_q_packed[query_ids], dtype=np.uint8),
                np.ascontiguousarray(onesQ_u32[query_ids], dtype=np.uint32),
            )

    output_path = Path(smiles_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _worker(idx: int, path: str, batch_size: int, on_bits: int, *args, **kwargs):
        query_ids, query_bytes, active_onesQ = query_payload_by_on_bits[on_bits]
        n_active_queries = len(query_ids)
        if batch_size == 0 or n_active_queries == 0:
            return batch_size, []

        chunk = np.load(path, mmap_mode="r")
        onesA = on_bits

        hit_capacity = min(
            max_hit_capacity_per_chunk,
            max(min_hit_capacity_per_chunk, n_active_queries * hit_capacity_per_query),
        )
        hit_query_ids = np.empty(hit_capacity, dtype=np.uint32)
        hit_positions = np.empty(hit_capacity, dtype=np.uint32)
        hit_scores = np.empty(hit_capacity, dtype=np.uint8)
        hit_count = np.zeros(1, dtype=np.uint64)
        overflow = np.zeros(1, dtype=np.uint8)

        n_hits = calculate_overlap_union_packed_with_scores(
            chunk,
            query_bytes,
            active_onesQ,
            onesA,
            lower_bound,
            upper_bound,
            hit_query_ids=hit_query_ids,
            hit_positions=hit_positions,
            hit_scores=hit_scores,
            hit_count=hit_count,
            overflow=overflow,
            n_threads=threads_per_worker,
        )

        if overflow[0]:
            actual_hits = int(hit_count[0])
            raise MemoryError(
                f"Chunk {idx} produced {actual_hits} hits, exceeding compact "
                f"hit_capacity_per_chunk={hit_capacity}. Increase "
                "hit_capacity_per_query or max_hit_capacity_per_chunk."
            )

        if n_hits:
            n_hits = int(n_hits)
            block = (
                idx,
                query_ids[hit_query_ids[:n_hits]].astype(np.int64, copy=False),
                hit_positions[:n_hits].copy(),
                hit_scores[:n_hits].copy(),
            )
            return batch_size, _format_smiles_block_tsv_chunks(x_b, block)

        return batch_size, []

    with output_path.open("w", encoding="utf-8") as out:
        out.write("chunk_idx\tquery_idx\tposition\tscore\tsmiles\tcompound_id\n")
        with ThreadPoolExecutor(max_workers=num_workers) as ex, tqdm(
            total=database_size
        ) as pbar:
            futures = [ex.submit(_worker, *args) for args in arguments]
            for fut in as_completed(futures):
                batch_size, text_chunks = fut.result()
                for text_chunk in text_chunks:
                    out.write(text_chunk)
                pbar.update(batch_size)

    return output_path
