from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import local
from typing import Optional, Union
import numpy as np
from tqdm import tqdm

from beast.tanimoto_cpp import (
    calculate_tanimoto_score_packed_f16,
    calculate_tanimoto_score_for_hits_packed,
    calculate_overlap_union_packed,
)
from pathlib import Path


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
        output_memmap: Optional filesystem path where the float16 score vector
            should be written as a NumPy memmap. If omitted, scores are kept in
            memory.
        num_workers: Number of Python worker threads used to process chunks.
        threads_per_worker: Number of native threads passed to the packed
            Tanimoto kernel for each chunk.

    Returns:
        A float16 NumPy array or memmap containing one score per database row.
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
        calculate_tanimoto_score_packed_f16(
            chunk,
            x_q_packed,
            onesQ,
            onesA,
            score_view,
            n_threads=threads_per_worker,
            fp_size=chunk.shape[1],
            n_rows=chunk.shape[0],
        )
        return batch_size

    if output_memmap is not None:
        scores_flat = np.memmap(
            filename=output_memmap, dtype=np.float16, shape=(database_size,), mode="w+"
        )
    else:
        scores_flat = np.empty(database_size, dtype=np.float16)

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

    x_q_packed, onesQ = prepare_query(x_q)
    bit_bound_range = [min(onesQ) * lower_bound, max(onesQ) / lower_bound]
    n_queries = len(onesQ)

    database_size = len(x_b)
    arguments_raw = x_b.chunk_info

    arguments = []
    max_batch = 0  # we need it for effective buffer caching in workers
    for i, path, batch_size, on_bits, start, end in arguments_raw:
        arguments.append((i, path, batch_size, on_bits, start, end))
        if batch_size > max_batch:
            max_batch = batch_size

    search_results = defaultdict(lambda: defaultdict(dict))
    buffer_cache = _make_thread_buffer_cache()

    def _worker(idx: int, path: str, batch_size: int, on_bits: int, *args, **kwargs):
        if (
            batch_size == 0
            or on_bits < bit_bound_range[0]
            or on_bits > bit_bound_range[1]
        ):
            return batch_size

        chunk = np.load(path, mmap_mode="r")
        onesA = on_bits

        hit_positions = buffer_cache("hit_positions", (n_queries, max_batch), np.uint32)
        # hit_positions = hit_positions[:, : chunk.shape[0]] #does not work because it is not contiguous in memory after slicing

        hit_counts = buffer_cache("hit_counts", (n_queries,), np.uint32)
        # hit_counts.fill(0) # we can skip this because calculate_overlap_union_packed will set hit_counts[i] to 0 for queries that have no hits in the chunk

        n_hits = calculate_overlap_union_packed(
            chunk,
            x_q_packed,
            onesQ,
            onesA,
            lower_bound,
            upper_bound,
            hit_positions=hit_positions,
            hit_counts=hit_counts,
            n_threads=threads_per_worker,
        )

        if n_hits:
            local_results = defaultdict(dict)
            for i in range(n_queries):
                if hit_counts[i] > 0:
                    if return_smiles:
                        local_results[i]["smiles"] = (
                            x_b.get_smiles_and_ids_by_ref_indices(
                                {idx: hit_positions[i, : hit_counts[i]]}
                            )[idx]
                        )
                    else:
                        local_results[i]["positions"] = hit_positions[
                            i, : hit_counts[i]
                        ].copy()

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
