import ctypes
import numpy as np
import os
from numpy.ctypeslib import ndpointer

_lib_path = os.path.join(os.path.dirname(__file__), "libtanimoto.so")
_lib = ctypes.CDLL(_lib_path)


def _load_symbol(name, argtypes, restype):
    try:
        symbol = getattr(_lib, name)
    except AttributeError:
        return None
    symbol.argtypes = argtypes
    symbol.restype = restype
    return symbol

# ---- Definitions ----

_calculate_overlap_union = _load_symbol("calculate_overlap_union", [
    ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # A
    ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), # union_indices
    ndpointer(ctypes.c_uint64, flags="C_CONTIGUOUS"), # union_offsets
    ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), # query_refs
    ctypes.c_size_t, # union_count
    ctypes.c_size_t, # refs_count
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # onesQ_ptr
    ctypes.c_float, # onesA
    ctypes.c_float, # lower_bound
    ctypes.c_float, # upper_bound
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)), # hit_positions_ptr
    ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), # hit_counts_ptr
    ctypes.c_size_t, # fp_size
    ctypes.c_size_t, # n_rows
    ctypes.c_size_t, # n_queries
    ctypes.c_int # n_threads
], ctypes.c_int)

# int calculate_overlap_union_packed(
#     const uint8_t* A_ptr,
#     const uint8_t* query_bytes_ptr,
#     const float* onesQ_ptr,
#     float onesA,
#     float lower_bound,
#     float upper_bound,
#     uint32_t* const* hit_positions_ptr,
#     uint32_t* hit_counts_ptr,
#     size_t fp_size,
#     size_t n_rows,
#     size_t n_queries,
#     int n_threads
# );
_calculate_overlap_union_packed = _load_symbol(
    "calculate_overlap_union_packed",
    [
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # A
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # query_bytes
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # onesQ_ptr
        ctypes.c_float, # onesA
        ctypes.c_float, # lower_bound
        ctypes.c_float, # upper_bound
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)), # hit_positions_ptr
        ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), # hit_counts_ptr
        ctypes.c_size_t, # fp_size
        ctypes.c_size_t, # n_rows
        ctypes.c_size_t, # n_queries
        ctypes.c_int # n_threads
    ],
    ctypes.c_int,
)

_calculate_overlap_union_packed_with_scores = _load_symbol(
    "calculate_overlap_union_packed_with_scores",
    [
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # A
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # query_bytes
        ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), # onesQ_ptr
        ctypes.c_uint32, # onesA
        ctypes.c_float, # lower_bound
        ctypes.c_float, # upper_bound
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)), # hit_positions_ptr
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), # hit_scores_ptr
        ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), # hit_counts_ptr
        ctypes.c_size_t, # fp_size
        ctypes.c_size_t, # n_rows
        ctypes.c_size_t, # n_queries
        ctypes.c_int # n_threads
    ],
    ctypes.c_int,
)


# void calculate_tanimoto_score(
#     const uint8_t* A_ptr,
#     const int32_t* query_indices_ptr,
#     float onesQ,
#     float onesA,
#     float* scores_out_ptr,
#     size_t fp_size,
#     size_t n_rows,
#     int n_threads
# );
_calculate_tanimoto_score = _load_symbol("calculate_tanimoto_score", [
    ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # A
    ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"), # query_indices
    ctypes.c_float, # onesQ
    ctypes.c_float, # onesA
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # scores_out
    ctypes.c_size_t, # fp_size
    ctypes.c_size_t, # n_rows
    ctypes.c_int # n_threads
], None)

# void calculate_tanimoto_score_f16(
#     const uint8_t* A_ptr,
#     const int32_t* query_indices_ptr,
#     float onesQ,
#     float onesA,
#     uint16_t* scores_out_ptr,
#     size_t fp_size,
#     size_t n_rows,
#     int n_threads
# );
_calculate_tanimoto_score_f16 = _load_symbol("calculate_tanimoto_score_f16", [
    ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # A
    ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"), # query_indices
    ctypes.c_float, # onesQ
    ctypes.c_float, # onesA
    ndpointer(np.float16, flags="C_CONTIGUOUS"), # scores_out
    ctypes.c_size_t, # fp_size
    ctypes.c_size_t, # n_rows
    ctypes.c_int # n_threads
], None)

# void calculate_tanimoto_score_packed_f16(
#     const uint8_t* A_ptr,
#     const uint8_t* query_bytes_ptr,
#     float onesQ,
#     float onesA,
#     uint16_t* scores_out_ptr,
#     size_t fp_size,
#     size_t n_rows,
#     int n_threads
# );
_calculate_tanimoto_score_packed_f16 = _load_symbol(
    "calculate_tanimoto_score_packed_f16",
    [
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # A
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # query_bytes
        ctypes.c_float, # onesQ
        ctypes.c_float, # onesA
        ndpointer(np.float16, flags="C_CONTIGUOUS"), # scores_out
        ctypes.c_size_t, # fp_size
        ctypes.c_size_t, # n_rows
        ctypes.c_int # n_threads
    ],
    None,
)

# void calculate_tanimoto_score_packed_u8(
#     const uint8_t* A_ptr,
#     const uint8_t* query_bytes_ptr,
#     uint32_t onesQ,
#     uint32_t onesA,
#     uint8_t* scores_out_ptr,
#     size_t fp_size,
#     size_t n_rows,
#     int n_threads
# );
_calculate_tanimoto_score_packed_u8 = _load_symbol(
    "calculate_tanimoto_score_packed_u8",
    [
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # A
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # query_bytes
        ctypes.c_uint32, # onesQ
        ctypes.c_uint32, # onesA
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # scores_out
        ctypes.c_size_t, # fp_size
        ctypes.c_size_t, # n_rows
        ctypes.c_int # n_threads
    ],
    None,
)

# int calculate_tanimoto_score_for_hits(
#     const uint8_t* A_ptr,
#     const int32_t* query_indices_ptr,
#     float onesQ,
#     float onesA,
#     float* scores_out_ptr,
#     float lower_bound,
#     float upper_bound,
#     uint32_t* hit_positions_ptr,
#     size_t n_rows,
#     size_t width_bytes,
#     int n_threads
# );
_calculate_tanimoto_score_for_hits = _load_symbol("calculate_tanimoto_score_for_hits", [
    ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # A
    ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"), # query_indices
    ctypes.c_float, # onesQ
    ctypes.c_float, # onesA
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # scores_out
    ctypes.c_float, # lower_bound
    ctypes.c_float, # upper_bound
    ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), # hit_positions
    ctypes.c_size_t, # n_rows
    ctypes.c_size_t, # width_bytes
    ctypes.c_int # n_threads
], ctypes.c_int)

# int calculate_tanimoto_score_for_hits_packed(
#     const uint8_t* A_ptr,
#     const uint8_t* query_bytes_ptr,
#     float onesQ,
#     float onesA,
#     float* scores_out_ptr,
#     float lower_bound,
#     float upper_bound,
#     uint32_t* hit_positions_ptr,
#     size_t fp_size,
#     size_t n_rows,
#     int n_threads
# );
_calculate_tanimoto_score_for_hits_packed = _load_symbol(
    "calculate_tanimoto_score_for_hits_packed",
    [
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # A
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # query_bytes
        ctypes.c_float, # onesQ
        ctypes.c_float, # onesA
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # scores_out
        ctypes.c_float, # lower_bound
        ctypes.c_float, # upper_bound
        ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), # hit_positions
        ctypes.c_size_t, # fp_size
        ctypes.c_size_t, # n_rows
        ctypes.c_int # n_threads
    ],
    ctypes.c_int,
)

# int calculate_tanimoto_score_for_hits_with_buffers(
#     const uint8_t* A_ptr,
#     const int32_t* query_indices_ptr,
#     float onesQ,
#     float onesA,
#     float* scores_out_ptr,
#     float lower_bound,
#     float upper_bound,
#     uint32_t* hit_positions_ptr,
#     size_t n_rows,
#     size_t width_bytes,
#     int n_threads,
#     uint8_t* counts_ptr,
#     uint8_t* counts_per_thread_ptr,
#     size_t counts_len,
#     size_t counts_per_thread_len
# );
_calculate_tanimoto_score_for_hits_with_buffers = _load_symbol(
    "calculate_tanimoto_score_for_hits_with_buffers",
    [
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # A
        ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"), # query_indices
        ctypes.c_float, # onesQ
        ctypes.c_float, # onesA
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), # scores_out
        ctypes.c_float, # lower_bound
        ctypes.c_float, # upper_bound
        ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), # hit_positions
        ctypes.c_size_t, # n_rows
        ctypes.c_size_t, # width_bytes
        ctypes.c_int, # n_threads
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # counts
        ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), # counts_per_thread
        ctypes.c_size_t, # counts_len
        ctypes.c_size_t, # counts_per_thread_len
    ],
    ctypes.c_int,
)


# ---- Wrappers matching old Cython signatures mostly ----

def calculate_tanimoto_score(
    A: np.ndarray,
    query_indices: np.ndarray,
    onesQ: float,
    onesA: float,
    scores_out: np.ndarray,
    n_threads: int = 0,
    fp_size: int | None = None,
    n_rows: int | None = None,
):
    if _calculate_tanimoto_score is None:
        raise RuntimeError(
            "calculate_tanimoto_score not available in libtanimoto.so"
        )
    if query_indices is None:
        raise ValueError("query_indices is required for calculate_tanimoto_score.")
    if scores_out is None:
        raise ValueError("scores_out is required for calculate_tanimoto_score.")

    A = np.ascontiguousarray(A, dtype=np.uint8)
    query_indices = np.ascontiguousarray(query_indices, dtype=np.int32)
    scores_out = np.ascontiguousarray(scores_out, dtype=np.float32)

    if fp_size is None:
        fp_size = A.shape[1]
    if n_rows is None:
        n_rows = A.shape[0]

    _calculate_tanimoto_score(
        A,
        query_indices,
        float(onesQ),
        float(onesA),
        scores_out,
        fp_size,
        n_rows,
        n_threads
    )

def calculate_tanimoto_score_f16(
    A: np.ndarray,
    query_indices: np.ndarray,
    onesQ: float,
    onesA: float,
    scores_out: np.ndarray,
    n_threads: int = 0,
    fp_size: int | None = None,
    n_rows: int | None = None,
):
    if _calculate_tanimoto_score_f16 is None:
        raise RuntimeError(
            "calculate_tanimoto_score_f16 not available in libtanimoto.so"
        )
    if query_indices is None:
        raise ValueError("query_indices is required for calculate_tanimoto_score_f16.")
    if scores_out is None:
        raise ValueError("scores_out is required for calculate_tanimoto_score_f16.")

    A = np.ascontiguousarray(A, dtype=np.uint8)
    query_indices = np.ascontiguousarray(query_indices, dtype=np.int32)
    scores_out = np.ascontiguousarray(scores_out, dtype=np.float16)

    if fp_size is None:
        fp_size = A.shape[1]
    if n_rows is None:
        n_rows = A.shape[0]

    _calculate_tanimoto_score_f16(
        A,
        query_indices,
        float(onesQ),
        float(onesA),
        scores_out,
        fp_size,
        n_rows,
        n_threads
    )

def calculate_tanimoto_score_packed_f16(
    A: np.ndarray,
    query_bytes: np.ndarray,
    onesQ: float,
    onesA: float,
    scores_out: np.ndarray,
    n_threads: int = 0,
    fp_size: int | None = None,
    n_rows: int | None = None,
):
    if _calculate_tanimoto_score_packed_f16 is None:
        raise RuntimeError(
            "calculate_tanimoto_score_packed_f16 not available in libtanimoto.so"
        )
    if query_bytes is None:
        raise ValueError(
            "query_bytes is required for calculate_tanimoto_score_packed_f16."
        )
    if scores_out is None:
        raise ValueError(
            "scores_out is required for calculate_tanimoto_score_packed_f16."
        )

    A = np.ascontiguousarray(A, dtype=np.uint8)
    query_bytes = np.ascontiguousarray(query_bytes, dtype=np.uint8)
    scores_out = np.ascontiguousarray(scores_out, dtype=np.float16)

    if fp_size is None:
        fp_size = A.shape[1]
    if n_rows is None:
        n_rows = A.shape[0]

    _calculate_tanimoto_score_packed_f16(
        A,
        query_bytes,
        float(onesQ),
        float(onesA),
        scores_out,
        fp_size,
        n_rows,
        n_threads
    )

def calculate_tanimoto_score_packed_u8(
    A: np.ndarray,
    query_bytes: np.ndarray,
    onesQ: int,
    onesA: int,
    scores_out: np.ndarray,
    n_threads: int = 0,
    fp_size: int | None = None,
    n_rows: int | None = None,
):
    if _calculate_tanimoto_score_packed_u8 is None:
        raise RuntimeError(
            "calculate_tanimoto_score_packed_u8 not available in libtanimoto.so"
        )
    if query_bytes is None:
        raise ValueError(
            "query_bytes is required for calculate_tanimoto_score_packed_u8."
        )
    if scores_out is None:
        raise ValueError(
            "scores_out is required for calculate_tanimoto_score_packed_u8."
        )

    A = np.ascontiguousarray(A, dtype=np.uint8)
    query_bytes = np.ascontiguousarray(query_bytes, dtype=np.uint8)
    scores_out = np.ascontiguousarray(scores_out, dtype=np.uint8)

    if fp_size is None:
        fp_size = A.shape[1]
    if n_rows is None:
        n_rows = A.shape[0]

    _calculate_tanimoto_score_packed_u8(
        A,
        query_bytes,
        int(onesQ),
        int(onesA),
        scores_out,
        fp_size,
        n_rows,
        n_threads
    )


def calculate_tanimoto_score_packed_u8_unchecked(
    A: np.ndarray,
    query_bytes: np.ndarray,
    onesQ: int,
    onesA: int,
    scores_out: np.ndarray,
    n_threads: int = 0,
    fp_size: int | None = None,
    n_rows: int | None = None,
):
    """Call the packed u8 scorer for arrays already validated by the caller."""
    if _calculate_tanimoto_score_packed_u8 is None:
        raise RuntimeError(
            "calculate_tanimoto_score_packed_u8 not available in libtanimoto.so"
        )

    if fp_size is None:
        fp_size = A.shape[1]
    if n_rows is None:
        n_rows = A.shape[0]

    _calculate_tanimoto_score_packed_u8(
        A,
        query_bytes,
        int(onesQ),
        int(onesA),
        scores_out,
        fp_size,
        n_rows,
        n_threads
    )


def calculate_tanimoto_score_for_hits(
    A: np.ndarray,
    query_indices: np.ndarray,
    onesQ: float,
    onesA: float,
    scores_out: np.ndarray,
    n_threads: int = 0,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    hit_positions: np.ndarray = None,
    fp_size: int | None = None,
    width_bytes: int | None = None,
    counts: np.ndarray | None = None,
    counts_per_thread: np.ndarray | None = None,
) -> int:
    if _calculate_tanimoto_score_for_hits is None:
        raise RuntimeError(
            "calculate_tanimoto_score_for_hits not available in libtanimoto.so"
        )
    if query_indices is None:
        raise ValueError("query_indices is required for calculate_tanimoto_score_for_hits.")
    if hit_positions is None:
        raise ValueError("hit_positions is required for calculate_tanimoto_score_for_hits.")
    query_indices = np.ascontiguousarray(query_indices, dtype=np.int32)
    if fp_size is None:
        fp_size = A.shape[0]
    if width_bytes is None:
        width_bytes = A.shape[1]

    if counts is not None or counts_per_thread is not None:
        if counts is None or counts_per_thread is None:
            raise ValueError(
                "counts and counts_per_thread must be provided together."
            )
        counts = np.ascontiguousarray(counts, dtype=np.uint8)
        counts_per_thread = np.ascontiguousarray(counts_per_thread, dtype=np.uint8)
        if _calculate_tanimoto_score_for_hits_with_buffers is None:
            return _calculate_tanimoto_score_for_hits(
                A, query_indices, onesQ, onesA, scores_out,
                lower_bound, upper_bound,
                hit_positions,
                fp_size, width_bytes,
                n_threads,
            )
        return _calculate_tanimoto_score_for_hits_with_buffers(
            A, query_indices, onesQ, onesA, scores_out,
            lower_bound, upper_bound,
            hit_positions,
            fp_size, width_bytes,
            n_threads,
            counts, counts_per_thread,
            counts.size, counts_per_thread.size,
        )

    return _calculate_tanimoto_score_for_hits(
        A, query_indices, onesQ, onesA, scores_out,
        lower_bound, upper_bound,
        hit_positions,
        fp_size, width_bytes,
        n_threads,
    )


def calculate_tanimoto_score_for_hits_packed(
    A: np.ndarray,
    query_bytes: np.ndarray,
    onesQ: float,
    onesA: float,
    scores_out: np.ndarray,
    n_threads: int = 0,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    hit_positions: np.ndarray = None,
    fp_size: int | None = None,
    n_rows: int | None = None,
) -> int:
    if _calculate_tanimoto_score_for_hits_packed is None:
        raise RuntimeError(
            "calculate_tanimoto_score_for_hits_packed not available in libtanimoto.so"
        )
    if query_bytes is None:
        raise ValueError(
            "query_bytes is required for calculate_tanimoto_score_for_hits_packed."
        )
    if hit_positions is None:
        raise ValueError(
            "hit_positions is required for calculate_tanimoto_score_for_hits_packed."
        )
    if scores_out is None:
        raise ValueError(
            "scores_out is required for calculate_tanimoto_score_for_hits_packed."
        )

    A = np.ascontiguousarray(A, dtype=np.uint8)
    query_bytes = np.ascontiguousarray(query_bytes, dtype=np.uint8)
    scores_out = np.ascontiguousarray(scores_out, dtype=np.float32)
    hit_positions = np.ascontiguousarray(hit_positions, dtype=np.uint32)

    if fp_size is None:
        fp_size = A.shape[1]
    if n_rows is None:
        n_rows = A.shape[0]

    return _calculate_tanimoto_score_for_hits_packed(
        A,
        query_bytes,
        float(onesQ),
        float(onesA),
        scores_out,
        float(lower_bound),
        float(upper_bound),
        hit_positions,
        fp_size,
        n_rows,
        n_threads,
    )

def calculate_overlap_union(
    A: np.ndarray,
    union_indices: np.ndarray,
    union_offsets: np.ndarray,
    query_refs: np.ndarray,
    onesQ: np.ndarray,
    onesA: float,
    lower_bound: float,
    upper_bound: float,
    hit_positions: np.ndarray,
    hit_counts: np.ndarray,
    n_threads: int = 0
) -> int:
    if _calculate_overlap_union is None:
        raise RuntimeError(
            "calculate_overlap_union not available in libtanimoto.so"
        )
    if A.dtype != np.uint8 or not A.flags.c_contiguous:
        A = np.ascontiguousarray(A, dtype=np.uint8)
    union_indices = np.ascontiguousarray(union_indices, dtype=np.uint32)
    union_offsets = np.ascontiguousarray(union_offsets, dtype=np.uint64)
    query_refs = np.ascontiguousarray(query_refs, dtype=np.uint32)
    onesQ = np.ascontiguousarray(onesQ, dtype=np.float32)

    if union_offsets.size != union_indices.size + 1:
        raise ValueError("union_offsets must be length union_indices + 1.")
    if union_offsets.size and union_offsets[-1] != query_refs.size:
        raise ValueError("union_offsets[-1] must equal query_refs length.")

    if not isinstance(hit_positions, np.ndarray):
        raise ValueError("hit_positions must be a numpy array.")
    if hit_positions.dtype != np.uint32 or not hit_positions.flags.c_contiguous:
        raise ValueError("hit_positions must be a contiguous uint32 array.")
    if hit_counts.dtype != np.uint32 or not hit_counts.flags.c_contiguous:
        hit_counts = np.ascontiguousarray(hit_counts, dtype=np.uint32)

    n_queries = int(onesQ.shape[0])
    if hit_positions.shape[0] != n_queries:
        raise ValueError("hit_positions must have one row per query.")
    if hit_counts.shape[0] != n_queries:
        raise ValueError("hit_counts must have one entry per query.")

    hit_ptrs = (ctypes.POINTER(ctypes.c_uint32) * n_queries)()
    for i in range(n_queries):
        hit_ptrs[i] = hit_positions[i].ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint32)
        )

    n_rows = A.shape[0]
    fp_size = A.shape[1]

    return _calculate_overlap_union(
        A,
        union_indices,
        union_offsets,
        query_refs,
        union_indices.size,
        query_refs.size,
        onesQ,
        float(onesA),
        float(lower_bound),
        float(upper_bound),
        hit_ptrs,
        hit_counts,
        fp_size,
        n_rows,
        n_queries,
        n_threads
    )

def calculate_overlap_union_packed(
    A: np.ndarray,
    query_bytes: np.ndarray,
    onesQ: np.ndarray,
    onesA: float,
    lower_bound: float,
    upper_bound: float,
    hit_positions: np.ndarray,
    hit_counts: np.ndarray,
    n_threads: int = 0
) -> int:
    if _calculate_overlap_union_packed is None:
        raise RuntimeError(
            "calculate_overlap_union_packed not available in libtanimoto.so"
        )
    if A.dtype != np.uint8 or not A.flags.c_contiguous:
        A = np.ascontiguousarray(A, dtype=np.uint8)
    query_bytes = np.ascontiguousarray(query_bytes, dtype=np.uint8)
    onesQ = np.ascontiguousarray(onesQ, dtype=np.float32)

    if not isinstance(hit_positions, np.ndarray):
        raise ValueError("hit_positions must be a numpy array.")
    if hit_positions.dtype != np.uint32 or not hit_positions.flags.c_contiguous:
        raise ValueError("hit_positions must be a contiguous uint32 array.")
    if hit_counts.dtype != np.uint32 or not hit_counts.flags.c_contiguous:
        hit_counts = np.ascontiguousarray(hit_counts, dtype=np.uint32)

    n_queries = int(onesQ.shape[0])
    if query_bytes.shape[0] != n_queries:
        raise ValueError("query_bytes must have one row per query.")
    if hit_positions.shape[0] != n_queries:
        raise ValueError("hit_positions must have one row per query.")
    if hit_counts.shape[0] != n_queries:
        raise ValueError("hit_counts must have one entry per query.")

    hit_ptrs = (ctypes.POINTER(ctypes.c_uint32) * n_queries)()
    for i in range(n_queries):
        hit_ptrs[i] = hit_positions[i].ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint32)
        )

    n_rows = A.shape[0]
    fp_size = A.shape[1]

    return _calculate_overlap_union_packed(
        A,
        query_bytes,
        onesQ,
        float(onesA),
        float(lower_bound),
        float(upper_bound),
        hit_ptrs,
        hit_counts,
        fp_size,
        n_rows,
        n_queries,
        n_threads
    )


def calculate_overlap_union_packed_with_scores(
    A: np.ndarray,
    query_bytes: np.ndarray,
    onesQ: np.ndarray,
    onesA: float,
    lower_bound: float,
    upper_bound: float,
    hit_positions: np.ndarray,
    hit_scores: np.ndarray,
    hit_counts: np.ndarray,
    n_threads: int = 0,
) -> int:
    if _calculate_overlap_union_packed_with_scores is None:
        raise RuntimeError(
            "calculate_overlap_union_packed_with_scores not available in libtanimoto.so"
        )
    if lower_bound > upper_bound:
        raise ValueError("lower_bound must be less than or equal to upper_bound.")
    if A.dtype != np.uint8 or not A.flags.c_contiguous:
        A = np.ascontiguousarray(A, dtype=np.uint8)
    query_bytes = np.ascontiguousarray(query_bytes, dtype=np.uint8)
    onesQ = np.ascontiguousarray(onesQ, dtype=np.uint32)

    if not isinstance(hit_positions, np.ndarray):
        raise ValueError("hit_positions must be a numpy array.")
    if hit_positions.dtype != np.uint32 or not hit_positions.flags.c_contiguous:
        raise ValueError("hit_positions must be a contiguous uint32 array.")
    if not isinstance(hit_scores, np.ndarray):
        raise ValueError("hit_scores must be a numpy array.")
    if hit_scores.dtype != np.uint8 or not hit_scores.flags.c_contiguous:
        raise ValueError("hit_scores must be a contiguous uint8 array.")
    if hit_counts.dtype != np.uint32 or not hit_counts.flags.c_contiguous:
        hit_counts = np.ascontiguousarray(hit_counts, dtype=np.uint32)

    n_queries = int(onesQ.shape[0])
    if query_bytes.shape[0] != n_queries:
        raise ValueError("query_bytes must have one row per query.")
    if hit_positions.shape[0] != n_queries:
        raise ValueError("hit_positions must have one row per query.")
    if hit_scores.shape[0] != n_queries:
        raise ValueError("hit_scores must have one row per query.")
    if hit_scores.shape != hit_positions.shape:
        raise ValueError("hit_scores must have the same shape as hit_positions.")
    if hit_counts.shape[0] != n_queries:
        raise ValueError("hit_counts must have one entry per query.")

    hit_position_ptrs = (ctypes.POINTER(ctypes.c_uint32) * n_queries)()
    hit_score_ptrs = (ctypes.POINTER(ctypes.c_uint8) * n_queries)()
    for i in range(n_queries):
        hit_position_ptrs[i] = hit_positions[i].ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint32)
        )
        hit_score_ptrs[i] = hit_scores[i].ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )

    n_rows = A.shape[0]
    fp_size = A.shape[1]

    return _calculate_overlap_union_packed_with_scores(
        A,
        query_bytes,
        onesQ,
        int(onesA),
        float(lower_bound),
        float(upper_bound),
        hit_position_ptrs,
        hit_score_ptrs,
        hit_counts,
        fp_size,
        n_rows,
        n_queries,
        n_threads
    )
