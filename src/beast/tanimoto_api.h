#ifndef TANIMOTO_API_H
#define TANIMOTO_API_H

#include <cstdint>
#include <cstddef>

extern "C" {

/**
 * Calculates Tanimoto hits using precomputed union arrays from Python.
 */
int calculate_overlap_union(
    const uint8_t* A_ptr, // Row-packed bytes (8 items per byte, packed along axis=0)
    const uint32_t* union_indices_ptr,
    const uint64_t* union_offsets_ptr,
    const uint32_t* query_refs_ptr,
    size_t union_count,
    size_t refs_count,
    const float* onesQ_ptr,
    float onesA,
    float lower_bound,
    float upper_bound,
    uint64_t thr_lower_num,
    uint64_t thr_upper_num,
    uint64_t thr_den,
    uint32_t* const* hit_positions_ptr, // Output: per-query hit index arrays
    uint32_t* hit_counts_ptr,           // Output: number of hits per query
    size_t fp_size,
    size_t n_rows,
    size_t n_queries,
    int n_threads
);

/**
 * Calculates Tanimoto hits for packed queries against packed rows.
 * Input is bit-packed along axis=1 (shape: n_rows x fp_size_bytes).
 * Query is bit-packed along axis=1 (shape: n_queries x fp_size_bytes).
 */
int calculate_overlap_union_packed(
    const uint8_t* A_ptr, // Row-packed bytes (bit-packed along axis=1)
    const uint8_t* query_bytes_ptr,
    const float* onesQ_ptr,
    float onesA,
    float lower_bound,
    float upper_bound,
    uint64_t thr_lower_num,
    uint64_t thr_upper_num,
    uint64_t thr_den,
    uint32_t* const* hit_positions_ptr,
    uint32_t* hit_counts_ptr,
    size_t fp_size,
    size_t n_rows,
    size_t n_queries,
    int n_threads
);

/**
 * Calculates Tanimoto hits and uint8 centi-scores for packed queries.
 * Input is bit-packed along axis=1 (shape: n_rows x fp_size_bytes).
 * Query is bit-packed along axis=1 (shape: n_queries x fp_size_bytes).
 */
int calculate_overlap_union_packed_with_scores(
    const uint8_t* A_ptr,
    const uint8_t* query_bytes_ptr,
    const uint32_t* onesQ_ptr,
    uint32_t onesA,
    float lower_bound,
    float upper_bound,
    uint64_t thr_lower_num,
    uint64_t thr_upper_num,
    uint64_t thr_den,
    uint32_t* hit_query_ids_ptr,
    uint32_t* hit_positions_ptr,
    uint8_t* hit_scores_ptr,
    uint64_t hit_capacity,
    uint64_t* hit_count_ptr,
    uint8_t* overflow_ptr,
    size_t fp_size,
    size_t n_rows,
    size_t n_queries,
    int n_threads
);

/**
 * Calculates generic Tanimoto scores for a single query against N targets.
 */
void calculate_tanimoto_score(
    const uint8_t* A_ptr, // Row-packed bytes (8 items per byte, packed along axis=0)
    const int32_t* query_indices_ptr,
    float onesQ,
    float onesA,
    float* scores_out_ptr, // Output: Array of scores (length n_rows * 8)
    size_t fp_size,
    size_t n_rows,
    int n_threads
);

/**
 * Calculates Tanimoto scores for a single query against N targets.
 * Outputs IEEE-754 half-precision values.
 */
void calculate_tanimoto_score_f16(
    const uint8_t* A_ptr, // Row-packed bytes (8 items per byte, packed along axis=0)
    const int32_t* query_indices_ptr,
    float onesQ,
    float onesA,
    uint16_t* scores_out_ptr, // Output: Array of fp16 scores (length n_rows * 8)
    size_t fp_size,
    size_t n_rows,
    int n_threads
);

/**
 * Calculates Tanimoto scores for a single query against N targets.
 * Input is bit-packed along axis=1 (shape: n_rows x fp_size_bytes).
 * Query is bit-packed bytes (length fp_size_bytes).
 * Outputs IEEE-754 half-precision values.
 */
void calculate_tanimoto_score_packed_f16(
    const uint8_t* A_ptr, // Row-packed bytes (bit-packed along axis=1)
    const uint8_t* query_bytes_ptr,
    float onesQ,
    float onesA,
    uint16_t* scores_out_ptr, // Output: Array of fp16 scores (length n_rows)
    size_t fp_size,
    size_t n_rows,
    int n_threads
);

/**
 * Calculates Tanimoto scores for a single query against N targets.
 * Input is bit-packed along axis=1 (shape: n_rows x fp_size_bytes).
 * Query is bit-packed bytes (length fp_size_bytes).
 * Outputs centi-scores: round(score * 100), clamped to [0, 100].
 */
void calculate_tanimoto_score_packed_u8(
    const uint8_t* A_ptr, // Row-packed bytes (bit-packed along axis=1)
    const uint8_t* query_bytes_ptr,
    uint32_t onesQ,
    uint32_t onesA,
    uint8_t* scores_out_ptr, // Output: Array of uint8 centi-scores (length n_rows)
    size_t fp_size,
    size_t n_rows,
    int n_threads
);

/**
 * Calculates Tanimoto scores for a single query against N targets
 * and concurrently filters them based on popcount thresholds.
 * Returns the total number of hits found.
 */
int calculate_tanimoto_score_for_hits(
    const uint8_t* A_ptr, // Row-packed bytes (8 items per byte, packed along axis=0)
    const int32_t* query_indices_ptr,
    float onesQ,
    float onesA,
    float* scores_out_ptr,     // Output: Array of scores (length n_rows)
    float lower_bound,
    float upper_bound,
    uint64_t thr_lower_num,
    uint64_t thr_upper_num,
    uint64_t thr_den,
    uint32_t* hit_positions_ptr, // Output: Array of filtered hit indices
    size_t fp_size,
    size_t n_rows,
    int n_threads
);

/**
 * Calculates Tanimoto scores for a single query against N targets
 * for axis=1 bit-packed inputs. Returns the total number of hits found.
 */
int calculate_tanimoto_score_for_hits_packed(
    const uint8_t* A_ptr, // Row-packed bytes (bit-packed along axis=1)
    const uint8_t* query_bytes_ptr,
    float onesQ,
    float onesA,
    float* scores_out_ptr,     // Output: Array of scores (length n_rows)
    float lower_bound,
    float upper_bound,
    uint64_t thr_lower_num,
    uint64_t thr_upper_num,
    uint64_t thr_den,
    uint32_t* hit_positions_ptr, // Output: Array of filtered hit indices
    size_t fp_size,
    size_t n_rows,
    int n_threads
);

/**
 * Calculates Tanimoto scores for a single query against N targets
 * using caller-provided count buffers.
 * Returns the total number of hits found.
 */
// int calculate_tanimoto_score_for_hits_with_buffers(
//     const uint32_t* A_ptr, // TRANSPOSED data
//     const int32_t* query_indices_ptr,
//     float onesQ,
//     float onesA,
//     float* scores_out_ptr,     // Output: Array of scores (length n_rows)
//     float lower_bound,
//     float upper_bound,
//     uint32_t* mask_ptr,        // Output: Bitmask of filtered hits
//     uint32_t* hit_positions_ptr, // Output: Array of filtered hit indices
//     size_t fp_size,
//     size_t width_bytes,
//     int n_threads,
//     uint8_t* counts_ptr,
//     uint8_t* counts_per_thread_ptr,
//     size_t counts_len,
//     size_t counts_per_thread_len
// );

}

#endif // TANIMOTO_API_H
