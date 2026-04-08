#include "tanimoto_defs.h"
#include "tanimoto_api.h"
#include <cstring>
#include <vector>

// ---- Restrict helper ----

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

// ---- LUT for byte popcounts ----

struct PopcountLut8 {
    uint8_t data[256];
    PopcountLut8() : data{} {
        for (int v = 0; v < 256; ++v) {
            data[v] = (uint8_t)__builtin_popcount((unsigned int)v);
        }
    }
};

static const PopcountLut8& _popcount_lut8() {
    static const PopcountLut8 lut;
    return lut;
}

// ---- Internal Helpers ----

static ALWAYS_INLINE bool _compute_inter_bounds(
    float sumQA,
    float lower_bound,
    float upper_bound,
    int max_inter,
    int* lower_inter_i,
    int* upper_inter_i
) {
    if (lower_bound > upper_bound) {
        return false;
    }
    float lower_inter = sumQA * lower_bound / (1.0f + lower_bound);
    float upper_inter = sumQA * upper_bound / (1.0f + upper_bound);
    int lower_i = (int)lower_inter;
    if ((float)lower_i < lower_inter) {
        lower_i++;
    }
    int upper_i = (int)upper_inter;
    if ((float)upper_i > upper_inter) {
        upper_i--;
    }
    if (lower_i < 0) {
        lower_i = 0;
    }
    if (max_inter < 0) {
        max_inter = 0;
    }
    if (upper_i > max_inter) {
        upper_i = max_inter;
    }
    if (lower_i > upper_i) {
        return false;
    }
    *lower_inter_i = lower_i;
    *upper_inter_i = upper_i;
    return true;
}

// ---- Main Interface ----

extern "C" int calculate_overlap_union_packed(
    const uint8_t* RESTRICT A_ptr, // Row-packed bytes (bit-packed along axis=1)
    const uint8_t* RESTRICT query_bytes_ptr, // Packed queries (n_queries x fp_size_bytes)
    const float* RESTRICT onesQ_ptr,
    float onesA,
    float lower_bound,
    float upper_bound,
    uint32_t* const* hit_positions_ptr,
    uint32_t* RESTRICT hit_counts_ptr,
    size_t fp_size,
    size_t n_rows,
    size_t n_queries,
    int n_threads
) {
    if (!A_ptr || !query_bytes_ptr || !onesQ_ptr ||
        !hit_positions_ptr || !hit_counts_ptr) {
        return 0;
    }
    if (fp_size == 0 || n_rows == 0 || n_queries == 0) {
        return 0;
    }

    if (n_threads > 0) {
        int threads = clamp_omp_threads(n_threads);
        omp_set_num_threads(threads);
    }

    const size_t n_cols = fp_size;
    const float onesA_f = onesA;

    std::vector<float> sumQA(n_queries, 0.0f);
    std::vector<int> lower_inter_i(n_queries, 0);
    std::vector<int> upper_inter_i(n_queries, -1);
    std::vector<uint8_t> query_active(n_queries, 0);
    std::vector<size_t> active_queries;
    active_queries.reserve(n_queries);

    for (size_t q = 0; q < n_queries; ++q) {
        if (!hit_positions_ptr[q]) {
            hit_counts_ptr[q] = 0;
            continue;
        }
        float onesQ = onesQ_ptr[q];
        if (onesQ <= 0.0f) {
            hit_counts_ptr[q] = 0;
            continue;
        }
        float sum = onesQ + onesA_f;
        int max_inter = (int)onesQ;
        if (onesA_f < (float)max_inter) {
            max_inter = (int)onesA_f;
        }
        int lower_i = 0;
        int upper_i = -1;
        if (!_compute_inter_bounds(sum, lower_bound, upper_bound,
                                   max_inter, &lower_i, &upper_i)) {
            hit_counts_ptr[q] = 0;
            continue;
        }
        sumQA[q] = sum;
        lower_inter_i[q] = lower_i;
        upper_inter_i[q] = upper_i;
        query_active[q] = 1;
        active_queries.push_back(q);
    }

    if (active_queries.empty()) {
        return 0;
    }

    const int max_threads = omp_get_max_threads();
    struct ThreadHits {
        std::vector<uint32_t> items;
        std::vector<uint32_t> queries;
        std::vector<uint32_t> counts;
    };
    std::vector<ThreadHits> thread_hits((size_t)max_threads);
    for (int t = 0; t < max_threads; ++t) {
        thread_hits[(size_t)t].counts.assign(n_queries, 0);
    }

    const bool use_words = (n_cols % 8 == 0);
    const bool rows_aligned = use_words &&
        ((reinterpret_cast<uintptr_t>(A_ptr) &
          (alignof(uint64_t) - 1u)) == 0u);
    const bool query_aligned = use_words &&
        ((reinterpret_cast<uintptr_t>(query_bytes_ptr) &
          (alignof(uint64_t) - 1u)) == 0u);
    const size_t word_cols = use_words ? (n_cols / 8u) : 0u;
    const uint64_t* rows64 = rows_aligned
        ? reinterpret_cast<const uint64_t*>(A_ptr)
        : nullptr;
    const uint64_t* query_words_base = query_aligned
        ? reinterpret_cast<const uint64_t*>(query_bytes_ptr)
        : nullptr;
    const PopcountLut8& lut = _popcount_lut8();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ThreadHits& hits = thread_hits[(size_t)tid];

        #pragma omp for schedule(static)
        for (long row = 0; row < (long)n_rows; ++row) {
            const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
            const uint64_t* row_words = rows64
                ? rows64 + (size_t)row * word_cols
                : nullptr;

            for (size_t q_idx = 0; q_idx < active_queries.size(); ++q_idx) {
                size_t q = active_queries[q_idx];
                const int lower_i = lower_inter_i[q];
                const int upper_i = upper_inter_i[q];
                const float sum = sumQA[q];
                const float onesQ = onesQ_ptr[q];
                unsigned int inter = 0;

                if (row_words && query_words_base) {
                    const uint64_t* q_words =
                        query_words_base + q * word_cols;
                    for (size_t w = 0; w < word_cols; ++w) {
                        inter += popcount64(row_words[w] & q_words[w]);
                    }
                } else {
                    const uint8_t* q_bytes =
                        query_bytes_ptr + q * n_cols;
                    for (size_t col = 0; col < n_cols; ++col) {
                        uint8_t val = row_ptr[col];
                        inter += lut.data[(uint8_t)(val & q_bytes[col])];
                    }
                }

                if (inter < (unsigned int)lower_i ||
                    inter > (unsigned int)upper_i) {
                    continue;
                }

                float denominator = sum - (float)inter;

                float score = 0.0f;
                if (denominator > 0.0f) {
                    score = (float)inter / denominator;
                } else if (inter == 0 && onesQ == 0.0f && onesA_f == 0.0f) {
                    score = 1.0f;
                }

                if (score >= lower_bound && score <= upper_bound) {
                    hits.items.push_back((uint32_t)row);
                    hits.queries.push_back((uint32_t)q);
                    hits.counts[q] += 1;
                }
            }
        }
    }

    size_t total_hits = 0;
    std::vector<size_t> thread_offsets((size_t)max_threads * n_queries, 0);

    for (size_t q = 0; q < n_queries; ++q) {
        if (!query_active[q]) {
            hit_counts_ptr[q] = 0;
            continue;
        }

        size_t total_hits_q = 0;
        for (int t = 0; t < max_threads; ++t) {
            thread_offsets[(size_t)t * n_queries + q] = total_hits_q;
            total_hits_q += thread_hits[(size_t)t].counts[q];
        }
        hit_counts_ptr[q] = (uint32_t)total_hits_q;
        total_hits += total_hits_q;
    }

    for (int t = 0; t < max_threads; ++t) {
        ThreadHits& hits = thread_hits[(size_t)t];
        if (hits.items.empty()) {
            continue;
        }
        std::vector<size_t> write_pos(n_queries, 0);
        size_t base = (size_t)t * n_queries;
        for (size_t q = 0; q < n_queries; ++q) {
            write_pos[q] = thread_offsets[base + q];
        }
        for (size_t i = 0; i < hits.items.size(); ++i) {
            uint32_t q = hits.queries[i];
            uint32_t item = hits.items[i];
            uint32_t* hit_positions = hit_positions_ptr[q];
            hit_positions[write_pos[q]++] = item;
        }
    }

    return (int)total_hits;
}
