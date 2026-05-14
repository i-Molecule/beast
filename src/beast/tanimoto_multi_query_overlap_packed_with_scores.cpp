#include "tanimoto_defs.h"
#include "tanimoto_api.h"
#include <cstring>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

struct PopcountLut8WithScores {
    uint8_t data[256];
    PopcountLut8WithScores() : data{} {
        for (int v = 0; v < 256; ++v) {
            data[v] = (uint8_t)__builtin_popcount((unsigned int)v);
        }
    }
};

static const PopcountLut8WithScores& _popcount_lut8_with_scores() {
    static const PopcountLut8WithScores lut;
    return lut;
}

static ALWAYS_INLINE bool _compute_inter_bounds_u32(
    uint32_t sumQA,
    float lower_bound,
    float upper_bound,
    uint32_t max_inter,
    uint32_t* lower_inter,
    uint32_t* upper_inter
) {
    if (lower_bound > upper_bound) {
        return false;
    }

    float lower_f = (float)sumQA * lower_bound / (1.0f + lower_bound);
    float upper_f = (float)sumQA * upper_bound / (1.0f + upper_bound);

    uint32_t lower_i = (uint32_t)lower_f;
    if ((float)lower_i < lower_f) {
        lower_i++;
    }

    uint32_t upper_i = (uint32_t)upper_f;
    if ((float)upper_i > upper_f && upper_i > 0u) {
        upper_i--;
    }
    if (upper_i > max_inter) {
        upper_i = max_inter;
    }
    if (lower_i > upper_i) {
        return false;
    }

    *lower_inter = lower_i;
    *upper_inter = upper_i;
    return true;
}

static ALWAYS_INLINE uint8_t _centi_score_for_inter(
    uint32_t inter,
    uint32_t sumQA
) {
    const uint32_t denom = sumQA - inter;
    if (denom == 0u) {
        return 0u;
    }
    const uint64_t numerator = (uint64_t)inter * 100u;
    uint64_t score = (numerator + (uint64_t)denom / 2u) / (uint64_t)denom;
    if (score > 100u) {
        score = 100u;
    }
    return (uint8_t)score;
}

extern "C" int calculate_overlap_union_packed_with_scores(
    const uint8_t* RESTRICT A_ptr,
    const uint8_t* RESTRICT query_bytes_ptr,
    const uint32_t* RESTRICT onesQ_ptr,
    uint32_t onesA,
    float lower_bound,
    float upper_bound,
    uint32_t* const* hit_positions_ptr,
    uint8_t* const* hit_scores_ptr,
    uint32_t* RESTRICT hit_counts_ptr,
    size_t fp_size,
    size_t n_rows,
    size_t n_queries,
    int n_threads
) {
    if (!A_ptr || !query_bytes_ptr || !onesQ_ptr ||
        !hit_positions_ptr || !hit_scores_ptr || !hit_counts_ptr) {
        return 0;
    }
    if (fp_size == 0 || n_rows == 0 || n_queries == 0) {
        return 0;
    }
    if (lower_bound > upper_bound) {
        return 0;
    }

    if (n_threads > 0) {
        int threads = clamp_omp_threads(n_threads);
        omp_set_num_threads(threads);
    }

    const size_t n_cols = fp_size;
    std::vector<uint32_t> sumQA(n_queries, 0u);
    std::vector<uint32_t> lower_inter(n_queries, 0u);
    std::vector<uint32_t> upper_inter(n_queries, 0u);
    std::vector<uint8_t> query_active(n_queries, 0u);
    std::vector<size_t> active_queries;
    active_queries.reserve(n_queries);

    for (size_t q = 0; q < n_queries; ++q) {
        if (!hit_positions_ptr[q] || !hit_scores_ptr[q]) {
            hit_counts_ptr[q] = 0;
            continue;
        }
        const uint32_t onesQ = onesQ_ptr[q];
        if (onesQ == 0u) {
            hit_counts_ptr[q] = 0;
            continue;
        }

        const uint32_t sum = onesQ + onesA;
        uint32_t max_inter = onesQ;
        if (onesA < max_inter) {
            max_inter = onesA;
        }

        uint32_t lower_i = 0u;
        uint32_t upper_i = 0u;
        if (!_compute_inter_bounds_u32(
                sum,
                lower_bound,
                upper_bound,
                max_inter,
                &lower_i,
                &upper_i
            )) {
            hit_counts_ptr[q] = 0;
            continue;
        }

        sumQA[q] = sum;
        lower_inter[q] = lower_i;
        upper_inter[q] = upper_i;
        query_active[q] = 1u;
        active_queries.push_back(q);
    }

    if (active_queries.empty()) {
        return 0;
    }

    const int max_threads = omp_get_max_threads();
    struct ThreadHits {
        std::vector<uint32_t> items;
        std::vector<uint32_t> queries;
        std::vector<uint8_t> scores;
        std::vector<uint32_t> counts;
    };
    std::vector<ThreadHits> thread_hits((size_t)max_threads);
    for (int t = 0; t < max_threads; ++t) {
        thread_hits[(size_t)t].counts.assign(n_queries, 0u);
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
    const PopcountLut8WithScores& lut = _popcount_lut8_with_scores();

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
                const size_t q = active_queries[q_idx];
                unsigned int inter = 0u;

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
                        inter += lut.data[(uint8_t)(row_ptr[col] & q_bytes[col])];
                    }
                }

                if (inter < lower_inter[q] || inter > upper_inter[q]) {
                    continue;
                }

                hits.items.push_back((uint32_t)row);
                hits.queries.push_back((uint32_t)q);
                hits.scores.push_back(_centi_score_for_inter(inter, sumQA[q]));
                hits.counts[q] += 1u;
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
            size_t pos = write_pos[q]++;
            hit_positions_ptr[q][pos] = hits.items[i];
            hit_scores_ptr[q][pos] = hits.scores[i];
        }
    }

    return (int)total_hits;
}
