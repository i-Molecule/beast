#include "tanimoto_defs.h"
#include "tanimoto_api.h"
#include <algorithm>
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

// ---- LUT for bit counts ----

struct BitLut8 {
    alignas(16) uint8_t data[256][16];
    BitLut8() : data{} {
        for (int v = 0; v < 256; ++v) {
            data[v][0] = (v & 0x80) ? 1 : 0;
            data[v][1] = (v & 0x40) ? 1 : 0;
            data[v][2] = (v & 0x20) ? 1 : 0;
            data[v][3] = (v & 0x10) ? 1 : 0;
            data[v][4] = (v & 0x08) ? 1 : 0;
            data[v][5] = (v & 0x04) ? 1 : 0;
            data[v][6] = (v & 0x02) ? 1 : 0;
            data[v][7] = (v & 0x01) ? 1 : 0;
            for (int i = 8; i < 16; ++i) {
                data[v][i] = 0;
            }
        }
    }
};

static const BitLut8& _bit_lut8() {
    static const BitLut8 lut;
    return lut;
}

// ---- Internal Helpers ----

static ALWAYS_INLINE bool _compute_inter_bounds(
    float sumQA,
    float lower_bound,
    float upper_bound,
    uint64_t thr_lower_num,
    uint64_t thr_upper_num,
    uint64_t thr_den,
    int* lower_inter_i,
    int* upper_inter_i
) {
    if (lower_bound > upper_bound) {
        return false;
    }
    int64_t lower_64 = 0;
    int64_t upper_64 = 0;
    tanimoto_inter_bounds(
        (uint64_t)sumQA,
        lower_bound,
        upper_bound,
        thr_lower_num,
        thr_upper_num,
        thr_den,
        &lower_64,
        &upper_64
    );
    int lower_i = (int)lower_64;
    int upper_i = (int)upper_64;
    if (lower_i < 0) {
        lower_i = 0;
    }
    if (upper_i > 255) {
        upper_i = 255;
    }
    if (lower_i > upper_i) {
        return false;
    }
    *lower_inter_i = lower_i;
    *upper_inter_i = upper_i;
    return true;
}

// ---- Main Interface ----

extern "C" int calculate_overlap_union(
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
    uint32_t* const* hit_positions_ptr,
    uint32_t* hit_counts_ptr,
    size_t fp_size,
    size_t n_rows,
    size_t n_queries,
    int n_threads
) {
    if (!A_ptr || !union_indices_ptr || !union_offsets_ptr || !query_refs_ptr ||
        !onesQ_ptr || !hit_positions_ptr || !hit_counts_ptr) {
        return 0;
    }
    if (fp_size == 0 || n_rows == 0 || n_queries == 0 ||
        union_count == 0 || refs_count == 0) {
        return 0;
    }
    if ((size_t)union_offsets_ptr[union_count] != refs_count) {
        return 0;
    }

    if (n_threads > 0) {
        int threads = clamp_omp_threads(n_threads);
        omp_set_num_threads(threads);
    }

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
        size_t n_query_bits = static_cast<size_t>(onesQ);
        if (n_query_bits == 0) {
            hit_counts_ptr[q] = 0;
            continue;
        }
        float sum = onesQ + onesA;
        int lower_i = 0;
        int upper_i = -1;
        if (!_compute_inter_bounds(sum, lower_bound, upper_bound,
                                   thr_lower_num, thr_upper_num, thr_den,
                                   &lower_i, &upper_i)) {
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
    const BitLut8& lut = _bit_lut8();
    const size_t n_cols = fp_size;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ThreadHits& hits = thread_hits[(size_t)tid];
        std::vector<uint16_t> counts(n_queries * 8, 0);

        #pragma omp for schedule(static)
        for (long row_pack = 0; row_pack < (long)n_rows; ++row_pack) {
            std::fill(counts.begin(), counts.end(), 0);
            const uint8_t* row = A_ptr + (size_t)row_pack * n_cols;
            const size_t item_base = (size_t)row_pack << 3;

            for (size_t u = 0; u < union_count; ++u) {
                uint8_t val = row[union_indices_ptr[u]];
                if (val == 0) {
                    continue;
                }
                const uint8_t* lut0 = lut.data[val];
                size_t start = (size_t)union_offsets_ptr[u];
                size_t end = (size_t)union_offsets_ptr[u + 1];
                for (size_t idx = start; idx < end; ++idx) {
                    size_t q = (size_t)query_refs_ptr[idx];
                    uint16_t* counts_q = &counts[q * 8];
                    counts_q[0] += lut0[0];
                    counts_q[1] += lut0[1];
                    counts_q[2] += lut0[2];
                    counts_q[3] += lut0[3];
                    counts_q[4] += lut0[4];
                    counts_q[5] += lut0[5];
                    counts_q[6] += lut0[6];
                    counts_q[7] += lut0[7];
                }
            }

            for (size_t q_idx = 0; q_idx < active_queries.size(); ++q_idx) {
                size_t q = active_queries[q_idx];
                int lower_i = lower_inter_i[q];
                int upper_i = upper_inter_i[q];
                float sum = sumQA[q];
                float onesQ = onesQ_ptr[q];
                uint16_t* counts_q = &counts[q * 8];

                for (size_t bit = 0; bit < 8; ++bit) {
                    size_t item = item_base + bit;
                    unsigned int inter = counts_q[bit];
                    if (inter < (unsigned int)lower_i ||
                        inter > (unsigned int)upper_i) {
                        continue;
                    }

                    float denominator = sum - (float)inter;

                    float score = 0.0f;
                    if (denominator > 0.0f) {
                        score = (float)inter / denominator;
                    } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                        score = 1.0f;
                    }

                    if (score >= lower_bound && score <= upper_bound) {
                        hits.items.push_back((uint32_t)item);
                        hits.queries.push_back((uint32_t)q);
                        hits.counts[q] += 1;
                    }
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
