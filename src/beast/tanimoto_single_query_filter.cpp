#include "tanimoto_defs.h"
#include "tanimoto_api.h"
#include <cstring>
#include <cstdio>
#include <cstdio>
#include <vector>
#if defined(__SSE2__)
#include <immintrin.h>
#endif

// ---- Restrict helper ----

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

// ---- LUT for SIMD bit counts ----

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

// ---- Main Interface ----

extern "C" int calculate_tanimoto_score_for_hits(
    const uint8_t* RESTRICT A_ptr, // Row-packed bytes (8 items per byte, packed along axis=0)
    const int32_t* RESTRICT query_indices_ptr,
    float onesQ,
    float onesA,
    float* RESTRICT scores_out_ptr,
    float lower_bound,
    float upper_bound,
    uint64_t thr_lower_num,
    uint64_t thr_upper_num,
    uint64_t thr_den,
    uint32_t* RESTRICT hit_positions_ptr,
    size_t fp_size,
    size_t n_rows,
    int n_threads
) {
    if (!A_ptr || !hit_positions_ptr) {
        return 0;
    }
    if (fp_size == 0) {
        return 0;
    }
    if (n_rows == 0) {
        return 0;
    }

    if (n_threads > 0) {
        int threads = clamp_omp_threads(n_threads);
        omp_set_num_threads(threads);
    }

    size_t n_cols = fp_size;

    if (!query_indices_ptr) {
        return 0;
    }
    size_t n_query_bits = static_cast<size_t>(onesQ);
    if (n_query_bits == 0) {
        return 0;
    }

    const float sumQA = onesQ + onesA;
    int64_t lower_inter_64 = 0;
    int64_t upper_inter_64 = 0;
    tanimoto_inter_bounds(
        (uint64_t)sumQA,
        lower_bound,
        upper_bound,
        thr_lower_num,
        thr_upper_num,
        thr_den,
        &lower_inter_64,
        &upper_inter_64
    );
    int lower_inter_i = (int)lower_inter_64;
    int upper_inter_i = (int)upper_inter_64;
    if (lower_inter_i < 0) {
        lower_inter_i = 0;
    }
    if (upper_inter_i > 255) {
        upper_inter_i = 255;
    }
    if (lower_inter_i > upper_inter_i) {
        return 0;
    }

    const int max_threads = omp_get_max_threads();
    std::vector<std::vector<uint32_t>> thread_hits(max_threads);

#if defined(__SSE2__)

    const bool use_simd = (n_query_bits <= 255);
    if (use_simd) {
        const BitLut8& lut = _bit_lut8();
        const __m128i bias = _mm_set1_epi8((char)0x80);
        const __m128i lower_v = _mm_set1_epi8((char)lower_inter_i);
        const __m128i upper_v = _mm_set1_epi8((char)upper_inter_i);
        const __m128i low_b = _mm_xor_si128(lower_v, bias);
        const __m128i high_b = _mm_xor_si128(upper_v, bias);
        const __m128i all_ones = _mm_set1_epi8((char)0xFF);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::vector<uint32_t>& hits = thread_hits[tid];
            #pragma omp for schedule(static)
            for (long row_pack = 0; row_pack < (long)n_rows; ++row_pack) {
                __m128i counts0 = _mm_setzero_si128();
                const uint8_t* row = A_ptr + (size_t)row_pack * n_cols;

                for (size_t q_idx = 0; q_idx < n_query_bits; ++q_idx) {
                    uint8_t val = row[(size_t)query_indices_ptr[q_idx]];
                    __m128i v0 = _mm_load_si128(
                        (const __m128i*)lut.data[val]
                    );
                    counts0 = _mm_add_epi8(counts0, v0);
                }

                __m128i inter_b = _mm_xor_si128(counts0, bias);
                __m128i low_gt_inter = _mm_cmpgt_epi8(low_b, inter_b);
                __m128i inter_gt_high = _mm_cmpgt_epi8(inter_b, high_b);
                __m128i pass = _mm_andnot_si128(
                    _mm_or_si128(low_gt_inter, inter_gt_high),
                    all_ones
                );
                int pass_mask = _mm_movemask_epi8(pass) & 0xFF;
                if (!pass_mask) {
                    continue;
                }

                alignas(16) uint8_t counts8[8];
                _mm_storel_epi64((__m128i*)counts8, counts0);

                const size_t item_base = (size_t)row_pack << 3;
                for (size_t bit = 0; bit < 8; ++bit) {
                    if (!(pass_mask & (1 << bit))) {
                        continue;
                    }
                    size_t item = item_base + bit;
                    unsigned int inter = counts8[bit];
                    float denominator = sumQA - (float)inter;

                    float score = 0.0f;
                    if (denominator > 0.0f) {
                        score = (float)inter / denominator;
                    } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                        score = 1.0f;
                    }

                    if (score >= lower_bound && score <= upper_bound) {
                        if (scores_out_ptr) {
                            scores_out_ptr[item] = score;
                        }
                        hits.push_back((uint32_t)item);
                    }
                }
            }
        }
    } else
#endif
    {
        const BitLut8& lut = _bit_lut8();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::vector<uint32_t>& hits = thread_hits[tid];
            #pragma omp for schedule(static)
            for (long row_pack = 0; row_pack < (long)n_rows; ++row_pack) {
                uint8_t counts[8] = {0};
                const uint8_t* row = A_ptr + (size_t)row_pack * n_cols;

                for (size_t q_idx = 0; q_idx < n_query_bits; ++q_idx) {
                    uint8_t val = row[(size_t)query_indices_ptr[q_idx]];
                    const uint8_t* lut0 = lut.data[val];
                    for (size_t bit = 0; bit < 8; ++bit) {
                        counts[bit] += lut0[bit];
                    }
                }

                const size_t item_base = (size_t)row_pack << 3;
                for (size_t bit = 0; bit < 8; ++bit) {
                    size_t item = item_base + bit;
                    unsigned int inter = counts[bit];
                    if (inter < (unsigned int)lower_inter_i ||
                        inter > (unsigned int)upper_inter_i) {
                        continue;
                    }

                    float denominator = sumQA - (float)inter;

                    float score = 0.0f;
                    if (denominator > 0.0f) {
                        score = (float)inter / denominator;
                    } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                        score = 1.0f;
                    }

                    if (score >= lower_bound && score <= upper_bound) {
                        if (scores_out_ptr) {
                            scores_out_ptr[item] = score;
                        }
                        hits.push_back((uint32_t)item);
                    }
                }
            }
        }
    }

    size_t total_hits = 0;
    std::vector<size_t> offsets((size_t)max_threads, 0);
    for (int t = 0; t < max_threads; ++t) {
        offsets[(size_t)t] = total_hits;
        total_hits += thread_hits[(size_t)t].size();
    }
    for (int t = 0; t < max_threads; ++t) {
        const std::vector<uint32_t>& hits = thread_hits[(size_t)t];
        if (!hits.empty()) {
            std::memcpy(
                hit_positions_ptr + offsets[(size_t)t],
                hits.data(),
                hits.size() * sizeof(uint32_t)
            );
        }
    }

    return (int)total_hits;
}
