#include "tanimoto_defs.h"
#include "tanimoto_api.h"
#include <cstddef>
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

// ---- Main Interface ----

extern "C" int calculate_tanimoto_score_for_hits_packed(
    const uint8_t* RESTRICT A_ptr, // Row-packed bytes (bit-packed along axis=1)
    const uint8_t* RESTRICT query_bytes_ptr,
    float onesQ,
    float onesA,
    float* RESTRICT scores_out_ptr,
    float lower_bound,
    float upper_bound,
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
    if (!query_bytes_ptr) {
        return 0;
    }
    if (onesQ <= 0.0f) {
        return 0;
    }

    const float sumQA = onesQ + onesA;
    const float lower_inter = sumQA * lower_bound / (1.0f + lower_bound);
    const float upper_inter = sumQA * upper_bound / (1.0f + upper_bound);
    int lower_inter_i = (int)lower_inter;
    if ((float)lower_inter_i < lower_inter) {
        lower_inter_i++;
    }
    int upper_inter_i = (int)upper_inter;
    if ((float)upper_inter_i > upper_inter) {
        upper_inter_i--;
    }
    if (lower_inter_i < 0) {
        lower_inter_i = 0;
    }
    int max_inter = (int)onesQ;
    if (onesA < (float)max_inter) {
        max_inter = (int)onesA;
    }
    if (max_inter < 0) {
        max_inter = 0;
    }
    if (upper_inter_i > max_inter) {
        upper_inter_i = max_inter;
    }
    if (lower_inter_i > upper_inter_i) {
        return 0;
    }

    const int max_threads = omp_get_max_threads();
    std::vector<std::vector<uint32_t>> thread_hits(max_threads);

    if (n_cols == 16) {
        uint64_t q0 = 0;
        uint64_t q1 = 0;
        const bool query_aligned =
            (reinterpret_cast<uintptr_t>(query_bytes_ptr) &
             (alignof(uint64_t) - 1u)) == 0u;
        if (query_aligned) {
            const uint64_t* q64 =
                reinterpret_cast<const uint64_t*>(query_bytes_ptr);
            q0 = q64[0];
            q1 = q64[1];
        } else {
            std::memcpy(&q0, query_bytes_ptr, sizeof(q0));
            std::memcpy(&q1, query_bytes_ptr + 8, sizeof(q1));
        }

        const bool rows_aligned =
            (reinterpret_cast<uintptr_t>(A_ptr) &
             (alignof(uint64_t) - 1u)) == 0u;
        if (rows_aligned) {
            const uint64_t* rows64 = reinterpret_cast<const uint64_t*>(A_ptr);
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                std::vector<uint32_t>& hits = thread_hits[tid];
                #pragma omp for schedule(static)
                for (long row = 0; row < (long)n_rows; ++row) {
                    const size_t idx = (size_t)row * 2u;
                    uint64_t r0 = rows64[idx];
                    uint64_t r1 = rows64[idx + 1u];
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1);
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
                            scores_out_ptr[(size_t)row] = score;
                        }
                        hits.push_back((uint32_t)row);
                    }
                }
            }
        } else {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                std::vector<uint32_t>& hits = thread_hits[tid];
                #pragma omp for schedule(static)
                for (long row = 0; row < (long)n_rows; ++row) {
                    const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
                    uint64_t r0 = 0;
                    uint64_t r1 = 0;
                    std::memcpy(&r0, row_ptr, sizeof(r0));
                    std::memcpy(&r1, row_ptr + 8, sizeof(r1));
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1);
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
                            scores_out_ptr[(size_t)row] = score;
                        }
                        hits.push_back((uint32_t)row);
                    }
                }
            }
        }
    } else if (n_cols % 8 == 0) {
        const size_t word_cols = n_cols / 8;
        const bool query_aligned =
            (reinterpret_cast<uintptr_t>(query_bytes_ptr) &
             (alignof(uint64_t) - 1u)) == 0u;
        const uint64_t* query_words = nullptr;
        std::vector<uint64_t> query_buf;
        if (query_aligned) {
            query_words = reinterpret_cast<const uint64_t*>(query_bytes_ptr);
        } else {
            query_buf.resize(word_cols);
            std::memcpy(query_buf.data(), query_bytes_ptr, n_cols);
            query_words = query_buf.data();
        }

        const bool rows_aligned =
            (reinterpret_cast<uintptr_t>(A_ptr) &
             (alignof(uint64_t) - 1u)) == 0u;
        if (rows_aligned && query_aligned) {
            const uint64_t* rows64 = reinterpret_cast<const uint64_t*>(A_ptr);
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                std::vector<uint32_t>& hits = thread_hits[tid];
                #pragma omp for schedule(static)
                for (long row = 0; row < (long)n_rows; ++row) {
                    const uint64_t* row_words = rows64 + (size_t)row * word_cols;
                    unsigned int inter = 0;
                    for (size_t w = 0; w < word_cols; ++w) {
                        inter += popcount64(row_words[w] & query_words[w]);
                    }
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
                            scores_out_ptr[(size_t)row] = score;
                        }
                        hits.push_back((uint32_t)row);
                    }
                }
            }
        } else {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                std::vector<uint32_t>& hits = thread_hits[tid];
                #pragma omp for schedule(static)
                for (long row = 0; row < (long)n_rows; ++row) {
                    const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
                    unsigned int inter = 0;
                    for (size_t w = 0; w < word_cols; ++w) {
                        uint64_t val = 0;
                        std::memcpy(&val, row_ptr + w * 8u, sizeof(val));
                        inter += popcount64(val & query_words[w]);
                    }
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
                            scores_out_ptr[(size_t)row] = score;
                        }
                        hits.push_back((uint32_t)row);
                    }
                }
            }
        }
    } else {
        const PopcountLut8& lut = _popcount_lut8();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::vector<uint32_t>& hits = thread_hits[tid];
            #pragma omp for schedule(static)
            for (long row = 0; row < (long)n_rows; ++row) {
                const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
                unsigned int inter = 0;

                for (size_t col = 0; col < n_cols; ++col) {
                    uint8_t val = row_ptr[col];
                    inter += lut.data[(uint8_t)(val & query_bytes_ptr[col])];
                }

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
                        scores_out_ptr[(size_t)row] = score;
                    }
                    hits.push_back((uint32_t)row);
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
