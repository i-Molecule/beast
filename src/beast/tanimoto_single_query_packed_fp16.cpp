#include "tanimoto_defs.h"
#include "tanimoto_api.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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

static ALWAYS_INLINE uint8_t quantize_score100_integer(
    uint32_t inter,
    uint32_t onesQ,
    uint32_t onesA
) {
    const uint32_t denom = onesQ + onesA - inter;
    if (denom == 0u) {
        return (inter == 0u && onesQ == 0u && onesA == 0u) ? 100u : 0u;
    }

    const uint64_t numerator = 200ull * static_cast<uint64_t>(inter) +
                               static_cast<uint64_t>(denom);
    const uint64_t denominator = 2ull * static_cast<uint64_t>(denom);
    uint64_t q = numerator / denominator;
    if (q > 100ull) {
        q = 100ull;
    }
    return static_cast<uint8_t>(q);
}

static std::vector<uint8_t> build_score100_lut(
    uint32_t onesQ,
    uint32_t onesA,
    size_t fp_size
) {
    const uint32_t fp_bits = fp_size > (std::numeric_limits<uint32_t>::max() / 8u)
        ? std::numeric_limits<uint32_t>::max()
        : static_cast<uint32_t>(fp_size * 8u);
    uint32_t max_inter = onesQ < onesA ? onesQ : onesA;
    if (max_inter > fp_bits) {
        max_inter = fp_bits;
    }

    std::vector<uint8_t> score_lut(static_cast<size_t>(max_inter) + 1u, 0);
    for (uint32_t inter = 0; inter <= max_inter; ++inter) {
        score_lut[inter] = quantize_score100_integer(inter, onesQ, onesA);
    }

    return score_lut;
}

template <size_t W>
static void score_packed_u8_words(
    const uint8_t* RESTRICT A_ptr,
    const uint8_t* RESTRICT query_bytes_ptr,
    uint8_t* RESTRICT scores_out_ptr,
    size_t n_rows,
    int threads,
    const std::vector<uint8_t>& score_lut
) {
    uint64_t q[W] = {};
    for (size_t w = 0; w < W; ++w) {
        std::memcpy(&q[w], query_bytes_ptr + (w * sizeof(uint64_t)), sizeof(uint64_t));
    }

    const bool rows_aligned =
        (reinterpret_cast<uintptr_t>(A_ptr) &
         (alignof(uint64_t) - 1u)) == 0u;

    if (rows_aligned) {
        const uint64_t* rows64 = reinterpret_cast<const uint64_t*>(A_ptr);
        if (threads > 1) {
            #pragma omp parallel for schedule(static)
            for (long row = 0; row < (long)n_rows; ++row) {
                const uint64_t* row_ptr = rows64 + ((size_t)row * W);
                unsigned int inter = 0;
                for (size_t w = 0; w < W; ++w) {
                    inter += (unsigned int)__builtin_popcountll(row_ptr[w] & q[w]);
                }
                scores_out_ptr[(size_t)row] = score_lut[inter];
            }
        } else {
            for (long row = 0; row < (long)n_rows; ++row) {
                const uint64_t* row_ptr = rows64 + ((size_t)row * W);
                unsigned int inter = 0;
                for (size_t w = 0; w < W; ++w) {
                    inter += (unsigned int)__builtin_popcountll(row_ptr[w] & q[w]);
                }
                scores_out_ptr[(size_t)row] = score_lut[inter];
            }
        }
        return;
    }

    if (threads > 1) {
        #pragma omp parallel for schedule(static)
        for (long row = 0; row < (long)n_rows; ++row) {
            const uint8_t* row_ptr = A_ptr + ((size_t)row * W * sizeof(uint64_t));
            unsigned int inter = 0;
            for (size_t w = 0; w < W; ++w) {
                uint64_t r = 0;
                std::memcpy(&r, row_ptr + (w * sizeof(uint64_t)), sizeof(uint64_t));
                inter += (unsigned int)__builtin_popcountll(r & q[w]);
            }
            scores_out_ptr[(size_t)row] = score_lut[inter];
        }
    } else {
        for (long row = 0; row < (long)n_rows; ++row) {
            const uint8_t* row_ptr = A_ptr + ((size_t)row * W * sizeof(uint64_t));
            unsigned int inter = 0;
            for (size_t w = 0; w < W; ++w) {
                uint64_t r = 0;
                std::memcpy(&r, row_ptr + (w * sizeof(uint64_t)), sizeof(uint64_t));
                inter += (unsigned int)__builtin_popcountll(r & q[w]);
            }
            scores_out_ptr[(size_t)row] = score_lut[inter];
        }
    }
}

// ---- Float32 -> Float16 conversion ----

static ALWAYS_INLINE uint16_t float_to_fp16_bits(float value) {
#if defined(__F16C__)
    __m128 v = _mm_set_ss(value);
    __m128i h = _mm_cvtps_ph(
        v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );
    return static_cast<uint16_t>(_mm_cvtsi128_si32(h));
#elif defined(__FLT16_MAX__)
    _Float16 h = (_Float16)value;
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
#else
#error "No FP16 conversion path available (need F16C or _Float16)."
#endif
}

// ---- Main Interface ----

extern "C" void calculate_tanimoto_score_packed_f16(
    const uint8_t* RESTRICT A_ptr, // Row-packed bytes (bit-packed along axis=1)
    const uint8_t* RESTRICT query_bytes_ptr, // Bit-packed query bytes
    float onesQ,
    float onesA,
    uint16_t* RESTRICT scores_out_ptr, // Output: IEEE-754 half bits
    size_t fp_size,
    size_t n_rows,
    int n_threads
) {
    if (!A_ptr || !scores_out_ptr) {
        return;
    }
    if (fp_size == 0 || n_rows == 0) {
        return;
    }

    int threads = n_threads;
    if (n_rows < 100000) {
        // std::fprintf(stderr, "we use n_threads=1\n");
        threads = 1;
    }
    if (threads > 0) {
        threads = clamp_omp_threads(threads);
        omp_set_num_threads(threads);
    }

    const size_t n_cols = fp_size;
    const float onesQA = onesQ + onesA;
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
            if (threads > 1) {
                #pragma omp parallel for schedule(static)
                for (long row = 0; row < (long)n_rows; ++row) {
                    const size_t idx = (size_t)row * 2u;
                    uint64_t r0 = rows64[idx];
                    uint64_t r1 = rows64[idx + 1u];
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1);

                    float denom = onesQA - (float)inter;
                    float score = 0.0f;
                    if (denom > 0.0f) {
                        score = (float)inter / denom;
                    } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                        score = 1.0f;
                    }
                    scores_out_ptr[(size_t)row] = float_to_fp16_bits(score);
                }
            } else {
                for (long row = 0; row < (long)n_rows; ++row) {
                    const size_t idx = (size_t)row * 2u;
                    uint64_t r0 = rows64[idx];
                    uint64_t r1 = rows64[idx + 1u];
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1);

                    float denom = onesQA - (float)inter;
                    float score = 0.0f;
                    if (denom > 0.0f) {
                        score = (float)inter / denom;
                    } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                        score = 1.0f;
                    }
                    scores_out_ptr[(size_t)row] = float_to_fp16_bits(score);
                }
            }
        } else {
            if (threads > 1) {
                #pragma omp parallel for schedule(static)
                for (long row = 0; row < (long)n_rows; ++row) {
                    const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
                    uint64_t r0 = 0;
                    uint64_t r1 = 0;
                    std::memcpy(&r0, row_ptr, sizeof(r0));
                    std::memcpy(&r1, row_ptr + 8, sizeof(r1));
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1);

                    float denom = onesQA - (float)inter;
                    float score = 0.0f;
                    if (denom > 0.0f) {
                        score = (float)inter / denom;
                    } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                        score = 1.0f;
                    }
                    scores_out_ptr[(size_t)row] = float_to_fp16_bits(score);
                }
            } else {
                for (long row = 0; row < (long)n_rows; ++row) {
                    const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
                    uint64_t r0 = 0;
                    uint64_t r1 = 0;
                    std::memcpy(&r0, row_ptr, sizeof(r0));
                    std::memcpy(&r1, row_ptr + 8, sizeof(r1));
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1);

                    float denom = onesQA - (float)inter;
                    float score = 0.0f;
                    if (denom > 0.0f) {
                        score = (float)inter / denom;
                    } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                        score = 1.0f;
                    }
                    scores_out_ptr[(size_t)row] = float_to_fp16_bits(score);
                }
            }
        }
        return;
    }
    if (n_cols == 48) {
        uint64_t q0 = 0;
        uint64_t q1 = 0;
        uint64_t q2 = 0;
        uint64_t q3 = 0;
        uint64_t q4 = 0;
        uint64_t q5 = 0;
        const bool query_aligned =
            (reinterpret_cast<uintptr_t>(query_bytes_ptr) &
             (alignof(uint64_t) - 1u)) == 0u;
        if (query_aligned) {
            const uint64_t* q64 =
                reinterpret_cast<const uint64_t*>(query_bytes_ptr);
            q0 = q64[0];
            q1 = q64[1];
            q2 = q64[2];
            q3 = q64[3];
            q4 = q64[4];
            q5 = q64[5];
        } else {
            std::memcpy(&q0, query_bytes_ptr, sizeof(q0));
            std::memcpy(&q1, query_bytes_ptr + 8, sizeof(q1));
            std::memcpy(&q2, query_bytes_ptr + 16, sizeof(q2));
            std::memcpy(&q3, query_bytes_ptr + 24, sizeof(q3));
            std::memcpy(&q4, query_bytes_ptr + 32, sizeof(q4));
            std::memcpy(&q5, query_bytes_ptr + 40, sizeof(q5));
        }

        const bool rows_aligned =
            (reinterpret_cast<uintptr_t>(A_ptr) &
             (alignof(uint64_t) - 1u)) == 0u;
        if (rows_aligned) {
            const uint64_t* rows64 = reinterpret_cast<const uint64_t*>(A_ptr);
            if (threads > 1) {
                #pragma omp parallel for schedule(static)
                for (long row = 0; row < (long)n_rows; ++row) {
                    const size_t idx = (size_t)row * 6u;
                    uint64_t r0 = rows64[idx];
                    uint64_t r1 = rows64[idx + 1u];
                    uint64_t r2 = rows64[idx + 2u];
                    uint64_t r3 = rows64[idx + 3u];
                    uint64_t r4 = rows64[idx + 4u];
                    uint64_t r5 = rows64[idx + 5u];
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1) +
                        (unsigned int)__builtin_popcountll(r2 & q2) +
                        (unsigned int)__builtin_popcountll(r3 & q3) +
                        (unsigned int)__builtin_popcountll(r4 & q4) +
                        (unsigned int)__builtin_popcountll(r5 & q5);

                    float denom = onesQA - (float)inter;
                    float score = 0.0f;
                    if (denom > 0.0f) {
                        score = (float)inter / denom;
                    } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                        score = 1.0f;
                    }
                    scores_out_ptr[(size_t)row] = float_to_fp16_bits(score);
                }
            } else {
                for (long row = 0; row < (long)n_rows; ++row) {
                    const size_t idx = (size_t)row * 6u;
                    uint64_t r0 = rows64[idx];
                    uint64_t r1 = rows64[idx + 1u];
                    uint64_t r2 = rows64[idx + 2u];
                    uint64_t r3 = rows64[idx + 3u];
                    uint64_t r4 = rows64[idx + 4u];
                    uint64_t r5 = rows64[idx + 5u];
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1) +
                        (unsigned int)__builtin_popcountll(r2 & q2) +
                        (unsigned int)__builtin_popcountll(r3 & q3) +
                        (unsigned int)__builtin_popcountll(r4 & q4) +
                        (unsigned int)__builtin_popcountll(r5 & q5);

                    float denom = onesQA - (float)inter;
                    float score = 0.0f;
                    if (denom > 0.0f) {
                        score = (float)inter / denom;
                    } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                        score = 1.0f;
                    }
                    scores_out_ptr[(size_t)row] = float_to_fp16_bits(score);
                }
            }
        } else {
            if (threads > 1) {
                #pragma omp parallel for schedule(static)
                for (long row = 0; row < (long)n_rows; ++row) {
                    const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
                    uint64_t r0 = 0;
                    uint64_t r1 = 0;
                    uint64_t r2 = 0;
                    uint64_t r3 = 0;
                    uint64_t r4 = 0;
                    uint64_t r5 = 0;
                    std::memcpy(&r0, row_ptr, sizeof(r0));
                    std::memcpy(&r1, row_ptr + 8, sizeof(r1));
                    std::memcpy(&r2, row_ptr + 16, sizeof(r2));
                    std::memcpy(&r3, row_ptr + 24, sizeof(r3));
                    std::memcpy(&r4, row_ptr + 32, sizeof(r4));
                    std::memcpy(&r5, row_ptr + 40, sizeof(r5));
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1) +
                        (unsigned int)__builtin_popcountll(r2 & q2) +
                        (unsigned int)__builtin_popcountll(r3 & q3) +
                        (unsigned int)__builtin_popcountll(r4 & q4) +
                        (unsigned int)__builtin_popcountll(r5 & q5);

                    float denom = onesQA - (float)inter;
                    float score = 0.0f;
                    if (denom > 0.0f) {
                        score = (float)inter / denom;
                    } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                        score = 1.0f;
                    }
                    scores_out_ptr[(size_t)row] = float_to_fp16_bits(score);
                }
            } else {
                for (long row = 0; row < (long)n_rows; ++row) {
                    const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
                    uint64_t r0 = 0;
                    uint64_t r1 = 0;
                    uint64_t r2 = 0;
                    uint64_t r3 = 0;
                    uint64_t r4 = 0;
                    uint64_t r5 = 0;
                    std::memcpy(&r0, row_ptr, sizeof(r0));
                    std::memcpy(&r1, row_ptr + 8, sizeof(r1));
                    std::memcpy(&r2, row_ptr + 16, sizeof(r2));
                    std::memcpy(&r3, row_ptr + 24, sizeof(r3));
                    std::memcpy(&r4, row_ptr + 32, sizeof(r4));
                    std::memcpy(&r5, row_ptr + 40, sizeof(r5));
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1) +
                        (unsigned int)__builtin_popcountll(r2 & q2) +
                        (unsigned int)__builtin_popcountll(r3 & q3) +
                        (unsigned int)__builtin_popcountll(r4 & q4) +
                        (unsigned int)__builtin_popcountll(r5 & q5);

                    float denom = onesQA - (float)inter;
                    float score = 0.0f;
                    if (denom > 0.0f) {
                        score = (float)inter / denom;
                    } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                        score = 1.0f;
                    }
                    scores_out_ptr[(size_t)row] = float_to_fp16_bits(score);
                }
            }
        }
        return;
    }

    const PopcountLut8& lut = _popcount_lut8();
    if (threads > 1) {
        #pragma omp parallel for schedule(static)
        for (long row = 0; row < (long)n_rows; ++row) {
            const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
            unsigned int inter = 0;

            for (size_t col = 0; col < n_cols; ++col) {
                uint8_t val = row_ptr[col];
                inter += lut.data[(uint8_t)(val & query_bytes_ptr[col])];
            }

            float denom = onesQA - (float)inter;
            float score = 0.0f;
            if (denom > 0.0f) {
                score = (float)inter / denom;
            } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                score = 1.0f;
            }
            scores_out_ptr[(size_t)row] = float_to_fp16_bits(score);
        }
    } else {
        for (long row = 0; row < (long)n_rows; ++row) {
            const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
            unsigned int inter = 0;

            for (size_t col = 0; col < n_cols; ++col) {
                uint8_t val = row_ptr[col];
                inter += lut.data[(uint8_t)(val & query_bytes_ptr[col])];
            }

            float denom = onesQA - (float)inter;
            float score = 0.0f;
            if (denom > 0.0f) {
                score = (float)inter / denom;
            } else if (inter == 0 && onesQ == 0.0f && onesA == 0.0f) {
                score = 1.0f;
            }
            scores_out_ptr[(size_t)row] = float_to_fp16_bits(score);
        }
    }
}

extern "C" void calculate_tanimoto_score_packed_u8(
    const uint8_t* RESTRICT A_ptr, // Row-packed bytes (bit-packed along axis=1)
    const uint8_t* RESTRICT query_bytes_ptr, // Bit-packed query bytes
    uint32_t onesQ,
    uint32_t onesA,
    uint8_t* RESTRICT scores_out_ptr, // Output: uint8 centi-scores
    size_t fp_size,
    size_t n_rows,
    int n_threads
) {
    if (!A_ptr || !query_bytes_ptr || !scores_out_ptr) {
        return;
    }
    if (fp_size == 0 || n_rows == 0) {
        return;
    }

    int threads = n_threads;
    if (n_rows < 100000) {
        threads = 1;
    }
    if (threads > 0) {
        threads = clamp_omp_threads(threads);
        omp_set_num_threads(threads);
    }

    const size_t n_cols = fp_size;
    const std::vector<uint8_t> score_lut = build_score100_lut(
        onesQ, onesA, fp_size
    );

    if (n_cols == 8) {
        score_packed_u8_words<1>(
            A_ptr, query_bytes_ptr, scores_out_ptr, n_rows, threads, score_lut
        );
        return;
    }

    if (n_cols == 16) {
        uint64_t q0 = 0;
        uint64_t q1 = 0;
        std::memcpy(&q0, query_bytes_ptr, sizeof(q0));
        std::memcpy(&q1, query_bytes_ptr + 8, sizeof(q1));

        const bool rows_aligned =
            (reinterpret_cast<uintptr_t>(A_ptr) &
             (alignof(uint64_t) - 1u)) == 0u;
        if (rows_aligned) {
            const uint64_t* rows64 = reinterpret_cast<const uint64_t*>(A_ptr);
            if (threads > 1) {
                #pragma omp parallel for schedule(static)
                for (long row = 0; row < (long)n_rows; ++row) {
                    const size_t idx = (size_t)row * 2u;
                    uint64_t r0 = rows64[idx];
                    uint64_t r1 = rows64[idx + 1u];
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1);
                    scores_out_ptr[(size_t)row] = score_lut[inter];
                }
            } else {
                for (long row = 0; row < (long)n_rows; ++row) {
                    const size_t idx = (size_t)row * 2u;
                    uint64_t r0 = rows64[idx];
                    uint64_t r1 = rows64[idx + 1u];
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1);
                    scores_out_ptr[(size_t)row] = score_lut[inter];
                }
            }
        } else {
            if (threads > 1) {
                #pragma omp parallel for schedule(static)
                for (long row = 0; row < (long)n_rows; ++row) {
                    const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
                    uint64_t r0 = 0;
                    uint64_t r1 = 0;
                    std::memcpy(&r0, row_ptr, sizeof(r0));
                    std::memcpy(&r1, row_ptr + 8, sizeof(r1));
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1);
                    scores_out_ptr[(size_t)row] = score_lut[inter];
                }
            } else {
                for (long row = 0; row < (long)n_rows; ++row) {
                    const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
                    uint64_t r0 = 0;
                    uint64_t r1 = 0;
                    std::memcpy(&r0, row_ptr, sizeof(r0));
                    std::memcpy(&r1, row_ptr + 8, sizeof(r1));
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1);
                    scores_out_ptr[(size_t)row] = score_lut[inter];
                }
            }
        }
        return;
    }

    if (n_cols == 32) {
        score_packed_u8_words<4>(
            A_ptr, query_bytes_ptr, scores_out_ptr, n_rows, threads, score_lut
        );
        return;
    }

    if (n_cols == 48) {
        uint64_t q0 = 0;
        uint64_t q1 = 0;
        uint64_t q2 = 0;
        uint64_t q3 = 0;
        uint64_t q4 = 0;
        uint64_t q5 = 0;
        std::memcpy(&q0, query_bytes_ptr, sizeof(q0));
        std::memcpy(&q1, query_bytes_ptr + 8, sizeof(q1));
        std::memcpy(&q2, query_bytes_ptr + 16, sizeof(q2));
        std::memcpy(&q3, query_bytes_ptr + 24, sizeof(q3));
        std::memcpy(&q4, query_bytes_ptr + 32, sizeof(q4));
        std::memcpy(&q5, query_bytes_ptr + 40, sizeof(q5));

        const bool rows_aligned =
            (reinterpret_cast<uintptr_t>(A_ptr) &
             (alignof(uint64_t) - 1u)) == 0u;
        if (rows_aligned) {
            const uint64_t* rows64 = reinterpret_cast<const uint64_t*>(A_ptr);
            if (threads > 1) {
                #pragma omp parallel for schedule(static)
                for (long row = 0; row < (long)n_rows; ++row) {
                    const size_t idx = (size_t)row * 6u;
                    uint64_t r0 = rows64[idx];
                    uint64_t r1 = rows64[idx + 1u];
                    uint64_t r2 = rows64[idx + 2u];
                    uint64_t r3 = rows64[idx + 3u];
                    uint64_t r4 = rows64[idx + 4u];
                    uint64_t r5 = rows64[idx + 5u];
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1) +
                        (unsigned int)__builtin_popcountll(r2 & q2) +
                        (unsigned int)__builtin_popcountll(r3 & q3) +
                        (unsigned int)__builtin_popcountll(r4 & q4) +
                        (unsigned int)__builtin_popcountll(r5 & q5);
                    scores_out_ptr[(size_t)row] = score_lut[inter];
                }
            } else {
                for (long row = 0; row < (long)n_rows; ++row) {
                    const size_t idx = (size_t)row * 6u;
                    uint64_t r0 = rows64[idx];
                    uint64_t r1 = rows64[idx + 1u];
                    uint64_t r2 = rows64[idx + 2u];
                    uint64_t r3 = rows64[idx + 3u];
                    uint64_t r4 = rows64[idx + 4u];
                    uint64_t r5 = rows64[idx + 5u];
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1) +
                        (unsigned int)__builtin_popcountll(r2 & q2) +
                        (unsigned int)__builtin_popcountll(r3 & q3) +
                        (unsigned int)__builtin_popcountll(r4 & q4) +
                        (unsigned int)__builtin_popcountll(r5 & q5);
                    scores_out_ptr[(size_t)row] = score_lut[inter];
                }
            }
        } else {
            if (threads > 1) {
                #pragma omp parallel for schedule(static)
                for (long row = 0; row < (long)n_rows; ++row) {
                    const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
                    uint64_t r0 = 0;
                    uint64_t r1 = 0;
                    uint64_t r2 = 0;
                    uint64_t r3 = 0;
                    uint64_t r4 = 0;
                    uint64_t r5 = 0;
                    std::memcpy(&r0, row_ptr, sizeof(r0));
                    std::memcpy(&r1, row_ptr + 8, sizeof(r1));
                    std::memcpy(&r2, row_ptr + 16, sizeof(r2));
                    std::memcpy(&r3, row_ptr + 24, sizeof(r3));
                    std::memcpy(&r4, row_ptr + 32, sizeof(r4));
                    std::memcpy(&r5, row_ptr + 40, sizeof(r5));
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1) +
                        (unsigned int)__builtin_popcountll(r2 & q2) +
                        (unsigned int)__builtin_popcountll(r3 & q3) +
                        (unsigned int)__builtin_popcountll(r4 & q4) +
                        (unsigned int)__builtin_popcountll(r5 & q5);
                    scores_out_ptr[(size_t)row] = score_lut[inter];
                }
            } else {
                for (long row = 0; row < (long)n_rows; ++row) {
                    const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
                    uint64_t r0 = 0;
                    uint64_t r1 = 0;
                    uint64_t r2 = 0;
                    uint64_t r3 = 0;
                    uint64_t r4 = 0;
                    uint64_t r5 = 0;
                    std::memcpy(&r0, row_ptr, sizeof(r0));
                    std::memcpy(&r1, row_ptr + 8, sizeof(r1));
                    std::memcpy(&r2, row_ptr + 16, sizeof(r2));
                    std::memcpy(&r3, row_ptr + 24, sizeof(r3));
                    std::memcpy(&r4, row_ptr + 32, sizeof(r4));
                    std::memcpy(&r5, row_ptr + 40, sizeof(r5));
                    unsigned int inter =
                        (unsigned int)__builtin_popcountll(r0 & q0) +
                        (unsigned int)__builtin_popcountll(r1 & q1) +
                        (unsigned int)__builtin_popcountll(r2 & q2) +
                        (unsigned int)__builtin_popcountll(r3 & q3) +
                        (unsigned int)__builtin_popcountll(r4 & q4) +
                        (unsigned int)__builtin_popcountll(r5 & q5);
                    scores_out_ptr[(size_t)row] = score_lut[inter];
                }
            }
        }
        return;
    }

    if (n_cols == 64) {
        score_packed_u8_words<8>(
            A_ptr, query_bytes_ptr, scores_out_ptr, n_rows, threads, score_lut
        );
        return;
    }

    const PopcountLut8& lut = _popcount_lut8();
    if (threads > 1) {
        #pragma omp parallel for schedule(static)
        for (long row = 0; row < (long)n_rows; ++row) {
            const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
            unsigned int inter = 0;

            for (size_t col = 0; col < n_cols; ++col) {
                uint8_t val = row_ptr[col];
                inter += lut.data[(uint8_t)(val & query_bytes_ptr[col])];
            }

            scores_out_ptr[(size_t)row] = score_lut[inter];
        }
    } else {
        for (long row = 0; row < (long)n_rows; ++row) {
            const uint8_t* row_ptr = A_ptr + (size_t)row * n_cols;
            unsigned int inter = 0;

            for (size_t col = 0; col < n_cols; ++col) {
                uint8_t val = row_ptr[col];
                inter += lut.data[(uint8_t)(val & query_bytes_ptr[col])];
            }

            scores_out_ptr[(size_t)row] = score_lut[inter];
        }
    }
}
