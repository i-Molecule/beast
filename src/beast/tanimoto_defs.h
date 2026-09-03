#ifndef TANIMOTO_DEFS_H
#define TANIMOTO_DEFS_H

#include <cstdint>
#include <cmath>
#include <vector>
#include <omp.h>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanForward)
static inline int __builtin_ctz(uint32_t x) {
    unsigned long ret;
    _BitScanForward(&ret, x);
    return (int)ret;
}
static inline int __builtin_popcount(uint32_t x) {
    return (int)__popcnt(x);
}
#ifdef _WIN64
static inline int __builtin_popcountll(uint64_t x) {
    return (int)__popcnt64(x);
}
#else
static inline int __builtin_popcountll(uint64_t x) {
    return (int)__popcnt((uint32_t)x) + (int)__popcnt((uint32_t)(x >> 32));
}
#endif
#endif

// ---- popcount helpers ----

// Force inlining for these small helpers
#if defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE inline
#endif

static ALWAYS_INLINE int clamp_omp_threads(int requested, size_t max_tasks = 0) {
    if (requested <= 0) {
        return 0;
    }
    int max_threads = omp_get_num_procs();
    if (max_threads <= 0) {
        max_threads = 1;
    }
    if (requested > max_threads) {
        requested = max_threads;
    }
    if (max_tasks > 0 && (size_t)requested > max_tasks) {
        requested = (int)max_tasks;
        if (requested < 1) {
            requested = 1;
        }
    }
    return requested;
}

static ALWAYS_INLINE unsigned int popcount32(uint32_t x) {
    return __builtin_popcount(x);
}

static ALWAYS_INLINE unsigned int popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}

static ALWAYS_INLINE unsigned int ctz32(uint32_t x) {
    return __builtin_ctz(x);
}

// ---- threshold bounds ----

// A row clears the Tanimoto threshold t = num/den when
//   inter / (sumQA - inter) >= num/den   <=>   inter * (den + num) >= num * sumQA,
// so the smallest admissible intersection is ceil(num * sumQA / (den + num)) and
// the largest is floor of the same expression.
//
// Evaluating that in float misrounds whenever the exact bound is an integer:
// 0.4f and 1.4f are both inexact, so sumQA * 0.4f / 1.4f yields 24.000002 rather
// than 24 and the round-up pushes the bound to 25. Every row whose Tanimoto sits
// exactly on the threshold is then dropped. The bound is therefore derived in
// exact integer arithmetic from a rational threshold supplied by the caller.
//
// thr_den == 0 means the caller did not supply one; the float path is kept for
// those callers so the ABI stays usable without a rational threshold.
static ALWAYS_INLINE void tanimoto_inter_bounds(
    uint64_t sum_qa,
    float lower_bound,
    float upper_bound,
    uint64_t thr_lower_num,
    uint64_t thr_upper_num,
    uint64_t thr_den,
    int64_t* lower_out,
    int64_t* upper_out
) {
    if (thr_den != 0u) {
        const uint64_t lower_scale = thr_den + thr_lower_num;
        const uint64_t upper_scale = thr_den + thr_upper_num;
        *lower_out = (int64_t)(
            (thr_lower_num * sum_qa + lower_scale - 1u) / lower_scale
        );
        *upper_out = (int64_t)((thr_upper_num * sum_qa) / upper_scale);
        return;
    }

    const float sum_f = (float)sum_qa;
    const float lower_f = sum_f * lower_bound / (1.0f + lower_bound);
    const float upper_f = sum_f * upper_bound / (1.0f + upper_bound);
    int64_t lower_i = (int64_t)lower_f;
    if ((float)lower_i < lower_f) {
        lower_i++;
    }
    int64_t upper_i = (int64_t)upper_f;
    if ((float)upper_i > upper_f) {
        upper_i--;
    }
    *lower_out = lower_i;
    *upper_out = upper_i;
}

// ---- tiny inner helpers ----

static ALWAYS_INLINE unsigned long long _row_pc64(const uint64_t* ai, size_t W) {
    if (W == 2) {
        return popcount64(ai[0]) + popcount64(ai[1]);
    }
    unsigned long long s = 0;
    for (size_t w = 0; w < W; ++w) {
        s += popcount64(ai[w]);
    }
    return s;
}

static ALWAYS_INLINE unsigned long long _row_pc32(const uint32_t* ai, size_t W) {
    if (W == 4) {
        return popcount32(ai[0]) + popcount32(ai[1]) +
               popcount32(ai[2]) + popcount32(ai[3]);
    }
    unsigned long long s = 0;
    for (size_t w = 0; w < W; ++w) {
        s += popcount32(ai[w]);
    }
    return s;
}

static ALWAYS_INLINE unsigned long long _inter_pc64(const uint64_t* ai, const uint64_t* bj, size_t W) {
    if (W == 2) {
        return popcount64(ai[0] & bj[0]) + popcount64(ai[1] & bj[1]);
    }
    unsigned long long s = 0;
    for (size_t w = 0; w < W; ++w) {
        s += popcount64(ai[w] & bj[w]);
    }
    return s;
}

static ALWAYS_INLINE unsigned long long _inter_pc32(const uint32_t* ai, const uint32_t* bj, size_t W) {
    if (W == 4) {
        return popcount32(ai[0] & bj[0]) + popcount32(ai[1] & bj[1]) +
               popcount32(ai[2] & bj[2]) + popcount32(ai[3] & bj[3]);
    }
    unsigned long long s = 0;
    for (size_t w = 0; w < W; ++w) {
        s += popcount32(ai[w] & bj[w]);
    }
    return s;
}

// Cached intersection helpers to avoid pointer indirection for query row when W is small

static ALWAYS_INLINE unsigned long long _inter_pc64_cached(
    const uint64_t* ai, size_t W,
    const uint64_t q0, const uint64_t q1, const uint64_t q2,
    const uint64_t q3, const uint64_t q4, const uint64_t q5,
    const uint64_t* full_q = nullptr // Fallback
) {
    switch (W) {
        case 2: return popcount64(ai[0] & q0) + popcount64(ai[1] & q1);
        case 3: return popcount64(ai[0] & q0) + popcount64(ai[1] & q1) + popcount64(ai[2] & q2);
        case 4: return popcount64(ai[0] & q0) + popcount64(ai[1] & q1) + popcount64(ai[2] & q2) + popcount64(ai[3] & q3);
        case 5: return popcount64(ai[0] & q0) + popcount64(ai[1] & q1) + popcount64(ai[2] & q2) + popcount64(ai[3] & q3) + popcount64(ai[4] & q4);
        case 6: return popcount64(ai[0] & q0) + popcount64(ai[1] & q1) + popcount64(ai[2] & q2) + popcount64(ai[3] & q3) + popcount64(ai[4] & q4) + popcount64(ai[5] & q5);
        default: return _inter_pc64(ai, full_q, W);
    }
}

static ALWAYS_INLINE unsigned long long _inter_pc32_cached(
    const uint32_t* ai, size_t W,
    const uint32_t q0, const uint32_t q1, const uint32_t q2,
    const uint32_t q3, const uint32_t q4, const uint32_t q5,
    const uint32_t* full_q = nullptr // Fallback
) {
    switch (W) {
        case 2: return popcount32(ai[0] & q0) + popcount32(ai[1] & q1);
        case 3: return popcount32(ai[0] & q0) + popcount32(ai[1] & q1) + popcount32(ai[2] & q2);
        case 4: return popcount32(ai[0] & q0) + popcount32(ai[1] & q1) + popcount32(ai[2] & q2) + popcount32(ai[3] & q3);
        case 5: return popcount32(ai[0] & q0) + popcount32(ai[1] & q1) + popcount32(ai[2] & q2) + popcount32(ai[3] & q3) + popcount32(ai[4] & q4);
        case 6: return popcount32(ai[0] & q0) + popcount32(ai[1] & q1) + popcount32(ai[2] & q2) + popcount32(ai[3] & q3) + popcount32(ai[4] & q4) + popcount32(ai[5] & q5);
        default: return _inter_pc32(ai, full_q, W);
    }
}

#endif // TANIMOTO_DEFS_H
