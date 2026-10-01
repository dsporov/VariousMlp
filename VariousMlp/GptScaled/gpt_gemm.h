// GptScaled's own copy of Karpathy/gpt_gemm.h, taken at commit 512eaaa and
// otherwise unchanged, so that the two diff cleanly. gpt_scaled compiles only
// the files in this directory, which keeps it independent of Karpathy/: a
// change on either side stays on that side. Where the comments below mention
// other files, they mean the ones in Karpathy/.
//
// gpt_gemm.h - a cache-blocked, vectorised matrix multiply for gpt_model.h.
//
// The layers in gpt_model.h are written the way the arithmetic reads: for every
// row, for every input channel, add a scaled row of weights into the output.
// That is correct and it threads well, and on the i9 it reaches about 2% of
// what the chip can do. Two things stand between it and the rest:
//
//   - every row of the input streams the whole weight matrix past the cache,
//     so at the scaled-up width the arithmetic waits on memory rather than the
//     other way round.
//   - neither /arch:AVX2 nor switching the scalar to float moved the measured
//     rate, which says the compiler is not turning those loops into vector
//     code on its own.
//
// This file is the standard answer to both, the one every BLAS since GotoBLAS
// has used. The multiply is cut into blocks sized to the caches - a KC-deep
// slab of B that lives in L3, an MC x KC block of A that lives in L2, a
// KC x NR sliver of B that lives in L1 - and each block is copied ("packed")
// into exactly the order the innermost loop will read it. The innermost loop
// then holds a 6 x 16 tile of C in registers while KC rank-one updates stream
// through it, and that tile is the only place the floating point units see
// any work.
//
// Everything is row-major, with a transpose flag per operand, because the
// three multiplies of a Linear layer want A*B, A*B^T and A^T*B and packing is
// the natural place to absorb the difference. C is always added into, never
// overwritten: every caller in gpt_model.h hands in either a buffer that
// Matrix::resize has just zeroed or a gradient that is meant to be summed into.
//
// Each element of C is produced by exactly one thread, summing its KC blocks in
// order, so the result is the same whatever size the team is - the property
// the scalar loops were threaded to keep. It is not the same as the scalar
// loops' result, though: the vector path fuses each multiply-add and every
// path sums in blocks, so the two agree to float rounding rather than bit for
// bit. gpt_bench.h measures by how much.

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

// The vector path wants FMA as well as AVX2. MSVC has no macro for FMA, but
// /arch:AVX2 promises it; gcc and clang define __FMA__ under -mfma or any
// -march that has it, -march=native on the i9 included.
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#define GPT_GEMM_AVX2 1
#include <immintrin.h>
#else
#define GPT_GEMM_AVX2 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gpt {
namespace blocked {

// The register tile. AVX2 has sixteen 256-bit registers: twelve hold the
// 6 x 16 tile of C as six rows of two 8-float vectors, two hold the current
// row of the B sliver, and one holds the broadcast element of A.
constexpr int kMR = 6;
constexpr int kNR = 16;

// The cache blocks, from the inside out. A KC x NR sliver of B is 16KB, half
// of the i9's 32KB L1, which leaves room for the A panel streaming past it. An
// MC x KC block of A is 120KB, small enough that the two hyperthreads sharing
// a core's 256KB L2 both fit. KC x NC of packed B is 3MB, a fifth of the L3.
constexpr int kMC = 120;
constexpr int kKC = 256;
constexpr int kNC = 3072;

static_assert(kMC % kMR == 0, "an A block has to be a whole number of panels");
static_assert(kNC % kNR == 0, "a B slab has to be a whole number of slivers");

// Below this many multiply-adds a multiply stays on the calling thread. Handing
// out work costs a few microseconds, which is about what a quarter of a million
// multiply-adds take on one core.
constexpr double kMinThreadedMacs = 262144.0;

inline bool in_parallel() {
#ifdef _OPENMP
    return omp_in_parallel() != 0;
#else
    return false;
#endif
}

inline int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline int team_index() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

inline int team_size() {
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

inline int round_up(int v, int to) {
    return (v + to - 1) / to * to;
}

// op(A) is m x k. Copies rows [i0, i0 + mc) and columns [p0, p0 + kc) of it
// into panels of kMR rows, each stored column by column, so that the tile
// reads kMR consecutive values for every step along k. Rows past the end are
// zeros, which is what lets the tile always compute a full 6 x 16.
template <class T>
void pack_a(const T* a, int lda, bool trans, int i0, int mc, int p0, int kc,
    T* out) {
    for (int ir = 0; ir < mc; ir += kMR) {
        T* panel = out + static_cast<size_t>(ir) * kc;
        const int rows = std::min(kMR, mc - ir);
        if (trans) {
            // op(A)(i, p) = A[p][i], so one step along k is one stored row,
            // and the panel's kMR values for it sit side by side.
            for (int p = 0; p < kc; ++p) {
                const T* src = a + static_cast<size_t>(p0 + p) * lda + i0 + ir;
                T* dst = panel + static_cast<size_t>(p) * kMR;
                for (int r = 0; r < rows; ++r) {
                    dst[r] = src[r];
                }
                for (int r = rows; r < kMR; ++r) {
                    dst[r] = T(0);
                }
            }
        }
        else {
            for (int r = 0; r < kMR; ++r) {
                if (r < rows) {
                    const T* src = a + static_cast<size_t>(i0 + ir + r) * lda + p0;
                    for (int p = 0; p < kc; ++p) {
                        panel[static_cast<size_t>(p) * kMR + r] = src[p];
                    }
                }
                else {
                    for (int p = 0; p < kc; ++p) {
                        panel[static_cast<size_t>(p) * kMR + r] = T(0);
                    }
                }
            }
        }
    }
}

// op(B) is k x n. Copies rows [p0, p0 + kc) and columns [j0, j0 + cols) of it
// into one sliver of kNR columns stored row by row, zero past the last column.
template <class T>
void pack_b(const T* b, int ldb, bool trans, int p0, int kc, int j0, int cols,
    T* out) {
    if (trans) {
        // op(B)(p, j) = B[j][p]: each column of the sliver is a stored row.
        for (int c = 0; c < kNR; ++c) {
            if (c < cols) {
                const T* src = b + static_cast<size_t>(j0 + c) * ldb + p0;
                for (int p = 0; p < kc; ++p) {
                    out[static_cast<size_t>(p) * kNR + c] = src[p];
                }
            }
            else {
                for (int p = 0; p < kc; ++p) {
                    out[static_cast<size_t>(p) * kNR + c] = T(0);
                }
            }
        }
    }
    else {
        for (int p = 0; p < kc; ++p) {
            const T* src = b + static_cast<size_t>(p0 + p) * ldb + j0;
            T* dst = out + static_cast<size_t>(p) * kNR;
            for (int c = 0; c < cols; ++c) {
                dst[c] = src[c];
            }
            for (int c = cols; c < kNR; ++c) {
                dst[c] = T(0);
            }
        }
    }
}

// C (kMR x kNR, row stride ldc) += alpha * the product of one packed A panel
// and one packed B sliver, kc deep. Written so that a compiler can vectorise
// the inner loop by itself: the accumulator is a local array, which nothing
// else can alias, and its row is exactly kNR wide. This is what a double build
// runs, and what any machine without AVX2 runs.
template <class T>
inline void micro_tile_portable(int kc, const T* a, const T* b, T alpha, T* c,
    int ldc) {
    T acc[kMR][kNR] = {};
    for (int p = 0; p < kc; ++p) {
        for (int r = 0; r < kMR; ++r) {
            const T ar = a[r];
            for (int j = 0; j < kNR; ++j) {
                acc[r][j] += ar * b[j];
            }
        }
        a += kMR;
        b += kNR;
    }
    for (int r = 0; r < kMR; ++r) {
        T* row = c + static_cast<size_t>(r) * ldc;
        for (int j = 0; j < kNR; ++j) {
            row[j] += alpha * acc[r][j];
        }
    }
}

#if GPT_GEMM_AVX2
inline void store_row_avx2(float* row, __m256 lo, __m256 hi, __m256 alpha) {
    _mm256_storeu_ps(row, _mm256_fmadd_ps(alpha, lo, _mm256_loadu_ps(row)));
    _mm256_storeu_ps(row + 8, _mm256_fmadd_ps(alpha, hi, _mm256_loadu_ps(row + 8)));
}

// The same tile with the twelve accumulators named one by one, so that they
// live in registers for the whole of the k loop instead of wherever the
// compiler would have put an array. Each step along k is two loads of B, six
// broadcasts of A and twelve fused multiply-adds - 192 flops for 8 loads.
// The loads are unaligned because std::vector promises 16 bytes, not 32; on
// anything since Haswell an unaligned load of aligned data costs nothing extra.
inline void micro_tile_avx2(int kc, const float* a, const float* b, float alpha,
    float* c, int ldc) {
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();

    for (int p = 0; p < kc; ++p) {
        const __m256 b0 = _mm256_loadu_ps(b);
        const __m256 b1 = _mm256_loadu_ps(b + 8);
        __m256 ar = _mm256_broadcast_ss(a);
        c00 = _mm256_fmadd_ps(ar, b0, c00);
        c01 = _mm256_fmadd_ps(ar, b1, c01);
        ar = _mm256_broadcast_ss(a + 1);
        c10 = _mm256_fmadd_ps(ar, b0, c10);
        c11 = _mm256_fmadd_ps(ar, b1, c11);
        ar = _mm256_broadcast_ss(a + 2);
        c20 = _mm256_fmadd_ps(ar, b0, c20);
        c21 = _mm256_fmadd_ps(ar, b1, c21);
        ar = _mm256_broadcast_ss(a + 3);
        c30 = _mm256_fmadd_ps(ar, b0, c30);
        c31 = _mm256_fmadd_ps(ar, b1, c31);
        ar = _mm256_broadcast_ss(a + 4);
        c40 = _mm256_fmadd_ps(ar, b0, c40);
        c41 = _mm256_fmadd_ps(ar, b1, c41);
        ar = _mm256_broadcast_ss(a + 5);
        c50 = _mm256_fmadd_ps(ar, b0, c50);
        c51 = _mm256_fmadd_ps(ar, b1, c51);
        a += kMR;
        b += kNR;
    }

    const __m256 al = _mm256_set1_ps(alpha);
    const size_t stride = static_cast<size_t>(ldc);
    store_row_avx2(c, c00, c01, al);
    store_row_avx2(c + stride, c10, c11, al);
    store_row_avx2(c + 2 * stride, c20, c21, al);
    store_row_avx2(c + 3 * stride, c30, c31, al);
    store_row_avx2(c + 4 * stride, c40, c41, al);
    store_row_avx2(c + 5 * stride, c50, c51, al);
}
#endif

template <class T>
inline void micro_tile(int kc, const T* a, const T* b, T alpha, T* c, int ldc) {
    micro_tile_portable(kc, a, b, alpha, c, ldc);
}

template <class T>
inline const char* kernel_name() {
    return "portable 6x16 tile";
}

#if GPT_GEMM_AVX2
// micro_tile is a plain overload, so float picks it by ordinary overload
// resolution and every other type falls through to the template. kernel_name
// has no arguments to overload on, so it is specialised instead.
inline void micro_tile(int kc, const float* a, const float* b, float alpha,
    float* c, int ldc) {
    micro_tile_avx2(kc, a, b, alpha, c, ldc);
}

template <>
inline const char* kernel_name<float>() {
    return "AVX2/FMA 6x16 tile";
}
#endif

// One MC-row block of A against one NR-column sliver of B: a column of tiles.
// A tile hanging off the bottom or right edge of C is computed in full into a
// scratch tile and only its valid part added in, so the tile itself never has
// to know about edges.
template <class T>
void compute_strip(const T* apack, int mc, const T* bsliver, int cols, int kc,
    T alpha, T* c, int ldc) {
    for (int ir = 0; ir < mc; ir += kMR) {
        const int rows = std::min(kMR, mc - ir);
        const T* panel = apack + static_cast<size_t>(ir) * kc;
        T* ctile = c + static_cast<size_t>(ir) * ldc;
        if (rows == kMR && cols == kNR) {
            micro_tile(kc, panel, bsliver, alpha, ctile, ldc);
            continue;
        }
        T scratch[kMR * kNR] = {};
        micro_tile(kc, panel, bsliver, alpha, scratch, kNR);
        for (int r = 0; r < rows; ++r) {
            for (int j = 0; j < cols; ++j) {
                ctile[static_cast<size_t>(r) * ldc + j] += scratch[r * kNR + j];
            }
        }
    }
}

// The packing buffers belong to the thread using them, and grow to the largest
// multiply that thread has seen. Packed B is shared with the rest of the team
// for the length of one call, which is safe because the calling thread's
// buffer is the one handed out and it does not start another multiply until
// this one has returned.
template <class T>
std::vector<T>& packed_a() {
    thread_local std::vector<T> buf;
    return buf;
}

template <class T>
std::vector<T>& packed_b() {
    thread_local std::vector<T> buf;
    return buf;
}

// C (m x n) += alpha * op(A) * op(B), where op(A) is m x k and op(B) is k x n.
// lda, ldb and ldc are the row strides of the matrices as they are stored, so
// a transposed operand's stride is the one of its untransposed shape. A stride
// wider than the row is fine, which is how attention hands in one head's
// slice of every row.
//
// may_thread says whether this is a place a team can be started. Attention
// calls this once per (batch, head) from inside a loop that is already split
// across threads, and those calls have to stay on the thread that made them.
template <class T>
void gemm(bool trans_a, bool trans_b, int m, int n, int k, T alpha,
    const T* a, int lda, const T* b, int ldb, T* c, int ldc, bool may_thread) {
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }

    std::vector<T>& bbuf = packed_b<T>();
    const size_t b_size = static_cast<size_t>(kKC) * round_up(std::min(n, kNC), kNR);
    if (bbuf.size() < b_size) {
        bbuf.resize(b_size);
    }
    T* bpack = bbuf.data();
    const size_t a_size = static_cast<size_t>(kMC) * kKC;

    const bool threaded = may_thread && !in_parallel() && max_threads() > 1
        && static_cast<double>(m) * n * k >= kMinThreadedMacs;

    if (!threaded) {
        std::vector<T>& abuf = packed_a<T>();
        if (abuf.size() < a_size) {
            abuf.resize(a_size);
        }
        T* apack = abuf.data();
        for (int jc = 0; jc < n; jc += kNC) {
            const int nc = std::min(kNC, n - jc);
            for (int pc = 0; pc < k; pc += kKC) {
                const int kc = std::min(kKC, k - pc);
                for (int jr = 0; jr < nc; jr += kNR) {
                    pack_b(b, ldb, trans_b, pc, kc, jc + jr, std::min(kNR, nc - jr),
                        bpack + static_cast<size_t>(jr) * kc);
                }
                for (int ic = 0; ic < m; ic += kMC) {
                    const int mc = std::min(kMC, m - ic);
                    pack_a(a, lda, trans_a, ic, mc, pc, kc, apack);
                    for (int jr = 0; jr < nc; jr += kNR) {
                        compute_strip(apack, mc, bpack + static_cast<size_t>(jr) * kc,
                            std::min(kNR, nc - jr), kc, alpha,
                            c + static_cast<size_t>(ic) * ldc + jc + jr, ldc);
                    }
                }
            }
        }
        return;
    }

    // The team shares one packed slab of B and splits the work under it as
    // (A block, B sliver) pairs. Pairs rather than A blocks alone, because a
    // weight gradient has only fan_in rows - four blocks at the video's final
    // width - and sixteen threads need more pieces than that. Each thread takes
    // a contiguous run of pairs, so it mostly stays on one A block and packs it
    // once; a block that straddles two threads' runs is packed by both.
    const int m_blocks = (m + kMC - 1) / kMC;
#pragma omp parallel
    {
        const int tid = team_index();
        const int nth = team_size();
        std::vector<T>& abuf = packed_a<T>();
        if (abuf.size() < a_size) {
            abuf.resize(a_size);
        }
        T* apack = abuf.data();

        for (int jc = 0; jc < n; jc += kNC) {
            const int nc = std::min(kNC, n - jc);
            const int slivers = (nc + kNR - 1) / kNR;
            for (int pc = 0; pc < k; pc += kKC) {
                const int kc = std::min(kKC, k - pc);
                for (int s = tid; s < slivers; s += nth) {
                    const int jr = s * kNR;
                    pack_b(b, ldb, trans_b, pc, kc, jc + jr, std::min(kNR, nc - jr),
                        bpack + static_cast<size_t>(jr) * kc);
                }
#pragma omp barrier

                const int pairs = m_blocks * slivers;
                const int first = static_cast<int>(static_cast<long long>(pairs) * tid / nth);
                const int last = static_cast<int>(static_cast<long long>(pairs) * (tid + 1) / nth);
                int packed = -1;
                for (int t = first; t < last; ++t) {
                    const int block = t / slivers;
                    const int ic = block * kMC;
                    const int mc = std::min(kMC, m - ic);
                    if (block != packed) {
                        pack_a(a, lda, trans_a, ic, mc, pc, kc, apack);
                        packed = block;
                    }
                    const int jr = (t % slivers) * kNR;
                    compute_strip(apack, mc, bpack + static_cast<size_t>(jr) * kc,
                        std::min(kNR, nc - jr), kc, alpha,
                        c + static_cast<size_t>(ic) * ldc + jc + jr, ldc);
                }
                // Nobody repacks B for the next slab until everyone is done
                // reading this one.
#pragma omp barrier
            }
        }
    }
}

}  // namespace blocked
}  // namespace gpt
