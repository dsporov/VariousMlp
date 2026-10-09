// gpt_gemm.h - a cache-blocked, vectorised matrix multiply for gpt_model.h.
//
// The layers in gpt_model.h are written in the same order as the arithmetic:
// for every row, for every input channel, add a scaled row of weights to the
// output. That code is correct and works well with threads, but on the i9 it
// reaches only about 2% of the chip's peak speed. There are two reasons:
//
//   - every row of the input reads the whole weight matrix through the cache.
//     So at the scaled-up width, the arithmetic waits for memory.
//   - neither /arch:AVX2 nor switching the scalar type to float changed the
//     measured speed. This shows that the compiler does not turn those loops
//     into vector code by itself.
//
// This file uses the standard solution to both problems, which every BLAS
// library has used since GotoBLAS. The multiply is cut into blocks that fit
// the caches: a KC-deep slab of B that stays in L3, an MC x KC block of A that
// stays in L2, and a KC x NR sliver of B that stays in L1. Each block is
// copied ("packed") into exactly the order in which the innermost loop reads
// it. The innermost loop keeps a 6 x 16 tile of C in registers while it adds
// KC rank-one updates to it. All of the floating point work happens in that
// tile.
//
// Everything is row-major, with a transpose flag for each operand. The three
// multiplies of a Linear layer need A*B, A*B^T and A^T*B, and packing is the
// natural place to handle that difference. The result is always added to C;
// C is never overwritten. Every caller in gpt_model.h passes either a buffer
// that Matrix::resize has just zeroed, or a gradient that is meant to be
// summed into.
//
// Each element of C is computed by exactly one thread, which adds up its KC
// blocks in order. So the result is the same for any number of threads. The
// threaded scalar loops were written to have the same property. The result is
// not identical to the scalar loops' result, though. The vector path fuses
// each multiply and add, and every path sums in blocks, so the two results
// agree to float rounding, not bit for bit. gpt_bench.h measures how large the
// difference is.

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

// The vector path needs FMA as well as AVX2. MSVC has no macro for FMA, but
// /arch:AVX2 guarantees it. gcc and clang define __FMA__ under -mfma, or under
// any -march that includes FMA, such as -march=native on the i9.
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

// The cache blocks, from the innermost out. A KC x NR sliver of B is 16KB,
// half of the i9's 32KB L1 cache, which leaves room for the A panel that is
// read alongside it. An MC x KC block of A is 120KB. That is small enough for
// the blocks of both hyperthreads of a core to fit in the core's 256KB L2
// cache. The packed KC x NC slab of B is 3MB, a fifth of the L3 cache.
constexpr int kMC = 120;
constexpr int kKC = 256;
constexpr int kNC = 3072;

static_assert(kMC % kMR == 0, "an A block has to be a whole number of panels");
static_assert(kNC % kNR == 0, "a B slab has to be a whole number of slivers");

// A multiply with fewer multiply-adds than this stays on the calling thread.
// Distributing work to threads costs a few microseconds, which is about the
// time one core needs for a quarter of a million multiply-adds.
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

// op(A) is m x k. Copies rows [i0, i0 + mc) and columns [p0, p0 + kc) of op(A)
// into panels of kMR rows. Each panel is stored column by column, so that the
// tile reads kMR consecutive values for every step along k. Rows past the end
// of A are filled with zeros, so the tile can always compute a full 6 x 16
// block.
template <class T>
void pack_a(const T* a, int lda, bool trans, int i0, int mc, int p0, int kc,
    T* out) {
    for (int ir = 0; ir < mc; ir += kMR) {
        T* panel = out + static_cast<size_t>(ir) * kc;
        const int rows = std::min(kMR, mc - ir);
        if (trans) {
            // op(A)(i, p) = A[p][i], so one step along k is one stored row of
            // A, and the panel's kMR values for that step are next to each
            // other in that row.
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

// op(B) is k x n. Copies rows [p0, p0 + kc) and columns [j0, j0 + cols) of
// op(B) into one sliver of kNR columns, stored row by row. Columns past the
// end are filled with zeros.
template <class T>
void pack_b(const T* b, int ldb, bool trans, int p0, int kc, int j0, int cols,
    T* out) {
    if (trans) {
        // op(B)(p, j) = B[j][p], so each column of the sliver is a stored
        // row of B.
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

// C (kMR x kNR, row stride ldc) += alpha * (one packed A panel) * (one packed
// B sliver), kc deep. It is written so that a compiler can vectorise the inner
// loop by itself: the accumulator is a local array, so no pointer can alias
// it, and its rows are exactly kNR wide. A double build runs this version, and
// so does any machine without AVX2.
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

// The same tile, with the twelve accumulators as twelve named variables. This
// keeps them in registers for the whole k loop; the compiler might put an
// array in memory instead. Each step along k is two loads of B, six broadcasts
// of A and twelve fused multiply-adds: 192 flops for 8 loads. The loads are
// unaligned because std::vector guarantees 16-byte alignment, not 32. On every
// CPU since Haswell, an unaligned load of aligned data costs nothing extra.
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
// micro_tile is a plain overload, so normal overload resolution picks it for
// float, and every other type uses the template. kernel_name has no arguments
// to overload on, so it is specialised instead.
inline void micro_tile(int kc, const float* a, const float* b, float alpha,
    float* c, int ldc) {
    micro_tile_avx2(kc, a, b, alpha, c, ldc);
}

template <>
inline const char* kernel_name<float>() {
    return "AVX2/FMA 6x16 tile";
}
#endif

// Multiplies one MC-row block of A by one NR-column sliver of B, which gives
// one column of tiles. A tile that extends past the bottom or right edge of C
// is computed in full into a scratch tile, and only its valid part is added to
// C. So the tile code never has to handle edges.
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

// Each thread has its own packing buffers, which grow to fit the largest
// multiply that thread has done. During one call, the packed B is shared with
// the rest of the team. This is safe because the shared buffer belongs to the
// calling thread, and that thread does not start another multiply until this
// one has returned.
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
// lda, ldb and ldc are the row strides of the matrices as they are stored. So
// the stride of a transposed operand is the stride of its untransposed shape.
// A stride wider than the row is allowed; attention uses this to pass one
// head's slice of every row.
//
// may_thread says whether this call may start a team of threads. Attention
// calls this once per (batch, head) from inside a loop that is already divided
// across threads. Those calls have to stay on the thread that made them.
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

    // The team shares one packed slab of B and divides the work on it into
    // (A block, B sliver) pairs. The work is divided into pairs, not only into
    // A blocks, because a weight gradient has only fan_in rows. At the video's
    // final width that is four blocks, and sixteen threads need more pieces
    // than that. Each thread takes a contiguous range of pairs, so it mostly
    // works on one A block and packs it once. If a block is in the ranges of
    // two threads, both of them pack it.
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
                // No thread packs B for the next slab until every thread has
                // finished reading this one.
#pragma omp barrier
            }
        }
    }
}

}  // namespace blocked
}  // namespace gpt
