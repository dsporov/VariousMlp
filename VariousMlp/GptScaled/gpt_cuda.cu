// GptScaled's own copy of makemore/gpt_cuda.cu, taken at commit 512eaaa. The
// one change since, the mixed dropout seeds in drop(), went into both copies,
// so the two still diff cleanly. gpt_scaled compiles only the files in this
// directory, which keeps it independent of makemore/: a change on either side
// stays on that side. Where the comments below mention other files, they mean
// the ones in makemore/.
//
// gpt_cuda.cu - the model on the GPU, every forward and backward pass a kernel.
//
// The same arithmetic as gpt_model.h, layer for layer, but where that file
// writes a loop over rows this one launches a grid of threads and lets each
// take a row, an element or a tile. Three shapes of kernel cover the model:
//
//   - elementwise: the embedding lookup, ReLU, dropout, the residual adds and
//     the optimiser. One thread per element.
//   - a warp per row: layer norm, the causal softmax, cross entropy. A row is
//     at most a few hundred numbers, so the 32 threads of a warp each take a
//     strided share and combine their partial sums with shuffles, never
//     touching shared memory.
//   - the multiply, which is where the time goes. gemm_kernel below is the
//     textbook tiled version: a 256-thread block owns a 64 x 64 tile of C,
//     stages 16-deep slabs of A and B through shared memory, and each thread
//     keeps a 4 x 4 corner of the tile in registers. It is the GPU edition of
//     gpt_gemm.h's blocking - shared memory playing L1, registers playing
//     registers - and it is batched, so that one launch covers every
//     (batch, head) of attention.
//
// Two things are done the slow way on purpose, for the same reason the CPU
// threading was: so that the same inputs give the same bits every run.
// Column sums (bias and layer norm gradients) add up fixed chunks of rows and
// then the chunks in order, rather than racing atomicAdds into one total. And
// the token embedding's gradient, where many positions share a row, is summed
// per vocabulary row over a list of positions sorted on the host, rather than
// scattered with atomics. Dropout draws its mask from a hash of a seed and the
// element index, which needs no generator state shared between threads.
//
// Everything runs on the default stream, so every kernel sees the finished
// output of the one before it with no synchronisation written out. The host
// waits only where it has to: reading the loss back, and sync().
//
// Needs a CUDA 12.x toolkit for a Pascal card like the GTX 1050 Ti, which
// CUDA 13 no longer supports. CMake builds the gpt_cuda target when it finds
// nvcc.

#include "gpt_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "check.h"

// A failed CUDA call is a broken program, not a number that drifted, so it
// goes through CHECK like any other structural failure.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t cuda_err_ = (call);                                  \
        CHECK(cuda_err_ == cudaSuccess,                                        \
            #call << ": " << cudaGetErrorString(cuda_err_));                   \
    } while (false)

// Launches report bad configurations through cudaGetLastError, and faults
// inside a kernel at the next call that waits for it.
#define CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

namespace gpt_cuda {
namespace detail {

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kRowsPerBlock = kThreads / kWarp;

// The multiply's tiling: a kTile x kTile tile of C per block, kTileK deep per
// stage, and a kPerThread x kPerThread share of the tile per thread, which
// makes the block kGemmSide threads on a side.
constexpr int kTile = 64;
constexpr int kTileK = 16;
constexpr int kPerThread = 4;
constexpr int kGemmSide = kTile / kPerThread;
constexpr int kGemmThreads = kGemmSide * kGemmSide;

// Rows per partial sum in a column sum.
constexpr int kColsumRows = 64;

// nn.LayerNorm's epsilon and gpt::AdamW's defaults, as gpt_model.h has them.
constexpr float kLnEps = 1e-5f;
constexpr float kBeta1 = 0.9f;
constexpr float kBeta2 = 0.999f;
constexpr float kAdamEps = 1e-8f;
constexpr float kWeightDecay = 0.01f;

constexpr unsigned long long kDropoutSeed = 1337ull;
constexpr unsigned long long kGolden = 0x9E3779B97F4A7C15ull;

inline unsigned blocks_for(size_t n, int per_block) {
    return static_cast<unsigned>((n + static_cast<size_t>(per_block) - 1)
        / static_cast<size_t>(per_block));
}

// Device memory that frees itself. It grows and never shrinks, and growing
// does not keep the contents: every buffer here is either written in full by
// the next kernel or cleared by the code that owns it.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    ~DeviceBuffer() {
        if (p_) {
            cudaFree(p_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_) {
        o.p_ = nullptr;
        o.n_ = 0;
    }

    void reserve(size_t n) {
        if (n <= n_) {
            return;
        }
        if (p_) {
            CUDA_CHECK(cudaFree(p_));
            p_ = nullptr;
            n_ = 0;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p_), n * sizeof(T)));
        n_ = n;
    }

    T* get() const { return p_; }

private:
    T* p_ = nullptr;
    size_t n_ = 0;
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__device__ inline size_t element_index() {
    return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ inline int warp_row() {
    return static_cast<int>(blockIdx.x) * kRowsPerBlock
        + static_cast<int>(threadIdx.x) / kWarp;
}

__device__ inline int lane_index() {
    return static_cast<int>(threadIdx.x) % kWarp;
}

// A butterfly, so every lane ends up holding the whole warp's total.
__device__ inline float warp_sum(float v) {
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ inline float warp_max(float v) {
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

// Where operand z of a batched multiply starts. z splits into
// (z / inner, z % inner), which for attention is (batch, head), and each of
// the three matrices moves by its own stride along each.
struct Batch {
    int count = 1;
    int inner = 1;
    long long a_outer = 0;
    long long a_inner = 0;
    long long b_outer = 0;
    long long b_inner = 0;
    long long c_outer = 0;
    long long c_inner = 0;
};

// C (m x n) = alpha * op(A) op(B) [+ bias per column] [+ C], row-major, with
// op a transpose or not as the template says. lda, ldb, ldc are the stored
// row strides, as in gpt_gemm.h.
template <bool kTransA, bool kTransB>
__global__ void gemm_kernel(int m, int n, int k, float alpha,
    const float* __restrict__ a, int lda, const float* __restrict__ b, int ldb,
    float* c, int ldc, const float* __restrict__ bias, bool accumulate,
    Batch batch) {
    const long long z = blockIdx.z;
    const long long outer = z / batch.inner;
    const long long inner = z % batch.inner;
    a += outer * batch.a_outer + inner * batch.a_inner;
    b += outer * batch.b_outer + inner * batch.b_inner;
    c += outer * batch.c_outer + inner * batch.c_inner;

    // One kTile x kTileK slab of op(A) and one kTileK x kTile slab of op(B) at
    // a time, both stored k-major so that the inner loop reads a row of each.
    // The extra column keeps the threads that fill a column of a slab from all
    // landing in the same shared memory bank.
    __shared__ float as[kTileK][kTile + 1];
    __shared__ float bs[kTileK][kTile + 1];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tid = ty * kGemmSide + tx;
    const int row0 = static_cast<int>(blockIdx.y) * kTile;
    const int col0 = static_cast<int>(blockIdx.x) * kTile;

    float acc[kPerThread][kPerThread];
#pragma unroll
    for (int r = 0; r < kPerThread; ++r) {
#pragma unroll
        for (int q = 0; q < kPerThread; ++q) {
            acc[r][q] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < k; k0 += kTileK) {
        // Each slab is walked in the order the stored matrix is laid out, so
        // neighbouring threads read neighbouring addresses whichever way the
        // operand is transposed. Past the edge of a matrix the slab is zero.
        for (int e = tid; e < kTile * kTileK; e += kGemmThreads) {
            const int i = kTransA ? e % kTile : e / kTileK;
            const int kk = kTransA ? e / kTile : e % kTileK;
            const int gi = row0 + i;
            const int gk = k0 + kk;
            float v = 0.0f;
            if (gi < m && gk < k) {
                v = kTransA ? a[static_cast<size_t>(gk) * lda + gi]
                            : a[static_cast<size_t>(gi) * lda + gk];
            }
            as[kk][i] = v;
        }
        for (int e = tid; e < kTile * kTileK; e += kGemmThreads) {
            const int j = kTransB ? e / kTileK : e % kTile;
            const int kk = kTransB ? e % kTileK : e / kTile;
            const int gj = col0 + j;
            const int gk = k0 + kk;
            float v = 0.0f;
            if (gj < n && gk < k) {
                v = kTransB ? b[static_cast<size_t>(gj) * ldb + gk]
                            : b[static_cast<size_t>(gk) * ldb + gj];
            }
            bs[kk][j] = v;
        }
        __syncthreads();

        // Thread (tx, ty) owns rows ty, ty+16, ... and columns tx, tx+16, ...
        // of the tile: strided rather than a contiguous 4 x 4, so that a warp
        // reading its row of the slab hits sixteen consecutive words.
#pragma unroll
        for (int kk = 0; kk < kTileK; ++kk) {
            float ar[kPerThread];
            float br[kPerThread];
#pragma unroll
            for (int r = 0; r < kPerThread; ++r) {
                ar[r] = as[kk][ty + r * kGemmSide];
                br[r] = bs[kk][tx + r * kGemmSide];
            }
#pragma unroll
            for (int r = 0; r < kPerThread; ++r) {
#pragma unroll
                for (int q = 0; q < kPerThread; ++q) {
                    acc[r][q] = fmaf(ar[r], br[q], acc[r][q]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < kPerThread; ++r) {
        const int gi = row0 + ty + r * kGemmSide;
        if (gi >= m) {
            continue;
        }
#pragma unroll
        for (int q = 0; q < kPerThread; ++q) {
            const int gj = col0 + tx + q * kGemmSide;
            if (gj >= n) {
                continue;
            }
            float v = alpha * acc[r][q];
            if (bias) {
                v += bias[gj];
            }
            float* dst = c + static_cast<size_t>(gi) * ldc + gj;
            *dst = accumulate ? *dst + v : v;
        }
    }
}

// Python: tok_emb + pos_emb, one thread per (position, channel).
__global__ void embed_forward(const int* __restrict__ tokens,
    const float* __restrict__ wte, const float* __restrict__ wpe,
    float* __restrict__ out, int n, int T, int C) {
    const size_t e = element_index();
    if (e >= static_cast<size_t>(n) * C) {
        return;
    }
    const int i = static_cast<int>(e / C);
    const int c = static_cast<int>(e % C);
    out[e] = wte[static_cast<size_t>(tokens[i]) * C + c]
        + wpe[static_cast<size_t>(i % T) * C + c];
}

// The token table's gradient, one block per vocabulary row. order lists the
// positions holding each token, grouped by token and ascending within a
// group, and start[v] is where token v's group begins - so a thread owns one
// channel of one row, and walks that row's positions in the same order every
// run.
__global__ void embed_backward_tokens(const float* __restrict__ dx,
    const int* __restrict__ order, const int* __restrict__ start, int C,
    float* __restrict__ dwte) {
    const int v = static_cast<int>(blockIdx.x);
    for (int c = static_cast<int>(threadIdx.x); c < C; c += blockDim.x) {
        float s = 0.0f;
        for (int j = start[v]; j < start[v + 1]; ++j) {
            s += dx[static_cast<size_t>(order[j]) * C + c];
        }
        dwte[static_cast<size_t>(v) * C + c] += s;
    }
}

// First half of a column sum: the sum of x (or of x * y) down one column over
// one chunk of kColsumRows rows. Threads run along the columns, so every step
// down the chunk is one coalesced read of a row.
__global__ void colsum_partial(const float* __restrict__ x,
    const float* __restrict__ y, int rows, int cols, float* __restrict__ partial) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (col >= cols) {
        return;
    }
    const int r0 = static_cast<int>(blockIdx.y) * kColsumRows;
    const int r1 = min(rows, r0 + kColsumRows);
    float s = 0.0f;
    for (int r = r0; r < r1; ++r) {
        const size_t e = static_cast<size_t>(r) * cols + col;
        s += y ? x[e] * y[e] : x[e];
    }
    partial[static_cast<size_t>(blockIdx.y) * cols + col] = s;
}

// Second half: the chunks, in order, added into out.
__global__ void colsum_finish(const float* __restrict__ partial, int chunks,
    int cols, float* __restrict__ out) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (col >= cols) {
        return;
    }
    float s = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        s += partial[static_cast<size_t>(c) * cols + col];
    }
    out[col] += s;
}

// Python: nn.LayerNorm(n_embd), a warp per row.
__global__ void layernorm_forward(const float* __restrict__ x,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ xhat, float* __restrict__ invstd, float* __restrict__ out,
    int rows, int C) {
    const int row = warp_row();
    if (row >= rows) {
        return;
    }
    const int lane = lane_index();
    const float* xr = x + static_cast<size_t>(row) * C;

    float sum = 0.0f;
    for (int c = lane; c < C; c += kWarp) {
        sum += xr[c];
    }
    const float mean = warp_sum(sum) / C;

    float var = 0.0f;
    for (int c = lane; c < C; c += kWarp) {
        const float d = xr[c] - mean;
        var += d * d;
    }
    var = warp_sum(var) / C;

    const float inv = 1.0f / sqrtf(var + kLnEps);
    if (lane == 0) {
        invstd[row] = inv;
    }
    float* xh = xhat + static_cast<size_t>(row) * C;
    float* o = out + static_cast<size_t>(row) * C;
    for (int c = lane; c < C; c += kWarp) {
        const float h = (xr[c] - mean) * inv;
        xh[c] = h;
        o[c] = gamma[c] * h + beta[c];
    }
}

// gpt_model.h's formula, a warp per row:
//   dx = invstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
// gamma's and beta's gradients are column sums, done separately.
__global__ void layernorm_backward(const float* __restrict__ dout,
    const float* __restrict__ xhat, const float* __restrict__ invstd,
    const float* __restrict__ gamma, float* __restrict__ dx, int rows, int C) {
    const int row = warp_row();
    if (row >= rows) {
        return;
    }
    const int lane = lane_index();
    const float* di = dout + static_cast<size_t>(row) * C;
    const float* xh = xhat + static_cast<size_t>(row) * C;

    float sum_d = 0.0f;
    float sum_dx = 0.0f;
    for (int c = lane; c < C; c += kWarp) {
        const float dxhat = di[c] * gamma[c];
        sum_d += dxhat;
        sum_dx += dxhat * xh[c];
    }
    const float mean_d = warp_sum(sum_d) / C;
    const float mean_dx = warp_sum(sum_dx) / C;
    const float inv = invstd[row];

    float* out = dx + static_cast<size_t>(row) * C;
    for (int c = lane; c < C; c += kWarp) {
        const float dxhat = di[c] * gamma[c];
        out[c] = inv * (dxhat - mean_d - xh[c] * mean_dx);
    }
}

// In place over rows of already-scaled scores, a warp per row. Row r is query
// position r % T, which may see keys 0..t; the rest of the row becomes zero
// probability, which is what lets the backward pass and the multiply by V
// ignore the future without being told about it.
__global__ void causal_softmax_forward(float* att, int rows, int T) {
    const int row = warp_row();
    if (row >= rows) {
        return;
    }
    const int lane = lane_index();
    const int t = row % T;
    float* a = att + static_cast<size_t>(row) * T;

    float mx = -INFINITY;
    for (int u = lane; u <= t; u += kWarp) {
        mx = fmaxf(mx, a[u]);
    }
    mx = warp_max(mx);

    float sum = 0.0f;
    for (int u = lane; u <= t; u += kWarp) {
        const float e = expf(a[u] - mx);
        a[u] = e;
        sum += e;
    }
    sum = warp_sum(sum);

    for (int u = lane; u <= t; u += kWarp) {
        a[u] /= sum;
    }
    for (int u = t + 1 + lane; u < T; u += kWarp) {
        a[u] = 0.0f;
    }
}

// ds = p * (dp - sum(p * dp)) over the visible part of each row; zero beyond.
__global__ void causal_softmax_backward(const float* __restrict__ p,
    const float* __restrict__ dp, float* __restrict__ ds, int rows, int T) {
    const int row = warp_row();
    if (row >= rows) {
        return;
    }
    const int lane = lane_index();
    const int t = row % T;
    const size_t base = static_cast<size_t>(row) * T;

    float dot = 0.0f;
    for (int u = lane; u <= t; u += kWarp) {
        dot += p[base + u] * dp[base + u];
    }
    dot = warp_sum(dot);

    for (int u = lane; u < T; u += kWarp) {
        ds[base + u] = u <= t ? p[base + u] * (dp[base + u] - dot) : 0.0f;
    }
}

// Python: F.cross_entropy, a warp per prediction. The per-row losses go back
// to the host to be summed in order; the gradient, softmax minus the one-hot
// target over the n rows averaged, is written straight away when asked for.
__global__ void cross_entropy(const float* __restrict__ logits,
    const int* __restrict__ targets, float* __restrict__ row_loss,
    float* __restrict__ dlogits, int rows, int V, float inv_n) {
    const int row = warp_row();
    if (row >= rows) {
        return;
    }
    const int lane = lane_index();
    const float* l = logits + static_cast<size_t>(row) * V;

    float mx = -INFINITY;
    for (int j = lane; j < V; j += kWarp) {
        mx = fmaxf(mx, l[j]);
    }
    mx = warp_max(mx);

    float sum = 0.0f;
    for (int j = lane; j < V; j += kWarp) {
        sum += expf(l[j] - mx);
    }
    sum = warp_sum(sum);

    const int y = targets[row];
    if (lane == 0) {
        row_loss[row] = logf(sum) + mx - l[y];
    }
    if (dlogits) {
        float* dl = dlogits + static_cast<size_t>(row) * V;
        for (int j = lane; j < V; j += kWarp) {
            dl[j] = (expf(l[j] - mx) / sum - (j == y ? 1.0f : 0.0f)) * inv_n;
        }
    }
}

__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, size_t n) {
    const size_t e = element_index();
    if (e < n) {
        out[e] = a[e] + b[e];
    }
}

__global__ void add_into_kernel(float* __restrict__ a, const float* __restrict__ b, size_t n) {
    const size_t e = element_index();
    if (e < n) {
        a[e] += b[e];
    }
}

__global__ void multiply_kernel(const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, size_t n) {
    const size_t e = element_index();
    if (e < n) {
        out[e] = a[e] * b[e];
    }
}

__global__ void multiply_into_kernel(float* __restrict__ a, const float* __restrict__ b,
    size_t n) {
    const size_t e = element_index();
    if (e < n) {
        a[e] *= b[e];
    }
}

__global__ void relu_inplace(float* x, size_t n) {
    const size_t e = element_index();
    if (e < n) {
        x[e] = fmaxf(x[e], 0.0f);
    }
}

// d *= (out > 0), where out is the ReLU's own output, as in gpt_model.h.
__global__ void relu_backward(float* __restrict__ d, const float* __restrict__ out,
    size_t n) {
    const size_t e = element_index();
    if (e < n && !(out[e] > 0.0f)) {
        d[e] = 0.0f;
    }
}

// splitmix64's finaliser: every bit of the input reaches every bit of the
// output. On the host as well, because drop() mixes its seeds with it.
__host__ __device__ inline unsigned long long mix64(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// A well-mixed 64-bit value per (seed, index) with no state to share, of
// which the top 24 bits make a float in [0, 1).
__device__ inline float uniform01(unsigned long long seed, unsigned long long i) {
    const unsigned long long z = mix64(seed + (i + 1ull) * kGolden);
    return static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
}

// Inverted dropout. The mask holds the scale for survivors and zero for the
// rest, so the backward pass is one multiply by it.
__global__ void dropout_forward(const float* __restrict__ x, float* __restrict__ out,
    float* __restrict__ mask, size_t n, float p, float scale,
    unsigned long long seed) {
    const size_t e = element_index();
    if (e >= n) {
        return;
    }
    const float keep = uniform01(seed, e) >= p ? scale : 0.0f;
    mask[e] = keep;
    out[e] = x[e] * keep;
}

// gpt::AdamW::step, one thread per parameter across every tensor at once.
__global__ void adamw(float* __restrict__ p, const float* __restrict__ g,
    float* __restrict__ m, float* __restrict__ v, size_t n, float lr, float bc1,
    float bc2) {
    const size_t e = element_index();
    if (e >= n) {
        return;
    }
    const float ge = g[e];
    const float me = kBeta1 * m[e] + (1.0f - kBeta1) * ge;
    const float ve = kBeta2 * v[e] + (1.0f - kBeta2) * ge * ge;
    m[e] = me;
    v[e] = ve;
    float pe = p[e];
    pe -= lr * kWeightDecay * pe;
    pe -= lr * (me / bc1) / (sqrtf(ve / bc2) + kAdamEps);
    p[e] = pe;
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------

void gemm(bool trans_a, bool trans_b, int m, int n, int k, float alpha,
    const float* a, int lda, const float* b, int ldb, float* c, int ldc,
    bool accumulate, const float* bias = nullptr, const Batch& batch = Batch()) {
    const dim3 block(kGemmSide, kGemmSide);
    const dim3 grid(blocks_for(static_cast<size_t>(n), kTile),
        blocks_for(static_cast<size_t>(m), kTile), static_cast<unsigned>(batch.count));
    if (trans_a && trans_b) {
        gemm_kernel<true, true><<<grid, block>>>(m, n, k, alpha, a, lda, b, ldb,
            c, ldc, bias, accumulate, batch);
    }
    else if (trans_a) {
        gemm_kernel<true, false><<<grid, block>>>(m, n, k, alpha, a, lda, b, ldb,
            c, ldc, bias, accumulate, batch);
    }
    else if (trans_b) {
        gemm_kernel<false, true><<<grid, block>>>(m, n, k, alpha, a, lda, b, ldb,
            c, ldc, bias, accumulate, batch);
    }
    else {
        gemm_kernel<false, false><<<grid, block>>>(m, n, k, alpha, a, lda, b, ldb,
            c, ldc, bias, accumulate, batch);
    }
    CHECK_LAUNCH();
}

// Where one operand of a per-(batch, head) multiply starts, in its outer
// (batch) and inner (head) strides.
struct Stride {
    long long outer;
    long long inner;
};

// The indices of one block's tensors in the flat parameter buffer, in the
// order gpt::Gpt collects them.
struct Slots {
    int ln1_g, ln1_b;
    int wk, wq, wv, wp, bp;
    int ln2_g, ln2_b;
    int w1, b1, w2, b2;
};

// What one block keeps between its forward and backward passes. r is the
// feedforward's hidden layer after the ReLU, which is all its backward needs.
// The p* and *_mask buffers are only allocated when the model has dropout.
struct BlockActs {
    DeviceBuffer<float> ln1_xhat, ln1_inv, h1;
    DeviceBuffer<float> q, k, v;
    DeviceBuffer<float> p, pd, attn_mask;
    DeviceBuffer<float> o, a, ad, resid_mask, mid;
    DeviceBuffer<float> ln2_xhat, ln2_inv, h2;
    DeviceBuffer<float> r, f2, f2d, ffwd_mask, out;
};

}  // namespace detail

using namespace detail;

struct Model::Impl {
    Shape s;
    int hs = 0;

    // Every parameter lives in one flat buffer, and so do the gradients and
    // the two Adam moments, which makes zero_grad one memset and the
    // optimiser one launch.
    std::vector<size_t> sizes;
    std::vector<size_t> offsets;
    size_t total = 0;
    int wte = 0;
    int wpe = 0;
    std::vector<Slots> slots;
    int lnf_g = 0;
    int lnf_b = 0;
    int wlm = 0;
    int blm = 0;
    DeviceBuffer<float> params, grads, adam_m, adam_v;
    int step = 0;

    bool training = true;
    bool dropping = false;          // whether the last forward pass drew masks
    unsigned long long draws = 0;   // masks drawn so far, for their seeds

    int B = 0;
    int T = 0;
    std::vector<BlockActs> acts;
    DeviceBuffer<float> emb, lnf_xhat, lnf_inv, hf, logits, dlogits, row_loss;
    DeviceBuffer<int> tokens, targets, order, starts;

    // Backward scratch, shared by every block because only one block's
    // backward pass runs at a time. ping and pong carry the residual stream's
    // gradient from one block down to the next.
    DeviceBuffer<float> d_ping, d_pong, d_mid, d_tmp, d_ff, d_h;
    DeviceBuffer<float> d_o, d_q, d_k, d_v, d_att, d_s, partial;

    std::vector<float> host_loss;
    std::vector<int> host_order, host_starts;

    explicit Impl(const Shape& shape) : s(shape) {
        CHECK(s.n_head > 0 && s.n_embd % s.n_head == 0,
            s.n_embd << " channels do not divide into " << s.n_head << " heads");
        CHECK(s.block > 0 && s.vocab > 0 && s.n_layer >= 0, "a model with context "
            << s.block << ", vocabulary " << s.vocab << " and " << s.n_layer << " blocks");
        CHECK(s.dropout >= 0.0f && s.dropout < 1.0f, "dropout probability " << s.dropout);
        CUDA_CHECK(cudaSetDevice(0));
        hs = s.n_embd / s.n_head;

        const size_t C = static_cast<size_t>(s.n_embd);
        const size_t V = static_cast<size_t>(s.vocab);
        wte = add_tensor(V * C);
        wpe = add_tensor(static_cast<size_t>(s.block) * C);
        for (int l = 0; l < s.n_layer; ++l) {
            Slots sl;
            sl.ln1_g = add_tensor(C);
            sl.ln1_b = add_tensor(C);
            sl.wk = add_tensor(C * C);
            sl.wq = add_tensor(C * C);
            sl.wv = add_tensor(C * C);
            sl.wp = add_tensor(C * C);
            sl.bp = add_tensor(C);
            sl.ln2_g = add_tensor(C);
            sl.ln2_b = add_tensor(C);
            sl.w1 = add_tensor(C * 4 * C);
            sl.b1 = add_tensor(4 * C);
            sl.w2 = add_tensor(4 * C * C);
            sl.b2 = add_tensor(C);
            slots.push_back(sl);
        }
        lnf_g = add_tensor(C);
        lnf_b = add_tensor(C);
        wlm = add_tensor(C * V);
        blm = add_tensor(V);

        params.reserve(total);
        grads.reserve(total);
        adam_m.reserve(total);
        adam_v.reserve(total);
        CUDA_CHECK(cudaMemset(params.get(), 0, total * sizeof(float)));
        CUDA_CHECK(cudaMemset(grads.get(), 0, total * sizeof(float)));
        CUDA_CHECK(cudaMemset(adam_m.get(), 0, total * sizeof(float)));
        CUDA_CHECK(cudaMemset(adam_v.get(), 0, total * sizeof(float)));
        acts.resize(static_cast<size_t>(s.n_layer));
    }

    int add_tensor(size_t n) {
        offsets.push_back(total);
        sizes.push_back(n);
        total += n;
        return static_cast<int>(sizes.size()) - 1;
    }

    float* P(int slot) { return params.get() + offsets[static_cast<size_t>(slot)]; }
    float* G(int slot) { return grads.get() + offsets[static_cast<size_t>(slot)]; }

    // Grows every activation and scratch buffer to hold a batch of B windows
    // of T positions.
    void ensure(int batch, int time) {
        const size_t n = static_cast<size_t>(batch) * time;
        const size_t C = static_cast<size_t>(s.n_embd);
        const size_t V = static_cast<size_t>(s.vocab);
        const size_t att = static_cast<size_t>(batch) * s.n_head * time * time;
        const bool with_dropout = s.dropout > 0.0f;
        for (BlockActs& a : acts) {
            a.ln1_xhat.reserve(n * C);
            a.ln1_inv.reserve(n);
            a.h1.reserve(n * C);
            a.q.reserve(n * C);
            a.k.reserve(n * C);
            a.v.reserve(n * C);
            a.p.reserve(att);
            a.o.reserve(n * C);
            a.a.reserve(n * C);
            a.mid.reserve(n * C);
            a.ln2_xhat.reserve(n * C);
            a.ln2_inv.reserve(n);
            a.h2.reserve(n * C);
            a.r.reserve(n * 4 * C);
            a.f2.reserve(n * C);
            a.out.reserve(n * C);
            if (with_dropout) {
                a.pd.reserve(att);
                a.attn_mask.reserve(att);
                a.ad.reserve(n * C);
                a.resid_mask.reserve(n * C);
                a.f2d.reserve(n * C);
                a.ffwd_mask.reserve(n * C);
            }
        }
        emb.reserve(n * C);
        lnf_xhat.reserve(n * C);
        lnf_inv.reserve(n);
        hf.reserve(n * C);
        logits.reserve(n * V);
        dlogits.reserve(n * V);
        row_loss.reserve(n);
        tokens.reserve(n);
        targets.reserve(n);
        order.reserve(n);
        starts.reserve(V + 1);
        d_ping.reserve(n * C);
        d_pong.reserve(n * C);
        d_mid.reserve(n * C);
        d_tmp.reserve(n * C);
        d_ff.reserve(n * 4 * C);
        d_h.reserve(n * C);
        d_o.reserve(n * C);
        d_q.reserve(n * C);
        d_k.reserve(n * C);
        d_v.reserve(n * C);
        d_att.reserve(att);
        d_s.reserve(att);
    }

    // Head h of batch b, as a T x hs matrix inside a (B, T, C) tensor: rows
    // n_embd apart, starting b*T*C + h*hs in.
    Stride head_slice() const {
        return { static_cast<long long>(T) * s.n_embd, static_cast<long long>(hs) };
    }

    // The T x T attention matrix of (b, h) inside a (B, heads, T, T) tensor.
    Stride head_square() const {
        return { static_cast<long long>(s.n_head) * T * T, static_cast<long long>(T) * T };
    }

    Batch per_head(Stride a, Stride b, Stride c) const {
        Batch batch;
        batch.count = B * s.n_head;
        batch.inner = s.n_head;
        batch.a_outer = a.outer;
        batch.a_inner = a.inner;
        batch.b_outer = b.outer;
        batch.b_inner = b.inner;
        batch.c_outer = c.outer;
        batch.c_inner = c.inner;
        return batch;
    }

    void layernorm(const float* x, int g, int b, float* xhat, float* inv, float* out,
        int rows) {
        layernorm_forward<<<blocks_for(static_cast<size_t>(rows), kRowsPerBlock),
            kThreads>>>(x, P(g), P(b), xhat, inv, out, rows, s.n_embd);
        CHECK_LAUNCH();
    }

    // Adds the column sums of x, or of x * y, into out.
    void colsum(const float* x, const float* y, int rows, int cols, float* out) {
        const int chunks = (rows + kColsumRows - 1) / kColsumRows;
        partial.reserve(static_cast<size_t>(chunks) * cols);
        const dim3 grid(blocks_for(static_cast<size_t>(cols), kThreads),
            static_cast<unsigned>(chunks));
        colsum_partial<<<grid, kThreads>>>(x, y, rows, cols, partial.get());
        CHECK_LAUNCH();
        colsum_finish<<<blocks_for(static_cast<size_t>(cols), kThreads), kThreads>>>(
            partial.get(), chunks, cols, out);
        CHECK_LAUNCH();
    }

    void layernorm_back(const float* dout, const float* xhat, const float* inv,
        int g, int b, float* dx, int rows) {
        layernorm_backward<<<blocks_for(static_cast<size_t>(rows), kRowsPerBlock),
            kThreads>>>(dout, xhat, inv, P(g), dx, rows, s.n_embd);
        CHECK_LAUNCH();
        colsum(dout, xhat, rows, s.n_embd, G(g));
        colsum(dout, nullptr, rows, s.n_embd, G(b));
    }

    // out = x W (+ b) over n rows, in -> out_w wide: dW += x^T dout, db the
    // column sums of dout, and dx = dout W^T, or += when accumulate_dx. b < 0
    // for a projection without a bias.
    void linear_back(const float* x, const float* dout, int n, int in, int out_w,
        int w, int b, float* dx, bool accumulate_dx) {
        gemm(true, false, in, out_w, n, 1.0f, x, in, dout, out_w, G(w), out_w, true);
        if (b >= 0) {
            colsum(dout, nullptr, n, out_w, G(b));
        }
        gemm(false, true, n, in, out_w, 1.0f, dout, out_w, P(w), out_w, dx, in,
            accumulate_dx);
    }

    // Draws a mask into mask and returns the dropped copy of x, or returns x
    // itself when this pass drops nothing.
    const float* drop(const float* x, DeviceBuffer<float>& out,
        DeviceBuffer<float>& mask, size_t n) {
        if (!dropping) {
            return x;
        }
        ++draws;
        // uniform01 hashes seed + (i + 1) * kGolden, so seeds that are
        // themselves kGolden apart make every mask the one before it shifted
        // by one element: a block's feed-forward mask would be its attention
        // output's one channel along, and each step's masks the last step's a
        // few dozen along. Mixed, the seeds are unrelated, and two masks
        // overlap only if theirs happen to differ by fewer than n multiples
        // of kGolden - for the largest mask, about one chance in 10^12.
        const unsigned long long seed = mix64(kDropoutSeed + draws * kGolden);
        dropout_forward<<<blocks_for(n, kThreads), kThreads>>>(x, out.get(), mask.get(),
            n, s.dropout, 1.0f / (1.0f - s.dropout), seed);
        CHECK_LAUNCH();
        return out.get();
    }

    // The backward of drop(): d times the mask, into scratch, or d itself.
    const float* undrop(const float* d, const DeviceBuffer<float>& mask,
        DeviceBuffer<float>& scratch, size_t n) {
        if (!dropping) {
            return d;
        }
        multiply_kernel<<<blocks_for(n, kThreads), kThreads>>>(d, mask.get(), scratch.get(), n);
        CHECK_LAUNCH();
        return scratch.get();
    }

    // Python:
    //   x = x + self.sa(self.ln1(x))
    //   x = x + self.ffwd(self.ln2(x))
    void block_forward(int l, const float* x) {
        BlockActs& a = acts[static_cast<size_t>(l)];
        const Slots& sl = slots[static_cast<size_t>(l)];
        const int n = B * T;
        const int C = s.n_embd;
        const int C4 = 4 * C;
        const size_t nc = static_cast<size_t>(n) * C;
        const int att_rows = B * s.n_head * T;
        const size_t att = static_cast<size_t>(att_rows) * T;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hs));

        layernorm(x, sl.ln1_g, sl.ln1_b, a.ln1_xhat.get(), a.ln1_inv.get(), a.h1.get(), n);
        gemm(false, false, n, C, C, 1.0f, a.h1.get(), C, P(sl.wq), C, a.q.get(), C, false);
        gemm(false, false, n, C, C, 1.0f, a.h1.get(), C, P(sl.wk), C, a.k.get(), C, false);
        gemm(false, false, n, C, C, 1.0f, a.h1.get(), C, P(sl.wv), C, a.v.get(), C, false);

        // wei = q @ k^T * hs**-0.5 for every (batch, head) in one launch, then
        // the mask and the softmax in place.
        gemm(false, true, T, T, hs, scale, a.q.get(), C, a.k.get(), C, a.p.get(), T,
            false, nullptr, per_head(head_slice(), head_slice(), head_square()));
        causal_softmax_forward<<<blocks_for(static_cast<size_t>(att_rows), kRowsPerBlock),
            kThreads>>>(a.p.get(), att_rows, T);
        CHECK_LAUNCH();
        const float* pd = drop(a.p.get(), a.pd, a.attn_mask, att);

        // out = wei @ v, each head writing its own hs columns of o.
        gemm(false, false, T, hs, T, 1.0f, pd, T, a.v.get(), C, a.o.get(), C,
            false, nullptr, per_head(head_square(), head_slice(), head_slice()));
        gemm(false, false, n, C, C, 1.0f, a.o.get(), C, P(sl.wp), C, a.a.get(), C,
            false, P(sl.bp));
        const float* ad = drop(a.a.get(), a.ad, a.resid_mask, nc);
        add_kernel<<<blocks_for(nc, kThreads), kThreads>>>(x, ad, a.mid.get(), nc);
        CHECK_LAUNCH();

        layernorm(a.mid.get(), sl.ln2_g, sl.ln2_b, a.ln2_xhat.get(), a.ln2_inv.get(),
            a.h2.get(), n);
        gemm(false, false, n, C4, C, 1.0f, a.h2.get(), C, P(sl.w1), C4, a.r.get(), C4,
            false, P(sl.b1));
        relu_inplace<<<blocks_for(static_cast<size_t>(n) * C4, kThreads), kThreads>>>(
            a.r.get(), static_cast<size_t>(n) * C4);
        CHECK_LAUNCH();
        gemm(false, false, n, C, C4, 1.0f, a.r.get(), C4, P(sl.w2), C, a.f2.get(), C,
            false, P(sl.b2));
        const float* f2d = drop(a.f2.get(), a.f2d, a.ffwd_mask, nc);
        add_kernel<<<blocks_for(nc, kThreads), kThreads>>>(a.mid.get(), f2d, a.out.get(), nc);
        CHECK_LAUNCH();
    }

    // The residual adds fan out in the forward pass, so the gradient arriving
    // at a block's output reaches mid both through the feedforward branch and
    // straight along the skip, and mid's reaches the input the same way.
    void block_backward(int l, const float* dout, float* dx) {
        BlockActs& a = acts[static_cast<size_t>(l)];
        const Slots& sl = slots[static_cast<size_t>(l)];
        const int n = B * T;
        const int C = s.n_embd;
        const int C4 = 4 * C;
        const size_t nc = static_cast<size_t>(n) * C;
        const size_t nc4 = static_cast<size_t>(n) * C4;
        const int att_rows = B * s.n_head * T;
        const size_t att = static_cast<size_t>(att_rows) * T;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hs));

        // out = mid + drop(relu(h2 W1 + b1) W2 + b2)
        const float* df2 = undrop(dout, a.ffwd_mask, d_tmp, nc);
        linear_back(a.r.get(), df2, n, C4, C, sl.w2, sl.b2, d_ff.get(), false);
        relu_backward<<<blocks_for(nc4, kThreads), kThreads>>>(d_ff.get(), a.r.get(), nc4);
        CHECK_LAUNCH();
        linear_back(a.h2.get(), d_ff.get(), n, C, C4, sl.w1, sl.b1, d_h.get(), false);
        layernorm_back(d_h.get(), a.ln2_xhat.get(), a.ln2_inv.get(), sl.ln2_g, sl.ln2_b,
            d_mid.get(), n);
        add_into_kernel<<<blocks_for(nc, kThreads), kThreads>>>(d_mid.get(), dout, nc);
        CHECK_LAUNCH();

        // mid = x + drop(o Wp + bp)
        const float* da = undrop(d_mid.get(), a.resid_mask, d_tmp, nc);
        linear_back(a.o.get(), da, n, C, C, sl.wp, sl.bp, d_o.get(), false);

        // o = wei @ v: dwei = dO v^T, and dv = wei^T dO. dwei comes out above
        // the diagonal too, where the softmax backward never looks.
        const float* pd = dropping ? a.pd.get() : a.p.get();
        gemm(false, true, T, T, hs, 1.0f, d_o.get(), C, a.v.get(), C, d_att.get(), T,
            false, nullptr, per_head(head_slice(), head_slice(), head_square()));
        gemm(true, false, T, hs, T, 1.0f, pd, T, d_o.get(), C, d_v.get(), C,
            false, nullptr, per_head(head_square(), head_slice(), head_slice()));
        if (dropping) {
            multiply_into_kernel<<<blocks_for(att, kThreads), kThreads>>>(d_att.get(),
                a.attn_mask.get(), att);
            CHECK_LAUNCH();
        }
        causal_softmax_backward<<<blocks_for(static_cast<size_t>(att_rows), kRowsPerBlock),
            kThreads>>>(a.p.get(), d_att.get(), d_s.get(), att_rows, T);
        CHECK_LAUNCH();

        // scores = scale * q k^T: dq = scale * ds k, dk = scale * ds^T q.
        gemm(false, false, T, hs, T, scale, d_s.get(), T, a.k.get(), C, d_q.get(), C,
            false, nullptr, per_head(head_square(), head_slice(), head_slice()));
        gemm(true, false, T, hs, T, scale, d_s.get(), T, a.q.get(), C, d_k.get(), C,
            false, nullptr, per_head(head_square(), head_slice(), head_slice()));

        // h1 fed all three projections, so all three gradients come back to it.
        linear_back(a.h1.get(), d_q.get(), n, C, C, sl.wq, -1, d_h.get(), false);
        linear_back(a.h1.get(), d_k.get(), n, C, C, sl.wk, -1, d_h.get(), true);
        linear_back(a.h1.get(), d_v.get(), n, C, C, sl.wv, -1, d_h.get(), true);
        layernorm_back(d_h.get(), a.ln1_xhat.get(), a.ln1_inv.get(), sl.ln1_g, sl.ln1_b,
            dx, n);
        add_into_kernel<<<blocks_for(nc, kThreads), kThreads>>>(dx, d_mid.get(), nc);
        CHECK_LAUNCH();
    }

    float forward(const int* x, const int* y, int batch, int time, bool want_grad) {
        CHECK(batch > 0 && time > 0 && time <= s.block, "a batch of " << batch
            << " windows of " << time << " positions, with a position table of "
            << s.block << " rows");
        B = batch;
        T = time;
        ensure(B, T);
        dropping = training && s.dropout > 0.0f;
        const int n = B * T;
        const int C = s.n_embd;
        const int V = s.vocab;

        CUDA_CHECK(cudaMemcpy(tokens.get(), x, static_cast<size_t>(n) * sizeof(int),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(targets.get(), y, static_cast<size_t>(n) * sizeof(int),
            cudaMemcpyHostToDevice));
        embed_forward<<<blocks_for(static_cast<size_t>(n) * C, kThreads), kThreads>>>(
            tokens.get(), P(wte), P(wpe), emb.get(), n, T, C);
        CHECK_LAUNCH();

        const float* cur = emb.get();
        for (int l = 0; l < s.n_layer; ++l) {
            block_forward(l, cur);
            cur = acts[static_cast<size_t>(l)].out.get();
        }
        layernorm(cur, lnf_g, lnf_b, lnf_xhat.get(), lnf_inv.get(), hf.get(), n);
        gemm(false, false, n, V, C, 1.0f, hf.get(), C, P(wlm), V, logits.get(), V,
            false, P(blm));

        const float inv_n = 1.0f / static_cast<float>(n);
        cross_entropy<<<blocks_for(static_cast<size_t>(n), kRowsPerBlock), kThreads>>>(
            logits.get(), targets.get(), row_loss.get(),
            want_grad ? dlogits.get() : nullptr, n, V, inv_n);
        CHECK_LAUNCH();

        // Summed in order on the host, the way gpt_model.h's cross_entropy
        // adds up its per-row losses, so the total does not depend on how the
        // rows were scheduled. The copy waits for every kernel above.
        host_loss.resize(static_cast<size_t>(n));
        CUDA_CHECK(cudaMemcpy(host_loss.data(), row_loss.get(),
            static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
        float total_loss = 0.0f;
        for (float lv : host_loss) {
            total_loss += lv;
        }
        return total_loss * inv_n;
    }

    // The positions holding each token, grouped by token and ascending within
    // a group: a counting sort, done on the host because the tokens came from
    // there and it is one pass over them.
    void upload_token_order(const int* x, int n) {
        const int V = s.vocab;
        host_starts.assign(static_cast<size_t>(V) + 1, 0);
        for (int i = 0; i < n; ++i) {
            ++host_starts[static_cast<size_t>(x[i]) + 1];
        }
        for (int v = 0; v < V; ++v) {
            host_starts[static_cast<size_t>(v) + 1] += host_starts[static_cast<size_t>(v)];
        }
        std::vector<int> next(host_starts.begin(), host_starts.end() - 1);
        host_order.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            host_order[static_cast<size_t>(next[static_cast<size_t>(x[i])]++)] = i;
        }
        CUDA_CHECK(cudaMemcpy(order.get(), host_order.data(),
            static_cast<size_t>(n) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(starts.get(), host_starts.data(),
            host_starts.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    void backward(const int* x) {
        const int n = B * T;
        const int C = s.n_embd;
        const int V = s.vocab;

        linear_back(hf.get(), dlogits.get(), n, C, V, wlm, blm, d_tmp.get(), false);
        layernorm_back(d_tmp.get(), lnf_xhat.get(), lnf_inv.get(), lnf_g, lnf_b,
            d_ping.get(), n);

        float* d = d_ping.get();
        float* spare = d_pong.get();
        for (int l = s.n_layer - 1; l >= 0; --l) {
            block_backward(l, d, spare);
            std::swap(d, spare);
        }

        // The token table by sorted position lists; the position table is a
        // column sum over the B windows, each of which uses rows 0..T-1.
        upload_token_order(x, n);
        embed_backward_tokens<<<static_cast<unsigned>(V), kThreads>>>(d, order.get(),
            starts.get(), C, G(wte));
        CHECK_LAUNCH();
        colsum(d, nullptr, B, T * C, G(wpe));
    }
};

Model::Model(const Shape& shape) : impl_(new Impl(shape)) {
}

Model::~Model() = default;

const std::vector<size_t>& Model::tensor_sizes() const {
    return impl_->sizes;
}

void Model::load(const std::vector<std::vector<float>>& weights) {
    Impl& m = *impl_;
    CHECK(weights.size() == m.sizes.size(), weights.size()
        << " tensors for a model with " << m.sizes.size());
    std::vector<float> flat(m.total);
    for (size_t i = 0; i < weights.size(); ++i) {
        CHECK(weights[i].size() == m.sizes[i], "tensor " << i << " has "
            << weights[i].size() << " entries, this model's " << m.sizes[i]);
        std::copy(weights[i].begin(), weights[i].end(),
            flat.begin() + static_cast<std::ptrdiff_t>(m.offsets[i]));
    }
    CUDA_CHECK(cudaMemcpy(m.params.get(), flat.data(), m.total * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(m.adam_m.get(), 0, m.total * sizeof(float)));
    CUDA_CHECK(cudaMemset(m.adam_v.get(), 0, m.total * sizeof(float)));
    m.step = 0;
}

void Model::read_grads(std::vector<std::vector<float>>& out) const {
    const Impl& m = *impl_;
    std::vector<float> flat(m.total);
    CUDA_CHECK(cudaMemcpy(flat.data(), m.grads.get(), m.total * sizeof(float),
        cudaMemcpyDeviceToHost));
    out.resize(m.sizes.size());
    for (size_t i = 0; i < m.sizes.size(); ++i) {
        const auto first = flat.begin() + static_cast<std::ptrdiff_t>(m.offsets[i]);
        out[i].assign(first, first + static_cast<std::ptrdiff_t>(m.sizes[i]));
    }
}

void Model::zero_grad() {
    CUDA_CHECK(cudaMemset(impl_->grads.get(), 0, impl_->total * sizeof(float)));
}

float Model::forward_backward(const int* x, const int* y, int B, int T) {
    const float loss = impl_->forward(x, y, B, T, true);
    impl_->backward(x);
    return loss;
}

float Model::forward_loss(const int* x, const int* y, int B, int T) {
    return impl_->forward(x, y, B, T, false);
}

void Model::set_training(bool training) {
    impl_->training = training;
}

void Model::adamw_step(float lr) {
    Impl& m = *impl_;
    ++m.step;
    // Both moments start at zero, so early steps are biased toward it, and
    // dividing by 1 - beta^t undoes that exactly.
    const float bc1 = 1.0f - std::pow(kBeta1, static_cast<float>(m.step));
    const float bc2 = 1.0f - std::pow(kBeta2, static_cast<float>(m.step));
    adamw<<<blocks_for(m.total, kThreads), kThreads>>>(m.params.get(), m.grads.get(),
        m.adam_m.get(), m.adam_v.get(), m.total, lr, bc1, bc2);
    CHECK_LAUNCH();
}

void Model::sync() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::string Model::describe() const {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    char buf[256];
    std::snprintf(buf, sizeof(buf), "CUDA on %s, compute %d.%d, %d SMs, %.1f GB",
        prop.name, prop.major, prop.minor, prop.multiProcessorCount,
        static_cast<double>(prop.totalGlobalMem) / 1e9);
    return std::string(buf);
}

}  // namespace gpt_cuda
