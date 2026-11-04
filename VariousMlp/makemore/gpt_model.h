// gpt_model.h - the Transformer itself, shared by every file that uses it.
//
// gpt.cpp is the walkthrough: it builds the model step by step as the video
// does, trains it, and checks the numbers. gpt_chat.cpp is a tool: it loads a
// checkpoint saved by the walkthrough and lets you type text for the model to
// continue. gpt_bench.h uses this model as the reference for checking the
// other implementations (blocked, ATen, CUDA). All of these files need the
// same model, and none of them is a port of a video. So the rule that each
// video's file can be read from start to finish on its own applies to
// gpt.cpp, not to this file.
//
// Everything a reader would want to read is still below, in order: the data,
// the layers, attention, the blocks, the optimiser and the checkpoint format,
// together with their explanations. gpt.cpp keeps only what the walkthrough
// itself needs: the bigram baseline, the two demonstrations, most of the cost
// model, and the numbers that the video reports.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "check.h"
#include "gpt_gemm.h"

// MSVC and gcc only vectorise a loop that stores through one pointer and loads
// through another if they can prove that the two do not overlap. const does
// not tell them that. Every use of RESTRICT below is on pointers into
// different vectors, and the comment at each use says which ones.
#if defined(_MSC_VER) || defined(__GNUC__)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

// The loops below are threaded with OpenMP. Nothing here is inside an #ifdef.
// A compiler without OpenMP ignores the pragmas, and every loop then runs
// exactly as it did before it was threaded. For the same reason, each loop
// kept its serial structure when it was parallelised, instead of being
// rewritten around the threads.
//
// The project passes /openmp:llvm, whose runtime has a better scheduler and
// cheaper barriers. But nothing below needs it: there is no collapse, no
// reduction and no array section, and every loop variable is a signed int. So
// everything here is within OpenMP 2.0, and plain /openmp builds it too. That
// matters because /openmp:llvm links libomp140.x86_64.dll, while /openmp links
// vcomp140.dll, which every machine with the VC redistributable already has.
#ifdef _OPENMP
#include <omp.h>
#endif

// A named namespace, not an anonymous one, because different translation units
// have to agree on these types. The free functions below are inline for the
// same reason.
namespace gpt {

// The scalar type, defined in one place. It was changed to float on
// 2026-09-18, after threading had given all the speed it could. An AVX2
// register holds 8 floats instead of 4 doubles, and each weight moves half as
// many bytes through the cache. So float should have helped with both of the
// remaining limits: vector width and memory traffic. Measured on the i9, it
// helped with neither: section 6 took 18.3s with float and 17.9s with double.
// This shows that the loops are not vectorised, or that at 32 channels they
// are too short for vectorisation to matter. It does not show that float was
// a mistake.
//
// The cost is in gradient_check(). That check depends on central differences
// being accurate to about 1e-9, which is true for double but not for float.
// So its kH and kTolerance are larger in a float build. With float, the check
// can no longer show that the gradients are exact; it can only catch clear
// errors, such as a wrong sign or a missing term. To switch back, change this
// one line. Checkpoints have sizeof(real) in their file name, so double and
// float builds keep separate files, and neither reads the other's files.
using real = float;

// Inside a parallel region, these describe the team. Outside a parallel
// region, and in a build without OpenMP, they describe the single running
// thread. So the serial case needs no separate code.
inline int thread_count() {
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

inline int thread_index() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// The number of threads that a large enough loop would get. The cost model in
// gpt.cpp prints it next to its throughput number.
inline int available_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Which code the multiplies below run. kScalar is the loops as written here;
// they produced every number that the walkthrough prints. kBlocked sends the
// Linear layers and the four products inside attention to the cache-blocked,
// vectorised multiply in gpt_gemm.h. The two agree to float rounding, not bit
// for bit. So the walkthrough stays on kScalar, because its reproduction
// checks are to four decimals. gpt_blocked.cpp is where kBlocked runs.
//
// This is a process-wide switch, not a constructor argument, because it is a
// property of the arithmetic, not of the model. gpt_bench.h changes it around
// single calls, to run one set of weights both ways.
enum class Kernels { kScalar, kBlocked };

inline Kernels& active_kernels() {
    static Kernels k = Kernels::kScalar;
    return k;
}

inline bool blocked_kernels() {
    return active_kernels() == Kernels::kBlocked;
}

// Starting a team of threads for a loop costs a few microseconds. So a loop
// only gets faster with threads if it has clearly more work than that. Three
// kinds of loop appear below, and they need different thresholds:
//
//   - a loop over the B*T rows of a tensor, where one iteration is the
//     arithmetic for a whole matrix row. A few dozen rows are already worth
//     dividing between threads.
//   - a flat elementwise loop, where one iteration is a single multiply. These
//     loops are limited by memory speed, and need thousands of iterations
//     before threads are worth their cost.
//   - attention, counted in (batch, head, query, key) entries. Its threads are
//     divided over the heads, but its work grows with the T x T square.
//
// The numbers are cautious on purpose, and at the video's small size several
// loops are below them and stay serial. That is correct, not a missed chance:
// 8192 elements divided between sixteen threads is 512 floats per thread,
// which is less work than starting the threads. These numbers are also the
// first thing to tune if a profile shows otherwise.
constexpr int kMinParallelRows = 32;
constexpr int kMinParallelElems = 16384;
constexpr long long kMinParallelWork = 16384;

// ---------------------------------------------------------------------------
// The data
// ---------------------------------------------------------------------------

// Python:
//   with open('input.txt', 'r', encoding='utf-8') as f:
//       text = f.read()
//   chars = sorted(list(set(text)))
//   vocab_size = len(chars)
//   stoi = {ch: i for i, ch in enumerate(chars)}
//   itos = {i: ch for i, ch in enumerate(chars)}
//   encode = lambda s: [stoi[c] for c in s]
//   decode = lambda l: ''.join([itos[i] for i in l])
//
// Unlike in the makemore files, there is no fixed alphabet that allows simple
// arithmetic, so the two maps are really built here. A table of 256 entries is
// used instead of a hash map, because the corpus is ASCII.
// Any plain text file works: the vocabulary is the set of distinct bytes in
// it. If you replace input.txt with War and Peace, the model trains on War and
// Peace. There are two things to know. First, every sanity::Expect below is
// tied to tiny Shakespeare and will report a mismatch for any other text,
// which is the purpose of Expect. Second, the tokens are bytes, not
// characters. So with a UTF-8 corpus in a non-Latin script, the model sees
// half a letter at a time.
constexpr const char* kCorpus = "input.txt";

// The vocabulary without the text it came from. Data below holds one, and
// every checkpoint holds one too. So a program that only samples from a saved
// model reads the vocabulary from the checkpoint and does not need the corpus.
// gpt_chat.cpp works this way.
struct Vocab {
    std::vector<char> itos;
    int stoi[256];
    int vocab = 0;

    // Builds both maps from a list of characters. The list must be in byte
    // order without repeats, which is what the scan of a corpus produces. The
    // checkpoint reader tests this itself, so a damaged file is rejected
    // before it gets here.
    void assign(const std::vector<char>& chars) {
        for (size_t i = 1; i < chars.size(); ++i) {
            CHECK(static_cast<unsigned char>(chars[i - 1])
                < static_cast<unsigned char>(chars[i]),
                "the vocabulary is not in byte order at entry " << i);
        }
        for (int& s : stoi) {
            s = -1;
        }
        itos = chars;
        vocab = static_cast<int>(chars.size());
        for (int i = 0; i < vocab; ++i) {
            stoi[static_cast<unsigned char>(chars[static_cast<size_t>(i)])] = i;
        }
    }

    char decode(int i) const {
        CHECK(i >= 0 && i < vocab, "token " << i << " is outside the vocabulary");
        return itos[static_cast<size_t>(i)];
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string s;
        s.reserve(ids.size());
        for (int i : ids) {
            s.push_back(decode(i));
        }
        return s;
    }
};

struct Data : Vocab {
    std::vector<int> train;
    std::vector<int> val;

    // The directory that input.txt was loaded from. Checkpoints are saved
    // there too. A run from Visual Studio and a run of the exe on its own have
    // different working directories, and this way each finds the other's
    // checkpoints.
    std::string dir;
};

// The data is searched for relative to a few likely working directories.
// Visual Studio runs with cwd = $(ProjectDir), a run of the exe on its own has
// cwd = x64\Debug, and running from the repo root is the other common case. A
// CMake build runs from somewhere in its build tree, which none of these paths
// reach. So CMake defines this directory explicitly, and it is tried first.
// gpt_chat.cpp looks for checkpoints in the same directories.
inline constexpr const char* kDataDirs[] = {
#ifdef VARIOUSMLP_DATA_DIR
    VARIOUSMLP_DATA_DIR "/",
#endif
    "makemore/",
    "",
    "VariousMlp/makemore/",
    "../../VariousMlp/makemore/",
};

// The vocabulary can contain any byte, so a byte outside printable ASCII is
// shown as an escape sequence instead of being sent to the console unchanged.
inline std::string printable(char c) {
    if (c == '\n') {
        return "\\n";
    }
    if (c == '\t') {
        return "\\t";
    }
    const unsigned char b = static_cast<unsigned char>(c);
    if (b >= 32 && b < 127) {
        return std::string(1, c);
    }
    char buf[8];
    std::snprintf(buf, sizeof(buf), "\\x%02x", b);
    return std::string(buf);
}

inline bool load_corpus(Data& d) {
    std::string text;
    for (const char* dir : kDataDirs) {
        const std::string path = std::string(dir) + kCorpus;
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            continue;
        }
        text.assign(std::istreambuf_iterator<char>(f),
            std::istreambuf_iterator<char>());
        if (!text.empty()) {
            printf("  loaded %zu characters from %s\n", text.size(), path.c_str());
            d.dir = dir;
            break;
        }
    }
    if (text.empty()) {
        return false;
    }

    // If a checkout changed the line endings, '\r' would become a 66th token
    // and change the vocabulary and every loss below. So '\r' is removed.
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());

    std::vector<bool> present(256, false);
    for (char c : text) {
        present[static_cast<unsigned char>(c)] = true;
    }
    std::vector<char> chars;
    for (int c = 0; c < 256; ++c) {
        if (present[static_cast<size_t>(c)]) {
            chars.push_back(static_cast<char>(c));
        }
    }
    d.assign(chars);

    // Python: data = torch.tensor(encode(text), dtype=torch.long)
    //         n = int(0.9 * len(data))
    //         train_data = data[:n]
    //         val_data = data[n:]
    //
    // The split is by position, not random. The validation set must be text
    // that the model has not trained on, and neighbouring characters of
    // Shakespeare depend strongly on each other.
    std::vector<int> ids;
    ids.reserve(text.size());
    for (char c : text) {
        const int id = d.stoi[static_cast<unsigned char>(c)];
        CHECK(id >= 0, "character " << static_cast<int>(static_cast<unsigned char>(c))
            << " was not in the vocabulary scan");
        ids.push_back(id);
    }

    const size_t n = static_cast<size_t>(0.9 * ids.size());
    d.train.assign(ids.begin(), ids.begin() + n);
    d.val.assign(ids.begin() + n, ids.end());
    return true;
}

// Python:
//   ix = torch.randint(len(data) - block_size, (batch_size,))
//   x = torch.stack([data[i:i+block_size] for i in ix])
//   y = torch.stack([data[i+1:i+block_size+1] for i in ix])
//
// One batch is B independent windows into the corpus. Each window of T
// characters is T separate training examples at once, because the causal mask
// lets position t see only positions 0..t. This is why a Transformer learns
// much more from one batch than the MLP did.
inline void sample_batch(const std::vector<int>& data, int B, int T,
    std::vector<int>& x, std::vector<int>& y, std::mt19937& g) {
    CHECK(static_cast<int>(data.size()) > T,
        "a split of " << data.size() << " tokens cannot hold a window of " << T + 1);

    x.resize(static_cast<size_t>(B) * T);
    y.resize(x.size());
    std::uniform_int_distribution<size_t> pick(0, data.size() - T - 1);
    for (int b = 0; b < B; ++b) {
        const size_t i = pick(g);
        for (int t = 0; t < T; ++t) {
            x[static_cast<size_t>(b) * T + t] = data[i + t];
            y[static_cast<size_t>(b) * T + t] = data[i + t + 1];
        }
    }
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

// Every activation in a Transformer has the shape (B, T, C): batch, position,
// channel. So the shape is stored with the data instead of being passed
// separately. Only attention needs to know where B ends and T begins.
// Everything else treats the tensor as B*T independent rows.
struct Matrix {
    int batch = 0;
    int time = 0;
    int cols = 0;
    std::vector<real> v;

    void resize(int b, int t, int c) {
        // The flat loops over v below index it with int, because OpenMP needs
        // a signed loop variable and every other count in this file is already
        // an int. That is enough for every model this port trains: the video's
        // final model has at most 25M entries in one tensor. But it is a real
        // limit, and it is better to fail here than to overflow silently.
        CHECK(static_cast<long long>(b) * t * c <= 2147483647LL,
            "a " << b << "x" << t << "x" << c
            << " tensor has more entries than an int can index");
        batch = b;
        time = t;
        cols = c;
        v.assign(static_cast<size_t>(b) * t * c, real(0));
    }

    int rows() const { return batch * time; }

    real* row(int b, int t) {
        return &v[(static_cast<size_t>(b) * time + t) * cols];
    }
    const real* row(int b, int t) const {
        return &v[(static_cast<size_t>(b) * time + t) * cols];
    }

    real* row(int i) { return &v[static_cast<size_t>(i) * cols]; }
    const real* row(int i) const { return &v[static_cast<size_t>(i) * cols]; }
};

struct Param {
    std::string name;
    std::vector<real>* data;
    std::vector<real>* grad;
};

// Softmax over n logits. Subtracting the maximum first costs almost nothing
// and prevents overflow in exp(). It does not change the result.
inline void softmax_row(const real* logits, real* probs, int n) {
    const real maxl = *std::max_element(logits, logits + n);
    real denom = 0;
    for (int j = 0; j < n; ++j) {
        probs[j] = std::exp(logits[j] - maxl);
        denom += probs[j];
    }
    for (int j = 0; j < n; ++j) {
        probs[j] /= denom;
    }
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

// out() is virtual because the composite layers below (attention, the
// feedforward, a whole block) end with a sublayer, and that sublayer's output
// is their output. Without a virtual out(), they would have to copy a
// (B, T, C) tensor in every step, only to return it from this accessor. At
// the video's final size, that copy would be 25MB.
class Layer {
public:
    virtual ~Layer() = default;

    virtual const Matrix& forward(const Matrix& x) = 0;
    virtual const Matrix& backward(const Matrix& dout) = 0;
    virtual void collect(std::vector<Param>&) {}
    virtual void set_training(bool) {}
    virtual const Matrix& out() const { return out_; }

protected:
    Matrix out_;
    Matrix dx_;
};

// Python:
//   nn.Linear(fan_in, fan_out, bias=...)
//   torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
//   torch.nn.init.zeros_(module.bias)
//
// The video's _init_weights, which it adds late and without much explanation:
// a standard deviation of 0.02 everywhere, instead of the Kaiming
// 1/sqrt(fan_in) from the previous video. For n_embd=384 those are 0.02 and
// 0.051. So it follows the same idea, "start each layer with smaller weights
// than seems necessary", but with one number for the whole model.
class Linear : public Layer {
public:
    Linear(int fan_in, int fan_out, bool has_bias, real init_std, std::mt19937& g)
        : fan_in_(fan_in), fan_out_(fan_out), has_bias_(has_bias),
        w_(static_cast<size_t>(fan_in) * static_cast<size_t>(fan_out)),
        dw_(w_.size(), real(0)),
        b_(has_bias ? static_cast<size_t>(fan_out) : size_t(0), real(0)),
        db_(b_.size(), real(0))
    {
        CHECK(fan_in > 0 && fan_out > 0,
            "a " << fan_in << "x" << fan_out << " linear layer");

        std::normal_distribution<real> randn(real(0), init_std);
        for (real& v : w_) {
            v = randn(g);
        }
        wname_ = "W " + std::to_string(fan_in) + "x" + std::to_string(fan_out);
        bname_ = "b " + std::to_string(fan_out);
    }

    const Matrix& forward(const Matrix& x) override {
        CHECK(x.cols == fan_in_, "a " << fan_in_ << "-wide layer was handed "
            << x.cols << " channels");

        // A pointer, not a copy. The layer that produced x owns it, and does
        // not change it until the next forward pass. By then, this layer's
        // backward pass has already run.
        x_ = &x;
        out_.resize(x.batch, x.time, fan_out_);
        const int n = x.rows();
        if (blocked_kernels()) {
            // Each row starts as a copy of the bias. Then x*W + b is the
            // add-into-C multiply that gpt_gemm.h provides.
            if (has_bias_) {
                for (int i = 0; i < n; ++i) {
                    std::copy(b_.begin(), b_.end(), out_.row(i));
                }
            }
            blocked::gemm(false, false, n, fan_out_, fan_in_, real(1),
                x.v.data(), fan_in_, w_.data(), fan_out_,
                out_.v.data(), fan_out_, true);
            return out_;
        }
        // One row per iteration, and no other iteration touches that row.
#pragma omp parallel for schedule(static) if(n >= kMinParallelRows)
        for (int i = 0; i < n; ++i) {
            const real* RESTRICT xi = x.row(i);
            real* RESTRICT oi = out_.row(i);
            if (has_bias_) {
                std::copy(b_.begin(), b_.end(), oi);
            }
            for (int k = 0; k < fan_in_; ++k) {
                const real xk = xi[k];
                const real* RESTRICT wk = &w_[static_cast<size_t>(k) * fan_out_];
                for (int j = 0; j < fan_out_; ++j) {
                    oi[j] += xk * wk[j];
                }
            }
        }
        return out_;
    }

    // Three passes, where the serial version had one. Each of the three
    // gradients that this layer produces needs a different loop to be the
    // parallel one.
    //
    // dx is computed per row and shares nothing. dW and db are sums over all
    // rows, so if the rows were divided between threads, all threads would
    // add into the same weights. A private copy of dW per thread, added
    // together afterwards, would cost fan_in*fan_out floats per thread (2.4MB
    // each at the video's final size), and it would change the order of the
    // sum. Dividing dW by input channel avoids both costs. Thread t owns whole
    // rows of dW, so two threads write to the same cache line only at the one
    // boundary between their ranges. Each row is still summed over the
    // examples in ascending order, by a single thread. All three passes
    // produce exactly the same bits as the fused serial loop.
    //
    // The cost of this division is that dout is read three times instead of
    // once. At the video's small size, dout stays in L2 the whole time, so
    // this costs nothing. At the scaled-up size the cost is real. It is the
    // same cache problem that blocking solves for the multiply itself.
    const Matrix& backward(const Matrix& dout) override {
        const Matrix& x = *x_;
        dx_.resize(x.batch, x.time, fan_in_);
        const int n = x.rows();

        if (blocked_kernels()) {
            // dx = dout * W^T and dW += x^T * dout: the same multiply with a
            // different operand transposed. The multiply now decides how to
            // divide the work between threads. It computes every output on
            // one thread, so neither sum depends on the number of threads.
            blocked::gemm(false, true, n, fan_in_, fan_out_, real(1),
                dout.v.data(), fan_out_, w_.data(), fan_out_,
                dx_.v.data(), fan_in_, true);
            blocked::gemm(true, false, fan_in_, fan_out_, n, real(1),
                x.v.data(), fan_in_, dout.v.data(), fan_out_,
                dw_.data(), fan_out_, true);
            accumulate_bias_grad(dout);
            return dx_;
        }

#pragma omp parallel for schedule(static) if(n >= kMinParallelRows)
        for (int i = 0; i < n; ++i) {
            // Distinct vectors: dout's storage and dx_'s.
            const real* RESTRICT di = dout.row(i);
            real* RESTRICT dxi = dx_.row(i);
            for (int k = 0; k < fan_in_; ++k) {
                const real* RESTRICT wk = &w_[static_cast<size_t>(k) * fan_out_];
                real acc = 0;
                for (int j = 0; j < fan_out_; ++j) {
                    acc += wk[j] * di[j];
                }
                dxi[k] = acc;
            }
        }

#pragma omp parallel for schedule(static) if(n >= kMinParallelRows)
        for (int k = 0; k < fan_in_; ++k) {
            // dwk points into dw_, di into dout's storage.
            real* RESTRICT dwk = &dw_[static_cast<size_t>(k) * fan_out_];
            for (int i = 0; i < n; ++i) {
                const real* RESTRICT di = dout.row(i);
                const real xk = x.row(i)[k];
                for (int j = 0; j < fan_out_; ++j) {
                    dwk[j] += xk * di[j];
                }
            }
        }

        accumulate_bias_grad(dout);
        return dx_;
    }

    void collect(std::vector<Param>& out) override {
        out.push_back({ wname_, &w_, &dw_ });
        if (has_bias_) {
            out.push_back({ bname_, &b_, &db_ });
        }
    }

private:
    // db is a column sum, and it stays serial. It is one pass over dout,
    // compared with the fan_in passes of the multiply, so it is a fraction of
    // a percent of the work here. Dividing fan_out values between sixteen
    // threads would put several threads on one cache line, which costs more
    // than it saves. Both kinds of kernel use this function.
    void accumulate_bias_grad(const Matrix& dout) {
        if (!has_bias_) {
            return;
        }
        const int n = dout.rows();
        for (int i = 0; i < n; ++i) {
            const real* di = dout.row(i);
            for (int j = 0; j < fan_out_; ++j) {
                db_[j] += di[j];
            }
        }
    }

    int fan_in_;
    int fan_out_;
    bool has_bias_;
    std::vector<real> w_, dw_;
    std::vector<real> b_, db_;
    std::string wname_, bname_;
    const Matrix* x_ = nullptr;
};

// Python: nn.LayerNorm(n_embd)
//
// This is batch normalisation from the previous video, but along the other
// axis. Batch normalisation normalised each feature across the examples of a
// batch. That made the examples depend on each other, and it needed a running
// mean and variance for evaluation. Layer normalisation normalises each row
// across its own channels. So every position is independent of every other,
// nothing has to be remembered between batches, and training and evaluation
// use the same code. For a model built for sequences of varying length, that
// difference is very important.
//
// torch.var is unbiased by default, but nn.LayerNorm divides by C, not C-1.
// That is why the backward pass here has no Bessel factor, while the one in
// BatchNorm1d had one.
class LayerNorm : public Layer {
public:
    explicit LayerNorm(int dim, real eps = real(1e-5))
        : dim_(dim), eps_(eps),
        gamma_(static_cast<size_t>(dim), real(1)), dgamma_(gamma_.size(), real(0)),
        beta_(static_cast<size_t>(dim), real(0)), dbeta_(beta_.size(), real(0))
    {
        CHECK(dim > 1, "layer normalisation over " << dim << " channel(s)");
        gname_ = "ln gamma " + std::to_string(dim);
        bname_ = "ln beta " + std::to_string(dim);
    }

    const Matrix& forward(const Matrix& x) override {
        CHECK(x.cols == dim_, "layer normalisation over " << dim_
            << " channels was handed " << x.cols);

        const int n = x.rows();
        out_.resize(x.batch, x.time, dim_);
        xhat_.resize(x.batch, x.time, dim_);
        invstd_.assign(static_cast<size_t>(n), real(0));

        const real inv_c = real(1) / dim_;
        // Each row is normalised using only its own channels. That property
        // is why LayerNorm was chosen for this model, and it also makes this
        // loop easy to parallelise.
#pragma omp parallel for schedule(static) if(n >= kMinParallelRows)
        for (int i = 0; i < n; ++i) {
            const real* xi = x.row(i);
            real mean = 0;
            for (int j = 0; j < dim_; ++j) {
                mean += xi[j];
            }
            mean *= inv_c;

            real var = 0;
            for (int j = 0; j < dim_; ++j) {
                const real d = xi[j] - mean;
                var += d * d;
            }
            var *= inv_c;

            const real inv = real(1) / std::sqrt(var + eps_);
            invstd_[static_cast<size_t>(i)] = inv;

            real* RESTRICT xh = xhat_.row(i);
            real* RESTRICT oi = out_.row(i);
            for (int j = 0; j < dim_; ++j) {
                xh[j] = (xi[j] - mean) * inv;
                oi[j] = gamma_[j] * xh[j] + beta_[j];
            }
        }
        return out_;
    }

    // Per row, with both means taken over the C channels:
    //
    //   dx = invstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
    //
    // The two subtracted terms come from the mean and the variance. If the
    // gradient shifted or scaled a whole row, the normalisation would undo
    // that change. So only the part of the change that is not a shift or a
    // scaling remains.
    const Matrix& backward(const Matrix& dout) override {
        const int n = dout.rows();
        dx_.resize(dout.batch, dout.time, dim_);

        const real inv_c = real(1) / dim_;
        // dx first, one row per thread. In the serial version, the same loop
        // also computed the two parameter gradients. But they are sums over
        // all rows, so they have to be computed in a separate loop.
#pragma omp parallel for schedule(static) if(n >= kMinParallelRows)
        for (int i = 0; i < n; ++i) {
            const real* RESTRICT di = dout.row(i);
            const real* RESTRICT xh = xhat_.row(i);
            real* RESTRICT dxi = dx_.row(i);

            real sum_d = 0;
            real sum_dx = 0;
            for (int j = 0; j < dim_; ++j) {
                const real dxhat = di[j] * gamma_[j];
                sum_d += dxhat;
                sum_dx += dxhat * xh[j];
            }
            const real mean_d = sum_d * inv_c;
            const real mean_dx = sum_dx * inv_c;
            const real inv = invstd_[static_cast<size_t>(i)];
            for (int j = 0; j < dim_; ++j) {
                const real dxhat = di[j] * gamma_[j];
                dxi[j] = inv * (dxhat - mean_d - xh[j] * mean_dx);
            }
        }

        // Then the parameter gradients, serially. Unlike in the Linear layer
        // above, there is no second axis to divide them along. There are only
        // dim_ of them, and several threads would share one cache line. The
        // arithmetic is also only a few tenths of a percent of the work of the
        // Linear layers before and after this one. A serial loop also keeps
        // the summation order of the serial build, so the loss curves of the
        // two builds can be compared.
        for (int i = 0; i < n; ++i) {
            const real* di = dout.row(i);
            const real* xh = xhat_.row(i);
            for (int j = 0; j < dim_; ++j) {
                dgamma_[j] += di[j] * xh[j];
                dbeta_[j] += di[j];
            }
        }
        return dx_;
    }

    void collect(std::vector<Param>& out) override {
        out.push_back({ gname_, &gamma_, &dgamma_ });
        out.push_back({ bname_, &beta_, &dbeta_ });
    }

private:
    int dim_;
    real eps_;
    std::vector<real> gamma_, dgamma_;
    std::vector<real> beta_, dbeta_;
    std::vector<real> invstd_;
    Matrix xhat_;
    std::string gname_, bname_;
};

// Python: nn.ReLU()
class ReLU : public Layer {
public:
    const Matrix& forward(const Matrix& x) override {
        out_.resize(x.batch, x.time, x.cols);
        const int n = static_cast<int>(x.v.size());
#pragma omp parallel for schedule(static) if(n >= kMinParallelElems)
        for (int k = 0; k < n; ++k) {
            out_.v[k] = x.v[k] > real(0) ? x.v[k] : real(0);
        }
        return out_;
    }

    const Matrix& backward(const Matrix& dout) override {
        dx_.resize(out_.batch, out_.time, out_.cols);
        const int n = static_cast<int>(dout.v.size());
#pragma omp parallel for schedule(static) if(n >= kMinParallelElems)
        for (int k = 0; k < n; ++k) {
            dx_.v[k] = out_.v[k] > real(0) ? dout.v[k] : real(0);
        }
        return dx_;
    }
};

// Python: nn.Dropout(p)
//
// Inverted dropout: during training, the kept activations are divided by 1-p.
// So evaluation, where nothing is dropped, needs no correction. The video adds
// dropout only at the scale-up, where the model has enough parameters to
// memorise the training set.
class Dropout : public Layer {
public:
    Dropout(real p, std::mt19937& g) : p_(p), rng_(g()) {
        CHECK(p >= real(0) && p < real(1), "dropout probability " << p);
    }

    void set_training(bool training) override { training_ = training; }

    const Matrix& forward(const Matrix& x) override {
        if (p_ <= real(0) || !training_) {
            passthrough_ = &x;
            return x;
        }
        passthrough_ = nullptr;

        const real keep = real(1) - p_;
        const real scale = real(1) / keep;
        out_.resize(x.batch, x.time, x.cols);
        mask_.assign(x.v.size(), real(0));
        // This loop is serial on purpose. The mask comes from one random
        // generator. With threads, they would race on its state, and every
        // run would draw a different mask. A reproducible parallel dropout
        // needs a counter-based generator keyed by the element index, which
        // would be a separate piece of work.
        std::uniform_real_distribution<real> u(real(0), real(1));
        for (size_t k = 0; k < x.v.size(); ++k) {
            if (u(rng_) < keep) {
                mask_[k] = scale;
                out_.v[k] = x.v[k] * scale;
            }
        }
        return out_;
    }

    const Matrix& backward(const Matrix& dout) override {
        if (passthrough_) {
            return dout;
        }
        dx_.resize(dout.batch, dout.time, dout.cols);
        const int n = static_cast<int>(dout.v.size());
#pragma omp parallel for schedule(static) if(n >= kMinParallelElems)
        for (int k = 0; k < n; ++k) {
            dx_.v[k] = dout.v[k] * mask_[k];
        }
        return dx_;
    }

    // When dropout is off, forward() returns its input unchanged instead of
    // copying it, so the base class's out_ would be empty. This override makes
    // out() return the correct tensor in that case.
    const Matrix& out() const override {
        return passthrough_ ? *passthrough_ : out_;
    }

private:
    real p_;
    bool training_ = true;
    std::mt19937 rng_;
    std::vector<real> mask_;
    const Matrix* passthrough_ = nullptr;
};

// ---------------------------------------------------------------------------
// Self-attention
// ---------------------------------------------------------------------------

// Python:
//   class Head(nn.Module):
//       def forward(self, x):
//           B, T, C = x.shape
//           k = self.key(x)       # (B, T, hs)
//           q = self.query(x)     # (B, T, hs)
//           wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # (B, T, T)
//           wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
//           wei = F.softmax(wei, dim=-1)
//           wei = self.dropout(wei)
//           v = self.value(x)
//           return wei @ v
//
//   class MultiHeadAttention(nn.Module):
//       def forward(self, x):
//           out = torch.cat([h(x) for h in self.heads], dim=-1)
//           return self.dropout(self.proj(out))
//
// The video writes one Head class and then a list of heads. This class runs
// all n_head heads at once, from a single key, query and value projection.
// That is the same arithmetic, with the concatenation done at the start
// instead of at the end. Head h owns channels [h*hs, (h+1)*hs) of each
// projection.
//
// Three things are easy to miss in the tensor expression above, so they are
// stated here:
//
//   - the mask is not an optimisation; it defines the training task. Position
//     t may only attend to 0..t. Because of that, the T positions of a window
//     are T separate examples, not one.
//   - the 1/sqrt(head_size) factor gives the scores unit variance when they
//     reach the softmax. Without it, a wide head produces large dot products,
//     the softmax output becomes almost one-hot, and the gradient through it
//     goes to zero. The previous video found the same problem with tanh.
//   - the masked entries are stored as zero probability, not as -inf, so the
//     backward pass can loop over u <= t and ignore the rest.
class CausalSelfAttention : public Layer {
public:
    CausalSelfAttention(int n_embd, int n_head, real dropout, real init_std,
        std::mt19937& g)
        : n_embd_(n_embd), n_head_(n_head), hs_(n_embd / n_head),
        key_(n_embd, n_embd, false, init_std, g),
        query_(n_embd, n_embd, false, init_std, g),
        value_(n_embd, n_embd, false, init_std, g),
        proj_(n_embd, n_embd, true, init_std, g),
        attn_drop_(dropout, g),
        resid_drop_(dropout, g)
    {
        CHECK(n_head > 0 && n_embd % n_head == 0,
            n_embd << " channels do not divide into " << n_head << " heads");
    }

    void set_training(bool training) override {
        attn_drop_.set_training(training);
        resid_drop_.set_training(training);
    }

    const Matrix& out() const override { return resid_drop_.out(); }

    const Matrix& forward(const Matrix& x) override {
        CHECK(x.cols == n_embd_, "attention over " << n_embd_
            << " channels was handed " << x.cols);

        const int B = x.batch;
        const int T = x.time;
        const real scale = real(1) / std::sqrt(static_cast<real>(hs_));

        const Matrix& q = query_.forward(x);
        const Matrix& k = key_.forward(x);
        const Matrix& v = value_.forward(x);

        // One row of att_ per (batch, head, query position). It holds that
        // position's probability distribution over the keys it may see.
        //
        // The batch and head loops are merged into one loop, so that OpenMP
        // has a single index to divide. The work is divided by (batch, head),
        // not by query position. The backward pass sums along the key axis, so
        // it cannot be divided by t, and the two passes have to use the same
        // division. Every (batch, head) does the same amount of work, so a
        // static schedule balances the threads exactly.
        const int bh_count = B * n_head_;
        const long long attn_work = static_cast<long long>(bh_count) * T * T;

        att_.resize(bh_count, T, T);
        if (blocked_kernels()) {
            // The scores for one (batch, head) are Q K^T, scaled. This is a
            // T x T product whose operands are that head's hs-wide slice of
            // every row, which is why the strides are n_embd. The whole
            // square is multiplied, and the masked half is then overwritten.
            // That wastes half of a product that is only a few percent of the
            // model's arithmetic, and in return the multiply gets a plain
            // rectangle. Each call stays on its own thread, because the
            // threads are already divided by head.
#pragma omp parallel for schedule(static) if(attn_work >= kMinParallelWork)
            for (int bh = 0; bh < bh_count; ++bh) {
                const int b = bh / n_head_;
                const int h = bh % n_head_;
                real* s = att_.row(bh, 0);
                blocked::gemm(false, true, T, T, hs_, scale,
                    q.row(b, 0) + h * hs_, n_embd_, k.row(b, 0) + h * hs_, n_embd_,
                    s, T, false);
                for (int t = 0; t < T; ++t) {
                    real* a = s + static_cast<size_t>(t) * T;
                    softmax_row(a, a, t + 1);
                    for (int u = t + 1; u < T; ++u) {
                        a[u] = real(0);
                    }
                }
            }
        }
        else {
#pragma omp parallel for schedule(static) if(attn_work >= kMinParallelWork)
            for (int bh = 0; bh < bh_count; ++bh) {
                const int b = bh / n_head_;
                const int h = bh % n_head_;
                for (int t = 0; t < T; ++t) {
                    real* a = att_.row(bh, t);
                    const real* qi = q.row(b, t) + h * hs_;
                    for (int u = 0; u <= t; ++u) {
                        const real* ki = k.row(b, u) + h * hs_;
                        real s = 0;
                        for (int dd = 0; dd < hs_; ++dd) {
                            s += qi[dd] * ki[dd];
                        }
                        a[u] = s * scale;
                    }
                    softmax_row(a, a, t + 1);
                    for (int u = t + 1; u < T; ++u) {
                        a[u] = real(0);        // masked future positions
                    }
                }
            }
        }

        const Matrix& a = attn_drop_.forward(att_);

        // Head h writes only channels [h*hs, (h+1)*hs) of heads_. So the
        // threads never write to the same elements, even though they share the
        // tensor.
        heads_.resize(B, T, n_embd_);
        if (blocked_kernels()) {
            // The attention probabilities are zero above the diagonal, so
            // multiplying the full square by V adds nothing from future
            // positions.
#pragma omp parallel for schedule(static) if(attn_work >= kMinParallelWork)
            for (int bh = 0; bh < bh_count; ++bh) {
                const int b = bh / n_head_;
                const int h = bh % n_head_;
                blocked::gemm(false, false, T, hs_, T, real(1),
                    a.row(bh, 0), T, v.row(b, 0) + h * hs_, n_embd_,
                    heads_.row(b, 0) + h * hs_, n_embd_, false);
            }
        }
        else {
#pragma omp parallel for schedule(static) if(attn_work >= kMinParallelWork)
            for (int bh = 0; bh < bh_count; ++bh) {
                const int b = bh / n_head_;
                const int h = bh % n_head_;
                for (int t = 0; t < T; ++t) {
                    const real* p = a.row(bh, t);
                    real* o = heads_.row(b, t) + h * hs_;
                    for (int u = 0; u <= t; ++u) {
                        const real* vi = v.row(b, u) + h * hs_;
                        const real w = p[u];
                        for (int dd = 0; dd < hs_; ++dd) {
                            o[dd] += w * vi[dd];
                        }
                    }
                }
            }
        }

        return resid_drop_.forward(proj_.forward(heads_));
    }

    const Matrix& backward(const Matrix& dout) override {
        const Matrix& dheads = proj_.backward(resid_drop_.backward(dout));

        const int B = dheads.batch;
        const int T = dheads.time;
        const real scale = real(1) / std::sqrt(static_cast<real>(hs_));

        const Matrix& a = attn_drop_.out();
        const Matrix& q = query_.out();
        const Matrix& k = key_.out();
        const Matrix& v = value_.out();

        const int bh_count = B * n_head_;
        const long long attn_work = static_cast<long long>(bh_count) * T * T;

        da_.resize(bh_count, T, T);
        dv_.resize(B, T, n_embd_);
        dq_.resize(B, T, n_embd_);
        dk_.resize(B, T, n_embd_);

        // out[b,t,h] = sum_{u<=t} a[b,h,t,u] * v[b,u,h]. So the gradient of
        // each attention probability is a dot product, and the gradient of v
        // at position u is a sum over all t >= u of a[t,u] * dout[t].
        //
        // That sum is why the work cannot be divided by query position: every
        // t from u upwards adds into dv at u. Dividing by (batch, head) avoids
        // the problem. Thread bh alone owns channels [h*hs, (h+1)*hs) of dv
        // for batch b, and it still goes through t in order. So the sums are
        // added in the same order as in the serial version.
        if (blocked_kernels()) {
            // dA = dO V^T and dV = A^T dO, per (batch, head). dA is also
            // computed above the diagonal, where nothing needs it. The softmax
            // backward below only reads u <= t, so those values are never
            // used. A^T dO needs no special care, because A is zero above the
            // diagonal.
#pragma omp parallel for schedule(static) if(attn_work >= kMinParallelWork)
            for (int bh = 0; bh < bh_count; ++bh) {
                const int b = bh / n_head_;
                const int h = bh % n_head_;
                const real* dO = dheads.row(b, 0) + h * hs_;
                blocked::gemm(false, true, T, T, hs_, real(1),
                    dO, n_embd_, v.row(b, 0) + h * hs_, n_embd_,
                    da_.row(bh, 0), T, false);
                blocked::gemm(true, false, T, hs_, T, real(1),
                    a.row(bh, 0), T, dO, n_embd_,
                    dv_.row(b, 0) + h * hs_, n_embd_, false);
            }
        }
        else {
#pragma omp parallel for schedule(static) if(attn_work >= kMinParallelWork)
            for (int bh = 0; bh < bh_count; ++bh) {
                const int b = bh / n_head_;
                const int h = bh % n_head_;
                for (int t = 0; t < T; ++t) {
                    const real* RESTRICT dO = dheads.row(b, t) + h * hs_;
                    const real* RESTRICT p = a.row(bh, t);
                    real* RESTRICT dA = da_.row(bh, t);
                    for (int u = 0; u <= t; ++u) {
                        const real* RESTRICT vi = v.row(b, u) + h * hs_;
                        real* RESTRICT dvi = dv_.row(b, u) + h * hs_;
                        real dot = 0;
                        const real w = p[u];
                        for (int dd = 0; dd < hs_; ++dd) {
                            dot += dO[dd] * vi[dd];
                            dvi[dd] += w * dO[dd];
                        }
                        dA[u] = dot;
                    }
                }
            }
        }

        const Matrix& dp = attn_drop_.backward(da_);

        // Softmax over a row: ds_u = p_u * (dp_u - sum_w p_w dp_w). The
        // subtracted term comes from the constraint that each row sums to one:
        // if one probability goes up, the others must go down.
        ds_.resize(bh_count, T, T);
#pragma omp parallel for schedule(static) if(attn_work >= kMinParallelWork)
        for (int bh = 0; bh < bh_count; ++bh) {
            for (int t = 0; t < T; ++t) {
                const real* p = att_.row(bh, t);
                const real* dpi = dp.row(bh, t);
                real* ds = ds_.row(bh, t);
                real dot = 0;
                for (int u = 0; u <= t; ++u) {
                    dot += p[u] * dpi[u];
                }
                for (int u = 0; u <= t; ++u) {
                    ds[u] = p[u] * (dpi[u] - dot);
                }
            }
        }

        // dk is summed along the key axis, like dv, so this loop is also
        // divided by (batch, head), for the same reason.
        if (blocked_kernels()) {
            // dQ = scale * dS K and dK = scale * dS^T Q. The resize set ds_
            // to zero, and only entries with u <= t were written. So future
            // positions add nothing to either product.
#pragma omp parallel for schedule(static) if(attn_work >= kMinParallelWork)
            for (int bh = 0; bh < bh_count; ++bh) {
                const int b = bh / n_head_;
                const int h = bh % n_head_;
                const real* ds = ds_.row(bh, 0);
                blocked::gemm(false, false, T, hs_, T, scale,
                    ds, T, k.row(b, 0) + h * hs_, n_embd_,
                    dq_.row(b, 0) + h * hs_, n_embd_, false);
                blocked::gemm(true, false, T, hs_, T, scale,
                    ds, T, q.row(b, 0) + h * hs_, n_embd_,
                    dk_.row(b, 0) + h * hs_, n_embd_, false);
            }
        }
        else {
#pragma omp parallel for schedule(static) if(attn_work >= kMinParallelWork)
            for (int bh = 0; bh < bh_count; ++bh) {
                const int b = bh / n_head_;
                const int h = bh % n_head_;
                for (int t = 0; t < T; ++t) {
                    const real* RESTRICT ds = ds_.row(bh, t);
                    const real* RESTRICT qi = q.row(b, t) + h * hs_;
                    real* RESTRICT dqi = dq_.row(b, t) + h * hs_;
                    for (int u = 0; u <= t; ++u) {
                        const real* RESTRICT ki = k.row(b, u) + h * hs_;
                        real* RESTRICT dki = dk_.row(b, u) + h * hs_;
                        const real s = ds[u] * scale;
                        for (int dd = 0; dd < hs_; ++dd) {
                            dqi[dd] += s * ki[dd];
                            dki[dd] += s * qi[dd];
                        }
                    }
                }
            }
        }

        // x was the input of all three projections, so its gradient is the
        // sum of all three.
        const Matrix& dxq = query_.backward(dq_);
        const Matrix& dxk = key_.backward(dk_);
        const Matrix& dxv = value_.backward(dv_);

        dx_.resize(B, T, n_embd_);
        const int n = static_cast<int>(dx_.v.size());
#pragma omp parallel for schedule(static) if(n >= kMinParallelElems)
        for (int i = 0; i < n; ++i) {
            dx_.v[i] = dxq.v[i] + dxk.v[i] + dxv.v[i];
        }
        return dx_;
    }

    void collect(std::vector<Param>& out) override {
        const size_t before = out.size();
        key_.collect(out);
        query_.collect(out);
        value_.collect(out);
        proj_.collect(out);
        static const char* kWhich[] = { "key", "query", "value", "proj" };
        for (size_t i = before; i < out.size(); ++i) {
            out[i].name = std::string(kWhich[std::min<size_t>(3, i - before)])
                + " " + out[i].name;
        }
    }

private:
    int n_embd_;
    int n_head_;
    int hs_;
    Linear key_, query_, value_, proj_;
    Dropout attn_drop_, resid_drop_;
    Matrix att_, heads_;
    Matrix da_, ds_, dq_, dk_, dv_;
};

// Python:
//   class FeedFoward(nn.Module):
//       def __init__(self, n_embd):
//           self.net = nn.Sequential(
//               nn.Linear(n_embd, 4 * n_embd),
//               nn.ReLU(),
//               nn.Linear(4 * n_embd, n_embd),
//               nn.Dropout(dropout),
//           )
//
// In attention, the positions exchange information with each other. In this
// layer, each position processes the information it received, on its own. The
// video adds this layer after it notices that the network goes directly from
// attention to prediction, with no computation in between. The factor of 4
// comes from the original paper.
class FeedForward : public Layer {
public:
    FeedForward(int n_embd, real dropout, real init_std, std::mt19937& g)
        : fc_(n_embd, 4 * n_embd, true, init_std, g),
        proj_(4 * n_embd, n_embd, true, init_std, g),
        drop_(dropout, g)
    {
    }

    void set_training(bool training) override { drop_.set_training(training); }
    const Matrix& out() const override { return drop_.out(); }

    const Matrix& forward(const Matrix& x) override {
        return drop_.forward(proj_.forward(relu_.forward(fc_.forward(x))));
    }

    const Matrix& backward(const Matrix& dout) override {
        return fc_.backward(relu_.backward(proj_.backward(drop_.backward(dout))));
    }

    void collect(std::vector<Param>& out) override {
        fc_.collect(out);
        proj_.collect(out);
    }

private:
    Linear fc_;
    ReLU relu_;
    Linear proj_;
    Dropout drop_;
};

// Python:
//   class Block(nn.Module):
//       def forward(self, x):
//           x = x + self.sa(self.ln1(x))
//           x = x + self.ffwd(self.ln2(x))
//           return x
//
// The two additions are the reason why a stack of these blocks can be trained
// at all. A residual connection gives the gradient a path from the loss to
// every layer that does not pass through any weights. So the signal that
// reaches block 0 is not a product of five Jacobians. The video first shows
// the failure (without residual connections, more blocks make the model worse,
// not better) and then fixes it.
//
// The normalisation is applied before each sublayer, not after it. The 2017
// paper does it the other way round. The video mentions that pre-norm is now
// the standard arrangement, because it keeps the residual path itself free of
// any normalisation.
class Block : public Layer {
public:
    Block(int n_embd, int n_head, real dropout, real init_std, std::mt19937& g)
        : ln1_(n_embd), attn_(n_embd, n_head, dropout, init_std, g),
        ln2_(n_embd), ffwd_(n_embd, dropout, init_std, g)
    {
    }

    void set_training(bool training) override {
        attn_.set_training(training);
        ffwd_.set_training(training);
    }

    const Matrix& out() const override { return out_; }

    const Matrix& forward(const Matrix& x) override {
        const int n = x.batch * x.time * x.cols;

        const Matrix& b = attn_.forward(ln1_.forward(x));
        mid_.resize(x.batch, x.time, x.cols);
#pragma omp parallel for schedule(static) if(n >= kMinParallelElems)
        for (int i = 0; i < n; ++i) {
            mid_.v[i] = x.v[i] + b.v[i];
        }

        const Matrix& d = ffwd_.forward(ln2_.forward(mid_));
        out_.resize(x.batch, x.time, x.cols);
#pragma omp parallel for schedule(static) if(n >= kMinParallelElems)
        for (int i = 0; i < n; ++i) {
            out_.v[i] = mid_.v[i] + d.v[i];
        }
        return out_;
    }

    // In the forward pass, each residual add uses its input twice, so in the
    // backward pass the two gradients are summed. The gradient at `out`
    // reaches `mid` through the feedforward branch, and also directly through
    // the skip connection.
    const Matrix& backward(const Matrix& dout) override {
        const int n = dout.batch * dout.time * dout.cols;

        const Matrix& dmid_branch = ln2_.backward(ffwd_.backward(dout));
        dmid_.resize(dout.batch, dout.time, dout.cols);
#pragma omp parallel for schedule(static) if(n >= kMinParallelElems)
        for (int i = 0; i < n; ++i) {
            dmid_.v[i] = dout.v[i] + dmid_branch.v[i];
        }

        const Matrix& dx_branch = ln1_.backward(attn_.backward(dmid_));
        dx_.resize(dout.batch, dout.time, dout.cols);
#pragma omp parallel for schedule(static) if(n >= kMinParallelElems)
        for (int i = 0; i < n; ++i) {
            dx_.v[i] = dmid_.v[i] + dx_branch.v[i];
        }
        return dx_;
    }

    void collect(std::vector<Param>& out) override {
        auto tag = [&out](size_t from, const char* prefix) {
            for (size_t i = from; i < out.size(); ++i) {
                out[i].name = std::string(prefix) + " " + out[i].name;
            }
        };
        size_t n = out.size();
        ln1_.collect(out);
        tag(n, "ln1");
        attn_.collect(out);          // names its own four projections
        n = out.size();
        ln2_.collect(out);
        tag(n, "ln2");
        n = out.size();
        ffwd_.collect(out);
        tag(n, "ffwd");
    }

private:
    LayerNorm ln1_;
    CausalSelfAttention attn_;
    LayerNorm ln2_;
    FeedForward ffwd_;
    Matrix mid_, dmid_;
};

// ---------------------------------------------------------------------------
// The models
// ---------------------------------------------------------------------------

constexpr real kInitStd = real(0.02);

// Cross entropy over B*T independent predictions. It is the same as in the
// makemore files, except that the vocabulary size is known only at run time.
//
// Python:
//   B, T, C = logits.shape
//   loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
inline real cross_entropy(const Matrix& logits, const int* targets, Matrix* dlogits) {
    const int n = logits.rows();
    const int vocab = logits.cols;
    CHECK(n > 0, "no predictions to score");

    if (dlogits) {
        dlogits->resize(logits.batch, logits.time, vocab);
    }

    const real inv_n = real(1) / n;

    // Each row is scored on its own, but two things here are shared: the
    // softmax needs a scratch row of probabilities, and the loss is a sum over
    // all rows. The scratch buffer is declared inside the parallel region, so
    // each thread allocates its own. The per-row losses go into an array,
    // which is added up afterwards in order. So the returned number is the
    // same for any number of threads.
    std::vector<real> row_loss(static_cast<size_t>(n));
#pragma omp parallel if(n >= kMinParallelRows)
    {
        std::vector<real> probs(static_cast<size_t>(vocab));
#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            softmax_row(logits.row(i), probs.data(), vocab);
            const int y = targets[i];
            row_loss[static_cast<size_t>(i)] =
                -std::log(probs[static_cast<size_t>(y)]);
            if (dlogits) {
                real* dl = dlogits->row(i);
                for (int j = 0; j < vocab; ++j) {
                    dl[j] = (probs[static_cast<size_t>(j)] - (j == y ? real(1) : real(0))) * inv_n;
                }
            }
        }
    }

    real total = 0;
    for (int i = 0; i < n; ++i) {
        total += row_loss[static_cast<size_t>(i)];
    }
    return total * inv_n;
}


struct Config {
    int n_embd;
    int n_head;
    int n_layer;
    int block;
    int batch;
    real dropout;
    real lr;
};

// The approximate number of floating point operations in one training step.
// The forward pass costs 2 operations (one multiply and one add) per
// parameter per token, and the backward pass costs twice that. This is the
// standard 6N estimate. The two embedding tables are left out, because they
// are lookups, not matrix multiplies. Attention's two products between
// activations use no parameters at all, and at context 256 they are too large
// to ignore.
//
// This function is here, not with the rest of the cost model in gpt.cpp,
// because gpt_bench.h also uses it to turn seconds into a rate.
inline double step_flops(const Config& cfg, size_t params, int vocab) {
    const double lookups = static_cast<double>(vocab + cfg.block) * cfg.n_embd;
    const double matmul = static_cast<double>(params) - lookups;
    const double tokens = static_cast<double>(cfg.batch) * cfg.block;
    const double attention = 6.0 * cfg.block * (cfg.block + 1.0)
        * cfg.n_embd * cfg.batch * cfg.n_layer;
    return 6.0 * matmul * tokens + attention;
}

// Python:
//   class GPTLanguageModel(nn.Module):
//       def __init__(self):
//           self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
//           self.position_embedding_table = nn.Embedding(block_size, n_embd)
//           self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
//           self.ln_f = nn.LayerNorm(n_embd)
//           self.lm_head = nn.Linear(n_embd, vocab_size)
//       def forward(self, idx, targets=None):
//           B, T = idx.shape
//           tok_emb = self.token_embedding_table(idx)              # (B, T, C)
//           pos_emb = self.position_embedding_table(torch.arange(T))  # (T, C)
//           x = tok_emb + pos_emb
//           x = self.blocks(x)
//           x = self.ln_f(x)
//           logits = self.lm_head(x)
//
// The position embedding tells attention where each character is. Without it,
// the model sees only an unordered set of characters: the dot product between
// a query and a key does not depend on which of them came first, so "abc" and
// "cba" would produce the same attention pattern. A learned vector per
// position is the simplest solution. It is also why block_size is a hard
// limit on the context: the table has no row for position 257.
class Gpt {
public:
    Gpt(const Config& cfg, int vocab, std::mt19937& g)
        : cfg_(cfg), vocab_(vocab),
        wte_(static_cast<size_t>(vocab) * cfg.n_embd),
        dwte_(wte_.size(), real(0)),
        wpe_(static_cast<size_t>(cfg.block) * cfg.n_embd),
        dwpe_(wpe_.size(), real(0)),
        ln_f_(cfg.n_embd),
        lm_head_(cfg.n_embd, vocab, true, kInitStd, g)
    {
        CHECK(cfg.n_layer >= 0, "a network of " << cfg.n_layer << " blocks");
        CHECK(cfg.block > 0 && cfg.batch > 0,
            "batch " << cfg.batch << " of " << cfg.block << " positions");

        std::normal_distribution<real> randn(real(0), kInitStd);
        for (real& v : wte_) {
            v = randn(g);
        }
        for (real& v : wpe_) {
            v = randn(g);
        }

        for (int i = 0; i < cfg.n_layer; ++i) {
            blocks_.push_back(std::make_unique<Block>(cfg.n_embd, cfg.n_head,
                cfg.dropout, kInitStd, g));
        }

        params_.push_back({ "wte " + std::to_string(vocab) + "x"
            + std::to_string(cfg.n_embd), &wte_, &dwte_ });
        params_.push_back({ "wpe " + std::to_string(cfg.block) + "x"
            + std::to_string(cfg.n_embd), &wpe_, &dwpe_ });
        for (size_t i = 0; i < blocks_.size(); ++i) {
            const size_t before = params_.size();
            blocks_[i]->collect(params_);
            for (size_t k = before; k < params_.size(); ++k) {
                params_[k].name = "block " + std::to_string(i) + " " + params_[k].name;
            }
        }
        ln_f_.collect(params_);
        lm_head_.collect(params_);
    }

    // params_ holds the addresses of wte_ and wpe_, so the object must not be
    // copied or moved.
    Gpt(const Gpt&) = delete;
    Gpt& operator=(const Gpt&) = delete;

    std::vector<Param>& params() { return params_; }
    const Config& config() const { return cfg_; }

    size_t parameter_count() const {
        size_t n = 0;
        for (const Param& p : params_) {
            n += p.data->size();
        }
        return n;
    }

    void set_training(bool training) {
        for (auto& b : blocks_) {
            b->set_training(training);
        }
    }

    void zero_grad() {
        // One tensor per iteration, with a dynamic schedule. The tensors
        // differ in size by a factor of about a thousand. With a static
        // schedule, one thread might get the token embedding table and
        // another thread two bias vectors.
        const int n = static_cast<int>(params_.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < n; ++i) {
            std::vector<real>& g = *params_[static_cast<size_t>(i)].grad;
            std::fill(g.begin(), g.end(), real(0));
        }
    }

    const Matrix& forward(const int* x, int B, int T) {
        CHECK(T > 0 && T <= cfg_.block, "a window of " << T
            << " positions, but the position table has " << cfg_.block << " rows");

        // One loop over (batch, position), in the same order as nested loops
        // would use: row i of the tensor is window i/T at position i%T.
        const int rows = B * T;
        emb_.resize(B, T, cfg_.n_embd);
#pragma omp parallel for schedule(static) if(rows >= kMinParallelRows)
        for (int i = 0; i < rows; ++i) {
            const real* tok = &wte_[static_cast<size_t>(x[i]) * cfg_.n_embd];
            const real* pos = &wpe_[static_cast<size_t>(i % T) * cfg_.n_embd];
            real* e = emb_.row(i);
            for (int c = 0; c < cfg_.n_embd; ++c) {
                e[c] = tok[c] + pos[c];
            }
        }

        const Matrix* cur = &emb_;
        for (auto& b : blocks_) {
            cur = &b->forward(*cur);
        }
        return lm_head_.forward(ln_f_.forward(*cur));
    }

    real forward_loss(const int* x, const int* y, int B, int T) {
        return cross_entropy(forward(x, B, T), y, nullptr);
    }

    real forward_backward(const int* x, const int* y, int B, int T) {
        const real loss = cross_entropy(forward(x, B, T), y, &dlogits_);

        const Matrix* d = &ln_f_.backward(lm_head_.backward(dlogits_));
        for (size_t i = blocks_.size(); i-- > 0; ) {
            d = &blocks_[i]->backward(*d);
        }

        // The gradients of the two embedding tables are added back by index,
        // and the += is essential. A character that appears eight times in a
        // batch contributes eight times, and all B windows use the same row of
        // the position table for position t.
        //
        // This is also why this is the only loop in the model that stays
        // serial. The repeated rows would cause conflicts between threads: two
        // positions with the same character, or the same position in
        // different windows, write to the same row. Every way of dividing the
        // loop has such conflicts. Also, this is only one pass over (B, T, C),
        // compared with the matrix multiplies before and after it, so it takes
        // little time anyway.
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                const real* de = d->row(b, t);
                real* dtok = &dwte_[static_cast<size_t>(x[b * T + t]) * cfg_.n_embd];
                real* dpos = &dwpe_[static_cast<size_t>(t) * cfg_.n_embd];
                for (int c = 0; c < cfg_.n_embd; ++c) {
                    dtok[c] += de[c];
                    dpos[c] += de[c];
                }
            }
        }
        return loss;
    }

    // Python:
    //   def generate(self, idx, max_new_tokens):
    //       for _ in range(max_new_tokens):
    //           idx_cond = idx[:, -block_size:]
    //           logits, loss = self(idx_cond)
    //           logits = logits[:, -1, :]
    //           probs = F.softmax(logits, dim=-1)
    //           idx_next = torch.multinomial(probs, num_samples=1)
    //           idx = torch.cat((idx, idx_next), dim=1)
    //
    // Cropping to the last block_size characters is necessary: position
    // block_size has no row in the position table. It also means that the
    // whole context is recomputed for every new character. That is why real
    // implementations cache the keys and values instead.
    // `prompt` is the starting context. An empty prompt starts from token 0.
    // That is the lowest byte in the corpus, which for most text files is the
    // newline character: the most neutral start this vocabulary has.
    //
    // Temperature divides the logits before the softmax. Below 1, it makes the
    // distribution sharper, toward the characters the model is most sure of.
    // Above 1, it makes the distribution flatter, toward uniform. The video
    // samples directly from the model and never mentions this setting. At the
    // character level it makes most of the difference between real words and
    // random letters. So the tool in gpt_chat.cpp needs it, even though the
    // walkthrough does not.
    std::vector<int> generate(const std::vector<int>& prompt, int count,
        real temperature, std::mt19937& g) {
        CHECK(temperature > real(0), "temperature " << temperature
            << " has to be positive");

        // Dropout has to be off. Otherwise the samples would come from a
        // different model than the one that was trained.
        set_training(false);

        std::vector<int> ctx = prompt.empty() ? std::vector<int>(1, 0) : prompt;
        std::vector<int> ids;
        std::vector<real> probs(static_cast<size_t>(vocab_));
        std::vector<real> scaled(static_cast<size_t>(vocab_));
        for (int i = 0; i < count; ++i) {
            const int T = static_cast<int>(std::min<size_t>(static_cast<size_t>(cfg_.block), ctx.size()));
            const int* window = ctx.data() + (ctx.size() - static_cast<size_t>(T));

            const Matrix& logits = forward(window, 1, T);
            const real* row = logits.row(0, T - 1);
            for (int j = 0; j < vocab_; ++j) {
                scaled[static_cast<size_t>(j)] = row[j] / temperature;
            }
            softmax_row(scaled.data(), probs.data(), vocab_);
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            const int next = dist(g);
            ids.push_back(next);
            ctx.push_back(next);
        }

        set_training(true);
        return ids;
    }

    // What the walkthrough uses: start from a newline, and sample directly
    // from the model, as the video does.
    std::vector<int> generate(int count, std::mt19937& g) {
        return generate(std::vector<int>(), count, real(1), g);
    }

private:
    Config cfg_;
    int vocab_;
    std::vector<real> wte_, dwte_;
    std::vector<real> wpe_, dwpe_;
    std::vector<std::unique_ptr<Block>> blocks_;
    LayerNorm ln_f_;
    Linear lm_head_;
    std::vector<Param> params_;
    Matrix emb_, dlogits_;
};

// ---------------------------------------------------------------------------
// The optimiser
// ---------------------------------------------------------------------------

// Python: torch.optim.AdamW(model.parameters(), lr=learning_rate)
//
// The makemore files used plain gradient descent with a hand-tuned step size.
// That does not work for a model whose parameters have very different
// gradient scales: an embedding row is used once per batch, while a weight
// matrix is used B*T times. Adam divides each parameter's step by that
// parameter's own recent gradient size, so one learning rate works for all of
// them.
//
// The W stands for decoupled weight decay. torch's AdamW shrinks each
// parameter toward zero in a separate step, instead of adding a term to the
// gradient. So the decay is not divided by sqrt(v). The default weight_decay
// is 0.01, and the video does not change it, so 0.01 is the value it used.
class AdamW {
public:
    AdamW(std::vector<Param>& params, real lr, real beta1 = real(0.9),
        real beta2 = real(0.999), real eps = real(1e-8), real wd = real(0.01))
        : params_(&params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), wd_(wd)
    {
        CHECK(lr > real(0), "learning rate " << lr);
        m_.resize(params.size());
        v_.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            m_[i].assign(params[i].data->size(), real(0));
            v_[i].assign(params[i].data->size(), real(0));
        }
    }

    void step() {
        ++t_;
        // Both moments start at zero, so in early steps they are biased toward
        // zero. Dividing by 1-beta^t removes that bias exactly.
        const real bc1 = real(1) - std::pow(beta1_, static_cast<real>(t_));
        const real bc2 = real(1) - std::pow(beta2_, static_cast<real>(t_));

        // Every parameter is updated independently of every other, so the
        // only question is how to divide the work. Dividing by tensor, not by
        // element within each tensor, costs one fork per step instead of one
        // per tensor. The dynamic schedule handles the very different tensor
        // sizes.
        const int np = static_cast<int>(params_->size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int k = 0; k < np; ++k) {
            const size_t kk = static_cast<size_t>(k);
            std::vector<real>& p = *(*params_)[kk].data;
            const std::vector<real>& g = *(*params_)[kk].grad;
            std::vector<real>& m = m_[kk];
            std::vector<real>& v = v_[kk];
            for (size_t i = 0; i < p.size(); ++i) {
                m[i] = beta1_ * m[i] + (real(1) - beta1_) * g[i];
                v[i] = beta2_ * v[i] + (real(1) - beta2_) * g[i] * g[i];
                const real mhat = m[i] / bc1;
                const real vhat = v[i] / bc2;
                p[i] -= lr_ * wd_ * p[i];
                p[i] -= lr_ * mhat / (std::sqrt(vhat) + eps_);
            }
        }
    }

    void set_lr(real lr) { lr_ = lr; }

private:
    std::vector<Param>* params_;
    real lr_, beta1_, beta2_, eps_, wd_;
    std::vector<std::vector<real>> m_, v_;
    int t_ = 0;
};

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

// Hours of training should not have to be repeated just because the program
// was restarted. The format describes itself, on purpose. The header holds the
// shape of the model and the size of its scalar type, and every tensor holds
// its own name and length. So a checkpoint for a model of a different shape is
// rejected, instead of being read as meaningless numbers. The header also holds
// the vocabulary. So a checkpoint can be used without the corpus it was
// trained on, and gpt_chat.cpp can open any checkpoint on its own.
//
// A missing or outdated checkpoint is not an error; it only means that the
// model has to be trained. So these functions report and return false instead
// of using CHECK. This is a third kind of problem, besides the two in check.h:
// not a broken program and not a wrong number, but a file that is missing or
// no longer matches.
//
// The bytes are raw `real` values (float by default) in native byte order.
// Both machines this runs on are x86-64, so endianness is ignored. The
// sizeof(real) in the header and in the file name stops a float build from
// reading a double build's file.

// The last byte is the version of the format. Version 3 added the vocabulary
// to the header.
constexpr char kCheckpointMagic[8] = { 'V', 'M', 'L', 'P', 'G', 'P', 'T', '3' };

// FNV-1a over the vocabulary and the corpus length. The header already checks
// the shape of the model, but the same shape does not mean the same corpus.
// Two corpora with the same number of distinct characters would produce
// checkpoints with the same name and a valid header. Loading one into the
// other would give a model whose token 37 means a different letter. Nothing
// in that failure looks like a failure; the losses are just bad. So it is
// worth eight bytes to make it impossible.
inline std::uint64_t corpus_fingerprint(const Data& d) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&h](unsigned char b) {
        h ^= b;
        h *= 1099511628211ull;
    };
    for (char c : d.itos) {
        mix(static_cast<unsigned char>(c));
    }
    const std::uint64_t n = d.train.size() + d.val.size();
    for (int i = 0; i < 8; ++i) {
        mix(static_cast<unsigned char>((n >> (i * 8)) & 0xffu));
    }
    return h;
}

template <class T>
inline void write_pod(std::ostream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
inline bool read_pod(std::istream& f, T& v) {
    return static_cast<bool>(f.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

inline std::string checkpoint_name(const Config& cfg, const Data& d) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "gpt_c%d_h%d_l%d_t%d_v%d_r%d_%08x.ckpt",
        cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block, d.vocab,
        static_cast<int>(sizeof(real)),
        static_cast<unsigned>(corpus_fingerprint(d) & 0xffffffffu));
    return std::string(buf);
}

inline bool save_params(const std::vector<Param>& params, const Config& cfg,
    const Data& d, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        printf("  could not open %s for writing\n", path.c_str());
        return false;
    }

    f.write(kCheckpointMagic, sizeof(kCheckpointMagic));
    write_pod(f, static_cast<std::int32_t>(sizeof(real)));
    write_pod(f, corpus_fingerprint(d));
    write_pod(f, static_cast<std::int32_t>(d.vocab));
    f.write(d.itos.data(), static_cast<std::streamsize>(d.itos.size()));
    write_pod(f, static_cast<std::int32_t>(cfg.n_embd));
    write_pod(f, static_cast<std::int32_t>(cfg.n_head));
    write_pod(f, static_cast<std::int32_t>(cfg.n_layer));
    write_pod(f, static_cast<std::int32_t>(cfg.block));
    write_pod(f, static_cast<std::int32_t>(params.size()));

    for (const Param& p : params) {
        write_pod(f, static_cast<std::int32_t>(p.name.size()));
        f.write(p.name.data(), static_cast<std::streamsize>(p.name.size()));
        write_pod(f, static_cast<std::int64_t>(p.data->size()));
        f.write(reinterpret_cast<const char*>(p.data->data()),
            static_cast<std::streamsize>(p.data->size() * sizeof(real)));
    }
    return static_cast<bool>(f);
}

// What the header of a checkpoint holds. Only the four fields of Config that
// change the shape of the weights are stored. Batch, dropout and the learning
// rate are training settings. So a model built from the header gets the
// values for sampling: batch 1, no dropout.
struct CheckpointHeader {
    Config cfg{ 0, 0, 0, 0, 1, real(0), real(0) };
    Vocab vocab;
    std::uint64_t fingerprint = 0;
    std::int32_t tensors = 0;
};

// Reads the header and leaves f at the first tensor. Returns nullptr if the
// header is valid, and otherwise the reason why it is not. The reason is
// returned instead of printed, because gpt_chat.cpp shows it in its list of
// checkpoints.
inline const char* read_checkpoint_header(std::istream& f, CheckpointHeader& h) {
    constexpr size_t kVersionByte = sizeof(kCheckpointMagic) - 1;
    char magic[sizeof(kCheckpointMagic)];
    if (!f.read(magic, sizeof(magic))
        || std::memcmp(magic, kCheckpointMagic, kVersionByte) != 0) {
        return "not a checkpoint written by this file";
    }
    if (magic[kVersionByte] != kCheckpointMagic[kVersionByte]) {
        return "it is in another version of the format";
    }

    std::int32_t scalar = 0, v = 0;
    if (!read_pod(f, scalar) || !read_pod(f, h.fingerprint) || !read_pod(f, v)) {
        return "the header is truncated";
    }
    if (scalar != static_cast<std::int32_t>(sizeof(real))) {
        return "it was written with a different scalar type";
    }
    if (v <= 0 || v > 256) {
        return "its vocabulary size is malformed";
    }
    std::vector<char> chars(static_cast<size_t>(v));
    if (!f.read(chars.data(), v)) {
        return "the header is truncated";
    }
    for (size_t i = 1; i < chars.size(); ++i) {
        if (static_cast<unsigned char>(chars[i - 1])
            >= static_cast<unsigned char>(chars[i])) {
            return "its vocabulary is not in byte order";
        }
    }
    h.vocab.assign(chars);

    std::int32_t embd = 0, head = 0, layer = 0, block = 0;
    if (!read_pod(f, embd) || !read_pod(f, head) || !read_pod(f, layer)
        || !read_pod(f, block) || !read_pod(f, h.tensors)) {
        return "the header is truncated";
    }
    // gpt_chat.cpp builds a model from these values. A shape that no model
    // can have is rejected here, so that it does not reach the CHECKs in the
    // Gpt constructor.
    if (embd <= 0 || head <= 0 || embd % head != 0 || layer < 0 || block <= 0) {
        return "its model shape is malformed";
    }
    h.cfg.n_embd = embd;
    h.cfg.n_head = head;
    h.cfg.n_layer = layer;
    h.cfg.block = block;
    return nullptr;
}

// Reads the tensors that follow the header into params. Returns nullptr on
// success, and otherwise the reason for the failure.
inline const char* read_tensors(std::istream& f, std::int32_t count,
    std::vector<Param>& params) {
    if (count != static_cast<std::int32_t>(params.size())) {
        return "it holds a different number of tensors";
    }

    // Read into a staging buffer first. If the file turns out to be truncated
    // in the middle, the model should stay as it was. It should not end up
    // with half of its weights from one model and half from another. The cost
    // is holding the parameters twice, which for the scaled-up model is 86MB.
    std::vector<std::vector<real>> staged(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        std::int32_t len = 0;
        if (!read_pod(f, len) || len < 0 || len > 256) {
            return "a tensor name is malformed";
        }
        std::string name(static_cast<size_t>(len), '\0');
        if (len > 0 && !f.read(&name[0], len)) {
            return "a tensor name is truncated";
        }
        std::int64_t n = 0;
        if (!read_pod(f, n) || n < 0) {
            return "a tensor length is malformed";
        }
        if (name != params[i].name
            || static_cast<size_t>(n) != params[i].data->size()) {
            return "its tensors do not line up with this model's";
        }
        staged[i].resize(static_cast<size_t>(n));
        if (n > 0 && !f.read(reinterpret_cast<char*>(staged[i].data()),
            static_cast<std::streamsize>(n * static_cast<std::int64_t>(sizeof(real))))) {
            return "it is truncated";
        }
    }

    for (size_t i = 0; i < params.size(); ++i) {
        *params[i].data = staged[i];
    }
    return nullptr;
}

// Loads a checkpoint into a model that training built. Training needs the
// corpus anyway, so the fingerprint is checked against it: a model must not
// resume on text it was not trained on.
inline bool load_params(std::vector<Param>& params, const Config& cfg, const Data& d,
    const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;            // no checkpoint, so the caller trains
    }

    auto reject = [&path](const char* why) {
        printf("  ignoring %s: %s\n", path.c_str(), why);
        return false;
    };

    CheckpointHeader h;
    if (const char* why = read_checkpoint_header(f, h)) {
        return reject(why);
    }
    if (h.fingerprint != corpus_fingerprint(d)) {
        return reject("it was trained on a different corpus");
    }
    if (h.vocab.vocab != d.vocab || h.cfg.n_embd != cfg.n_embd
        || h.cfg.n_head != cfg.n_head || h.cfg.n_layer != cfg.n_layer
        || h.cfg.block != cfg.block) {
        return reject("it was written for a differently shaped model");
    }
    if (const char* why = read_tensors(f, h.tensors, params)) {
        return reject(why);
    }
    return true;
}

// Builds a model from a checkpoint alone, without the corpus. The shape and
// the vocabulary come from the header, so this opens any checkpoint that
// save_params wrote. gpt_chat.cpp uses it. The fingerprint is not checked,
// because there is no corpus to compare it with. The token ids keep their
// meaning because the vocabulary is the one saved with the weights.
//
// Returns an empty pointer, after a report, if the file cannot be used.
// Otherwise vocab receives the vocabulary from the file.
inline std::unique_ptr<Gpt> load_model(const std::string& path, Vocab& vocab) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        printf("  could not open %s\n", path.c_str());
        return std::unique_ptr<Gpt>();
    }

    CheckpointHeader h;
    const char* why = read_checkpoint_header(f, h);
    if (why == nullptr) {
        // The file overwrites the random initial weights, so this seed has no
        // effect.
        std::mt19937 unused(1337);
        std::unique_ptr<Gpt> model(new Gpt(h.cfg, h.vocab.vocab, unused));
        why = read_tensors(f, h.tensors, model->params());
        if (why == nullptr) {
            vocab = h.vocab;
            return model;
        }
    }
    printf("  ignoring %s: %s\n", path.c_str(), why);
    return std::unique_ptr<Gpt>();
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

constexpr int kEvalIters = 200;      // the video's eval_iters
constexpr int kEvalSeed = 1337;      // the video's torch.manual_seed(1337)

struct Losses {
    real train;
    real val;
};

// Python:
//   @torch.no_grad()
//   def estimate_loss():
//       out = {}
//       model.eval()
//       for split in ['train', 'val']:
//           losses = torch.zeros(eval_iters)
//           for k in range(eval_iters):
//               X, Y = get_batch(split)
//               logits, loss = model(X, Y)
//               losses[k] = loss.item()
//           out[split] = losses.mean()
//       model.train()
//       return out
//
// The loss is averaged over kEvalIters random batches, not computed over the
// whole split. The loss of one minibatch is far too noisy to show a trend, and
// the whole split is 1.1 million positions. The seed is fixed, so that two
// models are scored on the same windows.
template <class Model>
Losses estimate_loss(Model& model, const Data& d, int B, int T) {
    model.set_training(false);

    std::mt19937 g(kEvalSeed);
    std::vector<int> x, y;
    Losses out{ real(0), real(0) };
    const std::vector<int>* splits[] = { &d.train, &d.val };
    real* into[] = { &out.train, &out.val };

    for (int s = 0; s < 2; ++s) {
        real total = 0;
        for (int k = 0; k < kEvalIters; ++k) {
            sample_batch(*splits[s], B, T, x, y, g);
            total += model.forward_loss(x.data(), y.data(), B, T);
        }
        *into[s] = total / kEvalIters;
    }

    model.set_training(true);
    return out;
}

struct TrainResult {
    real first_loss = real(0);
    real train_loss = real(0);
    real val_loss = real(0);
    real seconds = real(0);
};

// Python:
//   optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
//   for iter in range(max_iters):
//       if iter % eval_interval == 0:
//           losses = estimate_loss()
//           print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
//       xb, yb = get_batch('train')
//       logits, loss = model(xb, yb)
//       optimizer.zero_grad(set_to_none=True)
//       loss.backward()
//       optimizer.step()
template <class Model>
TrainResult train(Model& model, const Data& d, int B, int T, real lr, int iters,
    int report_every, std::mt19937& g) {
    CHECK(iters > 0, "asked for " << iters << " training steps");

    AdamW opt(model.params(), lr);
    std::vector<int> x, y;
    TrainResult r;
    const auto t0 = std::chrono::steady_clock::now();

    for (int it = 0; it < iters; ++it) {
        sample_batch(d.train, B, T, x, y, g);
        model.zero_grad();
        const real loss = model.forward_backward(x.data(), y.data(), B, T);
        opt.step();

        if (it == 0) {
            r.first_loss = loss;
        }
        if (report_every > 0 && (it + 1) % report_every == 0) {
            const Losses l = estimate_loss(model, d, B, T);
            printf("    step %6d  train %.4f  val %.4f  (gap %+.4f)\n",
                it + 1, l.train, l.val, l.val - l.train);
        }
    }

    r.seconds = std::chrono::duration<real>(
        std::chrono::steady_clock::now() - t0).count();
    const Losses l = estimate_loss(model, d, B, T);
    r.train_loss = l.train;
    r.val_loss = l.val;
    return r;
}

// ---------------------------------------------------------------------------
// The two configurations the video trains
// ---------------------------------------------------------------------------

// The small configuration, which is the one gpt.cpp trains. The video reaches
// it about two thirds of the way through, and gets a validation loss just
// under 2.0 with it before scaling up.
constexpr int kEmbed = 32;
constexpr int kHeads = 4;
constexpr int kLayers = 3;
constexpr int kBlock = 8;
constexpr int kBatch = 32;
constexpr real kDropout = real(0);
constexpr real kLr = real(1e-3);

constexpr int kIters = 5000;         // the video's max_iters
constexpr int kStageIters = 3000;    // intermediate models, sections 2-4
constexpr int kReportEvery = 1000;

// The scale-up at the end of the video: 10.8M parameters, validation loss
// 1.48, about fifteen minutes on an A100. gpt.cpp only records it: the full
// schedule is about 6 PFLOP, which is too much for the threaded scalar loops
// in this file. gpt_blocked.cpp measures what Kernels::kBlocked gains at this
// width. VariousMlp/GptScaled/gpt_scaled.cpp trains this configuration, with
// the CUDA model and the blocked multiply sharing each batch.
constexpr int kBigEmbed = 384;
constexpr int kBigHeads = 6;
constexpr int kBigLayers = 6;
constexpr int kBigBlock = 256;
constexpr int kBigBatch = 64;
constexpr real kBigDropout = real(0.2);
constexpr real kBigLr = real(3e-4);

}  // namespace gpt
