// gpt_model.h - the Transformer itself, shared by the two files that use it.
//
// gpt.cpp is the walkthrough: it builds the model up the way the video does,
// trains it, and checks the numbers. gpt_chat.cpp is a tool: it loads a
// checkpoint the walkthrough saved and lets you type at it. They need the same
// model, and a tool is not a video port, so the convention that each of these
// files is read start to finish on its own applies to gpt.cpp and not here.
//
// Everything a reader would want to read is still in order below - the data,
// the layers, attention, the blocks, the optimiser, the checkpoint format -
// and the commentary came with it. What stayed in gpt.cpp is only what the
// walkthrough itself needs: the bigram baseline, the two demonstrations, the
// cost model, and the numbers the video reads out.

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

// Neither MSVC nor gcc will vectorise a loop that stores through one pointer
// and loads through another unless it can prove the two do not overlap, and
// const says nothing about that. Every use below is on pointers into distinct
// vectors; the comment at each one says which.
#if defined(_MSC_VER) || defined(__GNUC__)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

// The loops below are threaded with OpenMP. Nothing here is guarded by an
// #ifdef, because a compiler without it ignores the pragmas and every loop runs
// exactly as it did when it was serial - which is also why the shape of each
// loop had to survive being parallelised rather than being rewritten around the
// threads.
//
// The project passes /openmp:llvm, whose runtime has the better scheduler and
// the cheaper barrier. Nothing below needs it though: no collapse, no
// reductions, no array sections, and every loop variable is a signed int, so
// this is all inside OpenMP 2.0 and plain /openmp builds it too. That matters
// because /openmp:llvm links libomp140.x86_64.dll while /openmp links the
// vcomp140.dll that every machine with the VC redistributable already has.
#ifdef _OPENMP
#include <omp.h>
#endif

// A named namespace rather than an anonymous one, because two translation units
// have to agree on these types. Free functions below are inline for the same
// reason.
namespace gpt {

// The scalar type, named once. Pascal consumer GPUs and every SIMD unit run
// float at twice the rate of double, so the performance pass flips this - but
// not before it has to: gradient_check() leans on central differences being
// good to about 1e-9, which is a property of double and not of float.
using real = double;

// Inside a parallel region these describe the team. Outside one - and in a
// build with no OpenMP at all - they describe the single thread that is
// running, which is what makes the serial path fall out of the same code.
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

// What a large enough loop would actually get, which is what the cost model in
// gpt.cpp wants to print alongside its throughput number.
inline int available_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Handing a loop to a team of threads costs a few microseconds, so a loop needs
// more than a few microseconds of work in it to come out ahead. Three shapes
// occur below and they need different thresholds:
//
//   - a loop over the B*T rows of a tensor, where one iteration is a whole
//     matrix row of arithmetic. A few dozen rows are already worth splitting.
//   - a flat elementwise loop, where one iteration is a single multiply. These
//     are memory-bound and need thousands of iterations before the fork pays
//     for itself.
//   - attention, counted in (batch, head, query, key) entries because its
//     parallelism is over the heads while its work is over the square.
//
// The numbers are deliberately cautious, and at the video's small size several
// loops fall under them and stay serial. That is the right answer rather than a
// missed opportunity: 8192 elements split sixteen ways is 512 doubles a thread,
// less work than the fork that handed them out. They are also the first thing
// to tune if a profile disagrees.
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
// Unlike the makemore files there is no fixed alphabet to do arithmetic on, so
// the two maps get built for real. 256 entries is a table rather than a hash
// map because the corpus is ASCII.
// Any plain text file works: the vocabulary is whichever distinct bytes turn up
// in it. Point this at War and Peace and the model trains on War and Peace. The
// two things to know are that every sanity::Expect below is pinned to tiny
// Shakespeare and will report a mismatch for anything else (which is what
// Expect is for), and that the tokens are bytes rather than characters - so a
// UTF-8 corpus in a non-Latin script gives the model half a letter at a time.
constexpr const char* kCorpus = "input.txt";

struct Data {
    std::vector<int> train;
    std::vector<int> val;
    std::vector<char> itos;
    int stoi[256];
    int vocab = 0;

    // The directory input.txt came from. Checkpoints go there too, so that a
    // run from Visual Studio and a run of the bare exe - which have different
    // working directories - find each other's work.
    std::string dir;

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

// The vocabulary can hold any byte, so anything outside printable ASCII is
// shown as an escape rather than sent to the console as-is.
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
    // Relative to a few plausible working directories: Visual Studio runs with
    // cwd = $(ProjectDir), a bare exe run has cwd = x64\Debug, and running from
    // the repo root is the other common case.
    static const char* kDirs[] = {
        "Karpathy/",
        "",
        "VariousMlp/Karpathy/",
        "../../VariousMlp/Karpathy/",
    };

    std::string text;
    for (const char* dir : kDirs) {
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

    // A checkout that rewrote the line endings would change the vocabulary and
    // every loss below it, so the '\r' goes rather than becoming a 66th token.
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());

    for (int& s : d.stoi) {
        s = -1;
    }
    std::vector<bool> present(256, false);
    for (char c : text) {
        present[static_cast<unsigned char>(c)] = true;
    }
    for (int c = 0; c < 256; ++c) {
        if (present[static_cast<size_t>(c)]) {
            d.stoi[c] = d.vocab++;
            d.itos.push_back(static_cast<char>(c));
        }
    }

    // Python: data = torch.tensor(encode(text), dtype=torch.long)
    //         n = int(0.9 * len(data))
    //         train_data = data[:n]
    //         val_data = data[n:]
    //
    // Split by position rather than at random: the point of the validation set
    // is text the model has not been trained on, and consecutive characters of
    // Shakespeare are anything but independent.
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
// means position t only ever saw positions 0..t - the reason a Transformer gets
// so much more out of a batch than the MLP did.
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

// Every activation in a Transformer is (B, T, C) - batch, position, channel -
// so the shape travels with the data rather than being passed alongside it.
// Attention is the only thing that cares about the split between B and T;
// everything else treats the tensor as B*T independent rows and never notices.
struct Matrix {
    int batch = 0;
    int time = 0;
    int cols = 0;
    std::vector<real> v;

    void resize(int b, int t, int c) {
        // The flat loops over v below index it with int, because OpenMP wants a
        // signed loop variable and every other count in this file is an int
        // already. That holds for anything this port trains - the video's final
        // model peaks at 25M entries in a tensor - but it is a real ceiling,
        // and one worth failing on rather than wrapping around at.
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

// Softmax over n logits. Subtracting the max first costs nothing and keeps
// exp() away from overflow; softmax is invariant to it.
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

// out() is virtual because the composite layers below - attention, the
// feedforward, a whole block - finish inside a sublayer and would otherwise
// have to copy a (B, T, C) tensor per step just to satisfy the accessor. At
// the video's final size that copy would be 50MB.
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
// The video's _init_weights, which arrives late and quietly: 0.02 everywhere
// rather than the Kaiming 1/sqrt(fan_in) of the previous video. For n_embd=384
// those are 0.02 against 0.051, so this is the same "start the layer smaller
// than you think" instinct, now applied as one number for the whole model.
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

        // A pointer, not a copy: the layer that produced x owns it and does not
        // touch it again until the next forward pass, by which time this
        // layer's backward has already run.
        x_ = &x;
        out_.resize(x.batch, x.time, fan_out_);
        const int n = x.rows();
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

    // Three passes where the serial version had one, because the three
    // gradients this layer produces each want a different loop to be the
    // parallel one.
    //
    // dx is per row and shares nothing. dW and db are sums over every row, so
    // handing rows to threads would have all of them accumulating into the same
    // weight. Giving each thread a private copy of dW to reduce afterwards
    // would cost fan_in*fan_out doubles per thread - 4.7MB apiece at the
    // video's final size - and would change the order the sum is taken in.
    // Splitting dW by input channel costs neither: thread t owns whole rows of
    // dW, so no two threads write the same cache line except at the one
    // boundary between their ranges, and each row is still summed over the
    // examples in ascending order by a single thread. All three passes produce
    // bit for bit what the fused serial loop produced.
    //
    // What the split does cost is that dout is streamed three times instead of
    // once. At the video's small size it sits in L2 throughout and this is
    // free; at the scaled-up size it is real, and it is the same cache problem
    // that blocking has to solve for the multiply itself.
    const Matrix& backward(const Matrix& dout) override {
        const Matrix& x = *x_;
        dx_.resize(x.batch, x.time, fan_in_);
        const int n = x.rows();

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

        // db is a column sum, and it stays serial. It is one pass over dout
        // against the fan_in passes above - a fraction of a percent of the work
        // here - and splitting fan_out doubles across sixteen threads would put
        // several of them on one cache line, which costs more than it saves.
        if (has_bias_) {
            for (int i = 0; i < n; ++i) {
                const real* di = dout.row(i);
                for (int j = 0; j < fan_out_; ++j) {
                    db_[j] += di[j];
                }
            }
        }
        return dx_;
    }

    void collect(std::vector<Param>& out) override {
        out.push_back({ wname_, &w_, &dw_ });
        if (has_bias_) {
            out.push_back({ bname_, &b_, &db_ });
        }
    }

private:
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
// Batch normalisation from the previous video, turned ninety degrees. That one
// normalised a feature across the examples in a batch, which coupled the
// examples together and forced a running mean and variance for evaluation.
// This one normalises a row across its own channels, so every position is
// independent of every other, there is nothing to remember between batches, and
// train and evaluation are the same code. For a model whose whole point is
// variable-length sequences that difference is decisive.
//
// torch.var is unbiased by default but nn.LayerNorm divides by C, not C-1,
// which is why there is no Bessel factor in the backward pass here and there
// was one in BatchNorm1d.
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
        // A row normalises against its own channels and nothing else, which is
        // the property that made LayerNorm the right choice here in the first
        // place and now makes it trivially parallel.
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
    // The two subtracted terms are the mean and the variance pulling back: a
    // row cannot be shifted or scaled by the gradient without the
    // normalisation undoing it, so only the shape of the change survives.
    const Matrix& backward(const Matrix& dout) override {
        const int n = dout.rows();
        dx_.resize(dout.batch, dout.time, dim_);

        const real inv_c = real(1) / dim_;
        // dx first, one row per thread. The two parameter gradients come out of
        // the same loop in the serial version, but they are sums over every
        // row, so they have to leave it.
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

        // And the parameter gradients serially. Unlike the Linear layer above
        // there is no second axis to split these along - there are dim_ of them
        // and threads would land several to a cache line - and the arithmetic
        // is a few tenths of a percent of what the Linear layers either side of
        // this one are doing. Keeping it serial also keeps the summation order
        // the one the serial build used, which is what makes the loss curve
        // comparable across the two.
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
// Inverted dropout: the surviving activations are divided by 1-p during
// training so that evaluation, where nothing is dropped, needs no correction.
// The video reaches for it only at the scale-up, where the model finally has
// enough parameters to memorise its way through the training set.
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
        // Serial, and not for want of trying: the mask is drawn from one
        // generator, so threads would both race on its state and draw a
        // different mask on every run. A reproducible parallel dropout needs a
        // counter-based generator keyed by element index, which is a different
        // piece of work from this one.
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

    // Disabled dropout returns its input untouched rather than copying it, so
    // the base class's out_ would be empty. This keeps out() honest.
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
// The video writes one Head and then a list of them; this runs all n_head at
// once out of a single key/query/value projection each, which is the same
// arithmetic with the concatenation done up front instead of at the end. Head h
// owns channels [h*hs, (h+1)*hs) of every one of them.
//
// Three things are worth naming because they are easy to lose in the tensor
// expression above:
//
//   - the mask is not an optimisation, it is the training signal. Position t
//     may only attend to 0..t, which is what makes all T positions of a window
//     T separate examples rather than one.
//   - the 1/sqrt(head_size) is there so the scores arrive at softmax with unit
//     variance. Without it a wide head produces large dot products, softmax
//     saturates toward one-hot, and the gradient through it vanishes - the
//     same failure the previous video diagnosed in tanh.
//   - the masked entries are stored as zero probability rather than -inf, so
//     the backward pass can loop over u <= t and ignore the rest.
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

        // One row of att_ per (batch, head, query position), holding that
        // position's distribution over the keys it is allowed to see.
        //
        // The batch and head loops are flattened into one so that OpenMP has a
        // single index to hand out. (batch, head) is the axis to split on - not
        // the query position - because the backward pass accumulates down the
        // key axis and so cannot be split by t, and the two passes have to
        // agree. Every (batch, head) does identical work, so a static schedule
        // balances the team exactly.
        const int bh_count = B * n_head_;
        const long long attn_work = static_cast<long long>(bh_count) * T * T;

        att_.resize(bh_count, T, T);
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
                    a[u] = real(0);        // the masked future
                }
            }
        }

        const Matrix& a = attn_drop_.forward(att_);

        // Head h writes channels [h*hs, (h+1)*hs) of heads_ and no others, so
        // the threads never meet even though they share the tensor.
        heads_.resize(B, T, n_embd_);
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

        // out[b,t,h] = sum_{u<=t} a[b,h,t,u] * v[b,u,h], so the weights pick up
        // a dot product and the values pick up a weighted sum down the column.
        //
        // That column sum is why the query position cannot be the parallel
        // axis: every t from u upwards adds into dv at u. Splitting by
        // (batch, head) leaves it alone - thread bh owns batch b's channels
        // [h*hs, (h+1)*hs) of dv outright, and still walks t in order, so the
        // sums come out in the order the serial version produced them.
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

        const Matrix& dp = attn_drop_.backward(da_);

        // Softmax over a row: ds_u = p_u * (dp_u - sum_w p_w dp_w). The
        // subtracted term is the constraint that the row sums to one, so
        // pushing one weight up has to pull the others down.
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

        // dk accumulates down the key axis the way dv did, so this splits by
        // (batch, head) for the same reason.
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

        // x fed all three projections, so all three gradients come back to it.
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
// Attention is where the positions talk to each other; this is where each
// position thinks about what it heard, on its own. The video adds it after
// noticing that the network goes straight from attending to predicting with no
// computation in between. The 4x is from the original paper.
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
// The two additions are the reason a stack of these trains at all. A residual
// connection gives the gradient a path from the loss to every layer that does
// not pass through any weights, so the signal that reaches block 0 is not the
// product of five Jacobians. The video demonstrates the failure first - blocks
// without residuals make the model worse, not better - and then fixes it.
//
// The normalisation is applied before each sublayer rather than after. The 2017
// paper does it the other way round; the video mentions that pre-norm is the
// arrangement everyone settled on, because it keeps the residual path itself
// free of any normalisation.
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

    // Each residual add is a fan-out in the forward pass, so it is a sum in the
    // backward pass: the gradient arriving at `out` reaches `mid` both through
    // the feedforward branch and directly along the skip.
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

// Cross entropy over B*T independent predictions. Identical to the makemore
// files apart from the vocabulary being a runtime number.
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

    // Every row scores on its own, but two things here are shared: the softmax
    // needs a scratch row of probabilities, and the loss is a sum over all the
    // rows. The scratch buffer is declared inside the parallel region so each
    // thread allocates its own, and the per-row losses go to an array that is
    // added up afterwards in order - so the number this returns is the same
    // number whatever size the team was.
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
// The position embedding is what tells attention where things are. Without it
// the model is a bag of characters: the dot product between a query and a key
// does not care which of them came first, so "abc" and "cba" would produce the
// same attention pattern. A learned vector per slot is the cheapest possible
// fix, and it is why block_size is a hard ceiling on the context - there is no
// row in the table for position 257.
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

    // params_ holds the addresses of wte_ and wpe_, so the object has to stay
    // where it is.
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
        // One tensor per iteration, dynamically scheduled: these differ in size
        // by three orders of magnitude, and a static split would hand one
        // thread the token embedding table and another two bias vectors.
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

        // Flattened over (batch, position), which is the same walk the nested
        // loops made: row i of the tensor is window i/T at slot i%T.
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

        // The two embedding tables scatter their gradients back by index, and
        // += is load-bearing: a character that appears eight times in a batch
        // contributes eight times, and every one of the B windows uses the same
        // row of the position table for slot t.
        //
        // Which is also why this is the one loop in the model that stays
        // serial. Those repeats are collisions: two positions holding the same
        // character, or the same slot in different windows, want the same row
        // at the same time, and there is no axis to split that does not have
        // them. It is one pass over (B, T, C) against the matrix multiplies on
        // either side of it, so it is not where the time is anyway.
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
    // Cropping to the last block_size is not a nicety: position block_size has
    // no row in the position table. It also means the whole prefix is
    // recomputed for every character, which is why real implementations cache
    // the keys and values instead.
    // `prompt` seeds the context; an empty one starts from token 0, which is
    // the lowest byte in the corpus and so, for any text file containing a
    // newline, a blank line - as neutral a thing to start from as this
    // vocabulary has.
    //
    // Temperature divides the logits before the softmax. Below 1 it sharpens
    // the distribution toward the characters the model is most sure of, above
    // 1 it flattens it toward uniform. The video samples straight from the
    // model and never mentions the knob; at character level it is most of the
    // difference between words and letter soup, so the tool in gpt_chat.cpp
    // wants it even though the walkthrough does not.
    std::vector<int> generate(const std::vector<int>& prompt, int count,
        real temperature, std::mt19937& g) {
        CHECK(temperature > real(0), "temperature " << temperature
            << " has to be positive");

        // Dropout has to be off, or the samples are drawn from a different
        // model than the one that was trained.
        set_training(false);

        std::vector<int> ctx = prompt.empty() ? std::vector<int>(1, 0) : prompt;
        std::vector<int> ids;
        std::vector<real> probs(static_cast<size_t>(vocab_));
        std::vector<real> scaled(static_cast<size_t>(vocab_));
        for (int i = 0; i < count; ++i) {
            const int T = static_cast<int>(std::min<size_t>(
                static_cast<size_t>(cfg_.block), ctx.size()));
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

    // What the walkthrough uses: start from a blank line, sample straight from
    // the model the way the video does.
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
// That does not survive a model whose parameters have wildly different gradient
// scales - an embedding row touched once per batch against a weight matrix
// touched B*T times. Adam divides each parameter's step by its own recent
// gradient magnitude, which makes one learning rate work for all of them.
//
// The W is decoupled weight decay: torch's AdamW shrinks the parameter toward
// zero as a separate step rather than by adding a term to the gradient, so the
// decay is not itself divided by sqrt(v). Default weight_decay is 0.01 and the
// video does not override it, so 0.01 is what actually ran.
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
        // Both moments start at zero, so early steps are biased toward it.
        // Dividing by 1-beta^t undoes that exactly.
        const real bc1 = real(1) - std::pow(beta1_, static_cast<real>(t_));
        const real bc2 = real(1) - std::pow(beta2_, static_cast<real>(t_));

        // Every parameter steps independently of every other, so the only
        // question is which axis to hand out. Splitting the tensors rather than
        // the elements within them costs one fork per step instead of one per
        // tensor, and dynamic scheduling covers their very different sizes.
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

// Nine hours of training is not something to repeat because the program was
// restarted. The format is deliberately self-describing: the header carries the
// shape of the model and the size of its scalar, and every tensor carries its
// own name and length, so a checkpoint belonging to a differently shaped model
// is rejected rather than reinterpreted as noise.
//
// A missing or stale checkpoint is not an error - it means "train instead" - so
// these report and return false rather than using CHECK. That is a third
// severity beyond the two in check.h: not a broken program, not a number that
// drifted, just a file that is not there or no longer fits.
//
// The bytes are raw doubles in native order. Both machines this runs on are
// x86-64, so endianness is ignored; the sizeof(real) in the header and in the
// filename is what stops a float build from reading a double build's file.

constexpr char kCheckpointMagic[8] = { 'V', 'M', 'L', 'P', 'G', 'P', 'T', '2' };

// FNV-1a over the vocabulary and the corpus length. The header already checks
// the shape of the model, but shape is not identity: two corpora with the same
// number of distinct characters would produce checkpoints with the same name
// and a header that validates, and loading one onto the other gives a model
// whose token 37 stands for a different letter. Nothing about that failure
// looks like a failure - the losses are simply bad - so it is worth eight bytes
// to make it impossible.
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

    char magic[sizeof(kCheckpointMagic)];
    if (!f.read(magic, sizeof(magic))
        || std::memcmp(magic, kCheckpointMagic, sizeof(magic)) != 0) {
        return reject("not a checkpoint written by this file");
    }

    std::int32_t scalar = 0, v = 0, embd = 0, head = 0, layer = 0, block = 0;
    std::int32_t count = 0;
    std::uint64_t fingerprint = 0;
    if (!read_pod(f, scalar) || !read_pod(f, fingerprint) || !read_pod(f, v)
        || !read_pod(f, embd) || !read_pod(f, head) || !read_pod(f, layer)
        || !read_pod(f, block) || !read_pod(f, count)) {
        return reject("the header is truncated");
    }
    if (scalar != static_cast<std::int32_t>(sizeof(real))) {
        return reject("it was written with a different scalar type");
    }
    if (fingerprint != corpus_fingerprint(d)) {
        return reject("it was trained on a different corpus");
    }
    if (v != d.vocab || embd != cfg.n_embd || head != cfg.n_head
        || layer != cfg.n_layer || block != cfg.block) {
        return reject("it was written for a differently shaped model");
    }
    if (count != static_cast<std::int32_t>(params.size())) {
        return reject("it holds a different number of tensors");
    }

    // Read into a staging buffer first. A file that turns out to be truncated
    // halfway through should leave the model as it was rather than half
    // overwritten with one model's weights and half with another's. The cost is
    // holding the parameters twice, which for the scaled-up model is 86MB.
    std::vector<std::vector<real>> staged(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        std::int32_t len = 0;
        if (!read_pod(f, len) || len < 0 || len > 256) {
            return reject("a tensor name is malformed");
        }
        std::string name(static_cast<size_t>(len), '\0');
        if (len > 0 && !f.read(&name[0], len)) {
            return reject("a tensor name is truncated");
        }
        std::int64_t n = 0;
        if (!read_pod(f, n) || n < 0) {
            return reject("a tensor length is malformed");
        }
        if (name != params[i].name
            || static_cast<size_t>(n) != params[i].data->size()) {
            return reject("its tensors do not line up with this model's");
        }
        staged[i].resize(static_cast<size_t>(n));
        if (n > 0 && !f.read(reinterpret_cast<char*>(staged[i].data()),
            static_cast<std::streamsize>(n * static_cast<std::int64_t>(sizeof(real))))) {
            return reject("it is truncated");
        }
    }

    for (size_t i = 0; i < params.size(); ++i) {
        *params[i].data = staged[i];
    }
    return true;
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
// Averaged over kEvalIters random batches rather than computed over the whole
// split: one minibatch loss is far too noisy to read a trend from, and the
// whole split is 1.1 million positions. The seed is fixed so that two models
// are scored on the same windows.
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

// The small one, which is what actually runs here. The video reaches it about
// two thirds of the way through and reads a validation loss just under 2.0 off
// it before scaling up.
constexpr int kEmbed = 32;
constexpr int kHeads = 4;
constexpr int kLayers = 3;
constexpr int kBlock = 8;
constexpr int kBatch = 32;
constexpr real kDropout = real(0);
constexpr real kLr = real(1e-3);

constexpr int kIters = 5000;         // the video's max_iters
constexpr int kStageIters = 3000;    // for the intermediate models in 2-4
constexpr int kReportEvery = 1000;

// The scale-up at the end of the video: 10.8M parameters, validation loss 1.48,
// about fifteen minutes on an A100. It is recorded rather than run - at 6
// PFLOP for the full schedule it needs the scalar loops in this file blocked
// and threaded first, which is a separate change.
constexpr int kBigEmbed = 384;
constexpr int kBigHeads = 6;
constexpr int kBigLayers = 6;
constexpr int kBigBlock = 256;
constexpr int kBigBatch = 64;
constexpr real kBigDropout = real(0.2);
constexpr real kBigLr = real(3e-4);

}  // namespace gpt
