// gpt.cpp - a C++ port of Andrej Karpathy's "Let's build GPT: from scratch, in
// code, spelled out" (video 7 of Neural Networks: Zero to Hero)
//
// The makemore files all answer the same question - how do you use more than
// one character of context - and all answer it the same way: staple the context
// together and push it through a fixed-width layer. That runs out of room fast.
// The bigram table needs 27^2 entries for two characters and 27^4 for four; the
// MLP fixes the parameter count but still smears the whole context into one
// vector before the network has looked at any of it.
//
// A Transformer lets the tokens look at each other first. Every position emits
// a query ("what am I looking for"), a key ("what do I have") and a value
// ("what I will tell you if you ask"), and the strength of the edge from
// position u to position t is the dot product of t's query with u's key. That
// is the whole idea; the rest of the file is bookkeeping around it.
//
// What the video builds, in order:
//   1. a bigram baseline on tiny Shakespeare, so there is a number to beat
//   2. the trick underneath attention: a running average over the past,
//      written first as a loop, then as a matrix multiply, then as a softmax
//   3. one head of self-attention
//   4. several heads in parallel, plus a per-position feedforward layer
//   5. residual connections, layer normalisation and dropout, which are what
//      let the thing be stacked
//
// Needs input.txt (tiny Shakespeare, 1115394 characters) next to this file.
//
// Nothing here uses an autograd engine, so every backward pass is written out:
// causal masked softmax, the three matmuls of attention, layer normalisation,
// the residual splits. gradient_check() below compares the lot against central
// differences, and is the reason to trust any of it.
//
// Runtime, at the video's small configuration: a few minutes in Release. The
// scaled-up configuration at the end of the video is four orders of magnitude
// more arithmetic and is not reachable until the scalar loops here are blocked
// and threaded - see kBig* below, which records it without running it.

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

namespace {

// The scalar type, named once. Pascal consumer GPUs and every SIMD unit run
// float at twice the rate of double, so the performance pass flips this - but
// not before it has to: gradient_check() leans on central differences being
// good to about 1e-9, which is a property of double and not of float.
using real = double;

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
std::string printable(char c) {
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

bool load_corpus(Data& d) {
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
void sample_batch(const std::vector<int>& data, int B, int T,
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
void softmax_row(const real* logits, real* probs, int n) {
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

    const Matrix& backward(const Matrix& dout) override {
        const Matrix& x = *x_;
        dx_.resize(x.batch, x.time, fan_in_);
        const int n = x.rows();
        for (int i = 0; i < n; ++i) {
            const real* RESTRICT di = dout.row(i);
            const real* RESTRICT xi = x.row(i);
            real* RESTRICT dxi = dx_.row(i);
            if (has_bias_) {
                for (int j = 0; j < fan_out_; ++j) {
                    db_[j] += di[j];
                }
            }
            for (int k = 0; k < fan_in_; ++k) {
                const real* RESTRICT wk = &w_[static_cast<size_t>(k) * fan_out_];
                real* RESTRICT dwk = &dw_[static_cast<size_t>(k) * fan_out_];
                const real xk = xi[k];
                real acc = 0;
                for (int j = 0; j < fan_out_; ++j) {
                    dwk[j] += xk * di[j];
                    acc += wk[j] * di[j];
                }
                dxi[k] = acc;
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
        for (int i = 0; i < n; ++i) {
            const real* RESTRICT di = dout.row(i);
            const real* RESTRICT xh = xhat_.row(i);
            real* RESTRICT dxi = dx_.row(i);

            real sum_d = 0;
            real sum_dx = 0;
            for (int j = 0; j < dim_; ++j) {
                const real dxhat = di[j] * gamma_[j];
                dgamma_[j] += di[j] * xh[j];
                dbeta_[j] += di[j];
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
        for (size_t k = 0; k < x.v.size(); ++k) {
            out_.v[k] = x.v[k] > real(0) ? x.v[k] : real(0);
        }
        return out_;
    }

    const Matrix& backward(const Matrix& dout) override {
        dx_.resize(out_.batch, out_.time, out_.cols);
        for (size_t k = 0; k < dout.v.size(); ++k) {
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
        for (size_t k = 0; k < dout.v.size(); ++k) {
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
        att_.resize(B * n_head_, T, T);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < n_head_; ++h) {
                const int bh = b * n_head_ + h;
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
        }

        const Matrix& a = attn_drop_.forward(att_);

        heads_.resize(B, T, n_embd_);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < n_head_; ++h) {
                const int bh = b * n_head_ + h;
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

        da_.resize(B * n_head_, T, T);
        dv_.resize(B, T, n_embd_);
        dq_.resize(B, T, n_embd_);
        dk_.resize(B, T, n_embd_);

        // out[b,t,h] = sum_{u<=t} a[b,h,t,u] * v[b,u,h], so the weights pick up
        // a dot product and the values pick up a weighted sum down the column.
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < n_head_; ++h) {
                const int bh = b * n_head_ + h;
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
        // subtracted term is the constraint that the row sums to one, so
        // pushing one weight up has to pull the others down.
        ds_.resize(B * n_head_, T, T);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < n_head_; ++h) {
                const int bh = b * n_head_ + h;
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
        }

        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < n_head_; ++h) {
                const int bh = b * n_head_ + h;
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

        // x fed all three projections, so all three gradients come back to it.
        const Matrix& dxq = query_.backward(dq_);
        const Matrix& dxk = key_.backward(dk_);
        const Matrix& dxv = value_.backward(dv_);

        dx_.resize(B, T, n_embd_);
        for (size_t i = 0; i < dx_.v.size(); ++i) {
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
        const Matrix& b = attn_.forward(ln1_.forward(x));
        mid_.resize(x.batch, x.time, x.cols);
        for (size_t i = 0; i < mid_.v.size(); ++i) {
            mid_.v[i] = x.v[i] + b.v[i];
        }

        const Matrix& d = ffwd_.forward(ln2_.forward(mid_));
        out_.resize(x.batch, x.time, x.cols);
        for (size_t i = 0; i < out_.v.size(); ++i) {
            out_.v[i] = mid_.v[i] + d.v[i];
        }
        return out_;
    }

    // Each residual add is a fan-out in the forward pass, so it is a sum in the
    // backward pass: the gradient arriving at `out` reaches `mid` both through
    // the feedforward branch and directly along the skip.
    const Matrix& backward(const Matrix& dout) override {
        const Matrix& dmid_branch = ln2_.backward(ffwd_.backward(dout));
        dmid_.resize(dout.batch, dout.time, dout.cols);
        for (size_t i = 0; i < dmid_.v.size(); ++i) {
            dmid_.v[i] = dout.v[i] + dmid_branch.v[i];
        }

        const Matrix& dx_branch = ln1_.backward(attn_.backward(dmid_));
        dx_.resize(dout.batch, dout.time, dout.cols);
        for (size_t i = 0; i < dx_.v.size(); ++i) {
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
real cross_entropy(const Matrix& logits, const int* targets, Matrix* dlogits) {
    const int n = logits.rows();
    const int vocab = logits.cols;
    CHECK(n > 0, "no predictions to score");

    if (dlogits) {
        dlogits->resize(logits.batch, logits.time, vocab);
    }

    const real inv_n = real(1) / n;
    real total = 0;
    std::vector<real> probs(static_cast<size_t>(vocab));
    for (int i = 0; i < n; ++i) {
        softmax_row(logits.row(i), probs.data(), vocab);
        const int y = targets[i];
        total += -std::log(probs[static_cast<size_t>(y)]);
        if (dlogits) {
            real* dl = dlogits->row(i);
            for (int j = 0; j < vocab; ++j) {
                dl[j] = (probs[static_cast<size_t>(j)] - (j == y ? real(1) : real(0))) * inv_n;
            }
        }
    }
    return total * inv_n;
}

// Python:
//   class BigramLanguageModel(nn.Module):
//       def __init__(self, vocab_size):
//           self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
//       def forward(self, idx, targets=None):
//           logits = self.token_embedding_table(idx)
//
// The video's starting point, and makemore part 1's neural net with a 65-token
// alphabet: the embedding table *is* the logits, so the model cannot see
// further back than one character however long the context window is.
class Bigram {
public:
    Bigram(int vocab, std::mt19937& g)
        : vocab_(vocab),
        table_(static_cast<size_t>(vocab) * vocab, real(0)),
        dtable_(table_.size(), real(0))
    {
        std::normal_distribution<real> randn(real(0), kInitStd);
        for (real& v : table_) {
            v = randn(g);
        }
        params_.push_back({ "token table " + std::to_string(vocab) + "x"
            + std::to_string(vocab), &table_, &dtable_ });
    }

    Bigram(const Bigram&) = delete;
    Bigram& operator=(const Bigram&) = delete;

    std::vector<Param>& params() { return params_; }
    size_t parameter_count() const { return table_.size(); }
    void set_training(bool) {}

    void zero_grad() {
        std::fill(dtable_.begin(), dtable_.end(), real(0));
    }

    const Matrix& forward(const int* x, int B, int T) {
        logits_.resize(B, T, vocab_);
        const int n = B * T;
        for (int i = 0; i < n; ++i) {
            const real* r = &table_[static_cast<size_t>(x[i]) * vocab_];
            std::copy(r, r + vocab_, logits_.row(i));
        }
        return logits_;
    }

    real forward_loss(const int* x, const int* y, int B, int T) {
        return cross_entropy(forward(x, B, T), y, nullptr);
    }

    real forward_backward(const int* x, const int* y, int B, int T) {
        const real loss = cross_entropy(forward(x, B, T), y, &dlogits_);
        const int n = B * T;
        for (int i = 0; i < n; ++i) {
            real* d = &dtable_[static_cast<size_t>(x[i]) * vocab_];
            const real* dl = dlogits_.row(i);
            for (int j = 0; j < vocab_; ++j) {
                d[j] += dl[j];
            }
        }
        return loss;
    }

    std::vector<int> generate(int count, std::mt19937& g) {
        std::vector<int> out;
        std::vector<real> probs(static_cast<size_t>(vocab_));
        int cur = 0;
        for (int i = 0; i < count; ++i) {
            softmax_row(&table_[static_cast<size_t>(cur) * vocab_],
                probs.data(), vocab_);
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            cur = dist(g);
            out.push_back(cur);
        }
        return out;
    }

private:
    int vocab_;
    std::vector<real> table_, dtable_;
    std::vector<Param> params_;
    Matrix logits_, dlogits_;
};

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
        for (Param& p : params_) {
            std::fill(p.grad->begin(), p.grad->end(), real(0));
        }
    }

    const Matrix& forward(const int* x, int B, int T) {
        CHECK(T > 0 && T <= cfg_.block, "a window of " << T
            << " positions, but the position table has " << cfg_.block << " rows");

        emb_.resize(B, T, cfg_.n_embd);
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                const real* tok = &wte_[static_cast<size_t>(x[b * T + t]) * cfg_.n_embd];
                const real* pos = &wpe_[static_cast<size_t>(t) * cfg_.n_embd];
                real* e = emb_.row(b, t);
                for (int c = 0; c < cfg_.n_embd; ++c) {
                    e[c] = tok[c] + pos[c];
                }
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
    std::vector<int> generate(int count, std::mt19937& g) {
        // Dropout has to be off, or the samples are drawn from a different
        // model than the one that was trained.
        set_training(false);

        std::vector<int> ids;
        // Token 0 is the lowest byte value in the corpus, which for any text
        // file containing a newline is '\n' - a blank line is as neutral a
        // thing to start from as this vocabulary has.
        std::vector<int> ctx(1, 0);
        std::vector<real> probs(static_cast<size_t>(vocab_));
        for (int i = 0; i < count; ++i) {
            const int T = static_cast<int>(std::min<size_t>(
                static_cast<size_t>(cfg_.block), ctx.size()));
            const int* window = ctx.data() + (ctx.size() - static_cast<size_t>(T));

            const Matrix& logits = forward(window, 1, T);
            softmax_row(logits.row(0, T - 1), probs.data(), vocab_);
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            const int next = dist(g);
            ids.push_back(next);
            ctx.push_back(next);
        }

        set_training(true);
        return ids;
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

        for (size_t k = 0; k < params_->size(); ++k) {
            std::vector<real>& p = *(*params_)[k].data;
            const std::vector<real>& g = *(*params_)[k].grad;
            std::vector<real>& m = m_[k];
            std::vector<real>& v = v_[k];
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
std::uint64_t corpus_fingerprint(const Data& d) {
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
void write_pod(std::ostream& f, const T& v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
bool read_pod(std::istream& f, T& v) {
    return static_cast<bool>(f.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

std::string checkpoint_name(const Config& cfg, const Data& d) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "gpt_c%d_h%d_l%d_t%d_v%d_r%d_%08x.ckpt",
        cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block, d.vocab,
        static_cast<int>(sizeof(real)),
        static_cast<unsigned>(corpus_fingerprint(d) & 0xffffffffu));
    return std::string(buf);
}

bool save_params(const std::vector<Param>& params, const Config& cfg,
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

bool load_params(std::vector<Param>& params, const Config& cfg, const Data& d,
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
// The trick underneath attention
// ---------------------------------------------------------------------------

// The video spends a while on this before showing a single query or key,
// because it is the part that looks like magic in the tensor expression. The
// question is: how does position t get a summary of positions 0..t, in a way
// that is one matrix multiply rather than a loop?
//
// Python:
//   # version 1: an explicit loop
//   for b in range(B):
//       for t in range(T):
//           xprev = x[b, :t+1]
//           xbow[b, t] = torch.mean(xprev, 0)
//
//   # version 2: the same thing as a matrix multiply
//   wei = torch.tril(torch.ones(T, T))
//   wei = wei / wei.sum(1, keepdim=True)
//   xbow2 = wei @ x
//
//   # version 3: the same weights, produced by a softmax
//   tril = torch.tril(torch.ones(T, T))
//   wei = torch.zeros((T, T))
//   wei = wei.masked_fill(tril == 0, float('-inf'))
//   wei = F.softmax(wei, dim=-1)
//   xbow3 = wei @ x
//
// Version 3 is the one that matters. The weights are no longer fixed at 1/(t+1)
// - they come out of a softmax over numbers that start at zero but need not
// stay there. Self-attention is version 3 with those numbers computed from the
// data instead of being zero, and nothing else about the shape changes.
void demonstrate_averaging() {
    constexpr int T = 6;
    constexpr int C = 2;

    std::mt19937 g(1337);
    std::normal_distribution<real> randn(real(0), real(1));
    real x[T][C];
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < C; ++c) {
            x[t][c] = randn(g);
        }
    }

    real loop[T][C] = {};
    for (int t = 0; t < T; ++t) {
        for (int u = 0; u <= t; ++u) {
            for (int c = 0; c < C; ++c) {
                loop[t][c] += x[u][c];
            }
        }
        for (int c = 0; c < C; ++c) {
            loop[t][c] /= (t + 1);
        }
    }

    // Version 3: zeros below the diagonal and -inf above it, through a softmax.
    real wei[T][T];
    for (int t = 0; t < T; ++t) {
        real scores[T];
        for (int u = 0; u < T; ++u) {
            scores[u] = u <= t ? real(0) : -std::numeric_limits<real>::infinity();
        }
        softmax_row(scores, wei[t], t + 1);
        for (int u = t + 1; u < T; ++u) {
            wei[t][u] = real(0);
        }
    }

    real matmul[T][C] = {};
    for (int t = 0; t < T; ++t) {
        for (int u = 0; u <= t; ++u) {
            for (int c = 0; c < C; ++c) {
                matmul[t][c] += wei[t][u] * x[u][c];
            }
        }
    }

    printf("  softmax of a masked row of zeros is a uniform average of the past:\n");
    for (int t = 0; t < T; ++t) {
        printf("    t=%d  weights", t);
        for (int u = 0; u < T; ++u) {
            printf(" %.3f", wei[t][u]);
        }
        printf("   loop (%+.4f, %+.4f)  matmul (%+.4f, %+.4f)\n",
            loop[t][0], loop[t][1], matmul[t][0], matmul[t][1]);
    }

    real worst = 0;
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < C; ++c) {
            worst = std::max(worst, std::fabs(loop[t][c] - matmul[t][c]));
        }
    }
    sanity::Expect chk;
    chk.that("the loop and the masked softmax agree", worst < real(1e-12));
    chk.report();
}

// ---------------------------------------------------------------------------
// Checking every derivative in the file
// ---------------------------------------------------------------------------

// Attention, layer normalisation and the residual splits are all hand-written
// here, and none of them fails loudly when it is wrong - a bad sign flip or a
// dropped term just trains a little worse, which is indistinguishable from a
// bad learning rate. So the whole model gets differentiated numerically on a
// deliberately tiny configuration and compared against what the backward passes
// produced.
//
// Dropout is set to zero for this: its mask is redrawn on every forward pass,
// so two forward passes at x+h and x-h would not be evaluating the same
// function. Its backward is a multiply by the stored mask, which is the one
// derivative in this file that is obvious by inspection.
void gradient_check(const Data& d, std::mt19937& g) {
    constexpr int kCheckBatch = 3;
    constexpr int kCheckTime = 5;
    constexpr int kProbesPerParam = 3;

    // Central differences cancel the leading error terms and spend about half
    // the mantissa doing it, so both the usable step size and the achievable
    // agreement follow the scalar type. double leaves room for a 1e-6 step and
    // lands near 1e-9; float cannot go below about 1e-3 without the subtraction
    // losing every significant digit, and lands two orders of magnitude worse.
    // The check is much weaker in float, but it still catches a wrong sign, a
    // dropped term or a missing Bessel factor, which is what it is for.
    constexpr bool kIsFloat = std::is_same<real, float>::value;
    constexpr real kH = kIsFloat ? real(1e-3) : real(1e-6);
    constexpr real kTolerance = kIsFloat ? real(3e-2) : real(1e-6);

    const Config tiny{ 8, 2, 2, kCheckTime, kCheckBatch, real(0), real(1e-3) };
    Gpt model(tiny, d.vocab, g);
    model.set_training(true);

    std::vector<int> x, y;
    std::mt19937 batch_rng(4242);
    sample_batch(d.train, kCheckBatch, kCheckTime, x, y, batch_rng);

    // gamma starts at exactly 1 and every bias at exactly 0. Those are the
    // values most likely to let a wrong expression agree with a right one by
    // accident, so nothing is left sitting on them.
    std::normal_distribution<real> jitter(real(0), real(0.3));
    for (Param& p : model.params()) {
        for (real& v : *p.data) {
            v += jitter(g);
        }
    }

    model.zero_grad();
    model.forward_backward(x.data(), y.data(), kCheckBatch, kCheckTime);

    std::vector<std::vector<real>> analytic;
    analytic.reserve(model.params().size());
    for (Param& p : model.params()) {
        analytic.push_back(*p.grad);
    }

    std::mt19937 probe_rng(20170612);   // the date on "Attention Is All You Need"
    real worst = 0;
    std::string worst_where;
    int probes = 0;

    for (size_t pi = 0; pi < model.params().size(); ++pi) {
        Param& p = model.params()[pi];
        std::uniform_int_distribution<size_t> pick(0, p.data->size() - 1);
        for (int t = 0; t < kProbesPerParam; ++t) {
            const size_t k = pick(probe_rng);
            const real saved = (*p.data)[k];

            (*p.data)[k] = saved + kH;
            const real up = model.forward_loss(x.data(), y.data(), kCheckBatch, kCheckTime);
            (*p.data)[k] = saved - kH;
            const real down = model.forward_loss(x.data(), y.data(), kCheckBatch, kCheckTime);
            (*p.data)[k] = saved;

            const real numeric = (up - down) / (real(2) * kH);
            const real got = analytic[pi][k];
            const real scale = std::max(real(1),
                std::max(std::fabs(got), std::fabs(numeric)));
            const real rel = std::fabs(got - numeric) / scale;
            if (rel > worst) {
                worst = rel;
                worst_where = p.name;
            }
            ++probes;
        }
    }

    printf("  %d probes across %zu parameter tensors (%zu parameters)\n",
        probes, model.params().size(), model.parameter_count());
    printf("  worst disagreement %.2e (%s)\n", worst,
        worst_where.empty() ? "none" : worst_where.c_str());

    sanity::Expect chk;
    chk.that("every hand-written gradient matches central differences",
        worst < kTolerance);
    chk.report();
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

// Roughly what one training step costs. The forward pass is two multiply-adds
// per parameter per token and the backward pass is twice that again, which is
// the standard 6N estimate; the two embedding tables come out because they are
// lookups rather than matrix multiplies. Attention's own two matmuls are not
// parameters at all, and at context 256 they stop being a rounding error.
double step_flops(const Config& cfg, size_t params, int vocab) {
    const double lookups = static_cast<double>(vocab + cfg.block) * cfg.n_embd;
    const double matmul = static_cast<double>(params) - lookups;
    const double tokens = static_cast<double>(cfg.batch) * cfg.block;
    const double attention = 6.0 * cfg.block * (cfg.block + 1.0)
        * cfg.n_embd * cfg.batch * cfg.n_layer;
    return 6.0 * matmul * tokens + attention;
}

// The same for a whole training run, including the evaluation passes, which
// are a sixth of the total and would otherwise make the throughput below look
// better than it is.
double run_flops(const Config& cfg, size_t params, int vocab, int iters,
    int report_every) {
    const double per_step = step_flops(cfg, params, vocab);
    const int reports = (report_every > 0 ? iters / report_every : 0) + 1;
    const double evals = static_cast<double>(reports) * kEvalIters * 2;
    return per_step * iters + per_step * evals / 3.0;
}

// Steps timed for the throughput figure in section 8. Enough to get past the
// first-touch page faults and the turbo ramp, few enough to be unnoticeable.
constexpr int kBenchSteps = 200;

// Timed on a throwaway model rather than on the trained one, so the number
// means the same thing whether section 6 trained or loaded a checkpoint - and
// so that measuring never moves weights that matter.
double measure_throughput(const Config& cfg, const Data& d, int vocab, int steps) {
    std::mt19937 g(1337);
    Gpt probe(cfg, vocab, g);
    AdamW opt(probe.params(), cfg.lr);
    std::vector<int> x, y;

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) {
        sample_batch(d.train, cfg.batch, cfg.block, x, y, g);
        probe.zero_grad();
        probe.forward_backward(x.data(), y.data(), cfg.batch, cfg.block);
        opt.step();
    }
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    return step_flops(cfg, probe.parameter_count(), vocab) * steps / secs;
}

// What the layers hold between their forward and backward passes, which is
// what actually decides whether a configuration fits in memory. Each block
// keeps on the order of 32 (B, T, C) tensors once the feedforward's 4x hidden
// width is counted four times over, plus six (B, heads, T, T) attention
// tensors - and every block keeps its own at the same time. Approximate, and
// deliberately on the high side.
double activation_bytes(const Config& cfg) {
    const double per_block = 32.0 * cfg.block * cfg.n_embd
        + 6.0 * cfg.n_head * cfg.block * static_cast<double>(cfg.block);
    return cfg.n_layer * sizeof(real) * cfg.batch * per_block;
}

// Flip this to actually train the video's final model. Section 8 prints what
// that would cost on the machine it is running on first, because at the
// throughput this file currently reaches the answer is measured in weeks.
constexpr bool kRunScaleUp = false;

// The numbers the video reads out, named so the Expect blocks below are claims
// rather than magic constants.
constexpr real kUniformLoss = real(4.1744);   // -log(1/65)
constexpr real kVideoBigram = real(2.48);
constexpr real kVideoSmallTrain = real(1.99);
constexpr real kVideoSmallGpt = real(2.08);   // validation, alongside train 1.99
constexpr real kVideoBigGpt = real(1.48);

}  // namespace

// ---------------------------------------------------------------------------
// The walkthrough
// ---------------------------------------------------------------------------

void test_gpt() {
    std::cout << "== 0. tiny shakespeare ==\n";
    Data d;
    if (!load_corpus(d)) {
        printf("  could not find %s - expected it next to gpt.cpp in Karpathy/\n",
            kCorpus);
        return;
    }
    printf("  vocabulary of %d characters: ", d.vocab);
    for (int i = 0; i < d.vocab; ++i) {
        printf("%s", printable(d.decode(i)).c_str());
    }
    printf("\n  train %zu tokens, val %zu tokens\n", d.train.size(), d.val.size());

    // One batch, spelled out the way the video does before it trusts get_batch:
    // a window of T characters is T separate examples, because position t only
    // ever gets to see 0..t.
    {
        std::mt19937 demo(1337);
        std::vector<int> x, y;
        sample_batch(d.train, 1, kBlock, x, y, demo);
        printf("  one window of %d characters is %d examples:\n", kBlock, kBlock);
        for (int t = 0; t < kBlock; ++t) {
            std::string ctx;
            for (int u = 0; u <= t; ++u) {
                ctx += printable(d.decode(x[static_cast<size_t>(u)]));
            }
            printf("    %-20s ---> %s\n", ctx.c_str(),
                printable(d.decode(y[static_cast<size_t>(t)])).c_str());
        }
    }

    // Pinned to tiny Shakespeare, which means pinned to the file rather than to
    // the program. Point kCorpus at anything else and these report a mismatch
    // and carry on, which is exactly what Expect is for - as do the loss checks
    // further down, since the video's numbers are numbers about this text.
    sanity::Expect data;
    data.equal("characters", static_cast<double>(d.train.size() + d.val.size()), 1115394);
    data.equal("vocabulary", static_cast<double>(d.vocab), 65);
    data.report();

    std::cout << "\n== 1. the bigram baseline ==\n";
    std::mt19937 g1(1337);
    Bigram bigram(d.vocab, g1);
    printf("  %zu parameters, loss before training %.4f (a uniform guess scores %.4f)\n",
        bigram.parameter_count(),
        estimate_loss(bigram, d, kBatch, kBlock).val, kUniformLoss);

    const TrainResult bigram_result =
        train(bigram, d, kBatch, kBlock, real(1e-2), kStageIters, kReportEvery, g1);
    printf("  train %.4f  val %.4f  (%.1fs)   the video reads ~%.2f\n",
        bigram_result.train_loss, bigram_result.val_loss,
        bigram_result.seconds, kVideoBigram);
    {
        std::mt19937 gs(1337);
        const std::string sample = d.decode(bigram.generate(200, gs));
        printf("  200 characters from it:\n---\n%s\n---\n", sample.c_str());
    }

    std::cout << "\n== 2. the trick: averaging the past ==\n";
    demonstrate_averaging();

    std::cout << "\n== 3. one head of self-attention ==\n";
    std::cout <<
        "  Every position emits a query, a key and a value. The weight position\n"
        "  t puts on position u is q[t] . k[u], scaled by 1/sqrt(head_size) and\n"
        "  pushed through a softmax over u <= t. Section 2's uniform average was\n"
        "  the special case where every score is zero.\n";
    {
        std::mt19937 g(1337);
        const Config one_head{ kEmbed, 1, 1, kBlock, kBatch, real(0), kLr };
        Gpt model(one_head, d.vocab, g);
        printf("  %d channels, 1 head, 1 block, %zu parameters\n",
            kEmbed, model.parameter_count());
        const TrainResult r =
            train(model, d, kBatch, kBlock, kLr, kStageIters, 0, g);
        printf("  train %.4f  val %.4f  (%.1fs)\n", r.train_loss, r.val_loss, r.seconds);
    }

    std::cout << "\n== 4. several heads, and something to think with ==\n";
    std::cout <<
        "  One head can only ask one question per position. Four heads of eight\n"
        "  channels cost the same arithmetic as one of thirty-two and let the\n"
        "  position ask four. The feedforward layer after them is where each\n"
        "  position gets to do anything with what it heard - without it the\n"
        "  model goes straight from attending to predicting.\n";
    {
        std::mt19937 g(1337);
        const Config multi{ kEmbed, kHeads, 1, kBlock, kBatch, real(0), kLr };
        Gpt model(multi, d.vocab, g);
        printf("  %d channels, %d heads, 1 block, %zu parameters\n",
            kEmbed, kHeads, model.parameter_count());
        const TrainResult r =
            train(model, d, kBatch, kBlock, kLr, kStageIters, 0, g);
        printf("  train %.4f  val %.4f  (%.1fs)\n", r.train_loss, r.val_loss, r.seconds);
    }

    std::cout << "\n== 5. checking the backward passes ==\n";
    {
        std::mt19937 g(1337);
        gradient_check(d, g);
    }

    std::cout << "\n== 6. the full model ==\n";
    printf("  %d channels, %d heads, %d blocks, context %d, batch %d, lr %.0e\n",
        kEmbed, kHeads, kLayers, kBlock, kBatch, kLr);
    std::mt19937 g6(1337);
    const Config cfg{ kEmbed, kHeads, kLayers, kBlock, kBatch, kDropout, kLr };
    Gpt model(cfg, d.vocab, g6);
    printf("  %zu parameters\n", model.parameter_count());

    const std::string ckpt = d.dir + checkpoint_name(cfg, d);
    TrainResult r;
    const bool loaded = load_params(model.params(), cfg, d, ckpt);
    if (loaded) {
        const Losses l = estimate_loss(model, d, kBatch, kBlock);
        r.train_loss = l.train;
        r.val_loss = l.val;
        printf("  loaded %s, so no training was needed\n", ckpt.c_str());
        printf("  train %.4f  val %.4f\n", r.train_loss, r.val_loss);
    }
    else {
        printf("  no checkpoint beside the data, so %d steps of training - "
            "this is the slow part\n", kIters);
        r = train(model, d, kBatch, kBlock, kLr, kIters, kReportEvery, g6);
        printf("  train %.4f  val %.4f  (%.1fs)\n",
            r.train_loss, r.val_loss, r.seconds);
        if (save_params(model.params(), cfg, d, ckpt)) {
            printf("  saved to %s - delete it to train again\n", ckpt.c_str());
        }
    }
    printf("  the video reads ~%.2f train / ~%.2f val here, and ~%.2f val after "
        "scaling to\n  %d channels / %d heads / %d blocks / context %d\n",
        kVideoSmallTrain, kVideoSmallGpt, kVideoBigGpt,
        kBigEmbed, kBigHeads, kBigLayers, kBigBlock);
    printf("  train and val have drifted %.4f apart, and that gap only grows "
        "from here\n", r.val_loss - r.train_loss);

    sanity::Expect chk;
    // Only meaningful when this run did the training; a loaded model has no
    // step 0 to have scored.
    if (!loaded) {
        chk.close("loss before training", r.first_loss, kUniformLoss, 0.6);
    }
    chk.that("the transformer beats the bigram baseline",
        r.val_loss < bigram_result.val_loss);
    // Not all of this gap is memorisation. The split is positional - the
    // validation set is the last tenth of the file - so it is different scenes
    // with different characters, and some of the distance is the model being
    // asked about text that is genuinely unlike what it trained on. What the
    // threshold is for is the runaway case: forty thousand parameters against a
    // million training tokens cannot memorise, and a gap several times this
    // size would mean something else is wrong. The video's own run of this
    // configuration sits at about 0.09, and it is the growth of this number
    // rather than its value that makes it add dropout before scaling up.
    chk.that("the train/val gap is drift, not memorisation",
        r.val_loss - r.train_loss < 0.25);
    chk.close("validation loss", r.val_loss, kVideoSmallGpt, 0.15);
    chk.report();

    std::cout << "\n== 7. generating ==\n";
    std::mt19937 gs(1337);
    const std::string sample = d.decode(model.generate(1000, gs));
    printf("---\n%s\n---\n", sample.c_str());
    std::cout <<
        "  Still nonsense, but it is now Shakespeare-shaped nonsense: speaker\n"
        "  names on their own line followed by a colon, blank lines between\n"
        "  speeches, words of plausible length. The bigram model above could not\n"
        "  do any of that, because none of it fits in one character of context.\n";

    std::cout << "\n== 8. what the scale-up costs ==\n";
    const Config big{ kBigEmbed, kBigHeads, kBigLayers, kBigBlock, kBigBatch,
        kBigDropout, kBigLr };
    printf("  the video's final model: %d channels, %d heads, %d blocks, "
        "context %d, batch %d,\n  dropout %.2f, lr %.0e\n",
        big.n_embd, big.n_head, big.n_layer, big.block, big.batch,
        big.dropout, big.lr);

    // Constructing it costs the weights and their gradients and nothing else -
    // the activation buffers below are only allocated by a forward pass, which
    // is the whole point of printing their size before running one.
    size_t big_params = 0;
    {
        std::mt19937 gp(1337);
        Gpt probe(big, d.vocab, gp);
        big_params = probe.parameter_count();
    }

    const double measured = measure_throughput(cfg, d, d.vocab, kBenchSteps);
    const double projected = run_flops(big, big_params, d.vocab, kIters, kReportEvery);
    const double bytes = activation_bytes(big);

    printf("  %zu parameters (the video reads 10.788929 M)\n", big_params);
    printf("  this machine runs the small model at %.2f GFLOP/s "
        "(timed over %d steps)\n", measured / 1e9, kBenchSteps);
    printf("  one step of the scaled-up model is %.2f TFLOP, so %d steps is "
        "%.1f days\n", step_flops(big, big_params, d.vocab) / 1e12, kIters,
        projected / measured / 86400.0);
    printf("  its activation buffers would need about %.1f GB at %zu bytes per "
        "scalar\n", bytes / 1e9, sizeof(real));
    std::cout <<
        "  Which is the honest state of this port: the model is right and the\n"
        "  arithmetic is not fast enough to use it. Every inner loop here is\n"
        "  scalar, single-threaded and unblocked, and the three fixes for that -\n"
        "  vectorising, threading, and blocking for cache - are worth about two\n"
        "  orders of magnitude between them. That is the difference between a\n"
        "  month and an overnight run, and it is a separate piece of work.\n";

    // A plain if, not `if constexpr`: the project sets no <LanguageStandard>,
    // so MSVC builds it as C++14. The branch folds away either way, and the
    // body still gets compiled and type-checked, which is the point of having
    // it here rather than in a comment.
    if (kRunScaleUp) {
        std::mt19937 gb(1337);
        Gpt big_model(big, d.vocab, gb);

        // The whole reason checkpoints exist: this branch is an overnight run
        // once, and a second or two every time after.
        const std::string bckpt = d.dir + checkpoint_name(big, d);
        if (load_params(big_model.params(), big, d, bckpt)) {
            const Losses l = estimate_loss(big_model, d, big.batch, big.block);
            printf("  loaded %s\n  train %.4f  val %.4f   "
                "the video reads ~%.2f\n",
                bckpt.c_str(), l.train, l.val, kVideoBigGpt);
        }
        else {
            const TrainResult br = train(big_model, d, big.batch, big.block,
                big.lr, kIters, kReportEvery, gb);
            printf("  train %.4f  val %.4f  (%.1fs)   the video reads ~%.2f\n",
                br.train_loss, br.val_loss, br.seconds, kVideoBigGpt);
            if (save_params(big_model.params(), big, d, bckpt)) {
                printf("  saved to %s\n", bckpt.c_str());
            }
        }

        std::mt19937 gbs(1337);
        printf("---\n%s\n---\n", d.decode(big_model.generate(2000, gbs)).c_str());
    }
    std::cout << "\n";
}
