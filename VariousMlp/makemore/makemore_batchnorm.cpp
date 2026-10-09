// makemore_batchnorm.cpp - a C++ port of Andrej Karpathy's makemore, part 3
// ("Building makemore Part 3: Activations & Gradients, BatchNorm", video 4 of
//  Neural Networks: Zero to Hero)
//
// The previous video built a network that works, but it did not examine two
// problems. Its loss starts at about 27, where a model that knows nothing at
// all should score 3.30. And its hidden tanh outputs are stuck at -1 and +1
// before the first training step. Neither problem stops the net from
// learning, but together they waste the first thousand steps and most of the
// hidden units. This video looks at both problems from the inside, and then
// fixes them.
//
// What it covers, in order:
//   1. why randn weights make the output layer confidently wrong
//   2. why the same weights saturate tanh, and what a dead neuron is
//   3. Kaiming init, gain / sqrt(fan_in): the principled version of the two
//      hand-picked factors that fix 1 and 2
//   4. batch normalisation, which normalises the preactivations directly,
//      instead of choosing weights so that they come out normalised
//   5. the diagnostics (activation, gradient, grad:data and update:data
//      statistics per layer) that make all of this visible in a deeper net
//
// This port is not a continuation of makemore_mlp.cpp but a rewrite of it.
// Batch normalisation is the first operation in the series that works down a
// column of the batch, not along a row. A forward pass per example cannot
// express it, so everything below processes a whole minibatch at a time.
//
// Needs names.txt (32033 names, one per line) next to this file.
//
// Section 6 trains five networks of kSteps minibatches each. That takes a few
// minutes in Release and much longer in Debug. Lower kSteps if you run it
// often: the order of the five networks is clear long before their losses
// are final.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "check.h"

namespace {

// ---------------------------------------------------------------------------
// The dataset
// ---------------------------------------------------------------------------

constexpr int kVocab = 27;    // '.' plus the 26 lowercase letters
constexpr int kBlock = 3;     // characters of context the model gets to see

// encode, decode, load_names and build_dataset are copies of the ones in
// makemore_mlp.cpp, for the same reason that file copies them from
// makemore_bigram.cpp: each video's file is meant to be read from start to
// finish on its own, in the same way as each video is watched.
//
// Python:
//   chars = sorted(list(set(''.join(words))))
//   stoi = {s: i + 1 for i, s in enumerate(chars)}
//   stoi['.'] = 0
//   itos = {i: s for s, i in stoi.items()}
inline int encode(char c) {
    CHECK(c == '.' || (c >= 'a' && c <= 'z'),
        "expected '.' or a-z, got '" << c << "' (byte "
        << static_cast<int>(static_cast<unsigned char>(c)) << ")");
    return c == '.' ? 0 : c - 'a' + 1;
}

inline char decode(int i) {
    CHECK(i >= 0 && i < kVocab, "token " << i << " is outside the vocabulary");
    return i == 0 ? '.' : static_cast<char>('a' + i - 1);
}

// Python: words = open('names.txt', 'r').read().splitlines()
std::vector<std::string> load_names() {
    // The data is searched for relative to a few likely working directories.
    // Visual Studio runs with cwd = $(ProjectDir), a run of the exe on its own
    // has cwd = x64\Debug, and running from the repo root is the other common
    // case. A CMake build runs from somewhere in its build tree, which none of
    // these paths reach. So CMake defines the directory explicitly, and it is
    // tried first.
    static const char* kCandidates[] = {
#ifdef VARIOUSMLP_DATA_DIR
        VARIOUSMLP_DATA_DIR "/names.txt",
#endif
        "makemore/names.txt",
        "names.txt",
        "VariousMlp/makemore/names.txt",
        "../../VariousMlp/makemore/names.txt",
    };

    for (const char* path : kCandidates) {
        std::ifstream f(path);
        if (!f) {
            continue;
        }

        std::vector<std::string> words;
        std::string line;
        size_t skipped = 0;
        while (std::getline(f, line)) {
            if (line.empty()) {
                continue;
            }
            if (line.back() == '\r') {
                line.pop_back();  // CRLF
            }
            // The model has exactly 27 tokens, so an unexpected uppercase
            // letter, digit or accented letter has no token.
            const bool clean = std::all_of(line.begin(), line.end(),
                [](char c) { return c >= 'a' && c <= 'z'; });
            if (clean) {
                words.push_back(line);
            }
            else {
                ++skipped;
            }
        }
        if (!words.empty()) {
            printf("  loaded %zu names from %s\n", words.size(), path);
            if (skipped) {
                printf("  skipped %zu line(s) with characters outside a-z\n", skipped);
            }
            return words;
        }
    }
    return {};
}

// One example per (context, next character) pair. xs holds kBlock token ids
// per example, one example after another; ys holds the one id that follows
// each context.
struct Dataset {
    std::vector<int> xs;
    std::vector<int> ys;

    size_t size() const { return ys.size(); }
    const int* context(size_t i) const { return &xs[i * kBlock]; }
};

// Python:
//   def build_dataset(words):
//       X, Y = [], []
//       for w in words:
//           context = [0] * block_size
//           for ch in w + '.':
//               ix = stoi[ch]
//               X.append(context)
//               Y.append(ix)
//               context = context[1:] + [ix]   # crop and append
//       return torch.tensor(X), torch.tensor(Y)
Dataset build_dataset(const std::vector<std::string>& words) {
    Dataset d;
    auto emit = [&](const int* ctx, int next) {
        d.xs.insert(d.xs.end(), ctx, ctx + kBlock);
        d.ys.push_back(next);
    };

    int ctx[kBlock];
    for (const auto& w : words) {
        std::fill(std::begin(ctx), std::end(ctx), 0);
        for (char c : w) {
            const int ix = encode(c);
            emit(ctx, ix);
            for (int j = 0; j + 1 < kBlock; ++j) {
                ctx[j] = ctx[j + 1];
            }
            ctx[kBlock - 1] = ix;
        }
        emit(ctx, 0);                              // the name ends
    }
    return d;
}

// Every token here becomes a raw offset into the embedding table, so a bad
// token would be an out-of-bounds read, not just a wrong answer. This is one
// pass per split before training starts, so the training loop itself needs no
// checks.
void validate_dataset(const Dataset& d, const char* what) {
    CHECK(d.size() > 0, what << ": no examples");
    CHECK(d.xs.size() == d.size() * kBlock,
        what << ": " << d.xs.size() << " context tokens for " << d.size()
        << " examples, expected " << d.size() * kBlock);

    for (size_t i = 0; i < d.xs.size(); ++i) {
        CHECK(d.xs[i] >= 0 && d.xs[i] < kVocab,
            what << ": context token " << i << " is " << d.xs[i]
            << ", outside [0, " << kVocab << ")");
    }
    for (size_t i = 0; i < d.ys.size(); ++i) {
        CHECK(d.ys[i] >= 0 && d.ys[i] < kVocab,
            what << ": target " << i << " is " << d.ys[i]
            << ", outside [0, " << kVocab << ")");
    }
}

// The split is by name, not by example, for the same reason as in the last
// video: all examples from one name share its letters.
//
// Python:
//   random.seed(42)
//   random.shuffle(words)
//   n1 = int(0.8 * len(words))
//   n2 = int(0.9 * len(words))
struct Splits {
    std::vector<std::string> train, dev, test;
};

Splits split_words(std::vector<std::string> words, std::mt19937& g) {
    CHECK(words.size() >= 10,
        "need at least 10 names for an 80/10/10 split, got " << words.size());

    std::shuffle(words.begin(), words.end(), g);
    const size_t n1 = static_cast<size_t>(0.8 * words.size());
    const size_t n2 = static_cast<size_t>(0.9 * words.size());

    Splits s;
    s.train.assign(words.begin(), words.begin() + n1);
    s.dev.assign(words.begin() + n1, words.begin() + n2);
    s.test.assign(words.begin() + n2, words.end());
    return s;
}

// ---------------------------------------------------------------------------
// A batch, and the layers that process one
// ---------------------------------------------------------------------------

// Rows are examples, columns are features. Batch normalisation is the reason
// this type exists: mean and variance are computed down a column, across the
// examples. It is the first operation in the series that makes one example's
// output depend on the other 31 examples in its batch.
struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> v;

    // Always fills with zeros. Layers that overwrite every element do not need
    // that, but it costs only a fraction of one matrix multiply, and a
    // half-filled buffer can then never be read as data.
    void resize(int r, int c) {
        rows = r;
        cols = c;
        v.assign(static_cast<size_t>(r) * static_cast<size_t>(c), 0.0);
    }

    double* row(int i) { return &v[static_cast<size_t>(i) * cols]; }
    const double* row(int i) const { return &v[static_cast<size_t>(i) * cols]; }
};

// A parameter and its gradient, together. The video's flat `parameters` list
// is exactly this: what the optimiser updates and the diagnostics measure,
// without the layer it belongs to, on purpose.
struct Param {
    std::string name;
    std::vector<double>* data;
    std::vector<double>* grad;
};

// Python:
//   class Linear:
//       def __call__(self, x):
//           self.out = x @ self.weight
//           ...
//
// Every layer keeps its own output, and that is not an implementation detail:
// the activation histograms in sections 1-5 and the per-layer statistics in
// section 7 all read `.out` after a forward pass. For the same reason,
// backward() is not virtual, so that it can record the incoming gradient.
class Layer {
public:
    virtual ~Layer() = default;

    virtual const char* kind() const = 0;
    virtual const Matrix& forward(const Matrix& x) = 0;
    virtual void collect(std::vector<Param>&) {}
    virtual void set_training(bool) {}

    const Matrix& backward(const Matrix& dout) {
        dout_ = &dout;
        return backward_impl(dout);
    }

    const Matrix& out() const { return out_; }

    // d(loss)/d(this layer's output), i.e. the video's `layer.out.grad`. It is
    // null until the first backward pass, and only valid until the next one.
    const Matrix* dout() const { return dout_; }

protected:
    virtual const Matrix& backward_impl(const Matrix& dout) = 0;

    Matrix out_;
    Matrix dx_;
    const Matrix* dout_ = nullptr;
};

// Python:
//   class Linear:
//       def __init__(self, fan_in, fan_out, bias=True):
//           self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
//           self.bias = torch.zeros(fan_out) if bias else None
//
// weight_scale is the one setting that the first half of the video changes.
// 1.0 is the naive init from part 2, 0.01 is the scaled-down output layer, 0.2
// is the hand-picked hidden layer, and gain / sqrt(fan_in) is Kaiming. All
// four are the same line with a different number in front of it.
class Linear : public Layer {
public:
    Linear(int fan_in, int fan_out, double weight_scale, double bias_scale,
        bool has_bias, std::mt19937& g)
        : fan_in_(fan_in), fan_out_(fan_out), has_bias_(has_bias),
        w_(static_cast<size_t>(fan_in) * static_cast<size_t>(fan_out)),
        dw_(w_.size(), 0.0),
        b_(has_bias ? static_cast<size_t>(fan_out) : 0, 0.0),
        db_(b_.size(), 0.0)
    {
        CHECK(fan_in > 0 && fan_out > 0,
            "a " << fan_in << "x" << fan_out << " linear layer");

        std::normal_distribution<double> randn(0.0, 1.0);
        for (double& v : w_) {
            v = randn(g) * weight_scale;
        }
        // A zero scale means the video's torch.zeros, not a scaled random
        // draw, so no numbers are taken from the generator. Then two recipes
        // that differ only in whether the bias exists still get the same
        // random numbers for everything else. That is what makes section 6 a
        // fair comparison.
        if (bias_scale != 0.0) {
            for (double& v : b_) {
                v = randn(g) * bias_scale;
            }
        }

        wname_ = "W " + std::to_string(fan_in) + "x" + std::to_string(fan_out);
        bname_ = "b " + std::to_string(fan_out);
    }

    const char* kind() const override { return "Linear"; }

    const Matrix& forward(const Matrix& x) override {
        CHECK(x.cols == fan_in_, "a " << fan_in_ << "-wide layer was handed "
            << x.cols << " inputs");

        // A pointer, not a copy. The layer that produced x owns it, and does
        // not change it until the next forward pass. By then, this layer's
        // backward pass has already run.
        x_ = &x;
        out_.resize(x.rows, fan_out_);
        for (int i = 0; i < x.rows; ++i) {
            const double* xi = x.row(i);
            double* oi = out_.row(i);
            if (has_bias_) {
                std::copy(b_.begin(), b_.end(), oi);
            }
            for (int k = 0; k < fan_in_; ++k) {
                const double xk = xi[k];
                const double* wk = &w_[static_cast<size_t>(k) * fan_out_];
                for (int j = 0; j < fan_out_; ++j) {
                    oi[j] += xk * wk[j];
                }
            }
        }
        return out_;
    }

    void collect(std::vector<Param>& out) override {
        out.push_back({ wname_, &w_, &dw_ });
        if (has_bias_) {
            out.push_back({ bname_, &b_, &db_ });
        }
    }

protected:
    const Matrix& backward_impl(const Matrix& dout) override {
        const Matrix& x = *x_;
        dx_.resize(x.rows, fan_in_);
        for (int i = 0; i < x.rows; ++i) {
            const double* di = dout.row(i);
            const double* xi = x.row(i);
            double* dxi = dx_.row(i);
            if (has_bias_) {
                for (int j = 0; j < fan_out_; ++j) {
                    db_[j] += di[j];
                }
            }
            for (int k = 0; k < fan_in_; ++k) {
                const double* wk = &w_[static_cast<size_t>(k) * fan_out_];
                double* dwk = &dw_[static_cast<size_t>(k) * fan_out_];
                const double xk = xi[k];
                double acc = 0.0;
                for (int j = 0; j < fan_out_; ++j) {
                    dwk[j] += xk * di[j];
                    acc += wk[j] * di[j];
                }
                dxi[k] = acc;
            }
        }
        return dx_;
    }

private:
    int fan_in_;
    int fan_out_;
    bool has_bias_;
    std::vector<double> w_, dw_;
    std::vector<double> b_, db_;
    std::string wname_, bname_;
    const Matrix* x_ = nullptr;
};

// Python:
//   class Tanh:
//       def __call__(self, x):
//           self.out = torch.tanh(x)
//           return self.out
class Tanh : public Layer {
public:
    const char* kind() const override { return "Tanh"; }

    const Matrix& forward(const Matrix& x) override {
        out_.resize(x.rows, x.cols);
        for (size_t k = 0; k < x.v.size(); ++k) {
            out_.v[k] = std::tanh(x.v[k]);
        }
        return out_;
    }

protected:
    // tanh' = 1 - tanh^2, and out_ already holds the tanh. The whole subject
    // of the video is in that factor: when |out| is near 1, the factor is near
    // 0, and almost no gradient passes back through the layer.
    const Matrix& backward_impl(const Matrix& dout) override {
        dx_.resize(out_.rows, out_.cols);
        for (size_t k = 0; k < dout.v.size(); ++k) {
            const double t = out_.v[k];
            dx_.v[k] = dout.v[k] * (1.0 - t * t);
        }
        return dx_;
    }
};

// Python:
//   class BatchNorm1d:
//       def __call__(self, x):
//           if self.training:
//               xmean = x.mean(0, keepdim=True)
//               xvar = x.var(0, keepdim=True)
//           else:
//               xmean = self.running_mean
//               xvar = self.running_var
//           xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
//           self.out = self.gamma * xhat + self.beta
//           if self.training:
//               with torch.no_grad():
//                   self.running_mean = (1 - m) * self.running_mean + m * xmean
//                   self.running_var = (1 - m) * self.running_var + m * xvar
//           return self.out
//
// torch.var is unbiased by default, so the division below is by n-1, not n.
// That choice also affects the backward pass, where it appears as the n/(n-1)
// factor, so the two have to agree.
class BatchNorm1d : public Layer {
public:
    BatchNorm1d(int dim, double eps, double momentum)
        : dim_(dim), eps_(eps), momentum_(momentum),
        gamma_(dim, 1.0), dgamma_(dim, 0.0),
        beta_(dim, 0.0), dbeta_(dim, 0.0),
        running_mean_(dim, 0.0), running_var_(dim, 1.0),
        invstd_(dim, 0.0)
    {
        CHECK(dim > 0, "batch normalisation over " << dim << " features");
        CHECK(eps > 0.0, "eps " << eps << " has to be positive to keep the "
            "division safe for a feature that is constant across the batch");
        CHECK(momentum > 0.0 && momentum <= 1.0,
            "momentum " << momentum << " is outside (0, 1]");

        gname_ = "gamma " + std::to_string(dim);
        bname_ = "beta " + std::to_string(dim);
    }

    const char* kind() const override { return "BatchNorm1d"; }
    void set_training(bool training) override { training_ = training; }

    const Matrix& forward(const Matrix& x) override {
        CHECK(x.cols == dim_, "batch normalisation over " << dim_
            << " features was handed " << x.cols);

        out_.resize(x.rows, dim_);
        if (!training_) {
            // The running estimates exist for exactly this call. A single
            // example has no batch to be normalised against. Batch
            // normalisation creates this problem, and the running estimates
            // are its workaround.
            for (int j = 0; j < dim_; ++j) {
                const double inv = 1.0 / std::sqrt(running_var_[j] + eps_);
                for (int i = 0; i < x.rows; ++i) {
                    out_.row(i)[j] =
                        gamma_[j] * (x.row(i)[j] - running_mean_[j]) * inv + beta_[j];
                }
            }
            return out_;
        }

        CHECK(x.rows > 1, "batch normalisation in training mode needs at least "
            "2 examples to have a variance, got " << x.rows);

        const double n = x.rows;
        xhat_.resize(x.rows, dim_);
        for (int j = 0; j < dim_; ++j) {
            double mean = 0.0;
            for (int i = 0; i < x.rows; ++i) {
                mean += x.row(i)[j];
            }
            mean /= n;

            double var = 0.0;
            for (int i = 0; i < x.rows; ++i) {
                const double d = x.row(i)[j] - mean;
                var += d * d;
            }
            var /= (n - 1.0);

            invstd_[j] = 1.0 / std::sqrt(var + eps_);
            for (int i = 0; i < x.rows; ++i) {
                const double xh = (x.row(i)[j] - mean) * invstd_[j];
                xhat_.row(i)[j] = xh;
                out_.row(i)[j] = gamma_[j] * xh + beta_[j];
            }

            running_mean_[j] = (1.0 - momentum_) * running_mean_[j] + momentum_ * mean;
            running_var_[j] = (1.0 - momentum_) * running_var_[j] + momentum_ * var;
        }
        return out_;
    }

    void collect(std::vector<Param>& out) override {
        out.push_back({ gname_, &gamma_, &dgamma_ });
        out.push_back({ bname_, &beta_, &dbeta_ });
    }

protected:
    // The one derivative in this repo that is not obvious from reading the
    // forward pass. Every example's normalised value depends on the mean and
    // the variance, and both depend on all n examples. So the gradient of one
    // output reaches back to the whole batch. Written out, with S1 and S2 the
    // two column sums:
    //
    //   dx = invstd/n * (n*dxhat - S1 - n/(n-1) * xhat * S2)
    //
    // which is the video's one-liner:
    //
    //   dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0)
    //             - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
    //
    // gradient_check() below compares it with central differences, because a
    // wrong version of this does not crash; it only trains a little worse.
    const Matrix& backward_impl(const Matrix& dout) override {
        CHECK(training_, "backward pass through batch normalisation in "
            "evaluation mode: the batch statistics it would differentiate "
            "were never computed");
        CHECK(dout.rows == xhat_.rows && dout.cols == dim_,
            "gradient of shape " << dout.rows << "x" << dout.cols
            << " for a " << xhat_.rows << "x" << dim_ << " forward pass");

        const int n = dout.rows;
        const double bessel = static_cast<double>(n) / (n - 1.0);
        dx_.resize(n, dim_);
        for (int j = 0; j < dim_; ++j) {
            double s1 = 0.0;
            double s2 = 0.0;
            for (int i = 0; i < n; ++i) {
                const double d = dout.row(i)[j];
                const double xh = xhat_.row(i)[j];
                dgamma_[j] += d * xh;
                dbeta_[j] += d;
                s1 += d * gamma_[j];
                s2 += d * gamma_[j] * xh;
            }
            for (int i = 0; i < n; ++i) {
                const double dxhat = dout.row(i)[j] * gamma_[j];
                dx_.row(i)[j] = invstd_[j] / n *
                    (n * dxhat - s1 - bessel * xhat_.row(i)[j] * s2);
            }
        }
        return dx_;
    }

private:
    int dim_;
    double eps_;
    double momentum_;
    bool training_ = true;
    std::vector<double> gamma_, dgamma_;
    std::vector<double> beta_, dbeta_;
    std::vector<double> running_mean_, running_var_;
    std::vector<double> invstd_;
    Matrix xhat_;
    std::string gname_, bname_;
};

// ---------------------------------------------------------------------------
// The network
// ---------------------------------------------------------------------------

// Softmax over one row of kVocab logits. Subtracting the row maximum first
// costs almost nothing and prevents overflow in exp(). It does not change the
// result.
//
// Python: probs = F.softmax(logits, dim=1)
void softmax(const double* logits, double* probs) {
    const double maxl = *std::max_element(logits, logits + kVocab);
    double denom = 0.0;
    for (int j = 0; j < kVocab; ++j) {
        probs[j] = std::exp(logits[j] - maxl);
        denom += probs[j];
    }
    for (int j = 0; j < kVocab; ++j) {
        probs[j] /= denom;
    }
}

// The embedding table stays outside the layer stack, as in the video. C is a
// lookup, not a matrix multiply, so there is no Linear layer for it. A later
// video gives it a layer of its own.
//
// Python:
//   emb = C[Xtr[ix]]
//   x = emb.view(emb.shape[0], -1)
//   for layer in layers:
//       x = layer(x)
//   loss = F.cross_entropy(x, Ytr[ix])
class Net {
public:
    Net(int embed, std::vector<std::unique_ptr<Layer>> layers, std::mt19937& g)
        : embed_(embed), layers_(std::move(layers)),
        C_(static_cast<size_t>(kVocab) * static_cast<size_t>(embed)),
        dC_(C_.size(), 0.0)
    {
        CHECK(embed > 0, "embedding width " << embed);
        CHECK(!layers_.empty(), "a network with no layers");

        std::normal_distribution<double> randn(0.0, 1.0);
        for (double& v : C_) {
            v = randn(g);
        }

        params_.push_back({ "C " + std::to_string(kVocab) + "x"
            + std::to_string(embed), &C_, &dC_ });
        for (size_t i = 0; i < layers_.size(); ++i) {
            const size_t before = params_.size();
            layers_[i]->collect(params_);
            for (size_t k = before; k < params_.size(); ++k) {
                params_[k].name = "layer " + std::to_string(i) + " " + params_[k].name;
            }
        }
    }

    // params_ holds the address of C_, so the object must not be copied or
    // moved.
    Net(const Net&) = delete;
    Net& operator=(const Net&) = delete;

    std::vector<Param>& params() { return params_; }
    const std::vector<std::unique_ptr<Layer>>& layers() const { return layers_; }
    const Matrix& logits() const { return *logits_; }

    size_t parameter_count() const {
        size_t n = 0;
        for (const Param& p : params_) {
            n += p.data->size();
        }
        return n;
    }

    // Python:
    //   for layer in layers:
    //       layer.training = False
    void set_training(bool training) {
        for (auto& layer : layers_) {
            layer->set_training(training);
        }
    }

    // Python:
    //   for p in parameters:
    //       p.grad = None
    void zero_grad() {
        for (Param& p : params_) {
            std::fill(p.grad->begin(), p.grad->end(), 0.0);
        }
    }

    // Python:
    //   for p in parameters:
    //       p.data += -lr * p.grad
    void step(double lr) {
        for (Param& p : params_) {
            std::vector<double>& d = *p.data;
            const std::vector<double>& g = *p.grad;
            for (size_t k = 0; k < d.size(); ++k) {
                d[k] -= lr * g[k];
            }
        }
    }

    const Matrix& forward(const Dataset& d, const size_t* rows, int n) {
        embed_rows(d, rows, n);
        const Matrix* x = &emb_;
        for (auto& layer : layers_) {
            x = &layer->forward(*x);
        }
        CHECK(x->cols == kVocab, "the last layer produces " << x->cols
            << " outputs, expected " << kVocab);
        logits_ = x;
        return *x;
    }

    double forward_loss(const Dataset& d, const size_t* rows, int n) {
        const Matrix& logits = forward(d, rows, n);
        return cross_entropy(logits, d, rows, n, nullptr);
    }

    // Returns the batch's mean loss, and adds d(loss)/d(parameter) into every
    // gradient. The caller is responsible for zero_grad(). So the gradients of
    // two half-batches can be added up for one step, which the gradient check
    // below needs.
    double forward_backward(const Dataset& d, const size_t* rows, int n) {
        const Matrix& logits = forward(d, rows, n);
        const double loss = cross_entropy(logits, d, rows, n, &dlogits_);

        const Matrix* g = &dlogits_;
        for (size_t k = layers_.size(); k-- > 0; ) {
            g = &layers_[k]->backward(*g);
        }
        scatter_embedding_grad(d, rows, n, *g);
        return loss;
    }

    // Mean negative log likelihood over a whole split, in nats. Evaluation runs
    // in chunks, not as one very large batch, because a hidden layer with
    // 22655 rows is 36MB of doubles. Batch normalisation in evaluation mode
    // treats the examples independently anyway, so the chunks give the same
    // result.
    //
    // Python:
    //   emb = C[Xdev]
    //   x = emb.view(emb.shape[0], -1)
    //   for layer in layers:
    //       x = layer(x)
    //   loss = F.cross_entropy(x, Ydev)
    double evaluate(const Dataset& d) {
        CHECK(d.size() > 0, "nothing to evaluate");

        constexpr int kChunk = 512;
        set_training(false);

        std::vector<size_t> rows(kChunk);
        double total = 0.0;
        size_t done = 0;
        while (done < d.size()) {
            const int n = static_cast<int>(std::min<size_t>(kChunk, d.size() - done));
            for (int i = 0; i < n; ++i) {
                rows[i] = done + static_cast<size_t>(i);
            }
            total += forward_loss(d, rows.data(), n) * n;
            done += static_cast<size_t>(n);
        }

        set_training(true);
        return total / static_cast<double>(d.size());
    }

    // Python:
    //   context = [0] * block_size
    //   while True:
    //       emb = C[torch.tensor([context])]
    //       x = emb.view(1, -1)
    //       for layer in layers:
    //           x = layer(x)
    //       probs = F.softmax(x, dim=1)
    //       ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    //       context = context[1:] + [ix]
    //       if ix == 0:
    //           break
    std::string sample(std::mt19937& g) {
        // A model that never produces '.' would loop forever. The longest name
        // in the dataset has 15 characters, so reaching this limit can only
        // mean that something is broken.
        constexpr size_t kMaxLength = 1000;

        set_training(false);

        Dataset one;
        one.xs.assign(kBlock, 0);
        one.ys.assign(1, 0);
        const size_t row = 0;

        std::string out;
        std::vector<double> probs(kVocab);
        for (;;) {
            const Matrix& logits = forward(one, &row, 1);
            softmax(logits.row(0), probs.data());
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            const int ix = dist(g);
            if (ix == 0) {
                break;
            }
            out.push_back(decode(ix));
            CHECK(out.size() < kMaxLength,
                "sampling ran past " << kMaxLength << " characters without an end token");
            for (int j = 0; j + 1 < kBlock; ++j) {
                one.xs[j] = one.xs[j + 1];
            }
            one.xs[kBlock - 1] = ix;
        }

        set_training(true);
        return out;
    }

    // The output of the first layer of this kind, for the diagnostics. Null if
    // the network has no such layer. That is how the sections that print tanh
    // statistics print nothing for a network without tanh.
    const Matrix* first_output_of(const char* which) const {
        for (const auto& layer : layers_) {
            if (std::string(layer->kind()) == which) {
                return &layer->out();
            }
        }
        return nullptr;
    }

private:
    void embed_rows(const Dataset& d, const size_t* rows, int n) {
        emb_.resize(n, kBlock * embed_);
        for (int i = 0; i < n; ++i) {
            const int* ctx = d.context(rows[i]);
            double* e = emb_.row(i);
            for (int j = 0; j < kBlock; ++j) {
                const double* c = &C_[static_cast<size_t>(ctx[j]) * embed_];
                std::copy(c, c + embed_, e + j * embed_);
            }
        }
    }

    // The embedding gradient is added back into C, and the += is essential. A
    // character that appears twice in one context contributes twice, and a
    // character that appears in several examples of the batch contributes once
    // per example.
    void scatter_embedding_grad(const Dataset& d, const size_t* rows, int n,
        const Matrix& demb) {
        for (int i = 0; i < n; ++i) {
            const int* ctx = d.context(rows[i]);
            const double* de = demb.row(i);
            for (int j = 0; j < kBlock; ++j) {
                double* dc = &dC_[static_cast<size_t>(ctx[j]) * embed_];
                for (int e = 0; e < embed_; ++e) {
                    dc[e] += de[j * embed_ + e];
                }
            }
        }
    }

    // Softmax and mean negative log likelihood over the batch. As in the two
    // previous files, the gradient is probs - onehot(target). The division by
    // n here makes the loss a mean, not a sum.
    //
    // Python: loss = F.cross_entropy(logits, Y)
    double cross_entropy(const Matrix& logits, const Dataset& d,
        const size_t* rows, int n, Matrix* dlogits) {
        if (dlogits) {
            dlogits->resize(n, kVocab);
        }

        const double inv_n = 1.0 / n;
        double total = 0.0;
        std::vector<double> probs(kVocab);
        for (int i = 0; i < n; ++i) {
            softmax(logits.row(i), probs.data());
            const int y = d.ys[rows[i]];
            total += -std::log(probs[y]);
            if (dlogits) {
                double* dl = dlogits->row(i);
                for (int j = 0; j < kVocab; ++j) {
                    dl[j] = (probs[j] - (j == y ? 1.0 : 0.0)) * inv_n;
                }
            }
        }
        return total * inv_n;
    }

    int embed_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::vector<double> C_, dC_;
    std::vector<Param> params_;
    Matrix emb_;
    Matrix dlogits_;
    const Matrix* logits_ = nullptr;
};

// ---------------------------------------------------------------------------
// The five initialisations the video works through
// ---------------------------------------------------------------------------

constexpr int kEmbed = 10;
constexpr int kHidden = 200;
constexpr int kBatch = 32;

constexpr double kBnEps = 1e-5;
constexpr double kBnMomentum = 0.1;    // the default in torch and in the video

// tanh compresses its input. So a layer whose initialisation should keep the
// spread of its input has to scale the input up a little before the tanh. 5/3
// is the value that torch.nn.init.calculate_gain returns for tanh. Without
// this gain, the activations of a deep stack fade out; with it, they keep
// their spread.
constexpr double kTanhGain = 5.0 / 3.0;
constexpr double kOutputGain = 0.1;    // the video's "make the last layer less confident"

// The two thresholds that the video uses in its plots: 0.97 for "how much of
// this layer is in the flat part of tanh", and 0.99 for the white pixels in
// plt.imshow(h.abs() > 0.99).
constexpr double kSaturated = 0.97;
constexpr double kDead = 0.99;

struct Recipe {
    const char* name;
    double w1_scale;
    double b1_scale;
    double w2_scale;
    double b2_scale;
    bool batchnorm;
};

std::vector<std::unique_ptr<Layer>> build_layers(const Recipe& r, std::mt19937& g) {
    std::vector<std::unique_ptr<Layer>> layers;

    // With batch normalisation, the hidden bias is useless: the mean
    // subtraction removes any constant added to the preactivations. So b1
    // would only waste computation on its gradient. The video removes it for
    // exactly that reason.
    layers.push_back(std::make_unique<Linear>(kBlock * kEmbed, kHidden,
        r.w1_scale, r.b1_scale, !r.batchnorm, g));
    if (r.batchnorm) {
        layers.push_back(std::make_unique<BatchNorm1d>(kHidden, kBnEps, kBnMomentum));
    }
    layers.push_back(std::make_unique<Tanh>());
    layers.push_back(std::make_unique<Linear>(kHidden, kVocab,
        r.w2_scale, r.b2_scale, true, g));
    return layers;
}

// sqrt is not constexpr, so the Kaiming scales are computed here, not declared
// above with the other constants.
std::vector<Recipe> recipes() {
    const double kaiming_hidden = kTanhGain / std::sqrt(static_cast<double>(kBlock * kEmbed));
    const double kaiming_out = kOutputGain / std::sqrt(static_cast<double>(kHidden));
    return {
        { "randn, exactly as in part 2",  1.0,            1.0,  1.0,          1.0, false },
        { "output layer squashed x0.01",  1.0,            1.0,  0.01,         0.0, false },
        { "hidden layer squashed x0.2",   0.2,            0.01, 0.01,         0.0, false },
        { "kaiming: gain / sqrt(fan_in)", kaiming_hidden, 0.0,  kaiming_out,  0.0, false },
        { "kaiming + batch norm",         kaiming_hidden, 0.0,  kaiming_out,  0.0, true  },
    };
}

// ---------------------------------------------------------------------------
// Looking at what a network is doing
// ---------------------------------------------------------------------------

struct Stats {
    double mean;
    double stdev;
};

Stats stats(const double* v, size_t n) {
    CHECK(n > 1, "standard deviation of " << n << " value(s)");

    double mean = 0.0;
    for (size_t k = 0; k < n; ++k) {
        mean += v[k];
    }
    mean /= static_cast<double>(n);

    double sq = 0.0;
    for (size_t k = 0; k < n; ++k) {
        const double d = v[k] - mean;
        sq += d * d;
    }
    return { mean, std::sqrt(sq / static_cast<double>(n - 1)) };
}

double fraction_above(const Matrix& m, double threshold) {
    size_t hits = 0;
    for (double v : m.v) {
        if (std::fabs(v) > threshold) {
            ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(m.v.size());
}

// The video's plt.imshow(h.abs() > 0.99, cmap='gray'). A column that is white
// from top to bottom is a neuron that is saturated for every example in the
// batch, so no gradient from this batch reaches its weights. Counting those
// columns gives the same information without the picture.
int dead_neurons(const Matrix& h, double threshold) {
    int dead = 0;
    for (int j = 0; j < h.cols; ++j) {
        bool all = true;
        for (int i = 0; i < h.rows && all; ++i) {
            all = std::fabs(h.row(i)[j]) > threshold;
        }
        if (all) {
            ++dead;
        }
    }
    return dead;
}

// A text version of the video's plt.hist(h.view(-1).tolist(), 50).
void print_histogram(const double* v, size_t n, double lo, double hi) {
    constexpr int kBuckets = 25;
    constexpr int kBarWidth = 46;

    CHECK(hi > lo, "histogram range [" << lo << ", " << hi << "]");

    std::vector<size_t> counts(kBuckets, 0);
    for (size_t k = 0; k < n; ++k) {
        const double t = (v[k] - lo) / (hi - lo);
        int b = static_cast<int>(t * kBuckets);
        b = std::min(kBuckets - 1, std::max(0, b));
        ++counts[b];
    }

    const size_t peak = *std::max_element(counts.begin(), counts.end());
    for (int b = 0; b < kBuckets; ++b) {
        const double edge = lo + (hi - lo) * b / kBuckets;
        const int bar = static_cast<int>(std::lround(
            static_cast<double>(counts[b]) / peak * kBarWidth));
        printf("      %+5.2f %s\n", edge, std::string(bar, '#').c_str());
    }
}

// Everything that the first half of the video reads from a single forward pass
// at step 0: the loss of a new network, how widely its logits spread, and what
// the hidden tanh does to the gradient that has to pass back through it.
void report_init(const char* label, Net& net, const Dataset& d,
    const size_t* rows, int n, bool histogram) {
    const double loss = net.forward_loss(d, rows, n);
    const Matrix& logits = net.logits();

    double max_abs = 0.0;
    for (double v : logits.v) {
        max_abs = std::max(max_abs, std::fabs(v));
    }

    printf("  %s\n", label);
    printf("    loss at step 0   %8.4f   (a model that knows nothing scores %.4f)\n",
        loss, std::log(static_cast<double>(kVocab)));
    printf("    largest logit    %8.4f\n", max_abs);

    const Matrix* h = net.first_output_of("Tanh");
    if (h) {
        const Stats s = stats(h->v.data(), h->v.size());
        printf("    tanh out: mean %+.3f  std %.3f  |h| > %.2f for %5.1f%%  "
            "dead neurons %d/%d\n",
            s.mean, s.stdev, kSaturated, fraction_above(*h, kSaturated) * 100.0,
            dead_neurons(*h, kDead), h->cols);
        if (histogram) {
            print_histogram(h->v.data(), h->v.size(), -1.0, 1.0);
        }
    }
}

// The small experiment that the video uses to motivate Kaiming init. If you
// multiply unit gaussians by a matrix of unit gaussians, the spread grows by
// sqrt(fan_in). So in a stack of such layers, the values either explode or
// (through tanh) collapse. Dividing by sqrt(fan_in) is the whole fix.
//
// Python:
//   x = torch.randn(1000, 10)
//   w = torch.randn(10, 200) / 10**0.5
//   y = x @ w
//   print(x.mean(), x.std())
//   print(y.mean(), y.std())
void why_divide_by_sqrt_fan_in(std::mt19937& g) {
    constexpr int kRows = 1000;
    constexpr int kFanIn = 30;
    constexpr int kFanOut = 200;

    std::normal_distribution<double> randn(0.0, 1.0);
    Matrix x;
    x.resize(kRows, kFanIn);
    for (double& v : x.v) {
        v = randn(g);
    }

    std::vector<double> w(static_cast<size_t>(kFanIn) * kFanOut);
    for (double& v : w) {
        v = randn(g);
    }

    const double scales[] = { 1.0, 1.0 / std::sqrt(static_cast<double>(kFanIn)) };
    const char* names[] = { "randn", "randn / sqrt(fan_in)" };

    printf("    x: %d x %d, std %.3f\n", kRows, kFanIn,
        stats(x.v.data(), x.v.size()).stdev);
    for (int s = 0; s < 2; ++s) {
        Matrix y;
        y.resize(kRows, kFanOut);
        for (int i = 0; i < kRows; ++i) {
            const double* xi = x.row(i);
            double* yi = y.row(i);
            for (int k = 0; k < kFanIn; ++k) {
                const double xk = xi[k] * scales[s];
                const double* wk = &w[static_cast<size_t>(k) * kFanOut];
                for (int j = 0; j < kFanOut; ++j) {
                    yi[j] += xk * wk[j];
                }
            }
        }
        printf("    x @ (%-20s): std %.3f\n", names[s],
            stats(y.v.data(), y.v.size()).stdev);
    }
    printf("    sqrt(fan_in) = %.3f, which is the factor the first row gained\n",
        std::sqrt(static_cast<double>(kFanIn)));
}

// Python:
//   for i, layer in enumerate(layers[:-1]):
//       if isinstance(layer, Tanh):
//           t = layer.out
//           print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (
//               i, layer.__class__.__name__, t.mean(), t.std(),
//               (t.abs() > 0.97).float().mean() * 100))
void report_activations(const Net& net) {
    printf("    activations, after the forward pass:\n");
    const auto& layers = net.layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        if (std::string(layers[i]->kind()) != "Tanh") {
            continue;
        }
        const Matrix& t = layers[i]->out();
        const Stats s = stats(t.v.data(), t.v.size());
        printf("      layer %2zu (%-11s) mean %+.2f  std %.2f  saturated %5.2f%%\n",
            i, layers[i]->kind(), s.mean, s.stdev,
            fraction_above(t, kSaturated) * 100.0);
    }
}

// The same loop over the gradient of each layer's output, instead of the
// output itself. The video's point is that these statistics should also stay
// similar across depth. If they shrink from layer to layer in the backward
// pass, the early layers hardly train.
//
// Python:
//   for i, layer in enumerate(layers[:-1]):
//       if isinstance(layer, Tanh):
//           t = layer.out.grad
//           print('layer %d (%10s): mean %+f, std %e' % (
//               i, layer.__class__.__name__, t.mean(), t.std()))
void report_gradients(const Net& net) {
    printf("    gradients of those same outputs, after the backward pass:\n");
    const auto& layers = net.layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        if (std::string(layers[i]->kind()) != "Tanh") {
            continue;
        }
        const Matrix* t = layers[i]->dout();
        CHECK(t != nullptr, "layer " << i << " has no gradient yet; "
            "report_gradients has to follow a backward pass");
        const Stats s = stats(t->v.data(), t->v.size());
        printf("      layer %2zu (%-11s) mean %+.3e  std %.3e\n",
            i, layers[i]->kind(), s.mean, s.stdev);
    }
}

// Python:
//   for i, p in enumerate(parameters):
//       t = p.grad
//       if p.ndim == 2:
//           print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (
//               tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
void report_grad_to_data(Net& net) {
    printf("    gradient-to-data ratio per parameter:\n");
    for (Param& p : net.params()) {
        if (p.data->size() < 2) {
            continue;
        }
        const Stats d = stats(p.data->data(), p.data->size());
        const Stats g = stats(p.grad->data(), p.grad->size());
        if (d.stdev < 1e-12) {
            printf("      %-24s grad std %.3e  (data is flat, no ratio)\n",
                p.name.c_str(), g.stdev);
            continue;
        }
        printf("      %-24s grad std %.3e  ratio %.3e\n",
            p.name.c_str(), g.stdev, g.stdev / d.stdev);
    }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

constexpr int kSteps = 50000;          // the video trains 200000
constexpr double kLearningRate = 0.1;
constexpr double kDecayedRate = 0.01;

struct TrainResult {
    double first_loss = 0.0;
    double last_loss = 0.0;
    double train_loss = 0.0;
    double dev_loss = 0.0;
    double seconds = 0.0;
};

// Python:
//   for i in range(max_steps):
//       ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
//       Xb, Yb = Xtr[ix], Ytr[ix]
//       emb = C[Xb]
//       x = emb.view(emb.shape[0], -1)
//       for layer in layers:
//           x = layer(x)
//       loss = F.cross_entropy(x, Yb)
//       for layer in layers:
//           layer.out.retain_grad()
//       for p in parameters:
//           p.grad = None
//       loss.backward()
//       lr = 0.1 if i < 100000 else 0.01
//       for p in parameters:
//           p.data += -lr * p.grad
TrainResult train(Net& net, const Dataset& data, const Dataset& dev, int steps,
    bool verbose, std::mt19937& g) {
    CHECK(steps > 0, "asked for " << steps << " training steps");
    CHECK(data.size() > 0, "no training examples");

    std::uniform_int_distribution<size_t> pick(0, data.size() - 1);
    std::vector<size_t> rows(kBatch);
    const int report_every = std::max(1, steps / 10);

    TrainResult r;
    double ema = 0.0;
    const auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step < steps; ++step) {
        for (int b = 0; b < kBatch; ++b) {
            rows[b] = pick(g);
        }

        net.zero_grad();
        const double loss = net.forward_backward(data, rows.data(), kBatch);
        net.step(step < steps / 2 ? kLearningRate : kDecayedRate);

        if (step == 0) {
            r.first_loss = loss;
            ema = loss;
        }
        else {
            ema += 0.001 * (loss - ema);
        }
        r.last_loss = loss;

        if (verbose && (step % report_every == 0 || step == steps - 1)) {
            printf("    step %6d  minibatch loss %.4f  (moving average %.4f)\n",
                step, loss, ema);
        }
    }

    r.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    r.train_loss = net.evaluate(data);
    r.dev_loss = net.evaluate(dev);
    return r;
}

// Python:
//   with torch.no_grad():
//       ud.append([((lr * p.grad).std() / p.data.std()).log10().item()
//                  for p in parameters])
//
// The rule of thumb that the video draws as a black line at -3: a step should
// change a parameter by about a thousandth of its own spread. Much more than
// that makes training unstable; much less means that the parameter hardly
// changes.
void report_update_to_data(Net& net, const Dataset& data, int steps,
    double lr, std::mt19937& g) {
    CHECK(steps > 0, "asked for " << steps << " probe steps");

    std::uniform_int_distribution<size_t> pick(0, data.size() - 1);
    std::vector<size_t> rows(kBatch);
    std::vector<double> sum(net.params().size(), 0.0);
    std::vector<int> counted(net.params().size(), 0);

    for (int step = 0; step < steps; ++step) {
        for (int b = 0; b < kBatch; ++b) {
            rows[b] = pick(g);
        }
        net.zero_grad();
        net.forward_backward(data, rows.data(), kBatch);

        for (size_t i = 0; i < net.params().size(); ++i) {
            Param& p = net.params()[i];
            if (p.data->size() < 2) {
                continue;
            }
            const Stats d = stats(p.data->data(), p.data->size());
            const Stats gr = stats(p.grad->data(), p.grad->size());
            // beta and the biases start at exactly zero. So in the first few
            // steps they have no spread to compare with, and those steps are
            // skipped instead of being reported as infinite.
            if (d.stdev < 1e-12 || gr.stdev <= 0.0) {
                continue;
            }
            sum[i] += std::log10(lr * gr.stdev / d.stdev);
            ++counted[i];
        }
        net.step(lr);
    }

    printf("    mean log10(lr * grad.std / data.std) over %d steps "
        "(the video's rule of thumb is -3):\n", steps);
    for (size_t i = 0; i < net.params().size(); ++i) {
        if (counted[i] == 0) {
            printf("      %-24s  (never had a spread to compare against)\n",
                net.params()[i].name.c_str());
            continue;
        }
        printf("      %-24s %+6.2f\n", net.params()[i].name.c_str(),
            sum[i] / counted[i]);
    }
}

// ---------------------------------------------------------------------------
// Checking the one derivative that is not obvious
// ---------------------------------------------------------------------------

// The backward pass of batch normalisation is the first derivative in this
// repo that is not obvious from reading the forward pass. A wrong version of
// it does not crash; it only trains a little worse, which looks exactly like a
// bad hyperparameter. So it is compared with the definition of a derivative.
//
// Central differences on a network that is very small on purpose: a wrong
// n/(n-1), a missing column sum or a missing gamma all show up as a
// disagreement in the first significant digit. So the tolerance can be tight
// and still never fail by accident.
void gradient_check(std::mt19937& g) {
    constexpr int kCheckBatch = 8;
    constexpr int kProbesPerParam = 4;
    constexpr int kTinyEmbed = 3;
    constexpr int kTinyHidden = 6;
    constexpr double kH = 1e-6;

    // A synthetic dataset: the check is about the arithmetic, not the names.
    std::uniform_int_distribution<int> token(0, kVocab - 1);
    Dataset d;
    for (int i = 0; i < kCheckBatch; ++i) {
        for (int j = 0; j < kBlock; ++j) {
            d.xs.push_back(token(g));
        }
        d.ys.push_back(token(g));
    }
    std::vector<size_t> rows(kCheckBatch);
    for (int i = 0; i < kCheckBatch; ++i) {
        rows[i] = static_cast<size_t>(i);
    }

    std::vector<std::unique_ptr<Layer>> layers;
    layers.push_back(std::make_unique<Linear>(kBlock * kTinyEmbed, kTinyHidden,
        0.5, 0.0, false, g));
    layers.push_back(std::make_unique<BatchNorm1d>(kTinyHidden, kBnEps, kBnMomentum));
    layers.push_back(std::make_unique<Tanh>());
    layers.push_back(std::make_unique<Linear>(kTinyHidden, kVocab, 0.5, 0.0, true, g));
    Net net(kTinyEmbed, std::move(layers), g);

    // gamma starts at 1, and beta and the biases at 0. With those values, a
    // wrong expression is most likely to agree with the correct one by
    // accident. So every parameter is moved away from its starting value.
    std::normal_distribution<double> jitter(0.0, 0.3);
    for (Param& p : net.params()) {
        for (double& v : *p.data) {
            v += jitter(g);
        }
    }

    net.zero_grad();
    net.forward_backward(d, rows.data(), kCheckBatch);

    std::vector<std::vector<double>> analytic;
    analytic.reserve(net.params().size());
    for (Param& p : net.params()) {
        analytic.push_back(*p.grad);
    }

    std::mt19937 probe_rng(20150211);   // the date of the batch norm paper
    double worst = 0.0;
    std::string worst_where;
    int probes = 0;

    for (size_t pi = 0; pi < net.params().size(); ++pi) {
        Param& p = net.params()[pi];
        std::uniform_int_distribution<size_t> pick(0, p.data->size() - 1);
        for (int t = 0; t < kProbesPerParam; ++t) {
            const size_t k = pick(probe_rng);
            const double saved = (*p.data)[k];

            // Training mode, on purpose: the backward pass differentiates the
            // batch statistics. The forward passes also change the running
            // estimates slightly, but nothing here reads them.
            (*p.data)[k] = saved + kH;
            const double up = net.forward_loss(d, rows.data(), kCheckBatch);
            (*p.data)[k] = saved - kH;
            const double down = net.forward_loss(d, rows.data(), kCheckBatch);
            (*p.data)[k] = saved;

            const double numeric = (up - down) / (2.0 * kH);
            const double got = analytic[pi][k];
            const double scale = std::max(1.0,
                std::max(std::fabs(got), std::fabs(numeric)));
            const double rel = std::fabs(got - numeric) / scale;
            if (rel > worst) {
                worst = rel;
                worst_where = p.name;
            }
            ++probes;
        }
    }

    printf("  %d probes across %zu parameter tensors\n", probes, net.params().size());
    printf("  worst disagreement %.2e (%s)\n", worst, worst_where.c_str());

    sanity::Expect chk;
    // Central differences at h = 1e-6 in double precision are accurate to
    // about 1e-9 here. A really wrong expression, such as a missing column sum
    // or an n where n-1 belongs, disagrees in the first digit or two. The
    // threshold is between the two, with a wide margin on both sides.
    chk.that("hand-written gradients match central differences", worst < 1e-6);
    chk.report();
}

// ---------------------------------------------------------------------------
// The deep network the diagnostics are for
// ---------------------------------------------------------------------------

constexpr int kDeepHidden = 100;
constexpr int kDeepLayers = 5;
constexpr int kProbeSteps = 2000;

// Python:
//   layers = [
//       Linear(n_embd * block_size, n_hidden), Tanh(),
//       Linear(           n_hidden, n_hidden), Tanh(),
//       Linear(           n_hidden, n_hidden), Tanh(),
//       Linear(           n_hidden, n_hidden), Tanh(),
//       Linear(           n_hidden, n_hidden), Tanh(),
//       Linear(           n_hidden, vocab_size),
//   ]
//   with torch.no_grad():
//       layers[-1].weight *= 0.1
//       for layer in layers[:-1]:
//           if isinstance(layer, Linear):
//               layer.weight *= 5/3
//
// One hidden layer is not enough to show anything. The point of the statistics
// is how they change with depth, and that needs several layers.
std::vector<std::unique_ptr<Layer>> build_deep(double gain, bool batchnorm,
    std::mt19937& g) {
    std::vector<std::unique_ptr<Layer>> layers;

    int fan_in = kBlock * kEmbed;
    for (int l = 0; l < kDeepLayers; ++l) {
        layers.push_back(std::make_unique<Linear>(fan_in, kDeepHidden,
            gain / std::sqrt(static_cast<double>(fan_in)), 0.0, !batchnorm, g));
        if (batchnorm) {
            layers.push_back(std::make_unique<BatchNorm1d>(kDeepHidden, kBnEps, kBnMomentum));
        }
        layers.push_back(std::make_unique<Tanh>());
        fan_in = kDeepHidden;
    }
    layers.push_back(std::make_unique<Linear>(fan_in, kVocab,
        kOutputGain / std::sqrt(static_cast<double>(fan_in)), 0.0, true, g));
    return layers;
}

// The video's numbers, named here so that the Expect blocks below read as
// claims instead of unexplained constants.
constexpr double kBigramLoss = 2.4541;     // what makemore_bigram.cpp scores
constexpr double kMlpDevLoss = 2.17;       // what makemore_mlp.cpp scores
constexpr double kVideoDevLoss = 2.10;     // the number this video finishes on

}  // namespace

// ---------------------------------------------------------------------------
// The walkthrough
// ---------------------------------------------------------------------------

void test_makemore_batchnorm() {
    std::cout << "== 0. the dataset ==\n";
    const std::vector<std::string> words = load_names();
    if (words.empty()) {
        std::cout << "  could not find names.txt - expected it next to "
            "makemore_batchnorm.cpp in makemore/\n";
        return;
    }

    // The same 80/10/10 split as in the last video. mt19937 and std::shuffle
    // do not reproduce Python's random numbers, so the splits contain other
    // names than in the video, and the losses differ in the last digit or two.
    std::mt19937 split_rng(42);
    const Splits splits = split_words(words, split_rng);
    const Dataset train_set = build_dataset(splits.train);
    const Dataset dev_set = build_dataset(splits.dev);
    const Dataset test_set = build_dataset(splits.test);

    validate_dataset(train_set, "train");
    validate_dataset(dev_set, "dev");
    validate_dataset(test_set, "test");

    printf("  train %7zu examples, dev %6zu, test %6zu (never touched below)\n",
        train_set.size(), dev_set.size(), test_set.size());

    sanity::Expect data;
    data.equal("examples in total", static_cast<double>(
        train_set.size() + dev_set.size() + test_set.size()), 228146);
    data.report();

    // One fixed batch, used by every network in sections 1 to 5, so that the
    // five sets of numbers differ only because of the initialisation.
    std::mt19937 batch_rng(2147483647);
    std::uniform_int_distribution<size_t> pick(0, train_set.size() - 1);
    std::vector<size_t> probe(kBatch);
    for (int b = 0; b < kBatch; ++b) {
        probe[b] = pick(batch_rng);
    }

    const std::vector<Recipe> all = recipes();
    CHECK(all.size() == 5u, "expected the video's five initialisations, got " << all.size());

    std::cout << "\n== 1. what randn init does before a single step ==\n";
    std::mt19937 g1(2147483647);
    Net naive(kEmbed, build_layers(all[0], g1), g1);
    printf("  %zu parameters\n", naive.parameter_count());
    report_init(all[0].name, naive, train_set, probe.data(), kBatch, true);
    std::cout <<
        "  Two separate problems, both visible above. The loss is an order of\n"
        "  magnitude above log(27) because the output layer starts with strong\n"
        "  opinions that are uncorrelated with the answer, and the histogram is\n"
        "  piled against -1 and +1 because the hidden preactivations are far too\n"
        "  wide for tanh. The first wastes the opening steps. The second is\n"
        "  the more expensive one: a unit out on the flat part of tanh\n"
        "  multiplies whatever gradient reaches it by nearly zero, so it barely\n"
        "  moves, and a unit that lands there for every example in a batch gets\n"
        "  nothing at all from that batch.\n";

    // The loss of a network that is confidently wrong has no upper limit. So
    // this check is about the direction, not about a specific number.
    sanity::Expect init;
    init.that("randn init starts far above a uniform guess",
        naive.forward_loss(train_set, probe.data(), kBatch)
        > 3.0 * std::log(static_cast<double>(kVocab)));
    init.that("randn init saturates most of the hidden layer",
        fraction_above(*naive.first_output_of("Tanh"), kSaturated) > 0.5);
    init.report();

    std::cout << "\n== 2. squashing the output layer ==\n";
    std::mt19937 g2(2147483647);
    Net tame_out(kEmbed, build_layers(all[1], g2), g2);
    report_init(all[1].name, tame_out, train_set, probe.data(), kBatch, false);
    std::cout <<
        "  W2 * 0.01 and b2 = 0 leave the logits nearly equal, which is exactly\n"
        "  what 'I have no idea' looks like. The hockey stick at the start of the\n"
        "  loss curve was the network spending steps undoing its own opinions.\n"
        "  Note that the tanh line is unchanged: this fixes one of the two.\n";

    std::cout << "\n== 3. squashing the hidden layer too ==\n";
    std::mt19937 g3(2147483647);
    Net tame_both(kEmbed, build_layers(all[2], g3), g3);
    report_init(all[2].name, tame_both, train_set, probe.data(), kBatch, true);
    std::cout <<
        "  W1 * 0.2 pulls the preactivations into the part of tanh that has a\n"
        "  derivative, and the histogram fills in. 0.2 is hand-picked, though,\n"
        "  and hand-picking one number per layer does not survive a deep net.\n";

    std::cout << "\n== 4. kaiming init ==\n";
    std::mt19937 g4(2147483647);
    why_divide_by_sqrt_fan_in(g4);
    Net kaiming(kEmbed, build_layers(all[3], g4), g4);
    report_init(all[3].name, kaiming, train_set, probe.data(), kBatch, false);
    printf("  the gain for tanh is %.4f (torch.nn.init.calculate_gain('tanh')),\n"
        "  so the hidden layer's scale is %.4f rather than the 0.2 above\n",
        kTanhGain, kTanhGain / std::sqrt(static_cast<double>(kBlock * kEmbed)));

    sanity::Expect fixed;
    fixed.close("squashed output layer starts near a uniform guess",
        tame_out.forward_loss(train_set, probe.data(), kBatch),
        std::log(static_cast<double>(kVocab)), 0.5);
    fixed.close("kaiming init starts near a uniform guess",
        kaiming.forward_loss(train_set, probe.data(), kBatch),
        std::log(static_cast<double>(kVocab)), 0.5);
    // This is not an absolute threshold. This hidden layer gets the raw
    // embedding as input, not the compressed output of another layer. So even
    // with a correctly scaled tanh, about a fifth of it is above 0.97. What
    // the gain improves is the comparison with the line above.
    fixed.that("kaiming more than halves the saturated fraction",
        fraction_above(*kaiming.first_output_of("Tanh"), kSaturated)
        < 0.5 * fraction_above(*naive.first_output_of("Tanh"), kSaturated));
    fixed.that("no neuron is saturated for the whole batch either way",
        dead_neurons(*kaiming.first_output_of("Tanh"), kDead) == 0
        && dead_neurons(*naive.first_output_of("Tanh"), kDead) == 0);
    fixed.report();

    std::cout << "\n== 5. batch normalisation ==\n";
    std::mt19937 g5(2147483647);
    Net bn(kEmbed, build_layers(all[4], g5), g5);
    report_init(all[4].name, bn, train_set, probe.data(), kBatch, true);
    std::cout <<
        "  Rather than arranging for the preactivations to arrive with the right\n"
        "  spread, batch normalisation measures what arrived and rescales it. The\n"
        "  net stops caring how its weights were initialised, which is the point:\n"
        "  the four recipes above are a hyperparameter, this is not.\n"
        "  The cost is that an example's output now depends on the other 31 in\n"
        "  its batch. The running mean and variance exist so that a single\n"
        "  example at evaluation time still has something to be normalised by,\n"
        "  and the coupling inside a batch turns out to act as a mild regulariser.\n";

    std::cout << "\n  checking the batch norm backward pass:\n";
    std::mt19937 gcheck(20150211);
    gradient_check(gcheck);

    std::cout << "\n== 6. training all five ==\n";
    printf("  %d steps of batch %d each, lr %.2f then %.2f - this is the slow part\n",
        kSteps, kBatch, kLearningRate, kDecayedRate);

    std::vector<TrainResult> results;
    std::unique_ptr<Net> batchnorm_net;
    for (size_t i = 0; i < all.size(); ++i) {
        // The same seed for every recipe, so the five networks differ only in
        // their initialisation. The order of the minibatches is identical for
        // all of them.
        std::mt19937 g(2147483647);
        auto net = std::make_unique<Net>(kEmbed, build_layers(all[i], g), g);
        printf("  [%zu/%zu] %s\n", i + 1, all.size(), all[i].name);
        // Only the first run prints its progress. Its loss falls very steeply
        // in the first steps, and sections 2 to 5 are about removing that.
        results.push_back(train(*net, train_set, dev_set, kSteps, i == 0, g));
        printf("        loss at step 0 %8.4f   train %.4f   dev %.4f   (%.1fs)\n",
            results.back().first_loss, results.back().train_loss,
            results.back().dev_loss, results.back().seconds);
        if (all[i].batchnorm) {
            batchnorm_net = std::move(net);
        }
    }

    printf("\n  %-32s %10s %9s %9s\n", "initialisation", "loss@0", "train", "dev");
    for (size_t i = 0; i < all.size(); ++i) {
        printf("  %-32s %10.4f %9.4f %9.4f\n", all[i].name, results[i].first_loss,
            results[i].train_loss, results[i].dev_loss);
    }
    printf("  for comparison: bigram %.4f, part 2's MLP %.4f, the video %.4f\n",
        kBigramLoss, kMlpDevLoss, kVideoDevLoss);
    std::cout <<
        "  The five land close together, which is the honest result for a net\n"
        "  this shallow - the video says so too. What the fixes buy here is the\n"
        "  first few thousand steps and the hidden units that would otherwise\n"
        "  have been dead. On the deeper net below they are the difference\n"
        "  between training and not.\n";

    double best_dev = results[0].dev_loss;
    for (const TrainResult& r : results) {
        best_dev = std::min(best_dev, r.dev_loss);
    }

    sanity::Expect trained;
    trained.that("every initialisation beats the bigram model",
        std::all_of(results.begin(), results.end(),
            [](const TrainResult& r) { return r.dev_loss < kBigramLoss; }));
    trained.that("randn init starts worst and the four fixes do not",
        std::all_of(results.begin() + 1, results.end(),
            [&](const TrainResult& r) { return r.first_loss < results[0].first_loss; }));
    // kSteps is a quarter of the video's schedule, so the losses are above the
    // video's 2.10, and the tolerance includes that difference. With a larger
    // kSteps, the result gets closer to 2.10.
    trained.close("best dev loss", best_dev, kVideoDevLoss, 0.25);
    trained.report();

    std::cout << "\n== 7. the same diagnostics on a deep net ==\n";
    std::cout <<
        "  Five hidden layers, where an initialisation that is slightly wrong\n"
        "  compounds five times instead of once.\n";

    const double gains[] = { 1.0, kTanhGain };
    for (double gain : gains) {
        std::mt19937 gd(2147483647);
        Net deep(kEmbed, build_deep(gain, false, gd), gd);
        printf("\n  gain %.3f, no batch norm, %zu parameters\n",
            gain, deep.parameter_count());
        deep.zero_grad();
        deep.forward_backward(train_set, probe.data(), kBatch);
        report_activations(deep);
        report_gradients(deep);
    }
    std::cout <<
        "\n  At gain 1.0 each layer shrinks what it was handed, so by the fifth\n"
        "  the activations have all but vanished and so has the gradient going\n"
        "  the other way. 5/3 is the number that cancels what tanh takes out.\n";

    {
        std::mt19937 gd(2147483647);
        Net deep(kEmbed, build_deep(kTanhGain, true, gd), gd);
        printf("\n  gain %.3f with batch norm, %zu parameters\n",
            kTanhGain, deep.parameter_count());
        deep.zero_grad();
        deep.forward_backward(train_set, probe.data(), kBatch);
        report_activations(deep);
        report_gradients(deep);
        report_grad_to_data(deep);
        std::cout << "\n";
        report_update_to_data(deep, train_set, kProbeSteps, kLearningRate, gd);
        std::cout <<
            "  Batch normalisation holds every layer at the same spread whatever\n"
            "  the gain was, which is why the video calls it the thing that makes\n"
            "  deep networks trainable rather than merely trainable-if-tuned.\n";
    }

    std::cout << "\n== 8. sampling from the batch-normalised net ==\n";
    CHECK(batchnorm_net != nullptr,
        "section 6 trained no batch-normalised network to sample from");
    std::mt19937 gs(2147483647);
    printf("  ");
    for (int i = 0; i < 20; ++i) {
        printf("%s ", batchnorm_net->sample(gs).c_str());
    }
    printf("\n  (sampling runs the net in evaluation mode: one name at a time has\n"
        "   no batch to normalise against, so the running estimates stand in)\n\n");
}
