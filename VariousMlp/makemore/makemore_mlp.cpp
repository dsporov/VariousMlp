// makemore_mlp.cpp - a C++ port of Andrej Karpathy's makemore, part 2
// ("Building makemore Part 2: MLP", video 3 of Neural Networks: Zero to Hero)
//
// The bigram model in makemore_bigram.cpp only ever sees one character of
// context, and the counting table it is equivalent to already has 27x27
// entries. Three characters of context would need 27^4 of them, most of which
// are never observed - so the video follows Bengio et al. 2003 instead: embed
// each character in a short vector, concatenate the last kBlock of them, and
// push that through one tanh hidden layer. The bigram net scored 2.4541 nats;
// this one gets under 2.2 on data it was never trained on.
//
// What the video builds, in order:
//   1. the dataset as (context, next character) pairs
//   2. an 80/10/10 train/dev/test split, so overfitting becomes measurable
//   3. the learning-rate sweep that picks the step size
//   4. a 2-dimensional embedding, small enough to plot, which sorts itself
//      into vowels and consonants with nobody asking it to
//   5. the same net scaled up: 10 embedding dimensions, 200 hidden units,
//      11897 parameters
//
// Needs names.txt (32033 names, one per line) next to this file.
//
// Section 4 trains kSteps minibatches and is by far the longest-running thing
// in this repo - minutes, not seconds, and Release only. Lower kSteps while
// you are iterating; the loss just ends up a little higher.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "check.h"

namespace {

// ---------------------------------------------------------------------------
// The dataset
// ---------------------------------------------------------------------------

constexpr int kVocab = 27;    // '.' plus the 26 lowercase letters
constexpr int kBlock = 3;     // characters of context the model gets to see

// encode, decode and load_names are deliberately a copy of the ones in
// makemore_bigram.cpp rather than a shared header: each video's file is meant
// to be read start to finish on its own, the way each video is watched.
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
    // Relative to a few plausible working directories: Visual Studio runs with
    // cwd = $(ProjectDir), a bare exe run has cwd = x64\Debug, and running from
    // the repo root is the other common case. A CMake build runs from somewhere
    // in its build tree, which none of those reach, so it names the directory
    // outright and that goes first.
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
            // The model has exactly 27 tokens, so a stray uppercase letter,
            // digit or accent has nowhere to go.
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

// One example per (context, next character) pair: xs holds kBlock token ids
// per example laid end to end, ys the single id that follows them.
struct Dataset {
    std::vector<int> xs;
    std::vector<int> ys;

    size_t size() const { return ys.size(); }
    const int* context(size_t i) const { return &xs[i * kBlock]; }
};

// The video's build_dataset: slide a window of kBlock characters across each
// name, starting from '...' and ending on the '.' that terminates it. A short
// name is not a special case, it simply keeps more of the padding.
//
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

// Python:
//   for x, y in zip(X[:n], Y[:n]):
//       print(''.join(itos[i.item()] for i in x), '--->', itos[y.item()])
void print_examples(const Dataset& d, size_t n) {
    for (size_t i = 0; i < std::min(n, d.size()); ++i) {
        const int* ctx = d.context(i);
        printf("    ");
        for (int j = 0; j < kBlock; ++j) {
            printf("%c", decode(ctx[j]));
        }
        printf(" ---> %c\n", decode(d.ys[i]));
    }
}

// Every token in here becomes a raw offset into the embedding table, so a bad
// one is an out-of-bounds read rather than a wrong answer. One pass per split
// before training starts; the training loop itself stays check-free.
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

// ---------------------------------------------------------------------------
// The split
// ---------------------------------------------------------------------------
//
// Splitting by name rather than by example is the point: every example cut
// from one name shares its letters, so scattering them across the splits would
// let the model meet the dev set during training.
//
// Python:
//   random.seed(42)
//   random.shuffle(words)
//   n1 = int(0.8 * len(words))
//   n2 = int(0.9 * len(words))
//   Xtr,  Ytr  = build_dataset(words[:n1])
//   Xdev, Ydev = build_dataset(words[n1:n2])
//   Xte,  Yte  = build_dataset(words[n2:])

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
// The network
// ---------------------------------------------------------------------------
//
// Bengio et al. 2003, exactly as the video assembles it. Torch does a whole
// batch at once, so every shape below leads with N; the C++ runs one example at
// a time, which is the same arithmetic with N = 1.
//
// Python:
//   emb = C[X]                                   # (N, kBlock, embed)
//   h = torch.tanh(emb.view(-1, 30) @ W1 + b1)   # (N, hidden); 30 = kBlock * embed
//   logits = h @ W2 + b2                         # (N, kVocab)
//   loss = F.cross_entropy(logits, Y)
//
// C is a lookup table, not a matrix multiply. The video spends a while on
// one_hot(x) @ C before pointing out that it only ever selects a row - the
// same observation the bigram file makes about its single weight matrix.

// Softmax over one row of kVocab logits. Subtracting the row max first costs
// nothing and keeps exp() away from overflow; softmax is invariant to it.
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

class Mlp {
public:
    // Per-example working memory. It lives outside the model so a caller
    // allocates it once for a whole training run rather than once per example.
    struct Scratch {
        std::vector<double> emb, hidden, logits, probs;
        std::vector<double> dlogits, dpre, demb;
    };

    Mlp(int embed, int hidden, std::mt19937& g)
        : embed_(embed), hidden_(hidden),
        C_(kVocab * embed), W1_(kBlock * embed * hidden), b1_(hidden),
        W2_(hidden * kVocab), b2_(kVocab)
    {
        CHECK(embed > 0 && hidden > 0, "embedding " << embed << " x hidden "
            << hidden << ": both have to be positive");

        // Python:
        //   C  = torch.randn((27, n_embd), generator=g)
        //   W1 = torch.randn((block_size * n_embd, n_hidden), generator=g)
        //   b1 = torch.randn(n_hidden, generator=g)
        //   W2 = torch.randn((n_hidden, 27), generator=g)
        //   b2 = torch.randn(27, generator=g)
        //   parameters = [C, W1, b1, W2, b2]
        //
        // Nothing here is scaled by fan-in, which leaves tanh saturated and the
        // starting loss absurdly high - that is the next video's subject, and
        // section 4 prints the evidence for it before training starts.
        std::normal_distribution<double> randn(0.0, 1.0);
        for (std::vector<double>* p : { &C_, &W1_, &b1_, &W2_, &b2_ }) {
            for (double& v : *p) {
                v = randn(g);
            }
        }

        dC_.assign(C_.size(), 0.0);
        dW1_.assign(W1_.size(), 0.0);
        db1_.assign(b1_.size(), 0.0);
        dW2_.assign(W2_.size(), 0.0);
        db2_.assign(b2_.size(), 0.0);
    }

    int embed() const { return embed_; }
    int hidden() const { return hidden_; }
    int inputs() const { return kBlock * embed_; }   // width of the concatenation
    // Python: sum(p.nelement() for p in parameters)
    size_t parameters() const {
        return C_.size() + W1_.size() + b1_.size() + W2_.size() + b2_.size();
    }

    double embedding(int token, int dim) const {
        CHECK(token >= 0 && token < kVocab, "token " << token << " is outside the vocabulary");
        CHECK(dim >= 0 && dim < embed_, "dimension " << dim << " is outside [0, " << embed_ << ")");
        return C_[token * embed_ + dim];
    }

    Scratch scratch() const {
        Scratch s;
        s.emb.assign(inputs(), 0.0);
        s.demb.assign(inputs(), 0.0);
        s.hidden.assign(hidden_, 0.0);
        s.dpre.assign(hidden_, 0.0);
        s.logits.assign(kVocab, 0.0);
        s.probs.assign(kVocab, 0.0);
        s.dlogits.assign(kVocab, 0.0);
        return s;
    }

    // The forward pass for one example. Unchecked on purpose: validate_dataset
    // has already established that every token indexes a real row of C, and
    // this runs kBatch times per step for hundreds of thousands of steps.
    //
    // Python:
    //   emb = C[X]
    //   h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    //   logits = h @ W2 + b2
    void forward(const int* ctx, Scratch& s) const {
        for (int j = 0; j < kBlock; ++j) {
            const double* row = &C_[ctx[j] * embed_];
            std::copy(row, row + embed_, &s.emb[j * embed_]);
        }

        std::copy(b1_.begin(), b1_.end(), s.hidden.begin());
        for (int k = 0; k < inputs(); ++k) {
            const double e = s.emb[k];
            const double* w = &W1_[k * hidden_];
            for (int h = 0; h < hidden_; ++h) {
                s.hidden[h] += e * w[h];
            }
        }
        for (int h = 0; h < hidden_; ++h) {
            s.hidden[h] = std::tanh(s.hidden[h]);
        }

        std::copy(b2_.begin(), b2_.end(), s.logits.begin());
        for (int h = 0; h < hidden_; ++h) {
            const double a = s.hidden[h];
            const double* w = &W2_[h * kVocab];
            for (int j = 0; j < kVocab; ++j) {
                s.logits[j] += a * w[j];
            }
        }
    }

    // Python: loss = F.cross_entropy(logits, Y)
    double loss(const int* ctx, int target, Scratch& s) const {
        forward(ctx, s);
        softmax(s.logits.data(), s.probs.data());
        return -std::log(s.probs[target]);
    }

    // Forward and backward for one example: returns that example's loss and
    // accumulates scale * d(loss)/d(parameter) into the gradients, so a caller
    // averaging over a minibatch passes scale = 1/batch.
    //
    // Written out by hand for the same reason as in the bigram file: an
    // autograd graph over 32 examples and 11897 parameters, rebuilt 200000
    // times, would dwarf the arithmetic it exists to organise.
    double forward_backward(const int* ctx, int target, double scale, Scratch& s) {
        forward(ctx, s);
        softmax(s.logits.data(), s.probs.data());

        // softmax followed by NLL collapses to probs - onehot(target).
        for (int j = 0; j < kVocab; ++j) {
            s.dlogits[j] = (s.probs[j] - (j == target ? 1.0 : 0.0)) * scale;
            db2_[j] += s.dlogits[j];
        }

        for (int h = 0; h < hidden_; ++h) {
            const double a = s.hidden[h];
            const double* w = &W2_[h * kVocab];
            double* dw = &dW2_[h * kVocab];
            double acc = 0.0;
            for (int j = 0; j < kVocab; ++j) {
                dw[j] += a * s.dlogits[j];
                acc += w[j] * s.dlogits[j];
            }
            // tanh' = 1 - tanh^2, and s.hidden already holds the tanh, so the
            // pre-activation never has to be kept around.
            s.dpre[h] = acc * (1.0 - a * a);
            db1_[h] += s.dpre[h];
        }

        for (int k = 0; k < inputs(); ++k) {
            const double e = s.emb[k];
            const double* w = &W1_[k * hidden_];
            double* dw = &dW1_[k * hidden_];
            double acc = 0.0;
            for (int h = 0; h < hidden_; ++h) {
                dw[h] += e * s.dpre[h];
                acc += w[h] * s.dpre[h];
            }
            s.demb[k] = acc;
        }

        // The embedding gradient scatters back into C, and += is load-bearing:
        // a character appearing twice in one context contributes twice.
        for (int j = 0; j < kBlock; ++j) {
            double* dc = &dC_[ctx[j] * embed_];
            const double* de = &s.demb[j * embed_];
            for (int e = 0; e < embed_; ++e) {
                dc[e] += de[e];
            }
        }

        return -std::log(s.probs[target]);
    }

    // Python:
    //   for p in parameters:
    //       p.grad = None
    void zero_grad() {
        for (std::vector<double>* p : { &dC_, &dW1_, &db1_, &dW2_, &db2_ }) {
            std::fill(p->begin(), p->end(), 0.0);
        }
    }

    // Python:
    //   for p in parameters:
    //       p.data += -lr * p.grad
    void step(double lr) {
        descend(C_, dC_, lr);
        descend(W1_, dW1_, lr);
        descend(b1_, db1_, lr);
        descend(W2_, dW2_, lr);
        descend(b2_, db2_, lr);
    }

    // Mean negative log likelihood over a whole split, in nats - the number
    // the video quotes for train and dev.
    //
    // Python:
    //   emb = C[Xdev]
    //   h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    //   logits = h @ W2 + b2
    //   loss = F.cross_entropy(logits, Ydev)
    double evaluate(const Dataset& d) const {
        CHECK(d.size() > 0, "nothing to evaluate");

        Scratch s = scratch();
        double total = 0.0;
        for (size_t i = 0; i < d.size(); ++i) {
            total += loss(d.context(i), d.ys[i], s);
        }
        return total / static_cast<double>(d.size());
    }

    // Python:
    //   context = [0] * block_size
    //   while True:
    //       emb = C[torch.tensor([context])]
    //       h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    //       logits = h @ W2 + b2
    //       probs = F.softmax(logits, dim=1)
    //       ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    //       context = context[1:] + [ix]
    //       out.append(ix)
    //       if ix == 0:
    //           break
    std::string sample(std::mt19937& g) const {
        // A model that never emits '.' would loop forever. The longest name in
        // the dataset is 15 characters, so this can only mean something broke.
        constexpr size_t kMaxLength = 1000;

        Scratch s = scratch();
        int ctx[kBlock];
        std::fill(std::begin(ctx), std::end(ctx), 0);

        std::string out;
        for (;;) {
            forward(ctx, s);
            softmax(s.logits.data(), s.probs.data());
            std::discrete_distribution<int> dist(s.probs.begin(), s.probs.end());
            const int ix = dist(g);
            if (ix == 0) {
                break;
            }
            out.push_back(decode(ix));
            CHECK(out.size() < kMaxLength,
                "sampling ran past " << kMaxLength << " characters without an end token");
            for (int j = 0; j + 1 < kBlock; ++j) {
                ctx[j] = ctx[j + 1];
            }
            ctx[kBlock - 1] = ix;
        }
        return out;
    }

private:
    static void descend(std::vector<double>& p, const std::vector<double>& d, double lr) {
        for (size_t i = 0; i < p.size(); ++i) {
            p[i] -= lr * d[i];
        }
    }

    int embed_;
    int hidden_;
    std::vector<double> C_, W1_, b1_, W2_, b2_;
    std::vector<double> dC_, dW1_, db1_, dW2_, db2_;
};

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

struct TrainConfig {
    int steps;
    int batch;
    double lr;
    double lr_decayed;   // the video's learning rate decay: one 10x cut
    int decay_at;
};

// Minibatch SGD, the other half of what makes this model tractable: 32
// examples give a gradient that points roughly the right way, and taking far
// more approximate steps beats taking a few exact ones.
//
// Python:
//   for i in range(200000):
//       ix = torch.randint(0, Xtr.shape[0], (32,))
//       emb = C[Xtr[ix]]
//       h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
//       logits = h @ W2 + b2
//       loss = F.cross_entropy(logits, Ytr[ix])
//       for p in parameters:
//           p.grad = None
//       loss.backward()
//       lr = 0.1 if i < 100000 else 0.01
//       for p in parameters:
//           p.data += -lr * p.grad
//       lossi.append(loss.item())
void train(Mlp& net, const Dataset& data, const TrainConfig& cfg, std::mt19937& g) {
    CHECK(cfg.steps > 0 && cfg.batch > 0,
        "asked for " << cfg.steps << " steps of batch " << cfg.batch);
    CHECK(data.size() > 0, "no training examples");

    std::uniform_int_distribution<size_t> pick(0, data.size() - 1);
    Mlp::Scratch s = net.scratch();
    const double scale = 1.0 / cfg.batch;
    const int report_every = std::max(1, cfg.steps / 10);

    // A 32-example loss is noisy enough that reading progress off it is
    // guesswork, so carry an exponential moving average alongside it. The
    // video reads the same trend off the scatter in its loss plot.
    double ema = 0.0;
    const auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step < cfg.steps; ++step) {
        net.zero_grad();
        double loss = 0.0;
        for (int b = 0; b < cfg.batch; ++b) {
            const size_t i = pick(g);
            loss += net.forward_backward(data.context(i), data.ys[i], scale, s);
        }
        loss *= scale;

        net.step(step < cfg.decay_at ? cfg.lr : cfg.lr_decayed);

        if (step == 0) {
            ema = loss;
        }
        else {
            ema += 0.001 * (loss - ema);
        }
        if (step % report_every == 0 || step == cfg.steps - 1) {
            printf("    step %6d  minibatch loss %.4f  (moving average %.4f)\n",
                step, loss, ema);
        }
    }

    const auto secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    printf("  %d steps of batch %d in %.1fs\n", cfg.steps, cfg.batch, secs);
}

// The video's learning-rate hunt: train a throwaway network for a thousand
// steps while raising the step size exponentially from 1e-3 to 1, and watch
// for where the loss stops falling and starts climbing again. Karpathy reads
// ~0.1 off the resulting plot; the bars below are that plot, one row per
// bucket of steps.
//
// Python:
//   lre = torch.linspace(-3, 0, 1000)
//   lrs = 10**lre
//   ...
//       lr = lrs[i]
//       lri.append(lre[i])
//       lossi.append(loss.item())
//   plt.plot(lri, lossi)
void find_learning_rate(const Dataset& data, int embed, int hidden, int batch,
    std::mt19937& g) {
    constexpr int kSweepSteps = 1000;
    constexpr int kBuckets = 20;
    constexpr double kLogLo = -3.0;      // 10^-3
    constexpr double kLogHi = 0.0;       // 10^0
    constexpr int kBarWidth = 40;

    CHECK(data.size() > 0, "no examples to sweep over");

    Mlp probe(embed, hidden, g);
    Mlp::Scratch s = probe.scratch();
    std::uniform_int_distribution<size_t> pick(0, data.size() - 1);
    const double scale = 1.0 / batch;

    std::vector<double> losses(kSweepSteps);
    std::vector<double> lrs(kSweepSteps);
    for (int step = 0; step < kSweepSteps; ++step) {
        const double t = static_cast<double>(step) / (kSweepSteps - 1);
        const double lr = std::pow(10.0, kLogLo + t * (kLogHi - kLogLo));

        probe.zero_grad();
        double loss = 0.0;
        for (int b = 0; b < batch; ++b) {
            const size_t i = pick(g);
            loss += probe.forward_backward(data.context(i), data.ys[i], scale, s);
        }
        probe.step(lr);

        lrs[step] = lr;
        losses[step] = loss * scale;
    }

    // Average within a bucket, because a single minibatch says very little. A
    // bucket that has gone infinite or NaN is a result too: that is the
    // right-hand end of the video's plot, where the step size is too large.
    const int per = kSweepSteps / kBuckets;
    std::vector<double> avg(kBuckets, 0.0);
    double worst = 0.0;
    for (int b = 0; b < kBuckets; ++b) {
        double sum = 0.0;
        bool finite = true;
        for (int i = b * per; i < (b + 1) * per; ++i) {
            finite = finite && std::isfinite(losses[i]);
            sum += losses[i];
        }
        avg[b] = finite ? sum / per : std::numeric_limits<double>::infinity();
        if (std::isfinite(avg[b])) {
            worst = std::max(worst, avg[b]);
        }
    }

    int best = 0;
    for (int b = 1; b < kBuckets; ++b) {
        if (avg[b] < avg[best]) {
            best = b;
        }
    }

    printf("  %d steps, learning rate rising from %.0e to %.0e:\n",
        kSweepSteps, lrs.front(), lrs.back());
    for (int b = 0; b < kBuckets; ++b) {
        const double lr = lrs[b * per + per / 2];
        if (!std::isfinite(avg[b])) {
            printf("    lr %7.4f  diverged\n", lr);
            continue;
        }
        const int bar = static_cast<int>(std::lround(avg[b] / worst * kBarWidth));
        printf("    lr %7.4f  loss %6.3f  %s%s\n", lr, avg[b],
            std::string(bar, '#').c_str(), b == best ? "  <- lowest" : "");
    }
    printf("  lowest bucket sits at lr %.3f; the video settles on 0.1\n",
        lrs[best * per + per / 2]);
}

// The payoff of a 2-dimensional embedding: printed as a scatter plot, the
// vowels have gathered in one region and the rare characters have been flung
// out to the edges. Nobody told the network that vowels are a category.
//
// Python:
//   plt.figure(figsize=(8, 8))
//   plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
//   for i in range(C.shape[0]):
//       plt.text(C[i, 0].item(), C[i, 1].item(), itos[i],
//                ha="center", va="center", color='white')
//   plt.grid('minor')
void print_embedding_scatter(const Mlp& net) {
    CHECK(net.embed() == 2,
        "the scatter plot needs a 2-dimensional embedding, got " << net.embed());

    constexpr int kWidth = 64;
    constexpr int kHeight = 21;

    double lo_x = net.embedding(0, 0), hi_x = lo_x;
    double lo_y = net.embedding(0, 1), hi_y = lo_y;
    for (int i = 1; i < kVocab; ++i) {
        lo_x = std::min(lo_x, net.embedding(i, 0));
        hi_x = std::max(hi_x, net.embedding(i, 0));
        lo_y = std::min(lo_y, net.embedding(i, 1));
        hi_y = std::max(hi_y, net.embedding(i, 1));
    }
    // A degenerate table collapsed onto a point would divide by zero here, and
    // a flat axis is still worth a picture with everything in one row.
    const double span_x = std::max(hi_x - lo_x, 1e-12);
    const double span_y = std::max(hi_y - lo_y, 1e-12);

    std::vector<std::string> grid(kHeight, std::string(kWidth, ' '));
    for (int i = 0; i < kVocab; ++i) {
        const double x = (net.embedding(i, 0) - lo_x) / span_x;
        const double y = (hi_y - net.embedding(i, 1)) / span_y;
        const int col = std::min(kWidth - 1, std::max(0,
            static_cast<int>(std::lround(x * (kWidth - 1)))));
        const int row = std::min(kHeight - 1, std::max(0,
            static_cast<int>(std::lround(y * (kHeight - 1)))));
        grid[row][col] = decode(i);
    }

    printf("  the 27 characters in embedding space (x %.2f..%.2f, y %.2f..%.2f):\n",
        lo_x, hi_x, lo_y, hi_y);
    for (const auto& line : grid) {
        printf("    |%s|\n", line.c_str());
    }
}

// ---------------------------------------------------------------------------
// The two configurations the video trains
// ---------------------------------------------------------------------------

// The small one exists for the picture: two embedding dimensions is what fits
// on a plot. The large one is the video's final model, and its 11897
// parameters are the number Karpathy reads out.
constexpr int kSmallEmbed = 2;
constexpr int kSmallHidden = 100;
constexpr int kSmallSteps = 30000;

constexpr int kEmbed = 10;
constexpr int kHidden = 200;
constexpr int kSteps = 200000;

constexpr int kBatch = 32;
constexpr double kLearningRate = 0.1;
constexpr double kDecayedRate = 0.01;

// The video's numbers, named here so the Expect blocks below read as claims
// rather than magic constants.
constexpr double kBigramLoss = 2.4541;     // what makemore_bigram.cpp scores
constexpr double kVideoDevLoss = 2.17;     // the number this video finishes on

}  // namespace

// ---------------------------------------------------------------------------
// The walkthrough
// ---------------------------------------------------------------------------

void test_makemore_mlp() {
    std::cout << "== 0. the dataset ==\n";
    const std::vector<std::string> words = load_names();
    if (words.empty()) {
        std::cout << "  could not find names.txt - expected it next to "
            "makemore_mlp.cpp in makemore/\n";
        return;
    }
    printf("  %zu names, %d characters of context per example\n", words.size(), kBlock);

    // The sliding window, spelled out for the first few names the way the
    // video does before it trusts build_dataset.
    const std::vector<std::string> first_few(
        words.begin(), words.begin() + std::min<size_t>(3, words.size()));
    std::cout << "  the window, for";
    for (const auto& w : first_few) {
        std::cout << " " << w;
    }
    std::cout << ":\n";
    const Dataset demo = build_dataset(first_few);
    print_examples(demo, demo.size());

    std::cout << "\n== 1. train / dev / test ==\n";
    // random.seed(42); random.shuffle(words) in the video. mt19937 and
    // std::shuffle do not reproduce Python's stream, so which names land in
    // which split differs: the per-split example counts below come out a few
    // hundred off the video's 182625 / 22655 / 22866, and the losses drift in
    // the last digit or two for the same reason.
    std::mt19937 split_rng(42);
    const Splits splits = split_words(words, split_rng);
    const Dataset train_set = build_dataset(splits.train);
    const Dataset dev_set = build_dataset(splits.dev);
    const Dataset test_set = build_dataset(splits.test);

    validate_dataset(train_set, "train");
    validate_dataset(dev_set, "dev");
    validate_dataset(test_set, "test");

    printf("  train %6zu names, %7zu examples\n", splits.train.size(), train_set.size());
    printf("  dev   %6zu names, %7zu examples\n", splits.dev.size(), dev_set.size());
    printf("  test  %6zu names, %7zu examples  (never touched below)\n",
        splits.test.size(), test_set.size());

    // Every name contributes one example per character plus one for the
    // terminating '.', so however the shuffle falls the three splits have to
    // add up to the 228146 bigrams the previous file counts.
    sanity::Expect data;
    data.equal("train names", static_cast<double>(splits.train.size()), 25626);
    data.equal("dev names", static_cast<double>(splits.dev.size()), 3203);
    data.equal("test names", static_cast<double>(splits.test.size()), 3204);
    data.equal("examples in total",
        static_cast<double>(train_set.size() + dev_set.size() + test_set.size()), 228146);
    data.report();

    std::cout << "\n== 2. finding a learning rate ==\n";
    std::mt19937 g(2147483647);
    find_learning_rate(train_set, kSmallEmbed, kSmallHidden, kBatch, g);

    std::cout << "\n== 3. two embedding dimensions ==\n";
    Mlp small(kSmallEmbed, kSmallHidden, g);
    printf("  %d embedding dims, %d hidden units, %zu parameters\n",
        small.embed(), small.hidden(), small.parameters());

    const TrainConfig small_cfg{ kSmallSteps, kBatch, kLearningRate, kDecayedRate,
        kSmallSteps * 2 / 3 };
    train(small, train_set, small_cfg, g);

    const double small_train = small.evaluate(train_set);
    const double small_dev = small.evaluate(dev_set);
    printf("  train %.4f  dev %.4f\n", small_train, small_dev);
    print_embedding_scatter(small);

    // Two dimensions is a real bottleneck - the video's reason for widening it
    // - but it should already be ahead of counting bigrams.
    sanity::Expect small_chk;
    small_chk.equal("parameters", static_cast<double>(small.parameters()), 3481);
    small_chk.that("beats the bigram model", small_dev < kBigramLoss);
    small_chk.report();

    std::cout << "\n== 4. scaling up ==\n";
    Mlp net(kEmbed, kHidden, g);
    printf("  %d embedding dims, %d hidden units, %zu parameters\n",
        net.embed(), net.hidden(), net.parameters());

    // -log(1/27) is what a model that knows nothing at all scores. The
    // untrained net is far worse than that: randn weights leave it confidently
    // wrong rather than uniformly unsure, which is where the next video starts.
    const double uniform_loss = std::log(static_cast<double>(kVocab));
    const double initial_dev = net.evaluate(dev_set);
    printf("  before training: dev %.4f  (a uniform guess would score %.4f)\n",
        initial_dev, uniform_loss);

    printf("  training %d steps, lr %.2f then %.2f - this is the slow part\n",
        kSteps, kLearningRate, kDecayedRate);
    const TrainConfig cfg{ kSteps, kBatch, kLearningRate, kDecayedRate, kSteps / 2 };
    train(net, train_set, cfg, g);

    const double train_loss = net.evaluate(train_set);
    const double dev_loss = net.evaluate(dev_set);
    printf("  train %.4f  dev %.4f   (the video finishes around %.2f on dev)\n",
        train_loss, dev_loss, kVideoDevLoss);
    printf("  bigram model, for comparison: %.4f\n", kBigramLoss);

    // The claim of the video, checked rather than asserted in a comment: three
    // characters of context and a hidden layer beat counting bigrams, and
    // train and dev staying close says the model is still too small to overfit
    // this dataset. The dev tolerance is wide because the split is not the
    // video's - see the note in section 1.
    sanity::Expect chk;
    chk.equal("parameters", static_cast<double>(net.parameters()), 11897);
    chk.that("training helped", dev_loss < initial_dev);
    chk.that("beats the bigram model", dev_loss < kBigramLoss);
    chk.close("dev loss", dev_loss, kVideoDevLoss, 0.1);
    chk.that("train and dev still track each other", dev_loss - train_loss < 0.2);
    chk.report();

    std::cout << "\n== 5. sampling ==\n";
    std::mt19937 gs(2147483647);
    printf("  ");
    for (int i = 0; i < 20; ++i) {
        printf("%s ", net.sample(gs).c_str());
    }
    printf("\n  (name-shaped now, where the bigram model produced consonant piles)\n\n");
}
