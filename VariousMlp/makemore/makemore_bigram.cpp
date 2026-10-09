// makemore_bigram.cpp - a C++ port of Andrej Karpathy's makemore, part 1
// ("The spelled-out intro to language modeling: building makemore", video 2 of
//  Neural Networks: Zero to Hero)
//
// The video builds the same bigram character-level language model twice:
//   1. count every bigram in the training set, then normalise the rows
//   2. fit a 27x27 weight matrix by gradient descent on the same bigrams
// Both reach a loss of ~2.454 nats, and that is the whole point: the neural
// net does not learn anything cleverer than counting; it *finds the counts
// again*. The goal of the file is to make the two numbers agree.
//
// Needs names.txt (32033 names, one per line) next to this file.
//
// Part 4 does full-batch gradient descent over all 228146 bigrams, 100 steps:
// ~6s in Release, ~24s in Debug. Use Release if you run it often.
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "check.h"

namespace {

// ---------------------------------------------------------------------------
// The dataset
// ---------------------------------------------------------------------------

constexpr int kVocab = 27;   // '.' plus the 26 lowercase letters

// The video's stoi / itos dicts. Because the alphabet is contiguous, simple
// arithmetic replaces the maps.
// Any character other than '.' and a-z would give an index outside
// [0, kVocab) and read past the end of the 27x27 arrays, so one comparison per
// character is worth it. load_names filters the dataset in advance; this
// check catches the remaining cases, such as hand-written test strings.
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
        if (!f) continue;

        std::vector<std::string> words;
        std::string line;
        size_t skipped = 0;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (line.back() == '\r')
                line.pop_back();  // CRLF
            // The model has exactly 27 tokens, so an unexpected uppercase
            // letter, digit or accented letter has no token. Such lines are
            // dropped here, so that encode() does not stop the whole run
            // because of one bad name.
            const bool clean = std::all_of(line.begin(), line.end(),
                [](char c) { return c >= 'a' && c <= 'z'; });
            if (clean)
                words.push_back(line);
            else
                ++skipped;
        }
        if (!words.empty()) {
            printf("  loaded %zu names from %s\n", words.size(), path);
            if (skipped) printf("  skipped %zu line(s) with characters outside a-z\n", skipped);
            return words;
        }
    }
    return {};
}

// Python:
//   chs = ['.'] + list(w) + ['.']
//   for ch1, ch2 in zip(chs, chs[1:]):
//       ix1 = stoi[ch1]
//       ix2 = stoi[ch2]
//
// Going through the word with a 'prev' index gives the same pairs, without
// building the padded list for every word.
template <class F>
void for_each_bigram(const std::string& w, F&& f) {
    int prev = 0;                                  // leading '.'
    for (char c : w) {
        const int cur = encode(c);
        f(prev, cur);
        prev = cur;
    }
    f(prev, 0);                                    // trailing '.'
}

// ---------------------------------------------------------------------------
// 1. Counting
// ---------------------------------------------------------------------------

// Python:
//   N = torch.zeros((27, 27), dtype=torch.int32)
//   for w in words:
//       ...
//           N[ix1, ix2] += 1
std::vector<int> count_bigrams(const std::vector<std::string>& words) {
    std::vector<int> N(kVocab * kVocab, 0);
    for (const auto& w : words) {
        for_each_bigram(w, [&](int a, int b) {
            ++N[a * kVocab + b];
        });
    }
    return N;
}

// Python: sorted(b.items(), key=lambda kv: -kv[1])
void print_top_bigrams(const std::vector<int>& N, int k) {
    CHECK(N.size() == kVocab * kVocab, "expected a " << kVocab << "x" << kVocab
        << " count matrix, got " << N.size() << " entries");
    CHECK(k > 0 && k <= static_cast<int>(N.size()),
        "asked for the top " << k << " of " << N.size() << " bigrams");

    std::vector<std::pair<int, int>> flat;         // (count, flat index)
    flat.reserve(N.size());
    for (int i = 0; i < static_cast<int>(N.size()); ++i) {
        flat.emplace_back(N[i], i);
    }
    std::sort(flat.begin(), flat.end(), std::greater<>());

    printf("  top %d bigrams:\n", k);
    for (int i = 0; i < k; ++i) {
        const int idx = flat[i].second;
        printf("    %c%c  %5d\n", decode(idx / kVocab), decode(idx % kVocab), flat[i].first);
    }
}

// A text version of the video's plt.imshow of the count matrix.
//
// Python:
//   plt.figure(figsize=(16, 16))
//   plt.imshow(N, cmap='Blues')
//   for i in range(27):
//       for j in range(27):
//           chstr = itos[i] + itos[j]
//           plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
//           plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
//   plt.axis('off')
void print_count_matrix(const std::vector<int>& N) {
    printf("  the 27x27 count matrix (rows = first char, cols = second):\n   ");
    for (int j = 0; j < kVocab; ++j) printf("%5c", decode(j));
    printf("\n");
    for (int i = 0; i < kVocab; ++i) {
        printf("  %c", decode(i));
        for (int j = 0; j < kVocab; ++j) printf("%5d", N[i * kVocab + j]);
        printf("\n");
    }
}

// The postcondition for any function that produces a probability
// distribution. It costs 27x27 additions, once per matrix. It catches a
// normalisation along the wrong axis, which otherwise only looks like a bad
// model.
void check_rows_sum_to_one(const std::vector<double>& P, const char* what) {
    CHECK(P.size() == kVocab * kVocab, what << ": expected a " << kVocab << "x"
        << kVocab << " matrix, got " << P.size() << " entries");
    for (int i = 0; i < kVocab; ++i) {
        double sum = 0.0;
        for (int j = 0; j < kVocab; ++j)
            sum += P[i * kVocab + j];
        CHECK(std::fabs(sum - 1.0) < 1e-12,
            what << ": row '" << decode(i) << "' sums to " << sum << ", not 1");
    }
}

// P = (N + smoothing), normalised per row. smoothing=0 is the raw model.
// smoothing=1 is the video's "model smoothing", which removes the infinite
// losses.
//
// Python:
//   P = (N + 1).float()
//   P /= P.sum(1, keepdims=True)
std::vector<double> row_normalise(const std::vector<int>& N, double smoothing) {
    CHECK(N.size() == kVocab * kVocab, "expected a " << kVocab << "x" << kVocab
        << " count matrix, got " << N.size() << " entries");
    CHECK(smoothing >= 0.0, "negative smoothing (" << smoothing << ") makes negative probabilities");

    std::vector<double> P(N.size());
    for (int i = 0; i < kVocab; ++i) {
        double sum = 0.0;
        for (int j = 0; j < kVocab; ++j) {
            sum += N[i * kVocab + j] + smoothing;
        }
        
        // A character that never starts a bigram has a row of zeros, and 0/0
        // would silently fill that row with NaN.
        CHECK(sum > 0.0, "no counts in the row for '" << decode(i)
            << "'; pass smoothing > 0 to make it a valid distribution");
        for (int j = 0; j < kVocab; ++j) {
            P[i * kVocab + j] = (N[i * kVocab + j] + smoothing) / sum;
        }
    }
    
    check_rows_sum_to_one(P, "normalised counts");
    return P;
}

// ---------------------------------------------------------------------------
// 2. Sampling
// ---------------------------------------------------------------------------

// torch.multinomial on one row. discrete_distribution normalises by itself, so
// the row can contain counts or probabilities.
//
// Python:
//   out = []
//   ix = 0
//   while True:
//       p = P[ix]
//       ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
//       out.append(itos[ix])
//       if ix == 0:
//           break
std::string sample_name(const std::vector<double>& P, std::mt19937& g) {
    CHECK(P.size() == kVocab * kVocab, "expected a " << kVocab << "x" << kVocab
        << " probability matrix, got " << P.size() << " entries");

    // A model that never produces the end token would loop forever. The
    // uniform model in the walkthrough already produces names of 26
    // characters on average. The limit is therefore high, so that reaching it
    // can only mean that something is broken.
    constexpr size_t kMaxLength = 1000;

    std::string out;
    int ix = 0;                                    // always start at '.'
    for (;;) {
        const double* row = &P[ix * kVocab];
        double sum = 0.0;
        for (int j = 0; j < kVocab; ++j) {
            sum += row[j];
        }
        CHECK(sum > 0.0, "row '" << decode(ix) << "' has no probability mass to sample from");

        std::discrete_distribution<int> dist(row, row + kVocab);
        ix = dist(g);
        if (ix == 0)
            break;                        // sampled the end token
        out.push_back(decode(ix));
        CHECK(out.size() < kMaxLength,
            "sampling ran past " << kMaxLength << " characters without an end token");
    }
    return out;
}

// ---------------------------------------------------------------------------
// 3. Evaluation: average negative log likelihood, in nats
// ---------------------------------------------------------------------------

// Python:
//   log_likelihood = 0.0
//   n = 0
//   for w in words:
//       ...
//           prob = P[ix1, ix2]
//           log_likelihood += torch.log(prob)
//           n += 1
//   nll = -log_likelihood
//   print(f'{nll / n}')
double average_nll(const std::vector<std::string>& words,
    const std::vector<double>& P,
    long long* bigrams_out = nullptr) {
    CHECK(!words.empty(), "no words to score");
    CHECK(P.size() == kVocab * kVocab, "expected a " << kVocab << "x" << kVocab
        << " probability matrix, got " << P.size() << " entries");

    double log_likelihood = 0.0;
    long long n = 0;
    for (const auto& w : words) {
        for_each_bigram(w, [&](int a, int b) {
            log_likelihood += std::log(P[a * kVocab + b]);
            ++n;
        });
    }

    // Every word contributes at least one bigram, so n == 0 would mean that
    // the padding in for_each_bigram is broken. (A bigram that never occurred
    // still gives -inf here, on purpose. The walkthrough shows this with
    // 'andrejq' before it introduces smoothing.)
    CHECK(n > 0, "scored " << words.size() << " words but counted no bigrams");

    if (bigrams_out)
        *bigrams_out = n;
    return -log_likelihood / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// 4. The same model as a neural net
// ---------------------------------------------------------------------------
//
// The network is one linear layer and nothing else. That is easy to miss in
// the video's `W = torch.randn((27, 27))`. The two types below make it
// explicit: 27 neurons with 27 weights each, no bias, and no activation
// function. The softmax in softmax_cross_entropy is the only nonlinearity in
// the model. So this is multinomial logistic regression. The hidden layer,
// and with it the name MLP, only comes in the next video.

// Softmax over one row of kVocab logits. Subtracting the row maximum first
// costs almost nothing and prevents overflow in exp(). It does not change the
// result.
//
// Python:
//   counts = logits.exp()
//   probs = counts / counts.sum(1, keepdims=True)
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

// A single neuron: kInputs weights and no bias. It is a view into the layer
// and does not own its weights. LinearLayer stores the weights of all 27
// neurons interleaved, so one neuron's weights form a strided column.
class Neuron {
public:
    static constexpr int kInputs = kVocab;

    Neuron(const double* first_weight, int stride)
        : w_(first_weight), stride_(stride)
    {
    }

    // The weight this neuron gives to `input`. It is also the neuron's output
    // for that input: with a one-hot input, the dot product reduces to one
    // term, because every other term is multiplied by zero. That is why
    // exp(W) ends up similar to the count matrix: each weight is really a log
    // count.
    double weight(int input) const {
        CHECK(input >= 0 && input < kInputs,
            "input " << input << " is outside [0, " << kInputs << ")");
        return w_[input * stride_];
    }

private:
    const double* w_;
    int stride_;
};

// The whole network: one layer, kNeurons wide. It holds the weights and their
// gradients next to each other, in the same way as micrograd's Value holds
// .data and .grad.
//
// The storage is input-major, w_[input * kNeurons + neuron], which is the
// transpose of "each neuron owns a contiguous array". That is on purpose. A
// one-hot input needs the weights of all 27 neurons for one character, and
// with this layout they form one contiguous row. The video makes the same
// point from the other direction: one_hot(x) @ W is an expensive way to write
// an array index.
class LinearLayer {
public:
    static constexpr int kNeurons = kVocab;
    static constexpr int kInputs = kVocab;
    static constexpr int kParameters = kNeurons * kInputs;

    explicit LinearLayer(std::mt19937& g)
        : w_(kParameters), dw_(kParameters, 0.0)
    {
        std::normal_distribution<double> randn(0.0, 1.0);
        for (auto& v : w_) {
            v = randn(g);
        }
    }

    Neuron neuron(int j) const {
        CHECK(j >= 0 && j < kNeurons,
            "neuron " << j << " is outside [0, " << kNeurons << ")");
        return Neuron(&w_[j], kNeurons);
    }

    // The logits for a one-hot input, one per neuron. There is no check here,
    // on purpose: this is the inner loop, and validate_examples has already
    // confirmed that every x and y in the training set is a valid row index.
    //
    // Python:
    //   xenc = F.one_hot(xs, num_classes=27).float()
    //   logits = xenc @ W
    const double* logits(int input) const {
        return &w_[input * kNeurons];
    }

    // Where the backward pass adds up d(loss)/d(logits) for that same row.
    double* dlogits(int input) {
        return &dw_[input * kNeurons];
    }

    void zero_grad() {
        std::fill(dw_.begin(), dw_.end(), 0.0);
    }

    // The video's 0.01 * (W**2).mean(): a force that pulls every weight
    // toward zero. In gradient descent, this has the same effect as add-one
    // smoothing of the counts. It is split into two functions, so that the
    // reported number and the gradient can be read separately. Both loops are
    // over 27x27 weights, which costs almost nothing compared with the pass
    // over the examples.
    double l2_penalty(double reg) const {
        double sq = 0.0;
        for (int k = 0; k < kParameters; ++k) {
            sq += w_[k] * w_[k];
        }
        return reg * sq / kParameters;
    }

    void add_l2_grad(double reg) {
        const double scale = reg * 2.0 / kParameters;
        for (int k = 0; k < kParameters; ++k) {
            dw_[k] += scale * w_[k];
        }
    }

    // Python: W.data += -50 * W.grad
    void step(double lr) {
        for (int k = 0; k < kParameters; ++k) {
            w_[k] -= lr * dw_[k];
        }
    }

    // The trained layer as a 27x27 probability matrix. With it, the count
    // model's scoring and sampling code can run unchanged on the neural net.
    std::vector<double> probabilities() const {
        std::vector<double> P(kParameters);
        for (int i = 0; i < kInputs; ++i) {
            softmax(logits(i), &P[i * kNeurons]);
        }
        return P;
    }

private:
    std::vector<double> w_;
    std::vector<double> dw_;
};

// Softmax and negative log likelihood for one example, in one function.
// Returns the loss, and adds scale * d(loss)/d(logits) into `dlogits`.
//
// For softmax followed by NLL, the gradient reduces to one line:
// probs - onehot(target). That is what loss.backward() computes in the video.
// Writing it by hand keeps this file independent of the micrograd engine.
// With micrograd, 228k examples x 27 classes would be ~14M Value nodes per
// step.
//
// Python: loss = -probs[torch.arange(num), ys].log().mean()
double softmax_cross_entropy(const double* logits, int target, double scale,
    double* dlogits) {
    double probs[kVocab];
    softmax(logits, probs);
    for (int j = 0; j < kVocab; ++j) {
        dlogits[j] += (probs[j] - (j == target ? 1.0 : 0.0)) * scale;
    }
    return -std::log(probs[target]);
}

// Shows the trained layer from the point of view of one neuron. The claim that
// these 27 objects are the whole model is easier to believe when you can see
// one of them. Neuron j votes for character j, so its strongest weights should
// belong to the characters that most often come before character j.
void print_neuron(const LinearLayer& layer, int j, int top_k) {
    CHECK(top_k > 0 && top_k <= Neuron::kInputs,
        "asked for the top " << top_k << " of " << Neuron::kInputs << " weights");

    const Neuron n = layer.neuron(j);
    std::vector<std::pair<double, int>> flat;      // (weight, input)
    flat.reserve(Neuron::kInputs);
    for (int i = 0; i < Neuron::kInputs; ++i) {
        flat.emplace_back(n.weight(i), i);
    }
    std::sort(flat.begin(), flat.end(), std::greater<>());

    printf("  neuron '%c' (%d weights, no bias), strongest inputs:",
        decode(j), Neuron::kInputs);
    for (int i = 0; i < top_k; ++i) {
        printf("  %c%c %+.2f", decode(flat[i].second), decode(j), flat[i].first);
    }
    printf("\n");
}

// Every index here becomes raw pointer arithmetic into the layer, so a bad
// index would be an out-of-bounds write, not just a wrong answer. This is one
// pass over the dataset before training starts, so the training loop itself
// needs no checks.
void validate_examples(const std::vector<int>& xs, const std::vector<int>& ys) {
    CHECK(!xs.empty(), "no training examples");
    CHECK(xs.size() == ys.size(),
        xs.size() << " inputs but " << ys.size() << " targets");

    for (size_t i = 0; i < xs.size(); ++i) {
        CHECK(xs[i] >= 0 && xs[i] < kVocab && ys[i] >= 0 && ys[i] < kVocab,
            "example " << i << " is (" << xs[i] << ", " << ys[i]
            << "), outside [0, " << kVocab << ")");
    }
}

void train_neural_net(const std::vector<std::string>& words, double count_model_loss) {
    // The training set: one example per bigram. xs and ys hold the same
    // characters, shifted by one. So a single stream of N+1 tokens could be
    // used for both, with xs reading from index 0 and ys from index 1. The '.'
    // between two names would then be both the end of one name and the start
    // of the next. Language models usually store their data that way, but
    // this file does not, on purpose: two named arrays match the video, and
    // the extra copy costs less than a megabyte.
    //
    // Python:
    //   xs, ys = [], []
    //   for w in words:
    //       ...
    //           xs.append(ix1)
    //           ys.append(ix2)
    //   xs = torch.tensor(xs)
    //   ys = torch.tensor(ys)
    //   num = xs.nelement()
    std::vector<int> xs, ys;
    for (const auto& w : words) {
        for_each_bigram(w, [&](int a, int b) {
            xs.push_back(a);
            ys.push_back(b);
        });
    }

    const size_t num = xs.size();
    printf("  %zu bigram examples\n", num);
    printf("  network: 1 linear layer, %d neurons x %d weights, no bias, "
        "no activation (%d parameters)\n",
        LinearLayer::kNeurons, LinearLayer::kInputs, LinearLayer::kParameters);

    validate_examples(xs, ys);

    // W = torch.randn((27, 27), requires_grad=True)
    std::mt19937 g(2147483647);
    LinearLayer layer(g);

    double first_loss = 0.0, last_loss = 0.0;

    const double inv_n = 1.0 / static_cast<double>(num);
    const double reg = 0.01;        // the video's 0.01 * (W**2).mean()
    const double lr = 50.0;         // W.data += -50 * W.grad
    const int steps = 100;

    const auto t0 = std::chrono::steady_clock::now();

    // Python:
    //   for k in range(100):
    //       xenc = F.one_hot(xs, num_classes=27).float()
    //       logits = xenc @ W
    //       counts = logits.exp()
    //       probs = counts / counts.sum(1, keepdims=True)
    //       loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    //       W.grad = None
    //       loss.backward()
    //       W.data += -50 * W.grad
    for (int step = 0; step < steps; ++step) {
        layer.zero_grad();

        // Forward and backward in the same pass: each example reads the row of
        // logits that its input character selects, and writes that row's
        // gradient directly.
        double loss = 0.0;
        for (size_t i = 0; i < num; ++i) {
            const int x = xs[i];
            const double* logits = layer.logits(x);
            double* dlogits = layer.dlogits(x);
            loss += softmax_cross_entropy(logits, ys[i], inv_n, dlogits);
        }
        // L2 regularisation, as an adjustable setting instead of a
        // hand-picked smoothing constant
        layer.add_l2_grad(reg);
        loss = loss * inv_n + layer.l2_penalty(reg);

        layer.step(lr);

        if (step == 0) {
            first_loss = loss;
        }
        last_loss = loss;

        if (step % 10 == 0 || step == steps - 1) {
            printf("    step %3d  loss %.6f (incl. regularisation)\n", step, loss);
        }
    }

    const auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("  trained in %.2fs\n", secs);

    // softmax(W) should now look like the count matrix: the net found the
    // counts again.
    const std::vector<double> P = layer.probabilities();
    check_rows_sum_to_one(P, "softmax of W");
    print_neuron(layer, encode('n'), 5);

    // A fair comparison: both numbers below are the plain nll, without the
    // regularisation term. The net is a little above the count model, because
    // the L2 term still smooths it. With reg = 0, the two numbers converge.
    long long n = 0;
    const double net_loss = average_nll(words, P, &n);
    printf("  count model loss %.6f  vs  neural net loss %.6f\n",
        count_model_loss, net_loss);

    // The claim that this whole file demonstrates, checked in code instead of
    // stated in a comment: gradient descent found the counts again. With
    // reg = 0.01, the net is about 0.021 nats above the count model after 100
    // steps, so the tolerance is just above that. If you raise reg or stop
    // training early, this check fails.
    sanity::Expect chk;
    chk.that("training loss went down", last_loss < first_loss);
    chk.close("net matches the count model", net_loss, count_model_loss, 0.03);
    chk.report();

    std::mt19937 gs(2147483647);
    printf("  samples from the trained net: ");
    for (int i = 0; i < 5; ++i) {
        printf("%s ", sample_name(P, gs).c_str());
    }
    printf("\n\n");
}

}  // namespace

// ---------------------------------------------------------------------------
// The walkthrough
// ---------------------------------------------------------------------------

void test_makemore() {
    std::cout << "== 0. the dataset ==\n";
    const std::vector<std::string> words = load_names();
    if (words.empty()) {
        std::cout << "  could not find names.txt - expected it next to "
            "makemore_bigram.cpp in makemore/\n";
        return;
    }

    // Python:
    //   len(words)
    //   min(len(w) for w in words)
    //   max(len(w) for w in words)
    size_t shortest = words[0].size(), longest = 0;
    for (const auto& w : words) {
        shortest = std::min(shortest, w.size());
        longest = std::max(longest, w.size());
    }
    std::cout << "  first 8: ";
    for (size_t i = 0; i < std::min<size_t>(8, words.size()); ++i)
        std::cout << words[i] << " ";
    printf("\n  lengths %zu..%zu\n", shortest, longest);

    // Everything below is checked against the video, and so against the
    // video's dataset. With a different names.txt, these checks are expected
    // to fail. That is why they use Expect, which reports and continues, and
    // not CHECK, which would stop the walkthrough.
    sanity::Expect data;
    data.equal("names loaded", static_cast<double>(words.size()), 32033);
    data.equal("shortest name", static_cast<double>(shortest), 2);
    data.equal("longest name", static_cast<double>(longest), 15);
    data.report();
    std::cout << "\n";

    std::cout << "== 1. counting bigrams ==\n";
    const std::vector<int> N = count_bigrams(words);
    print_top_bigrams(N, 10);

    // Both models are built on the counts, so the four counts that the video
    // mentions, and the total, are checked here.
    long long total = 0;
    for (int c : N)
        total += c;
    sanity::Expect counts;
    counts.equal("n. count", N[encode('n') * kVocab + encode('.')], 6763);
    counts.equal("a. count", N[encode('a') * kVocab + encode('.')], 6640);
    counts.equal("an count", N[encode('a') * kVocab + encode('n')], 5438);
    counts.equal(".a count", N[encode('.') * kVocab + encode('a')], 4410);
    counts.equal("total bigrams", static_cast<double>(total), 228146);
    counts.report();
    print_count_matrix(N);
    std::cout << "\n";

    std::cout << "== 2. sampling ==\n";
    std::mt19937 g(2147483647);

    // For comparison, the video first samples from a uniform model, to show
    // what an untrained model produces.
    //
    // Python: p = torch.ones(27) / 27.0
    const std::vector<double> uniform(kVocab * kVocab, 1.0 / kVocab);
    std::cout << "  uniform model:  ";
    for (int i = 0; i < 5; ++i)
        std::cout << sample_name(uniform, g) << " ";

    const std::vector<double> P = row_normalise(N, 0.0);
    std::cout << "\n  bigram model:   ";
    for (int i = 0; i < 5; ++i)
        std::cout << sample_name(P, g) << " ";
    std::cout << "\n  (names differ from the video: torch's RNG stream is not "
        "reproducible here)\n\n";

    std::cout << "== 3. how good is it? negative log likelihood ==\n";
    // The last digit differs from the video. The video adds up the log
    // likelihood in a Python loop over float32 scalars, and 228k additions to
    // a running total of ~560000 lose a few units of precision. This code uses
    // double.
    long long n = 0;
    const double nll = average_nll(words, P, &n);
    printf("  %lld bigrams, average nll %.4f nats   (video: 228146 bigrams, ~2.454)\n", n, nll);

    sanity::Expect loss;
    loss.equal("bigrams scored", static_cast<double>(n), 228146);
    loss.close("average nll", nll, 2.4541, 1e-3);
    loss.report();

    // A bigram that never occurs in the training set makes the loss infinite.
    const std::vector<std::string> probe = { "andrejq" };
    printf("  nll of 'andrejq' unsmoothed: %.4f  ('jq' was never seen)\n",
        average_nll(probe, P, nullptr));

    const std::vector<double> Ps = row_normalise(N, 1.0);
    printf("  nll of 'andrejq' smoothed:   %.4f\n", average_nll(probe, Ps, nullptr));
    printf("  whole dataset, smoothed:     %.4f\n\n", average_nll(words, Ps, &n));

    std::cout << "== 4. the same model, by gradient descent ==\n";
    train_neural_net(words, nll);
}
