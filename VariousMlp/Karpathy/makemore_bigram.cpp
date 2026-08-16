// makemore_bigram.cpp - a C++ port of Andrej Karpathy's makemore, part 1
// ("The spelled-out intro to language modeling: building makemore", video 2 of
//  Neural Networks: Zero to Hero)
//
// The video builds the same bigram character-level language model twice:
//   1. count every bigram in the training set, then normalise the rows
//   2. fit a 27x27 weight matrix by gradient descent on the same bigrams
// Both land on a loss of ~2.454 nats, which is the whole point: the neural net
// is not learning something cleverer than counting, it is *rediscovering* the
// counts. Getting the two numbers to agree is the payoff.
//
// Needs names.txt (32033 names, one per line) next to this file.
//
// Part 4 does full-batch gradient descent over all 228146 bigrams, 100 steps:
// ~6s in Release, ~24s in Debug. Prefer Release if you are iterating.
//
// Build:  g++ -std=c++17 -O2 makemore_bigram.cpp -o makemore
// Run:    ./makemore

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

namespace {

// ---------------------------------------------------------------------------
// The dataset
// ---------------------------------------------------------------------------

constexpr int kVocab = 27;   // '.' plus the 26 lowercase letters

// The video's stoi / itos dicts. A contiguous alphabet means we can just do
// arithmetic instead of building the maps.
inline int  encode(char c) { return c == '.' ? 0 : c - 'a' + 1; }
inline char decode(int i) { return i == 0 ? '.' : static_cast<char>('a' + i - 1); }

std::vector<std::string> load_names() {
    // Relative to a few plausible working directories: Visual Studio runs with
    // cwd = $(ProjectDir), a bare exe run has cwd = x64\Debug, and running from
    // the repo root is the other common case.
    static const char* kCandidates[] = {
        "Karpathy/names.txt",
        "names.txt",
        "VariousMlp/Karpathy/names.txt",
        "../../VariousMlp/Karpathy/names.txt",
    };

    for (const char* path : kCandidates) {
        std::ifstream f(path);
        if (!f) continue;

        std::vector<std::string> words;
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF
            if (!line.empty()) words.push_back(line);
        }
        if (!words.empty()) {
            printf("  loaded %zu names from %s\n", words.size(), path);
            return words;
        }
    }
    return {};
}

// Python: chs = ['.'] + list(w) + ['.']; zip(chs, chs[1:])
// Walking with a 'prev' index is the same traversal without materialising the
// padded list on every word.
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

std::vector<int> count_bigrams(const std::vector<std::string>& words) {
    std::vector<int> N(kVocab * kVocab, 0);
    for (const auto& w : words)
        for_each_bigram(w, [&](int a, int b) { ++N[a * kVocab + b]; });
    return N;
}

void print_top_bigrams(const std::vector<int>& N, int k) {
    std::vector<std::pair<int, int>> flat;         // (count, flat index)
    flat.reserve(N.size());
    for (int i = 0; i < static_cast<int>(N.size()); ++i) flat.emplace_back(N[i], i);
    std::sort(flat.begin(), flat.end(), std::greater<>());

    printf("  top %d bigrams:\n", k);
    for (int i = 0; i < k; ++i) {
        const int idx = flat[i].second;
        printf("    %c%c  %5d\n", decode(idx / kVocab), decode(idx % kVocab), flat[i].first);
    }
}

// The ASCII stand-in for the video's plt.imshow of the count matrix.
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

// P = (N + smoothing) normalised per row. smoothing=0 is the raw model,
// smoothing=1 is the video's "model smoothing" that kills the infinities.
std::vector<double> row_normalise(const std::vector<int>& N, double smoothing) {
    std::vector<double> P(N.size());
    for (int i = 0; i < kVocab; ++i) {
        double sum = 0.0;
        for (int j = 0; j < kVocab; ++j) sum += N[i * kVocab + j] + smoothing;
        for (int j = 0; j < kVocab; ++j) P[i * kVocab + j] = (N[i * kVocab + j] + smoothing) / sum;
    }
    return P;
}

// ---------------------------------------------------------------------------
// 2. Sampling
// ---------------------------------------------------------------------------

// torch.multinomial on one row. discrete_distribution normalises internally,
// so the row's weights can be counts or probabilities.
std::string sample_name(const std::vector<double>& P, std::mt19937& g) {
    std::string out;
    int ix = 0;                                    // always start at '.'
    for (;;) {
        const double* row = &P[ix * kVocab];
        std::discrete_distribution<int> dist(row, row + kVocab);
        ix = dist(g);
        if (ix == 0) break;                        // sampled the end token
        out.push_back(decode(ix));
    }
    return out;
}

// ---------------------------------------------------------------------------
// 3. Evaluation: average negative log likelihood, in nats
// ---------------------------------------------------------------------------

double average_nll(const std::vector<std::string>& words,
    const std::vector<double>& P,
    long long* bigrams_out = nullptr) {
    double log_likelihood = 0.0;
    long long n = 0;
    for (const auto& w : words) {
        for_each_bigram(w, [&](int a, int b) {
            log_likelihood += std::log(P[a * kVocab + b]);
            ++n;
            });
    }
    if (bigrams_out) *bigrams_out = n;
    return -log_likelihood / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// 4. The same model as a neural net
// ---------------------------------------------------------------------------

void train_neural_net(const std::vector<std::string>& words, double count_model_loss) {
    // The training set: one example per bigram.
    std::vector<int> xs, ys;
    for (const auto& w : words) {
        for_each_bigram(w, [&](int a, int b) { xs.push_back(a); ys.push_back(b); });
    }
    const size_t num = xs.size();
    printf("  %zu bigram examples\n", num);

    // W = torch.randn((27, 27), requires_grad=True)
    std::mt19937 g(2147483647);
    std::normal_distribution<double> randn(0.0, 1.0);
    std::vector<double> W(kVocab * kVocab);
    for (auto& w : W) w = randn(g);

    std::vector<double> dW(kVocab * kVocab);
    const double inv_n = 1.0 / static_cast<double>(num);
    const double reg = 0.01;        // the video's 0.01 * (W**2).mean()
    const double lr = 50.0;         // W.data += -50 * W.grad
    const int steps = 100;

    const auto t0 = std::chrono::steady_clock::now();

    for (int step = 0; step < steps; ++step) {
        std::fill(dW.begin(), dW.end(), 0.0);
        double loss = 0.0;
        double probs[kVocab];

        for (size_t i = 0; i < num; ++i) {
            const int x = xs[i], y = ys[i];

            // Forward. one_hot(x) @ W is just row x of W - as the video points
            // out, the matmul is an expensive way to spell an array index.
            const double* row = &W[x * kVocab];

            // Subtracting the row max before exp is a free guard against
            // overflow; softmax is invariant to it.
            const double maxl = *std::max_element(row, row + kVocab);
            double denom = 0.0;
            for (int j = 0; j < kVocab; ++j) {
                probs[j] = std::exp(row[j] - maxl);
                denom += probs[j];
            }
            for (int j = 0; j < kVocab; ++j) probs[j] /= denom;

            loss += -std::log(probs[y]);

            // Backward. For softmax + negative log likelihood the gradient
            // collapses to d(loss)/d(logits) = probs - onehot(y), averaged over
            // the batch. That is exactly what loss.backward() derives in the
            // video; written out by hand it is one line, and it keeps this file
            // independent of the micrograd engine (228k examples x 27 classes
            // would be ~14M Value nodes per step).
            double* drow = &dW[x * kVocab];
            for (int j = 0; j < kVocab; ++j)
                drow[j] += (probs[j] - (j == y ? 1.0 : 0.0)) * inv_n;
        }
        loss *= inv_n;

        // L2 regularisation: the "gravity" pulling W toward zero, which is the
        // gradient-descent equivalent of add-one smoothing on the counts.
        const double scale = 1.0 / (kVocab * kVocab);
        double sq = 0.0;
        for (double w : W) sq += w * w;
        loss += reg * sq * scale;
        for (int k = 0; k < kVocab * kVocab; ++k) dW[k] += reg * 2.0 * W[k] * scale;

        for (int k = 0; k < kVocab * kVocab; ++k) W[k] -= lr * dW[k];

        if (step % 10 == 0 || step == steps - 1)
            printf("    step %3d  loss %.6f (incl. regularisation)\n", step, loss);
    }

    const auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("  trained in %.2fs\n", secs);

    // W.exp() should now look like the count matrix: the net rediscovered it.
    std::vector<double> P(kVocab * kVocab);
    for (int i = 0; i < kVocab; ++i) {
        const double* row = &W[i * kVocab];
        const double maxl = *std::max_element(row, row + kVocab);
        double denom = 0.0;
        for (int j = 0; j < kVocab; ++j) { P[i * kVocab + j] = std::exp(row[j] - maxl); denom += P[i * kVocab + j]; }
        for (int j = 0; j < kVocab; ++j) P[i * kVocab + j] /= denom;
    }

    // Like-for-like: both numbers below are the plain nll with no regularisation
    // term. The net sits a little above the count model because the L2 pull is
    // still smoothing it - drop reg to 0 and the two converge.
    long long n = 0;
    printf("  count model loss %.6f  vs  neural net loss %.6f\n",
        count_model_loss, average_nll(words, P, &n));

    std::mt19937 gs(2147483647);
    printf("  samples from the trained net: ");
    for (int i = 0; i < 5; ++i) printf("%s ", sample_name(P, gs).c_str());
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
            "makemore_bigram.cpp in Karpathy/\n";
        return;
    }

    size_t shortest = words[0].size(), longest = 0;
    for (const auto& w : words) {
        shortest = std::min(shortest, w.size());
        longest = std::max(longest, w.size());
    }
    std::cout << "  first 8: ";
    for (int i = 0; i < 8; ++i) std::cout << words[i] << " ";
    printf("\n  lengths %zu..%zu   (expected 32033 names, 2..15)\n\n", shortest, longest);

    std::cout << "== 1. counting bigrams ==\n";
    const std::vector<int> N = count_bigrams(words);
    print_top_bigrams(N, 10);
    std::cout << "  (expected n. 6763, a. 6640, an 5438, .a 4410, ...)\n";
    print_count_matrix(N);
    std::cout << "\n";

    std::cout << "== 2. sampling ==\n";
    std::mt19937 g(2147483647);

    // For contrast, the video first samples from a uniform model to show what
    // "untrained" looks like.
    const std::vector<double> uniform(kVocab * kVocab, 1.0 / kVocab);
    std::cout << "  uniform model:  ";
    for (int i = 0; i < 5; ++i) std::cout << sample_name(uniform, g) << " ";

    const std::vector<double> P = row_normalise(N, 0.0);
    std::cout << "\n  bigram model:   ";
    for (int i = 0; i < 5; ++i) std::cout << sample_name(P, g) << " ";
    std::cout << "\n  (names differ from the video: torch's RNG stream is not "
        "reproducible here)\n\n";

    std::cout << "== 3. how good is it? negative log likelihood ==\n";
    // The last digit drifts from the video: it accumulates the log likelihood
    // in a Python loop over float32 scalars, where 228k sequential adds onto a
    // ~560000 running total lose a couple of units. This runs in double.
    long long n = 0;
    const double nll = average_nll(words, P, &n);
    printf("  %lld bigrams, average nll %.4f nats   (video: 228146 bigrams, ~2.454)\n", n, nll);

    // A bigram the training set never saw sends the loss to infinity.
    const std::vector<std::string> probe = { "andrejq" };
    printf("  nll of 'andrejq' unsmoothed: %.4f  ('jq' was never seen)\n",
        average_nll(probe, P, nullptr));

    const std::vector<double> Ps = row_normalise(N, 1.0);
    printf("  nll of 'andrejq' smoothed:   %.4f\n", average_nll(probe, Ps, nullptr));
    printf("  whole dataset, smoothed:     %.4f\n\n", average_nll(words, Ps, &n));

    std::cout << "== 4. the same model, by gradient descent ==\n";
    train_neural_net(words, nll);
}
