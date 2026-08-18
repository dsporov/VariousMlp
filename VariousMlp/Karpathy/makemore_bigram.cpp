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

// The video's stoi / itos dicts. A contiguous alphabet means we can just do
// arithmetic instead of building the maps.
// Anything outside '.' and a-z lands outside [0, kVocab) and indexes off the
// end of the 27x27 arrays, so it is worth a compare per character. load_names
// filters the dataset up front; this is the backstop for hand-written probes.
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
        size_t skipped = 0;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (line.back() == '\r')
                line.pop_back();  // CRLF
            // The model has exactly 27 tokens, so a stray uppercase letter,
            // digit or accent has nowhere to go. Drop those lines here rather
            // than letting encode() abort the whole run over one bad name.
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
    for (const auto& w : words) {
        for_each_bigram(w, [&](int a, int b) {
            ++N[a * kVocab + b];
        });
    }
    return N;
}

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

// The postcondition for anything claiming to produce a distribution. 27x27
// adds, run once per matrix, and it is the check that catches a normalisation
// written the wrong way round - which otherwise just looks like a bad model.
void check_rows_sum_to_one(const std::vector<double>& P, const char* what) {
    CHECK(P.size() == kVocab * kVocab, what << ": expected a " << kVocab << "x"
        << kVocab << " matrix, got " << P.size() << " entries");
    for (int i = 0; i < kVocab; ++i) {
        double sum = 0.0;
        for (int j = 0; j < kVocab; ++j) sum += P[i * kVocab + j];
        CHECK(std::fabs(sum - 1.0) < 1e-12,
            what << ": row '" << decode(i) << "' sums to " << sum << ", not 1");
    }
}

// P = (N + smoothing) normalised per row. smoothing=0 is the raw model,
// smoothing=1 is the video's "model smoothing" that kills the infinities.
std::vector<double> row_normalise(const std::vector<int>& N, double smoothing) {
    CHECK(N.size() == kVocab * kVocab, "expected a " << kVocab << "x" << kVocab
        << " count matrix, got " << N.size() << " entries");
    CHECK(smoothing >= 0.0, "negative smoothing (" << smoothing << ") makes negative probabilities");

    std::vector<double> P(N.size());
    for (int i = 0; i < kVocab; ++i) {
        double sum = 0.0;
        for (int j = 0; j < kVocab; ++j) sum += N[i * kVocab + j] + smoothing;
        // A character that never starts a bigram leaves an all-zero row, and
        // 0/0 would quietly fill it with NaN.
        CHECK(sum > 0.0, "no counts in the row for '" << decode(i)
            << "'; pass smoothing > 0 to make it a valid distribution");
        for (int j = 0; j < kVocab; ++j) P[i * kVocab + j] = (N[i * kVocab + j] + smoothing) / sum;
    }
    check_rows_sum_to_one(P, "normalised counts");
    return P;
}

// ---------------------------------------------------------------------------
// 2. Sampling
// ---------------------------------------------------------------------------

// torch.multinomial on one row. discrete_distribution normalises internally,
// so the row's weights can be counts or probabilities.
std::string sample_name(const std::vector<double>& P, std::mt19937& g) {
    CHECK(P.size() == kVocab * kVocab, "expected a " << kVocab << "x" << kVocab
        << " probability matrix, got " << P.size() << " entries");

    // A model that never emits the end token would loop forever. The uniform
    // model in the walkthrough already produces 26-character names on average,
    // so the cap has to be generous to only ever mean "something is broken".
    constexpr size_t kMaxLength = 1000;

    std::string out;
    int ix = 0;                                    // always start at '.'
    for (;;) {
        const double* row = &P[ix * kVocab];
        double sum = 0.0;
        for (int j = 0; j < kVocab; ++j) sum += row[j];
        CHECK(sum > 0.0, "row '" << decode(ix) << "' has no probability mass to sample from");

        std::discrete_distribution<int> dist(row, row + kVocab);
        ix = dist(g);
        if (ix == 0) break;                        // sampled the end token
        out.push_back(decode(ix));
        CHECK(out.size() < kMaxLength,
            "sampling ran past " << kMaxLength << " characters without an end token");
    }
    return out;
}

// ---------------------------------------------------------------------------
// 3. Evaluation: average negative log likelihood, in nats
// ---------------------------------------------------------------------------

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
    // Every word contributes at least one bigram, so n == 0 would mean the
    // padding in for_each_bigram stopped working. (An unseen bigram still gives
    // -inf here on purpose: that is the point the walkthrough makes with
    // 'andrejq' before it introduces smoothing.)
    CHECK(n > 0, "scored " << words.size() << " words but counted no bigrams");
    if (bigrams_out) *bigrams_out = n;
    return -log_likelihood / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// 4. The same model as a neural net
// ---------------------------------------------------------------------------

// Every index here becomes raw pointer arithmetic into W below, so a bad one is
// an out-of-bounds write rather than a wrong answer. One pass over the dataset,
// before training starts; the training loop itself stays check-free.
void validate_examples(const std::vector<int>& xs, const std::vector<int>& ys) {
    CHECK(!xs.empty(), "no training examples");
    CHECK(xs.size() == ys.size(),
        xs.size() << " inputs but " << ys.size() << " targets");
    for (size_t i = 0; i < xs.size(); ++i)
        CHECK(xs[i] >= 0 && xs[i] < kVocab && ys[i] >= 0 && ys[i] < kVocab,
            "example " << i << " is (" << xs[i] << ", " << ys[i]
            << "), outside [0, " << kVocab << ")");
}

void train_neural_net(const std::vector<std::string>& words, double count_model_loss) {
    // The training set: one example per bigram.
    std::vector<int> xs, ys;
    for (const auto& w : words) {
        for_each_bigram(w, [&](int a, int b) { xs.push_back(a); ys.push_back(b); });
    }
    const size_t num = xs.size();
    printf("  %zu bigram examples\n", num);

    validate_examples(xs, ys);

    // W = torch.randn((27, 27), requires_grad=True)
    std::mt19937 g(2147483647);
    std::normal_distribution<double> randn(0.0, 1.0);
    std::vector<double> W(kVocab * kVocab);
    for (auto& w : W) w = randn(g);

    double first_loss = 0.0, last_loss = 0.0;

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
        for (double w : W)
            sq += w * w;
        loss += reg * sq * scale;
        for (int k = 0; k < kVocab * kVocab; ++k)
            dW[k] += reg * 2.0 * W[k] * scale;

        for (int k = 0; k < kVocab * kVocab; ++k)
            W[k] -= lr * dW[k];

        if (step == 0) first_loss = loss;
        last_loss = loss;

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
    check_rows_sum_to_one(P, "softmax of W");

    // Like-for-like: both numbers below are the plain nll with no regularisation
    // term. The net sits a little above the count model because the L2 pull is
    // still smoothing it - drop reg to 0 and the two converge.
    long long n = 0;
    const double net_loss = average_nll(words, P, &n);
    printf("  count model loss %.6f  vs  neural net loss %.6f\n",
        count_model_loss, net_loss);

    // The claim the whole file is built to demonstrate, checked instead of
    // asserted in a comment: gradient descent rediscovered the counts. reg =
    // 0.01 leaves the net about 0.021 nats above the count model after 100
    // steps, so the tolerance sits just above that - raise reg, or stop
    // training early, and this is the line that notices.
    sanity::Expect chk;
    chk.that("training loss went down", last_loss < first_loss);
    chk.close("net matches the count model", net_loss, count_model_loss, 0.03);
    chk.report();

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
    for (size_t i = 0; i < std::min<size_t>(8, words.size()); ++i)
        std::cout << words[i] << " ";
    printf("\n  lengths %zu..%zu\n", shortest, longest);

    // Everything below is checked against the video, which means checked
    // against its dataset. Swap names.txt for another list and these are
    // expected to fail - hence Expect, which reports and carries on, rather
    // than CHECK, which would stop the walkthrough.
    sanity::Expect data;
    data.equal("names loaded", static_cast<double>(words.size()), 32033);
    data.equal("shortest name", static_cast<double>(shortest), 2);
    data.equal("longest name", static_cast<double>(longest), 15);
    data.report();
    std::cout << "\n";

    std::cout << "== 1. counting bigrams ==\n";
    const std::vector<int> N = count_bigrams(words);
    print_top_bigrams(N, 10);

    // The counting step is the foundation for both models, so pin the four
    // counts the video reads out loud plus the total.
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

    // For contrast, the video first samples from a uniform model to show what
    // "untrained" looks like.
    const std::vector<double> uniform(kVocab * kVocab, 1.0 / kVocab);
    std::cout << "  uniform model:  ";
    for (int i = 0; i < 5; ++i)
        std::cout << sample_name(uniform, g) << " ";

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

    sanity::Expect loss;
    loss.equal("bigrams scored", static_cast<double>(n), 228146);
    loss.close("average nll", nll, 2.4541, 1e-3);
    loss.report();

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
