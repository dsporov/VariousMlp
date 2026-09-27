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
// is the whole idea; the rest is bookkeeping around it.
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
// The model those five steps arrive at lives in gpt_model.h, because
// gpt_chat.cpp needs it too. This file is the walkthrough around it.
//
// Needs input.txt (tiny Shakespeare, 1115394 characters) next to this file.
//
// Nothing here uses an autograd engine, so every backward pass is written out:
// causal masked softmax, the three matmuls of attention, layer normalisation,
// the residual splits. gradient_check() below compares the lot against central
// differences, and is the reason to trust any of it.
//
// Runtime, at the video's small configuration: a few minutes in Release, and
// seconds after that - section 6 saves a checkpoint and loads it next time. The
// scaled-up configuration at the end of the video is four orders of magnitude
// more arithmetic and is not reachable until the scalar loops are blocked and
// threaded - see section 8, which measures this machine and says so.

#include "gpt_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "check.h"

using namespace gpt;

namespace {

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

// step_flops() in gpt_model.h, for a whole training run: the cost of one step,
// times the steps, plus the evaluation passes, which are a sixth of the total
// and would otherwise make the throughput below look better than it is.
double run_flops(const Config& cfg, size_t params, int vocab, int iters,
    int report_every) {
    const double per_step = step_flops(cfg, params, vocab);
    const int reports = (report_every > 0 ? iters / report_every : 0) + 1;
    const double evals = static_cast<double>(reports) * kEvalIters * 2;
    return per_step * iters + per_step * evals / 3.0;
}

// Steps timed for the throughput figure in section 8, after the one untimed
// warm-up step below. 200 was enough while every loop ran serially, because the
// window was then just under three seconds; threading cut it to well under one,
// short enough that scheduling and turbo jitter stopped averaging out. This is
// a few seconds threaded and is the one part of section 8 that a serial build
// notices.
constexpr int kBenchSteps = 1000;

// Timed on a throwaway model rather than on the trained one, so the number
// means the same thing whether section 6 trained or loaded a checkpoint - and
// so that measuring never moves weights that matter.
double measure_throughput(const Config& cfg, const Data& d, int vocab, int steps) {
    std::mt19937 g(1337);
    Gpt probe(cfg, vocab, g);
    AdamW opt(probe.params(), cfg.lr);
    std::vector<int> x, y;

    const auto one_step = [&]() {
        sample_batch(d.train, cfg.batch, cfg.block, x, y, g);
        probe.zero_grad();
        probe.forward_backward(x.data(), y.data(), cfg.batch, cfg.block);
        opt.step();
    };

    // The first step pays for everything that only ever happens once: the
    // activation buffers are allocated by that first forward pass and faulted
    // in a page at a time, and the OpenMP pool is built at its first parallel
    // region. Leaving those inside the clock is what made the printed rate
    // disagree with the wall clock section 6 had just demonstrated.
    one_step();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) {
        one_step();
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
    printf("  %d thread%s\n", available_threads(),
        available_threads() == 1 ? "" : "s");
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
        "(timed over %d steps on %d thread%s)\n", measured / 1e9, kBenchSteps,
        available_threads(), available_threads() == 1 ? "" : "s");
    printf("  one step of the scaled-up model is %.2f TFLOP, so %d steps is "
        "%.1f days\n", step_flops(big, big_params, d.vocab) / 1e12, kIters,
        projected / measured / 86400.0);
    printf("  its activation buffers would need about %.1f GB at %zu bytes per "
        "scalar\n", bytes / 1e9, sizeof(real));
    std::cout <<
        "  Which is the honest state of this port. Of the three things that\n"
        "  would fix it - vectorising, threading, and blocking for cache - the\n"
        "  second is done, and the rate above is what it left. The other two\n"
        "  are still open here: neither /arch:AVX2 nor dropping from double to\n"
        "  float has moved that rate, and every inner loop streams a whole\n"
        "  weight matrix past the cache for each row it handles. gpt_blocked\n"
        "  runs this same model through a cache-blocked, vectorised multiply\n"
        "  and measures what the pair is worth at the scaled-up width.\n";

    // Outside a template, `if constexpr` still compiles and type-checks the
    // branch it discards, which is the point of having it here rather than
    // in a comment.
    if constexpr (kRunScaleUp) {
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
