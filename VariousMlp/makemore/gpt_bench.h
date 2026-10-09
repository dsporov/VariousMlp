// gpt_bench.h - the test bench for every other implementation of the model.
//
// gpt_model.h is the reference. The walkthrough checks every backward pass in
// it against central differences, and every number that the walkthrough
// prints comes from it. The files that implement the same model in another
// way are checked against it here: gpt_blocked.cpp with a cache-blocked
// multiply, gpt_aten.cpp with ATen tensors, and gpt_cuda.cu on the GPU. They
// are not checked against central differences again. Agreement with a model
// that is already known to be correct is a much stricter test than a finite
// difference in float can be.
//
// There are three sections, each with the reference and the candidate side by
// side:
//
//   1. gradients. The candidate is loaded with the reference's weights, and
//      both run one forward and backward pass on the same batch. Every
//      gradient tensor has to agree to float rounding.
//   2. training. The walkthrough's model for 5000 steps, from the same weights
//      on the same batches. The two runs already differ in the last bits after
//      the first step, so the curves are not identical. But they have to reach
//      the same loss.
//   3. throughput, at the scaled-up model's width, where the multiplies are
//      large enough for a faster implementation to make a difference.
//
// A candidate is any class with the members below. It does not derive from a
// base class (duck typing), so each backend keeps its own types to itself.
// For example, the CUDA backend never shows a device pointer to the host code.
//
//   Candidate(...)                       shapes only; the weights come from load()
//   void load(const std::vector<Param>& weights)
//                                        the reference's weights, in its order
//   void read_grads(std::vector<std::vector<real>>& out)
//                                        the gradients, in the same order
//   void zero_grad()
//   real forward_backward(const int* x, const int* y, int B, int T)
//   real forward_loss(const int* x, const int* y, int B, int T)
//   void set_training(bool training)
//   void adamw_step(real lr)             runs where the weights are
//   void sync()                          waits for queued work (for timing)
//   std::string describe() const
//
// The sections take a factory, make(cfg, vocab), which returns a
// std::unique_ptr<Candidate>. So each backend decides for itself what else it
// needs to be constructed, for example a device.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "gpt_model.h"

namespace gpt {
namespace bench {

// A gradient tensor has to match the reference's to this fraction of its
// largest entry. Float rounding in a different order gives errors of about
// 1e-6 to 1e-5. A wrong or missing term gives errors of about 1.
constexpr double kGradTolerance = 1e-3;

// For the loss of one forward pass, with the same weights and the same batch.
constexpr double kLossTolerance = 1e-4;

// Two float runs of 5000 steps start to differ at the first step, like two
// runs from different seeds. So their final losses agree only to about the
// noise of an estimate over 200 batches. A mistake in a backward pass is
// already caught in section 1, so it cannot hide inside this tolerance.
constexpr double kTrainedTolerance = 0.05;

// The shape for section 3: the scaled-up model's width and depth, with a
// quarter of its context and batch. It is big enough for blocking to make a
// difference in the multiplies. It is small enough that the scalar reference
// finishes a step in seconds, and that the activations fit in the 4GB of a
// GTX 1050 Ti.
constexpr int kBenchBlock = 64;
constexpr int kBenchBatch = 16;

// Each model is timed until this much wall-clock time has passed. Before
// that, one untimed step takes the one-time costs: allocation, the first
// touch of new memory, and starting the threads.
constexpr double kBenchSeconds = 3.0;
constexpr int kBenchMaxSteps = 50;

static_assert(kIters % kReportEvery == 0,
    "section 2 reads its final losses off the last report");

using Clock = std::chrono::steady_clock;

inline double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// The reference, and also gpt_blocked.cpp's candidate: the walkthrough's Gpt
// and AdamW, with the process-wide kernel switch set around every call.
class CpuModel {
public:
    CpuModel(const Config& cfg, int vocab, Kernels kernels, std::mt19937& g)
        : kernels_(kernels), model_(cfg, vocab, g), opt_(model_.params(), cfg.lr)
    {
    }

    // opt_ holds addresses inside model_, so the object must not be copied.
    CpuModel(const CpuModel&) = delete;
    CpuModel& operator=(const CpuModel&) = delete;

    Gpt& model() { return model_; }

    void load(const std::vector<Param>& weights) {
        std::vector<Param>& mine = model_.params();
        CHECK(weights.size() == mine.size(), weights.size()
            << " tensors for a model with " << mine.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            CHECK(weights[i].data->size() == mine[i].data->size(), weights[i].name
                << " has " << weights[i].data->size() << " entries, this model's "
                << mine[i].data->size());
            *mine[i].data = *weights[i].data;
        }
    }

    void read_grads(std::vector<std::vector<real>>& out) {
        out.clear();
        for (const Param& p : model_.params()) {
            out.push_back(*p.grad);
        }
    }

    void zero_grad() { model_.zero_grad(); }

    real forward_backward(const int* x, const int* y, int B, int T) {
        const Using use(kernels_);
        return model_.forward_backward(x, y, B, T);
    }

    real forward_loss(const int* x, const int* y, int B, int T) {
        const Using use(kernels_);
        return model_.forward_loss(x, y, B, T);
    }

    void set_training(bool training) { model_.set_training(training); }

    void adamw_step(real lr) {
        opt_.set_lr(lr);
        opt_.step();
    }

    void sync() {}

    std::string describe() const {
        const int threads = available_threads();
        std::string s = kernels_ == Kernels::kBlocked
            ? std::string("blocked multiply, ") + blocked::kernel_name<real>()
            : std::string("scalar loops");
        s += ", " + std::to_string(threads) + (threads == 1 ? " thread" : " threads");
        return s;
    }

private:
    // Sets the switch during one call, and restores the previous value
    // afterwards.
    class Using {
    public:
        explicit Using(Kernels k) : saved_(active_kernels()) { active_kernels() = k; }
        ~Using() { active_kernels() = saved_; }
        Using(const Using&) = delete;
        Using& operator=(const Using&) = delete;

    private:
        Kernels saved_;
    };

    Kernels kernels_;
    Gpt model_;
    AdamW opt_;
};

// The same precaution as in gradient_check(): gamma starts at exactly 1 and
// every bias at exactly 0. With those values, a wrong expression is most
// likely to agree with the correct one by accident.
inline void jitter(std::vector<Param>& params, std::mt19937& g) {
    std::normal_distribution<real> noise(real(0), real(0.3));
    for (Param& p : params) {
        for (real& v : *p.data) {
            v += noise(g);
        }
    }
}

// The largest difference between two tensors, as a fraction of the largest
// entry in either one. It is relative to the whole tensor, not to each
// element. An entry that should be zero comes out as rounding noise on both
// sides, and the ratio of two noise values means nothing.
inline double tensor_error(const std::vector<real>& a, const std::vector<real>& b) {
    double diff = 0.0;
    double scale = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = static_cast<double>(a[i]);
        const double y = static_cast<double>(b[i]);
        diff = std::max(diff, std::fabs(x - y));
        scale = std::max(scale, std::max(std::fabs(x), std::fabs(y)));
    }
    return scale > 0.0 ? diff / scale : diff;
}

// The worst tensor_error across a whole model, and the tensor it was in.
inline double worst_error(const std::vector<Param>& names,
    const std::vector<std::vector<real>>& a, const std::vector<std::vector<real>>& b,
    std::string& where) {
    CHECK(a.size() == names.size() && b.size() == names.size(), a.size() << " and "
        << b.size() << " gradient tensors for a model with " << names.size());
    double worst = 0.0;
    where = "none";
    for (size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].size() == b[i].size(), names[i].name << ": " << a[i].size()
            << " entries against " << b[i].size());
        const double e = tensor_error(a[i], b[i]);
        if (e > worst) {
            worst = e;
            where = names[i].name;
        }
    }
    return worst;
}

// One timed optimiser step, from zeroing the gradients to the updated weights.
template <class Model>
double timed_step(Model& m, const std::vector<int>& x, const std::vector<int>& y,
    int B, int T, real lr) {
    const Clock::time_point t0 = Clock::now();
    m.zero_grad();
    m.forward_backward(x.data(), y.data(), B, T);
    m.adamw_step(lr);
    m.sync();
    return seconds_since(t0);
}

inline void print_config(const Config& cfg) {
    printf("  %d channels, %d heads, %d blocks, context %d, batch %d, dropout %.2f\n",
        cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block, cfg.batch,
        static_cast<double>(cfg.dropout));
}

inline bool load_data(Data& d) {
    std::cout << "== 0. tiny shakespeare ==\n";
    if (!load_corpus(d)) {
        printf("  could not find %s - expected it in makemore/\n", kCorpus);
        return false;
    }
    printf("  vocabulary of %d characters, %zu training tokens\n",
        d.vocab, d.train.size());
    return true;
}

template <class Candidate, class Make>
void compare_gradients(const Data& d, const char* label, Make make) {
    std::cout << "\n== 1. gradients against the reference ==\n";
    printf("  The reference's weights loaded into %s, one forward and backward\n"
        "  pass each on the same batch, every gradient tensor compared.\n", label);

    struct Case {
        const char* what;
        Config cfg;
        bool jittered;
    };
    const Case cases[] = {
        { "tiny, every tile ragged", { 8, 2, 2, 5, 3, real(0), kLr }, true },
        { "the walkthrough's model",
            { kEmbed, kHeads, kLayers, kBlock, kBatch, real(0), kLr }, false },
        { "the scaled-up width, 2 blocks",
            { kBigEmbed, kBigHeads, 2, kBenchBlock, 4, real(0), kBigLr }, false },
    };

    sanity::Expect chk;
    for (const Case& c : cases) {
        std::mt19937 g(1337);
        CpuModel ref(c.cfg, d.vocab, Kernels::kScalar, g);
        if (c.jittered) {
            jitter(ref.model().params(), g);
        }
        std::unique_ptr<Candidate> cand = make(c.cfg, d.vocab);
        cand->load(ref.model().params());

        std::vector<int> x, y;
        sample_batch(d.train, c.cfg.batch, c.cfg.block, x, y, g);

        ref.zero_grad();
        cand->zero_grad();
        const real ref_loss = ref.forward_backward(x.data(), y.data(),
            c.cfg.batch, c.cfg.block);
        const real cand_loss = cand->forward_backward(x.data(), y.data(),
            c.cfg.batch, c.cfg.block);

        std::vector<std::vector<real>> ref_grads, cand_grads;
        ref.read_grads(ref_grads);
        cand->read_grads(cand_grads);
        std::string where;
        const double worst = worst_error(ref.model().params(), ref_grads,
            cand_grads, where);

        printf("  %s, %zu parameters\n    loss %.6f against %.6f, worst gradient "
            "%.1e (%s)\n", c.what, ref.model().parameter_count(),
            static_cast<double>(ref_loss), static_cast<double>(cand_loss),
            worst, where.c_str());
        chk.close((std::string(c.what) + ": loss").c_str(),
            cand_loss, ref_loss, kLossTolerance);
        chk.that((std::string(c.what) + ": every gradient").c_str(),
            worst < kGradTolerance);
    }
    chk.report();
}

template <class Candidate, class Make>
void train_side_by_side(const Data& d, const char* label, Make make) {
    std::cout << "\n== 2. training the walkthrough's model both ways ==\n";
    const Config cfg{ kEmbed, kHeads, kLayers, kBlock, kBatch, kDropout, kLr };
    print_config(cfg);
    printf("  %d steps each, from the same weights on the same batches\n", kIters);

    // Seeded exactly like section 6 of the walkthrough, so the reference
    // column below is the same run that section prints.
    std::mt19937 g(1337);
    CpuModel ref(cfg, d.vocab, Kernels::kScalar, g);
    std::unique_ptr<Candidate> cand = make(cfg, d.vocab);
    cand->load(ref.model().params());

    std::vector<int> x, y;
    double ref_secs = 0.0;
    double cand_secs = 0.0;
    Losses ref_l{ real(0), real(0) };
    Losses cand_l{ real(0), real(0) };
    for (int it = 0; it < kIters; ++it) {
        sample_batch(d.train, cfg.batch, cfg.block, x, y, g);
        ref_secs += timed_step(ref, x, y, cfg.batch, cfg.block, cfg.lr);
        cand_secs += timed_step(*cand, x, y, cfg.batch, cfg.block, cfg.lr);
        if ((it + 1) % kReportEvery == 0) {
            ref_l = estimate_loss(ref, d, cfg.batch, cfg.block);
            cand_l = estimate_loss(*cand, d, cfg.batch, cfg.block);
            printf("    step %6d   reference %.4f / %.4f   %-9s %.4f / %.4f\n",
                it + 1, static_cast<double>(ref_l.train), static_cast<double>(ref_l.val),
                label, static_cast<double>(cand_l.train), static_cast<double>(cand_l.val));
        }
    }

    printf("  reference  %8.3f ms a step  (%s)\n",
        1e3 * ref_secs / kIters, ref.describe().c_str());
    printf("  %-9s  %8.3f ms a step  (%s)  %.2fx\n", label,
        1e3 * cand_secs / kIters, cand->describe().c_str(), ref_secs / cand_secs);

    sanity::Expect chk;
    chk.close("train loss lands where the reference's does",
        cand_l.train, ref_l.train, kTrainedTolerance);
    chk.close("validation loss lands where the reference's does",
        cand_l.val, ref_l.val, kTrainedTolerance);
    chk.report();
}

// Seconds per optimiser step, after one untimed step.
template <class Model>
double seconds_per_step(Model& m, const Data& d, const Config& cfg,
    std::mt19937& g, int& steps) {
    std::vector<int> x, y;
    sample_batch(d.train, cfg.batch, cfg.block, x, y, g);
    timed_step(m, x, y, cfg.batch, cfg.block, cfg.lr);

    double total = 0.0;
    steps = 0;
    while (steps == 0 || (total < kBenchSeconds && steps < kBenchMaxSteps)) {
        sample_batch(d.train, cfg.batch, cfg.block, x, y, g);
        total += timed_step(m, x, y, cfg.batch, cfg.block, cfg.lr);
        ++steps;
    }
    return total / steps;
}

template <class Candidate, class Make>
void throughput(const Data& d, const char* label, Make make) {
    std::cout << "\n== 3. throughput at the scaled-up width ==\n";
    const Config cfg{ kBigEmbed, kBigHeads, kBigLayers, kBenchBlock, kBenchBatch,
        kBigDropout, kBigLr };
    print_config(cfg);

    std::mt19937 g(1337);
    CpuModel ref(cfg, d.vocab, Kernels::kScalar, g);
    std::unique_ptr<Candidate> cand = make(cfg, d.vocab);
    cand->load(ref.model().params());

    const size_t params = ref.model().parameter_count();
    const double flops = step_flops(cfg, params, d.vocab);
    printf("  %zu parameters, %.1f GFLOP a step. Timing only: dropout draws\n"
        "  different masks on each side, so the losses are not comparable.\n",
        params, flops / 1e9);

    int ref_steps = 0;
    int cand_steps = 0;
    const double ref_s = seconds_per_step(ref, d, cfg, g, ref_steps);
    const double cand_s = seconds_per_step(*cand, d, cfg, g, cand_steps);
    printf("  reference  %9.1f ms a step  %8.1f GFLOP/s  (%d timed)\n",
        1e3 * ref_s, flops / ref_s / 1e9, ref_steps);
    printf("  %-9s  %9.1f ms a step  %8.1f GFLOP/s  (%d timed)  %.1fx\n", label,
        1e3 * cand_s, flops / cand_s / 1e9, cand_steps, ref_s / cand_s);

    // The full run differs from this shape only in context and batch. So its
    // parameter count is this shape's count plus the extra rows of the
    // position table.
    const Config big{ kBigEmbed, kBigHeads, kBigLayers, kBigBlock, kBigBatch,
        kBigDropout, kBigLr };
    const size_t big_params = params
        + static_cast<size_t>(kBigBlock - kBenchBlock) * static_cast<size_t>(kBigEmbed);
    const double big_run = step_flops(big, big_params, d.vocab) * kIters;
    printf("  at %s's rate the video's full run - %d steps at context %d, batch %d -\n"
        "  is %.1f hours of training steps, and %.1f at the reference's\n",
        label, kIters, kBigBlock, kBigBatch,
        big_run / (flops / cand_s) / 3600.0, big_run / (flops / ref_s) / 3600.0);
}

template <class Candidate, class Make>
void run_sections(const Data& d, const char* label, Make make) {
    compare_gradients<Candidate>(d, label, make);
    train_side_by_side<Candidate>(d, label, make);
    throughput<Candidate>(d, label, make);
}

}  // namespace bench
}  // namespace gpt
