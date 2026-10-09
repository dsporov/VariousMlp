// gpt_scaled.cpp - the video's scale-up, trained on the GPU and the CPU at once.
//
// At the end of the video, the walkthrough's model grows into a real one: 384
// channels, 6 heads, 6 blocks, a context of 256 characters, batches of 64 and
// dropout 0.2. That is 10.8M parameters. The video reaches a validation loss
// of 1.48 in about fifteen minutes on an A100. gpt.cpp only records that run;
// it does not perform it. This program performs it. It uses the two fastest
// implementations in the repository, the hand-written CUDA model and the
// cache-blocked AVX2 multiply, and shares every batch between them in the same
// way as gpt_split.cpp.
//
// gpt_split itself cannot train this shape on a GTX 1050 Ti. The parameters
// are not the problem: weights, gradients and both Adam moments together are
// 173MB. The activations are the problem. At context 256 the GPU model keeps
// about 85MB per window between its forward and backward passes. Two thirds
// of that is in (B, T, C) tensors, and the rest is in the (B, heads, T, T)
// attention matrices. At the start, gpt_split gives the GPU 38 windows, which
// need 3.2GB. The card has 4GB, and that memory also holds the CUDA context
// and, under Windows, the desktop. The GPU's share also grows later if the GPU
// turns out to be the faster side.
//
// So the GPU processes its windows in chunks of at most kGpuChunk windows per
// call, and every chunk reuses the same buffers. Nothing in gpt_cuda.cu had to
// change for this. Its backward passes add into the gradient instead of
// overwriting it, so the gradients of all chunks accumulate on the device and
// are copied to the host once per step. The one thing to get right is the
// scale factor for that sum. The loss of each call, and so its gradient, is a
// mean over the windows of that call only. So the accumulated gradient is a
// sum of means. With c chunks of k windows each, it is c times the mean over
// all c*k GPU windows. Its scale factor in the batch mean is therefore k/B,
// not c*k/B:
//
//     g = (k / B) * (g_1 + ... + g_c) + ((B - c*k) / B) * g_cpu
//
// This needs every chunk to have the same size, so the GPU's share is rounded
// down to a multiple of its chunk count: 38 windows become three chunks of 12.
// Section 1 compares the result with the scalar reference.
//
// Everything else works as in gpt_split. The CPU's copy of the weights is the
// master copy. AdamW runs on the CPU, and the weights are copied to the GPU
// after every step. The GPU's share of each batch moves toward the point where
// both sides finish at the same time.
//
// This directory is a separate project. check.h, gpt_gemm.h, gpt_model.h,
// gpt_cuda.h and gpt_cuda.cu are copies of the files in makemore/, and nothing
// here includes a file from makemore/. The only contact with the rest of the
// repository is through files on disk. The program reads makemore/input.txt.
// It saves the trained weights next to that file, in gpt_model.h's checkpoint
// format and under gpt_model.h's file name for this shape. gpt.cpp's scale-up
// branch looks for that file, and the chat tool can read it.
//
// Expect a run of a few hours: 5000 steps of 1.1 TFLOP each, plus the video's
// eleven evaluations of 200 batches on each of the training and validation
// sets.
//
// CMake only, because it has its own main, and built only when CUDA is found.
// Build the gpt_scaled target.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <future>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "check.h"
#include "gpt_cuda.h"
#include "gpt_model.h"

using namespace gpt;

namespace {

// The video's eval_interval. Its max_iters and eval_iters are kIters and
// kEvalIters in gpt_model.h, and the model's shape is the kBig constants.
constexpr int kEvalEvery = 500;

// Evaluations are 500 steps apart, which at this size means several minutes.
// So a progress line is printed more often than that.
constexpr int kProgressEvery = 100;

// The largest number of windows the GPU gets in one call. Its activation and
// scratch buffers are sized for the largest call, and at this shape they need
// about 85MB per window. Sixteen windows need 1.4GB, which leaves room on a 4GB
// card for the weights, the CUDA context and the desktop. Larger chunks would
// mean fewer kernel launches. But 16 windows are already 4096 rows, which
// gives each multiply far more tiles than the six SMs of the GTX 1050 Ti can
// run at once.
constexpr int kGpuChunk = 16;

// The GPU's share of each batch before any step has been timed, as in
// gpt_split.cpp. It comes from the speeds measured at context 64: 504 GFLOP/s
// on the GPU and 319 GFLOP/s for the blocked multiply. At context 256,
// attention is a larger part of the work. The share adjusts itself within a
// few dozen steps.
constexpr double kInitialGpuShare = 0.6;

// How much one step's timing changes the running averages. It is low enough
// that one slow step does not move the GPU's share much, and high enough that
// the share settles quickly.
constexpr double kSmoothing = 0.25;

// The thread that drives the GPU is busy for the whole GPU half, because the
// CUDA runtime waits for copies by spinning. So the CPU half uses one thread
// fewer, and the cores are not oversubscribed. Oversubscription slows an
// OpenMP team down a lot, because the team waits at every barrier for its
// slowest thread.
constexpr int kGpuDriverThreads = 1;

// The batch for section 1. It has few enough windows at full context that the
// scalar reference finishes them in seconds. The chunks are small enough that
// the GPU's share needs more than one call, because that is what the section
// checks.
constexpr int kCheckBatch = 6;
constexpr int kCheckChunk = 2;

// The tolerances from gpt_bench.h. A gradient tensor has to match the
// reference's to this fraction of its largest entry. Float rounding in a
// different order gives errors of about 1e-6 to 1e-5. A chunk with the wrong
// scale factor, or a lost chunk, gives errors of about 0.1.
constexpr double kGradTolerance = 1e-3;
constexpr double kLossTolerance = 1e-4;

// An untrained model gives every character about the same probability, so its
// loss starts near ln(vocab). It starts a little above that, because weights
// with a standard deviation of 0.02 already make the logits uneven.
constexpr double kStartTolerance = 0.2;

// The video's final validation loss, as recorded in gpt_model.h. This run has
// its own initial weights, batches and dropout masks, so its result should be
// close to this value, not equal to it.
constexpr double kVideoVal = 1.48;
constexpr double kVideoValTolerance = 0.05;

// The number of characters to sample from the trained model. Sampling runs on
// the CPU, because gpt_cuda.h returns losses and never logits.
constexpr int kSampleLength = 1000;

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// A running average. The first sample is used as it is.
class Smoothed {
public:
    void add(double sample) {
        value_ = seen_ ? value_ + kSmoothing * (sample - value_) : sample;
        seen_ = true;
    }

    double value() const {
        return value_;
    }

private:
    double value_ = 0.0;
    bool seen_ = false;
};

// Sets the OpenMP team size for this thread's parallel regions during one
// call, and restores the previous size afterwards.
class ThreadLimit {
public:
    explicit ThreadLimit(int n) : saved_(available_threads()) {
        set(n);
    }

    ~ThreadLimit() {
        set(saved_);
    }

    ThreadLimit(const ThreadLimit&) = delete;
    ThreadLimit& operator=(const ThreadLimit&) = delete;

private:
    static void set(int n) {
#ifdef _OPENMP
        omp_set_num_threads(n);
#else
        static_cast<void>(n);
#endif
    }

    int saved_;
};

// Selects which multiply gpt_model.h runs during one call, and restores the
// previous choice afterwards. The switch is process-wide, but only the main
// thread ever runs the CPU model. It uses the blocked multiply for the CPU
// half and for sampling, and the scalar loops for the reference in section 1.
class UsingKernels {
public:
    explicit UsingKernels(Kernels k) : saved_(active_kernels()) {
        active_kernels() = k;
    }

    ~UsingKernels() {
        active_kernels() = saved_;
    }

    UsingKernels(const UsingKernels&) = delete;
    UsingKernels& operator=(const UsingKernels&) = delete;

private:
    Kernels saved_;
};

// How the windows of a batch are divided. The GPU gets the first windows of
// the batch, in `chunks` calls of `chunk` windows each. The CPU gets the rest.
struct Division {
    int chunks;
    int chunk;

    int gpu_windows() const {
        return chunks * chunk;
    }
};

// The result of one side's part of a pass: the mean loss over its windows,
// and the time it took.
struct Half {
    real loss;
    double seconds;
};

// Where the time of a training step went, averaged over the last few steps.
struct StepTimes {
    int gpu_windows = 0;
    int gpu_chunks = 0;
    int gpu_chunk = 0;
    int cpu_windows = 0;
    double gpu_ms = 0.0;
    double cpu_ms = 0.0;
    double combine_ms = 0.0;
    double update_ms = 0.0;
};

// The model twice, once on each side, with every batch shared between them.
// It has the members that estimate_loss in gpt_model.h needs from a model, so
// the video's evaluation is shared between the two sides too.
class SplitModel {
public:
    SplitModel(const Config& cfg, int vocab, int gpu_chunk, std::mt19937& g)
        : gpu_(gpt_cuda::Shape{ cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block,
            vocab, static_cast<float>(cfg.dropout) }),
          cpu_(cfg, vocab, g),
          opt_(cpu_.params(), cfg.lr),
          gpu_chunk_(gpu_chunk),
          cpu_threads_(std::max(1, available_threads() - kGpuDriverThreads)) {
        CHECK(gpu_chunk > 0, "GPU calls of " << gpu_chunk << " windows");

        // combine() goes through the two sets of gradients together, so they
        // have to be the same tensors in the same order.
        const std::vector<size_t>& sizes = gpu_.tensor_sizes();
        const std::vector<Param>& tensors = cpu_.params();
        CHECK(sizes.size() == tensors.size(), "the GPU model has " << sizes.size()
            << " tensors and the CPU model " << tensors.size());
        for (size_t i = 0; i < sizes.size(); ++i) {
            CHECK(sizes[i] == tensors[i].data->size(), tensors[i].name << " has "
                << tensors[i].data->size() << " entries on the CPU and "
                << sizes[i] << " on the GPU");
        }
        upload_weights();
    }

    SplitModel(const SplitModel&) = delete;
    SplitModel& operator=(const SplitModel&) = delete;

    // The CPU's copy is the master copy. AdamW updates it, and it is the copy
    // that is saved and sampled from.
    Gpt& cpu_model() {
        return cpu_;
    }

    std::vector<Param>& params() {
        return cpu_.params();
    }

    // Copies another model's weights to both sides.
    void load(const std::vector<Param>& weights) {
        std::vector<Param>& mine = cpu_.params();
        CHECK(weights.size() == mine.size(), weights.size()
            << " tensors for a model with " << mine.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            CHECK(weights[i].data->size() == mine[i].data->size(), weights[i].name
                << " has " << weights[i].data->size() << " entries, this model's "
                << mine[i].data->size());
            *mine[i].data = *weights[i].data;
        }
        upload_weights();
    }

    // Loads a checkpoint in gpt_model.h's format. As in gpt_model.h, a
    // missing or outdated checkpoint is not an error. It only means that the
    // model has to be trained.
    bool load_checkpoint(const Config& cfg, const Data& d, const std::string& path) {
        if (!load_params(cpu_.params(), cfg, d, path)) {
            return false;
        }
        upload_weights();
        return true;
    }

    void zero_grad() {
        cpu_.zero_grad();
        gpu_.zero_grad();
        zeroed_ = true;
    }

    // How a batch of B windows is divided at the current share. If there are
    // at least two windows, each side gets at least one. Then both sides are
    // always timed, and the share can keep adjusting. A single window cannot
    // be divided, so it goes to the GPU. The GPU's share is then rounded down
    // to chunks of equal size, which combine() relies on.
    Division division(int B) const {
        CHECK(B > 0, "a batch of " << B << " windows");
        if (B == 1) {
            return Division{ 1, 1 };
        }
        const int wanted = std::clamp(static_cast<int>(std::lround(gpu_share_ * B)),
            1, B - 1);
        const int chunks = (wanted + gpu_chunk_ - 1) / gpu_chunk_;
        return Division{ chunks, wanted / chunks };
    }

    real forward_backward(const int* x, const int* y, int B, int T) {
        // Each side's gradient is scaled by its share of this batch, so a
        // second batch before zero_grad would scale the first one twice.
        CHECK(zeroed_, "forward_backward twice without zero_grad: the split "
            "combines one batch's gradients and does not accumulate them");
        zeroed_ = false;
        ++steps_;

        const Division div = division(B);
        const int n = div.gpu_windows();
        const auto [gpu, cpu] = both_halves(x, y, div, B, T, true);

        const Clock::time_point start = Clock::now();
        combine(div, B);
        const double combine_s = seconds_since(start);

        // The first step includes one-time costs on both sides, such as the
        // first touch of new memory. Counting it would distort the share.
        if (steps_ > 1 && n < B) {
            gpu_window_s_.add(gpu.seconds / n);
            cpu_window_s_.add(cpu.seconds / (B - n));
            gpu_share_ = cpu_window_s_.value()
                / (gpu_window_s_.value() + cpu_window_s_.value());

            gpu_ms_.add(1e3 * gpu.seconds);
            cpu_ms_.add(1e3 * cpu.seconds);
            combine_ms_.add(1e3 * combine_s);
            times_.gpu_windows = n;
            times_.gpu_chunks = div.chunks;
            times_.gpu_chunk = div.chunk;
            times_.cpu_windows = B - n;
            times_.gpu_ms = gpu_ms_.value();
            times_.cpu_ms = cpu_ms_.value();
            times_.combine_ms = combine_ms_.value();
        }
        return mean_loss(gpu.loss, cpu.loss, n, B);
    }

    real forward_loss(const int* x, const int* y, int B, int T) {
        const Division div = division(B);
        const auto [gpu, cpu] = both_halves(x, y, div, B, T, false);
        return mean_loss(gpu.loss, cpu.loss, div.gpu_windows(), B);
    }

    void set_training(bool training) {
        cpu_.set_training(training);
        gpu_.set_training(training);
    }

    void adamw_step() {
        const Clock::time_point start = Clock::now();
        opt_.step();
        upload_weights();
        if (steps_ > 1) {
            update_ms_.add(1e3 * seconds_since(start));
            times_.update_ms = update_ms_.value();
        }
    }

    const StepTimes& times() const {
        return times_;
    }

    int cpu_threads() const {
        return cpu_threads_;
    }

    std::string describe() const {
        char buf[160];
        std::snprintf(buf, sizeof(buf), ", at most %d windows a call; the rest of "
            "each batch on %s, %d threads", gpu_chunk_, blocked::kernel_name<real>(),
            cpu_threads_);
        return gpu_.describe() + buf;
    }

private:
    // Runs the GPU's windows on a separate thread and the rest on this
    // thread, at the same time.
    std::pair<Half, Half> both_halves(const int* x, const int* y, Division div,
        int B, int T, bool backward) {
        std::future<Half> gpu_half = std::async(std::launch::async,
            [this, x, y, div, T, backward] {
                return run_gpu(x, y, div, T, backward);
            });

        // If this half throws, the destructor of gpu_half waits for the GPU
        // half to finish before the exception leaves this function. So the
        // thread never outlives the buffers it writes to.
        Half cpu{ real(0), 0.0 };
        const int n = div.gpu_windows();
        if (n < B) {
            const size_t skip = static_cast<size_t>(n) * static_cast<size_t>(T);
            const ThreadLimit limit(cpu_threads_);
            const UsingKernels kernels(Kernels::kBlocked);
            const Clock::time_point start = Clock::now();
            cpu.loss = backward
                ? cpu_.forward_backward(x + skip, y + skip, B - n, T)
                : cpu_.forward_loss(x + skip, y + skip, B - n, T);
            cpu.seconds = seconds_since(start);
        }
        return { gpu_half.get(), cpu };
    }

    // Runs the GPU's windows one chunk at a time. The gradients accumulate on
    // the device from one call to the next, and are copied back once, after
    // the last call. Because every chunk has the same size, the mean of the
    // chunk losses is the mean over all of the GPU's windows.
    Half run_gpu(const int* x, const int* y, Division div, int T, bool backward) {
        const Clock::time_point start = Clock::now();
        const size_t stride = static_cast<size_t>(div.chunk) * static_cast<size_t>(T);
        double total = 0.0;
        for (int c = 0; c < div.chunks; ++c) {
            const size_t skip = static_cast<size_t>(c) * stride;
            const float loss = backward
                ? gpu_.forward_backward(x + skip, y + skip, div.chunk, T)
                : gpu_.forward_loss(x + skip, y + skip, div.chunk, T);
            total += static_cast<double>(loss);
        }
        if (backward) {
            // The copy waits for the last backward pass to finish.
            gpu_.read_grads(gpu_grads_);
        }
        return Half{ static_cast<real>(total / div.chunks), seconds_since(start) };
    }

    // The mean loss over the whole batch: each side's mean loss counts in
    // proportion to its number of windows. It is computed in double, because
    // section 1 requires it to match the reference to 1e-4.
    static real mean_loss(real gpu, real cpu, int n, int B) {
        const double sum = static_cast<double>(n) * static_cast<double>(gpu)
            + static_cast<double>(B - n) * static_cast<double>(cpu);
        return static_cast<real>(sum / static_cast<double>(B));
    }

    // Turns the CPU model's gradients, which are a mean over its own windows,
    // into the mean over the whole batch. The GPU's sum of chunk means is
    // scaled by k/B, as the comment at the top of this file explains. The work
    // is divided by tensor, as in AdamW, so there is only one OpenMP fork per
    // step.
    void combine(Division div, int B) {
        const int n = div.gpu_windows();
        const real wg = static_cast<real>(div.chunk) / static_cast<real>(B);
        const real wc = static_cast<real>(B - n) / static_cast<real>(B);
        std::vector<Param>& tensors = cpu_.params();
        const int np = static_cast<int>(tensors.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int k = 0; k < np; ++k) {
            const size_t kk = static_cast<size_t>(k);
            std::vector<real>& g = *tensors[kk].grad;
            const std::vector<float>& h = gpu_grads_[kk];
            for (size_t i = 0; i < g.size(); ++i) {
                g[i] = wc * g[i] + wg * static_cast<real>(h[i]);
            }
        }
    }

    // Gives the GPU a fresh copy of the master weights. load() also clears the
    // GPU's own AdamW moments, which this program never uses.
    void upload_weights() {
        const std::vector<Param>& tensors = cpu_.params();
        staged_.resize(tensors.size());
        const int np = static_cast<int>(tensors.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int k = 0; k < np; ++k) {
            const size_t kk = static_cast<size_t>(k);
            staged_[kk].assign(tensors[kk].data->begin(), tensors[kk].data->end());
        }
        gpu_.load(staged_);
    }

    gpt_cuda::Model gpu_;
    Gpt cpu_;
    AdamW opt_;
    const int gpu_chunk_;
    const int cpu_threads_;

    double gpu_share_ = kInitialGpuShare;
    Smoothed gpu_window_s_;
    Smoothed cpu_window_s_;
    Smoothed gpu_ms_;
    Smoothed cpu_ms_;
    Smoothed combine_ms_;
    Smoothed update_ms_;
    StepTimes times_;
    int steps_ = 0;
    bool zeroed_ = false;

    // Kept between steps so that their allocated memory is reused.
    std::vector<std::vector<float>> gpu_grads_;
    std::vector<std::vector<float>> staged_;
};

// The largest difference between two tensors, as a fraction of the largest
// entry in either one. This is the measure gpt_bench.h uses. It is relative to
// the whole tensor, not to each element. An entry that should be zero comes
// out as rounding noise on both sides, and the ratio of two noise values
// means nothing.
double tensor_error(const std::vector<real>& a, const std::vector<real>& b) {
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

// The worst tensor_error between two models' gradients, and the tensor it was
// in.
double worst_gradient_error(const std::vector<Param>& a, const std::vector<Param>& b,
    std::string& where) {
    CHECK(a.size() == b.size(), a.size() << " gradient tensors against " << b.size());
    double worst = 0.0;
    where = "none";
    for (size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].grad->size() == b[i].grad->size(), a[i].name << ": "
            << a[i].grad->size() << " entries against " << b[i].grad->size());
        const double e = tensor_error(*a[i].grad, *b[i].grad);
        if (e > worst) {
            worst = e;
            where = a[i].name;
        }
    }
    return worst;
}

void print_config(const Config& cfg) {
    printf("  %d channels, %d heads, %d blocks, context %d, batch %d, dropout %.2f, "
        "lr %.0e\n", cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block, cfg.batch,
        static_cast<double>(cfg.dropout), static_cast<double>(cfg.lr));
}

// Runs one forward and backward pass at the scale-up's width and context, on
// the GPU/CPU split model and on the scalar reference. Both start from the
// same weights and use the same batch, and every gradient tensor is compared.
// Dropout is off, because the two models would draw different masks.
void check_split(const Data& d) {
    std::cout << "\n== 1. the chunked split against the reference ==\n";
    const Config cfg{ kBigEmbed, kBigHeads, kBigLayers, kBigBlock, kCheckBatch,
        real(0), kBigLr };
    print_config(cfg);

    std::mt19937 g(1337);
    Gpt ref(cfg, d.vocab, g);

    // load() replaces the split model's own initial weights, so the
    // generator that creates them does not matter.
    std::mt19937 unused(0);
    SplitModel split(cfg, d.vocab, kCheckChunk, unused);
    split.load(ref.params());

    std::vector<int> x, y;
    sample_batch(d.train, cfg.batch, cfg.block, x, y, g);

    real ref_loss = real(0);
    {
        const UsingKernels kernels(Kernels::kScalar);
        ref.zero_grad();
        ref_loss = ref.forward_backward(x.data(), y.data(), cfg.batch, cfg.block);
    }

    // The share only changes after a step has been timed, so this is the
    // division that the pass below uses.
    const Division div = split.division(cfg.batch);
    split.zero_grad();
    const real split_loss = split.forward_backward(x.data(), y.data(), cfg.batch,
        cfg.block);

    std::string where;
    const double worst = worst_gradient_error(ref.params(), split.params(), where);
    printf("  %zu parameters; %d windows on the GPU in %d calls of %d, %d on the CPU\n",
        ref.parameter_count(), div.gpu_windows(), div.chunks, div.chunk,
        cfg.batch - div.gpu_windows());
    printf("  loss %.6f against %.6f, worst gradient %.1e (%s)\n",
        static_cast<double>(ref_loss), static_cast<double>(split_loss), worst,
        where.c_str());

    sanity::Expect chk;
    chk.close("the loss", split_loss, ref_loss, kLossTolerance);
    chk.that("every gradient", worst < kGradTolerance);
    chk.report();
}

// Whether the loop below evaluates the model before step `it`, following the
// video's schedule.
bool evaluates_at(int it) {
    return it % kEvalEvery == 0 || it == kIters - 1;
}

// How many of those evaluations happen at step `it` or later.
int evaluations_from(int it) {
    int n = 0;
    for (int j = it; j < kIters; ++j) {
        if (evaluates_at(j)) {
            ++n;
        }
    }
    return n;
}

// Python, the hyperparameters the video ends on:
//   batch_size = 64
//   block_size = 256
//   max_iters = 5000
//   eval_interval = 500
//   learning_rate = 3e-4
//   eval_iters = 200
//   n_embd = 384
//   n_head = 6
//   n_layer = 6
//   dropout = 0.2
//
// and the loop that runs them, which evaluates before every step whose number
// is a multiple of eval_interval, and before the last:
//   for iter in range(max_iters):
//       if iter % eval_interval == 0 or iter == max_iters - 1:
//           losses = estimate_loss()
//           print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
//       xb, yb = get_batch('train')
//       logits, loss = model(xb, yb)
//       optimizer.zero_grad(set_to_none=True)
//       loss.backward()
//       optimizer.step()
//
// Returns whether it trained. It does not train if an earlier run left a
// checkpoint to load.
bool train_or_load(const Data& d, const Config& cfg, SplitModel& model,
    std::mt19937& g) {
    std::cout << "\n== 2. the video's scale-up ==\n";
    print_config(cfg);
    const size_t params = model.cpu_model().parameter_count();
    const double flops = step_flops(cfg, params, d.vocab);
    printf("  %zu parameters (the video reads 10.788929 M), %.2f TFLOP a step\n",
        params, flops / 1e12);
    printf("  %s\n", model.describe().c_str());

    const std::string ckpt = d.dir + checkpoint_name(cfg, d);
    sanity::Expect chk;
    if (model.load_checkpoint(cfg, d, ckpt)) {
        const Losses l = estimate_loss(model, d, cfg.batch, cfg.block);
        printf("  loaded %s, so no training was needed\n", ckpt.c_str());
        printf("  train %.4f  val %.4f   the video reads ~%.2f val\n",
            static_cast<double>(l.train), static_cast<double>(l.val), kVideoVal);
        chk.close("validation loss", l.val, kVideoVal, kVideoValTolerance);
        chk.report();
        return false;
    }

    printf("  no checkpoint beside the data, so %d steps of training\n", kIters);
    std::vector<int> x, y;
    Losses first{ real(0), real(0) };
    Losses last{ real(0), real(0) };
    double step_s = 0.0;
    double eval_s = 0.0;
    int evals = 0;
    double recent_s = 0.0;
    double recent_loss = 0.0;
    const Clock::time_point t0 = Clock::now();
    for (int it = 0; it < kIters; ++it) {
        if (evaluates_at(it)) {
            const Clock::time_point e0 = Clock::now();
            last = estimate_loss(model, d, cfg.batch, cfg.block);
            eval_s += seconds_since(e0);
            ++evals;
            if (it == 0) {
                first = last;
            }
            printf("    step %d: train loss %.4f, val loss %.4f   (%.1f min in)\n",
                it, static_cast<double>(last.train), static_cast<double>(last.val),
                seconds_since(t0) / 60.0);
        }

        sample_batch(d.train, cfg.batch, cfg.block, x, y, g);
        const Clock::time_point s0 = Clock::now();
        model.zero_grad();
        const real loss = model.forward_backward(x.data(), y.data(), cfg.batch,
            cfg.block);
        model.adamw_step();
        const double s = seconds_since(s0);
        step_s += s;
        recent_s += s;
        recent_loss += static_cast<double>(loss);

        if ((it + 1) % kProgressEvery == 0) {
            // The remaining steps at the average step time so far, plus the
            // remaining evaluations at the average evaluation time.
            const double left_s = (kIters - it - 1) * step_s / (it + 1)
                + evaluations_from(it + 1) * eval_s / evals;
            printf("      %d steps, batch loss %.4f, %.0f ms a step, %.0f GFLOP/s, "
                "%d windows on the GPU, about %.0f min to go\n",
                it + 1, recent_loss / kProgressEvery, 1e3 * recent_s / kProgressEvery,
                flops * kProgressEvery / recent_s / 1e9, model.times().gpu_windows,
                left_s / 60.0);
            recent_s = 0.0;
            recent_loss = 0.0;
        }
    }

    printf("  %d steps in %.1f min: %.0f ms a step, %.0f GFLOP/s\n", kIters,
        step_s / 60.0, 1e3 * step_s / kIters, flops * kIters / step_s / 1e9);
    printf("  %d evaluations of %d batches on each split took another %.1f min\n",
        evals, kEvalIters, eval_s / 60.0);
    if (save_params(model.params(), cfg, d, ckpt)) {
        printf("  saved to %s - delete it to train again\n", ckpt.c_str());
    }

    chk.close("loss before training", first.val,
        std::log(static_cast<double>(d.vocab)), kStartTolerance);
    chk.close("validation loss", last.val, kVideoVal, kVideoValTolerance);
    chk.report();
    return true;
}

// A step takes as long as its slower half, plus the two serial stages after
// it. So this output shows where more speed could come from: a better balance
// between the two halves, or a faster exchange of gradients and weights.
void print_times(const SplitModel& model, bool trained) {
    std::cout << "\n== 3. where the training steps went ==\n";
    if (!trained) {
        printf("  nowhere: the weights came from a checkpoint\n");
        return;
    }
    const StepTimes& t = model.times();
    printf("  Smoothed over the last few steps.\n");
    printf("    GPU  %3d windows  %8.1f ms  %d calls of %d, then the gradients to "
        "the host\n", t.gpu_windows, t.gpu_ms, t.gpu_chunks, t.gpu_chunk);
    printf("    CPU  %3d windows  %8.1f ms  forward and backward, %d threads\n",
        t.cpu_windows, t.cpu_ms, model.cpu_threads());
    printf("    then             %8.1f ms  combining the two gradients\n",
        t.combine_ms);
    printf("    and              %8.1f ms  AdamW, then the weights to the GPU\n",
        t.update_ms);
    printf("  a step: about %.1f ms, %.1f of it after both halves are done\n",
        std::max(t.gpu_ms, t.cpu_ms) + t.combine_ms + t.update_ms,
        t.combine_ms + t.update_ms);
}

// Python: print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
void sample(const Data& d, SplitModel& model) {
    std::cout << "\n== 4. generating ==\n";
    std::mt19937 g(1337);
    const UsingKernels kernels(Kernels::kBlocked);
    const Clock::time_point t0 = Clock::now();
    const std::string text = d.decode(model.cpu_model().generate(kSampleLength, g));
    printf("---\n%s\n---\n", text.c_str());
    printf("  %d characters in %.1f s, each one a forward pass over up to %d of "
        "context\n", kSampleLength, seconds_since(t0), kBigBlock);
}

}  // namespace

int main() {
    // A failed CHECK, including a failed CUDA call on either thread, is
    // reported here and ends the program. Without this, it would reach
    // terminate().
    try {
        // Started before anything else, so the total at the end covers every
        // section, including the check and the sampling.
        const Clock::time_point start = Clock::now();
        printf("gpt_scaled: the video's scale-up, each batch shared between CUDA "
            "and the blocked multiply\n");

        std::cout << "\n== 0. tiny shakespeare ==\n";
        Data d;
        if (!load_corpus(d)) {
            printf("  could not find %s - expected it in makemore/\n", kCorpus);
            return 1;
        }
        printf("  vocabulary of %d characters, %zu training tokens\n", d.vocab,
            d.train.size());

        check_split(d);

        // Seeded in the same way as gpt.cpp's scale-up, so both programs start
        // from the same weights.
        const Config cfg{ kBigEmbed, kBigHeads, kBigLayers, kBigBlock, kBigBatch,
            kBigDropout, kBigLr };
        std::mt19937 g(1337);
        SplitModel model(cfg, d.vocab, kGpuChunk, g);
        const bool trained = train_or_load(d, cfg, model, g);
        print_times(model, trained);
        sample(d, model);

        const double total_s = seconds_since(start);
        printf("\n  the whole run took %.1f min (%.2f hours)\n", total_s / 60.0,
            total_s / 3600.0);
    }
    catch (const std::exception& e) {
        std::cerr << "\n*** " << e.what() << "\n";
        return 1;
    }
    return 0;
}
