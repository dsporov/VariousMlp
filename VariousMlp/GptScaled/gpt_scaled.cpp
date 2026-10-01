// gpt_scaled.cpp - the video's scale-up, trained on the GPU and the CPU at once.
//
// The end of the video turns the walkthrough's model into a real one: 384
// channels, 6 heads, 6 blocks, a context of 256 characters, batches of 64,
// dropout 0.2 - 10.8M parameters, validation loss 1.48, about fifteen minutes
// on an A100. gpt.cpp records that run rather than making it. This program
// makes it, with the two fastest things the repository has: the hand-written
// CUDA model and the cache-blocked AVX2 multiply, sharing every batch the way
// gpt_split.cpp shares it.
//
// gpt_split itself cannot train this shape on a GTX 1050 Ti, and not because
// of the parameters - weights, gradients and both Adam moments together are
// 173MB. It is the activations. At context 256 the GPU model keeps about 85MB
// a window between its forward and backward passes, two thirds of it in
// (B, T, C) tensors and the rest in (B, heads, T, T) attention squares, so
// the 38 windows of gpt_split's opening share want 3.2GB of a card that has
// 4GB and also holds the CUDA context and, under Windows, the desktop. And
// the share only grows from there when the GPU turns out to be the faster
// side.
//
// So the GPU takes its windows in chunks, at most kGpuChunk to a call, and
// every chunk reuses the same buffers. Nothing in gpt_cuda.cu had to change
// for that: its backward passes add into the gradient rather than overwrite
// it, so the chunks' gradients pile up on the device and cross to the host
// once a step. The one thing to get right is the weight. Each call's loss,
// and so its gradient, is a mean over that call's windows only, which makes
// the pile a sum of means. With c chunks of k windows it is c times the mean
// over the GPU's c*k windows, so its weight in the batch's mean is k/B rather
// than c*k/B:
//
//     g = (k / B) * (g_1 + ... + g_c) + ((B - c*k) / B) * g_cpu
//
// That needs every chunk the same size, so the GPU's share is rounded down to
// a multiple of its chunk count - 38 windows become three chunks of 12.
// Section 1 holds the result to the scalar reference.
//
// Everything else is gpt_split's: the CPU's copy of the weights is the
// master, AdamW runs there and the weights go to the GPU after every step,
// and the split moves toward where both sides finish together.
//
// This directory is a project of its own. check.h, gpt_gemm.h, gpt_model.h,
// gpt_cuda.h and gpt_cuda.cu are copies of the files in Karpathy/, and
// nothing here includes anything from there. It touches the rest of the
// repository only through files on disk: it reads Karpathy/input.txt, and it
// saves the trained weights beside it in gpt_model.h's checkpoint format,
// under gpt_model.h's name for this shape - the file gpt.cpp's scale-up
// branch looks for, and one the chat tool can read.
//
// Expect a few hours: 5000 steps of 1.1 TFLOP each, and the video's eleven
// evaluations of 200 batches on each split.
//
// CMake-only, because it has its own main, and built only when CUDA is found.
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

// Evaluations are 500 steps apart, which at this size is minutes rather than
// seconds, so a line of progress goes out more often than that.
constexpr int kProgressEvery = 100;

// The most windows the GPU is handed in one call. Its activations and scratch
// are sized by the largest call, at about 85MB a window at this shape, so
// sixteen is 1.4GB and leaves a 4GB card room for the weights, the CUDA
// context and the desktop. Larger chunks mean fewer launches, but 16 windows
// are already 4096 rows, which gives a multiply far more tiles than the
// GTX 1050 Ti's six SMs can run at once.
constexpr int kGpuChunk = 16;

// The GPU's share of each batch before anything has been timed, as in
// gpt_split.cpp: 504 GFLOP/s against the blocked multiply's 319, measured at
// context 64. Context 256 weighs attention more heavily, and the split finds
// its own level within a few dozen steps.
constexpr double kInitialGpuShare = 0.6;

// How far one step's timing moves the running estimates. Low enough that one
// slow step does not swing the split, high enough to settle quickly.
constexpr double kSmoothing = 0.25;

// The thread that drives the GPU is busy for the whole of its half, waiting
// on copies, which the CUDA runtime does by spinning. The CPU half gives up a
// thread to it rather than oversubscribing the cores, which an OpenMP team,
// waiting at every barrier for its slowest member, takes badly.
constexpr int kGpuDriverThreads = 1;

// Section 1's batch: few enough windows at full context that the scalar
// reference gets through them in seconds, in chunks small enough that the
// GPU's share takes more than one call - which is the thing being checked.
constexpr int kCheckBatch = 6;
constexpr int kCheckChunk = 2;

// gpt_bench.h's tolerances. A gradient tensor has to match the reference's to
// this fraction of its largest entry: float rounding in a different order
// lands around 1e-6 to 1e-5, and a chunk weighted wrongly, or lost, in the
// tenths.
constexpr double kGradTolerance = 1e-3;
constexpr double kLossTolerance = 1e-4;

// An untrained model gives every character about the same probability, so its
// loss starts near ln(vocab) - a little above, since weights of width 0.02
// already make the logits uneven.
constexpr double kStartTolerance = 0.2;

// The video's validation loss at the end, as gpt_model.h records it. This
// run's initial weights, batches and dropout masks are all its own, so it
// should land near that rather than on it.
constexpr double kVideoVal = 1.48;
constexpr double kVideoValTolerance = 0.05;

// Characters to sample from the trained model. On the CPU, because gpt_cuda.h
// hands back losses and never logits.
constexpr int kSampleLength = 1000;

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// A running average that takes its first sample as it is.
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

// Sets the OpenMP team size for this thread's parallel regions for the length
// of one call, and puts back what was there.
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

// Sets the multiply gpt_model.h runs for the length of one call, and puts
// back what was there. The switch is process-wide, but only the main thread
// ever runs the CPU model: the blocked multiply for the split's CPU half and
// for sampling, the scalar loops for section 1's reference.
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

// How a batch's windows are dealt out: chunks calls of chunk windows each on
// the GPU, from the front of the batch, and the rest on the CPU.
struct Division {
    int chunks;
    int chunk;

    int gpu_windows() const {
        return chunks * chunk;
    }
};

// One side's part of a pass: the mean loss over its windows, and how long it
// took.
struct Half {
    real loss;
    double seconds;
};

// Where the training steps went, smoothed over the last few.
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
// It has the members gpt_model.h's estimate_loss asks of a model, so the
// video's evaluation is shared the same way.
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

        // combine() walks the two sets of gradients side by side, so they
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

    // The CPU's copy is the master: the one AdamW steps, the one saved, and
    // the one sampled from.
    Gpt& cpu_model() {
        return cpu_;
    }

    std::vector<Param>& params() {
        return cpu_.params();
    }

    // Another model's weights, on both sides.
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

    // A checkpoint in gpt_model.h's format. A missing or stale one is not an
    // error, as there: it means train instead.
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

    // How a batch of B windows is dealt out at the current share. Both sides
    // get at least one window whenever there are two to go round, so that
    // both are always being timed and the split can keep moving; a single
    // window cannot be split, and goes to the GPU. The GPU's share is then
    // rounded down to equal chunks, which combine() relies on.
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

        // The first step pays for first touch on both sides, and would skew
        // the split if it counted.
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
    // The GPU's windows on a thread of their own and the rest on this one, at
    // the same time.
    std::pair<Half, Half> both_halves(const int* x, const int* y, Division div,
        int B, int T, bool backward) {
        std::future<Half> gpu_half = std::async(std::launch::async,
            [this, x, y, div, T, backward] {
                return run_gpu(x, y, div, T, backward);
            });

        // Should this half throw, gpu_half's destructor waits for the GPU's
        // before the exception leaves, so the thread never outlives the
        // buffers it writes.
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

    // The GPU's windows, a chunk at a time. The gradients pile up on the
    // device from one call to the next and come back once, after the last.
    // With every chunk the same size, the mean of their losses is the mean
    // over all of the GPU's windows.
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

    // Each side's mean, weighted by how many windows it saw. In double, since
    // section 1 holds this to 1e-4 of the reference's.
    static real mean_loss(real gpu, real cpu, int n, int B) {
        const double sum = static_cast<double>(n) * static_cast<double>(gpu)
            + static_cast<double>(B - n) * static_cast<double>(cpu);
        return static_cast<real>(sum / static_cast<double>(B));
    }

    // Turns the CPU model's gradients, a mean over its own windows, into the
    // mean over the whole batch, with the GPU's sum of chunk means weighted by
    // one chunk's size as the comment at the top works out. Split by tensor,
    // like AdamW, for one fork a step.
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

    // Kept between steps so that their capacity is too.
    std::vector<std::vector<float>> gpu_grads_;
    std::vector<std::vector<float>> staged_;
};

// The largest difference between two tensors, as a fraction of the largest
// entry in either: gpt_bench.h's measure. Relative to the tensor rather than
// element by element, because an entry that should be zero comes out as
// rounding noise on both sides and the ratio of two noises means nothing.
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

// One forward and backward pass at the scale-up's width and context, on the
// split and on the scalar reference, from the same weights on the same batch,
// and every gradient tensor compared. Without dropout, since the two would
// draw different masks.
void check_split(const Data& d) {
    std::cout << "\n== 1. the chunked split against the reference ==\n";
    const Config cfg{ kBigEmbed, kBigHeads, kBigLayers, kBigBlock, kCheckBatch,
        real(0), kBigLr };
    print_config(cfg);

    std::mt19937 g(1337);
    Gpt ref(cfg, d.vocab, g);

    // The split's own initial weights are thrown away by load(), so the
    // generator it draws them from does not matter.
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

    // The share only moves once a step has been timed, so this is the
    // division the pass below uses.
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

// Whether the loop below evaluates before step it: the video's schedule.
bool evaluates_at(int it) {
    return it % kEvalEvery == 0 || it == kIters - 1;
}

// How many of those evaluations are at step it or later.
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
// Returns whether it trained, which it does not when an earlier run left a
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
            // The steps left at the average step so far, and the evaluations
            // left at the average evaluation.
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

// A step is its slower half plus the two serial stages after it, so this says
// whether what is left to win is in the balance or in the exchange.
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
    // A CHECK that fires - a CUDA call that failed included, on either thread
    // - is reported and ends the program rather than being left to terminate().
    try {
        // Started before anything else, so the total at the end covers every
        // section, the check and the sample included.
        const Clock::time_point start = Clock::now();
        printf("gpt_scaled: the video's scale-up, each batch shared between CUDA "
            "and the blocked multiply\n");

        std::cout << "\n== 0. tiny shakespeare ==\n";
        Data d;
        if (!load_corpus(d)) {
            printf("  could not find %s - expected it in Karpathy/\n", kCorpus);
            return 1;
        }
        printf("  vocabulary of %d characters, %zu training tokens\n", d.vocab,
            d.train.size());

        check_split(d);

        // Seeded as gpt.cpp seeds its scale-up, so the two start from the
        // same weights.
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
