// gpt_split.cpp - every batch shared between the GPU and the CPU.
//
// gpt_cuda.cu and the blocked multiply can each train the model alone. Here
// they train it together, with data parallelism. In each step, the windows of
// the batch are divided between the two. Each side runs the forward and
// backward pass on its share, with its own copy of the weights. Then the two
// gradients are combined into one, before a single AdamW step. Each side's
// gradient is scaled by its share of the batch, because each side's loss, and
// so its gradient, is a mean over its own windows only. With n of the B
// windows on the GPU,
//
//     g = (n / B) * g_gpu + ((B - n) / B) * g_cpu
//
// This is the gradient of the mean over all B windows, which is what the
// reference computes. Section 1 of the bench checks exactly that.
//
// The CPU's copy of the weights is the master copy. AdamW runs on the CPU, and
// the result is copied to the GPU after every step. So there is one set of
// weights and one optimiser state. If each side ran its own AdamW on the same
// gradient, the two copies would slowly become different in the last bits,
// because nvcc fuses multiply-adds that the host compiler does not fuse. And
// nothing would detect it. There was also no other choice: gpt_cuda.h takes
// weights in and gives gradients out, and does nothing else. This target uses
// it unchanged, so that gpt_cuda stays exactly as it was. The cost is two full
// copies of the parameters across PCIe in every step, through the pageable
// buffers that load() and read_grads() allocate for themselves. The extra
// section that this program adds after the bench's three sections shows that
// cost.
//
// The GPU's share adapts. Every step times both halves, and the GPU's share
// moves toward the point where both halves finish at the same time. It starts
// from the speeds measured at the bench's width: 504 GFLOP/s on the
// GTX 1050 Ti and 319 GFLOP/s for the blocked multiply on the i9. So about 0.6
// of the windows go to the GPU. At the walkthrough's 32 channels, expect no
// gain: there the GPU is limited by the cost of kernel launches, and it is
// already slower than the blocked multiply alone.
//
// CMake only, because it has its own main. Build the gpt_split target.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "gpt_bench.h"
#include "gpt_cuda.h"
#include "gpt_model.h"

using namespace gpt;
using bench::Clock;
using bench::seconds_since;

namespace {

// The GPU's share of each batch before any step has been timed:
// 504 / (504 + 319), rounded.
constexpr double kInitialGpuShare = 0.6;

// How much one step's timing changes the running averages. It is low enough
// that one slow step (for example many page faults at once, or the OS running
// something else) does not move the share much. It is high enough that the
// share settles within the few dozen steps that section 3 times.
constexpr double kSmoothing = 0.25;

// The thread that drives the GPU is busy for the whole GPU half. It waits for
// copies, which the CUDA runtime does by spinning by default, and then it
// unpacks the gradients. So the CPU half uses one thread fewer, and the cores
// are not oversubscribed. Oversubscription slows an OpenMP team down a lot,
// because the team waits at every barrier for its slowest thread.
constexpr int kGpuDriverThreads = 1;

// A running average. The first sample is used as it is.
class Smoothed {
public:
    void add(double sample) {
        value_ = seen_ ? value_ + kSmoothing * (sample - value_) : sample;
        seen_ = true;
    }

    double value() const { return value_; }

private:
    double value_ = 0.0;
    bool seen_ = false;
};

// Sets the OpenMP team size for this thread's parallel regions during one
// call, and restores the previous size afterwards. The bench runs the
// reference on this same thread, and the reference should keep all cores.
class ThreadLimit {
public:
    explicit ThreadLimit(int n) : saved_(available_threads()) { set(n); }
    ~ThreadLimit() { set(saved_); }
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

// Where the time of the most recent backend's steps went, as running averages.
// The bench owns the backends and destroys each one at the end of its section.
// So each backend writes its numbers here, and main prints them after
// section 3.
struct SplitLog {
    int gpu_rows = 0;
    int cpu_rows = 0;
    int cpu_threads = 0;
    double gpu_ms = 0.0;
    double cpu_ms = 0.0;
    double combine_ms = 0.0;
    double update_ms = 0.0;
};

class SplitBackend {
public:
    SplitBackend(const Config& cfg, int vocab, std::mt19937& g, SplitLog& log)
        : gpu_(gpt_cuda::Shape{ cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block,
            vocab, static_cast<float>(cfg.dropout) }),
          cpu_(cfg, vocab, Kernels::kBlocked, g),
          cpu_threads_(std::max(1, available_threads() - kGpuDriverThreads)),
          log_(log)
    {
        // combine() goes through the two sets of gradients together, so they
        // have to be the same tensors in the same order.
        const std::vector<size_t>& sizes = gpu_.tensor_sizes();
        const std::vector<Param>& params = cpu_.model().params();
        CHECK(sizes.size() == params.size(), "the GPU model has " << sizes.size()
            << " tensors and the CPU model " << params.size());
        for (size_t i = 0; i < sizes.size(); ++i) {
            CHECK(sizes[i] == params[i].data->size(), params[i].name << " has "
                << params[i].data->size() << " entries on the CPU and "
                << sizes[i] << " on the GPU");
        }
    }

    SplitBackend(const SplitBackend&) = delete;
    SplitBackend& operator=(const SplitBackend&) = delete;

    void load(const std::vector<Param>& weights) {
        cpu_.load(weights);
        upload_weights();
    }

    // After forward_backward, the CPU model's gradients are the combined
    // gradients.
    void read_grads(std::vector<std::vector<real>>& out) { cpu_.read_grads(out); }

    void zero_grad() {
        cpu_.zero_grad();
        gpu_.zero_grad();
        zeroed_ = true;
    }

    real forward_backward(const int* x, const int* y, int B, int T) {
        // Each side's gradient is scaled by its share of this batch, so a
        // second batch before zero_grad would scale the first one twice.
        CHECK(zeroed_, "forward_backward twice without zero_grad: the split "
            "combines one batch's gradients and does not accumulate them");
        zeroed_ = false;
        ++steps_;

        const int n = gpu_rows(B);
        const auto [gpu, cpu] = both_halves(x, y, n, B, T, true);

        const Clock::time_point start = Clock::now();
        combine(n, B);
        const double combine_s = seconds_since(start);

        // The first step includes one-time costs on both sides: allocation,
        // and the first touch of new memory. Counting it would distort the
        // share.
        if (steps_ > 1 && n > 0 && n < B) {
            gpu_row_s_.add(gpu.seconds / n);
            cpu_row_s_.add(cpu.seconds / (B - n));
            gpu_share_ = cpu_row_s_.value() / (gpu_row_s_.value() + cpu_row_s_.value());

            gpu_ms_.add(1e3 * gpu.seconds);
            cpu_ms_.add(1e3 * cpu.seconds);
            combine_ms_.add(1e3 * combine_s);
            log_.gpu_rows = n;
            log_.cpu_rows = B - n;
            log_.cpu_threads = cpu_threads_;
            log_.gpu_ms = gpu_ms_.value();
            log_.cpu_ms = cpu_ms_.value();
            log_.combine_ms = combine_ms_.value();
        }
        return mean_loss(gpu.loss, cpu.loss, n, B);
    }

    real forward_loss(const int* x, const int* y, int B, int T) {
        const int n = gpu_rows(B);
        const auto [gpu, cpu] = both_halves(x, y, n, B, T, false);
        return mean_loss(gpu.loss, cpu.loss, n, B);
    }

    void set_training(bool training) {
        cpu_.set_training(training);
        gpu_.set_training(training);
    }

    void adamw_step(real lr) {
        const Clock::time_point start = Clock::now();
        cpu_.adamw_step(lr);
        upload_weights();
        if (steps_ > 1) {
            update_ms_.add(1e3 * seconds_since(start));
            log_.update_ms = update_ms_.value();
        }
    }

    void sync() { gpu_.sync(); }

    std::string describe() const {
        char buf[128];
        std::snprintf(buf, sizeof(buf), ", %.0f%% of each batch; the rest on %s, "
            "%d threads", 100.0 * gpu_share_, blocked::kernel_name<real>(),
            cpu_threads_);
        return gpu_.describe() + buf;
    }

private:
    // The result of one side's part of a pass: the mean loss over its
    // windows, and the time it took.
    struct Half {
        real loss;
        double seconds;
    };

    // If there are at least two windows, each side gets at least one. Then
    // both sides are always timed, and the share can keep adjusting. A single
    // window cannot be divided, so it goes to the GPU.
    int gpu_rows(int B) const {
        if (B < 2) {
            return B;
        }
        const int n = static_cast<int>(std::lround(gpu_share_ * B));
        return std::clamp(n, 1, B - 1);
    }

    // Runs the first n windows on the GPU, driven from a separate thread, and
    // the rest on this thread, at the same time. A side with no windows is
    // skipped.
    std::pair<Half, Half> both_halves(const int* x, const int* y, int n, int B,
        int T, bool backward) {
        std::future<Half> gpu_half;
        if (n > 0) {
            gpu_half = std::async(std::launch::async, [this, x, y, n, T, backward] {
                const Clock::time_point start = Clock::now();
                float loss = 0.0f;
                if (backward) {
                    loss = gpu_.forward_backward(x, y, n, T);
                    // The copy waits for the backward pass to finish.
                    gpu_.read_grads(gpu_grads_);
                }
                else {
                    loss = gpu_.forward_loss(x, y, n, T);
                }
                return Half{ static_cast<real>(loss), seconds_since(start) };
            });
        }

        // If this half throws, the destructor of gpu_half waits for the GPU
        // half to finish before the exception leaves this function. So the
        // thread never outlives the buffers it writes to.
        Half cpu{ real(0), 0.0 };
        if (n < B) {
            const size_t skip = static_cast<size_t>(n) * static_cast<size_t>(T);
            const ThreadLimit limit(cpu_threads_);
            const Clock::time_point start = Clock::now();
            cpu.loss = backward
                ? cpu_.forward_backward(x + skip, y + skip, B - n, T)
                : cpu_.forward_loss(x + skip, y + skip, B - n, T);
            cpu.seconds = seconds_since(start);
        }

        const Half gpu = n > 0 ? gpu_half.get() : Half{ real(0), 0.0 };
        return { gpu, cpu };
    }

    // The mean loss over the whole batch: each side's mean loss counts in
    // proportion to its number of windows. It is computed in double, because
    // the bench requires it to match the reference to 1e-4.
    static real mean_loss(real gpu, real cpu, int n, int B) {
        const double sum = static_cast<double>(n) * static_cast<double>(gpu)
            + static_cast<double>(B - n) * static_cast<double>(cpu);
        return static_cast<real>(sum / static_cast<double>(B));
    }

    // Turns the CPU model's gradients, which are a mean over its own windows,
    // into the mean over the whole batch. With no windows on the GPU, they
    // already are that mean. The work is divided by tensor, as in AdamW, so
    // there is only one OpenMP fork per step.
    void combine(int n, int B) {
        if (n == 0) {
            return;
        }
        const real wg = static_cast<real>(n) / static_cast<real>(B);
        const real wc = static_cast<real>(B - n) / static_cast<real>(B);
        std::vector<Param>& params = cpu_.model().params();
        const int np = static_cast<int>(params.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int k = 0; k < np; ++k) {
            const size_t kk = static_cast<size_t>(k);
            std::vector<real>& g = *params[kk].grad;
            const std::vector<float>& h = gpu_grads_[kk];
            for (size_t i = 0; i < g.size(); ++i) {
                g[i] = wc * g[i] + wg * static_cast<real>(h[i]);
            }
        }
    }

    // Gives the GPU a fresh copy of the master weights. load() also clears the
    // GPU's own AdamW moments, which this backend never uses.
    void upload_weights() {
        const std::vector<Param>& params = cpu_.model().params();
        staged_.resize(params.size());
        const int np = static_cast<int>(params.size());
#pragma omp parallel for schedule(dynamic, 1)
        for (int k = 0; k < np; ++k) {
            const size_t kk = static_cast<size_t>(k);
            staged_[kk].assign(params[kk].data->begin(), params[kk].data->end());
        }
        gpu_.load(staged_);
    }

    gpt_cuda::Model gpu_;
    bench::CpuModel cpu_;
    const int cpu_threads_;
    SplitLog& log_;

    double gpu_share_ = kInitialGpuShare;
    Smoothed gpu_row_s_;
    Smoothed cpu_row_s_;
    Smoothed gpu_ms_;
    Smoothed cpu_ms_;
    Smoothed combine_ms_;
    Smoothed update_ms_;
    int steps_ = 0;
    bool zeroed_ = false;

    // Kept between steps so that their allocated memory is reused.
    std::vector<std::vector<float>> gpu_grads_;
    std::vector<std::vector<float>> staged_;
};

// A step takes as long as its slower half, plus the two serial stages after
// it. So this output shows where more speed could come from: a better balance
// between the two halves, or a faster exchange of gradients and weights.
void print_log(const SplitLog& log) {
    std::cout << "\n== 4. where section 3's steps went ==\n";
    printf("  Smoothed over its last few steps.\n");
    printf("    GPU  %3d windows  %8.1f ms  forward, backward, gradients to the host\n",
        log.gpu_rows, log.gpu_ms);
    printf("    CPU  %3d windows  %8.1f ms  forward and backward, %d threads\n",
        log.cpu_rows, log.cpu_ms, log.cpu_threads);
    printf("    then             %8.1f ms  combining the two gradients\n",
        log.combine_ms);
    printf("    and              %8.1f ms  AdamW, then the weights to the GPU\n",
        log.update_ms);
    printf("  a step: about %.1f ms, %.1f of it after both halves are done\n",
        std::max(log.gpu_ms, log.cpu_ms) + log.combine_ms + log.update_ms,
        log.combine_ms + log.update_ms);
}

}  // namespace

int main() {
    // A failed CHECK, including a failed CUDA call on either thread, is
    // reported here and ends the program. Without this, it would reach
    // terminate().
    try {
        printf("gpt_split: each batch shared between CUDA and the blocked multiply\n");

        Data d;
        if (!bench::load_data(d)) {
            return 1;
        }

        // load() replaces the CPU side's own initial weights, so the
        // generator that creates them does not matter.
        SplitLog log;
        const auto make = [&log](const Config& cfg, int vocab) {
            std::mt19937 unused(0);
            return std::make_unique<SplitBackend>(cfg, vocab, unused, log);
        };
        {
            const Config probe_cfg{ kEmbed, kHeads, kLayers, kBlock, kBatch,
                kDropout, kLr };
            printf("  %s\n", make(probe_cfg, d.vocab)->describe().c_str());
        }
        bench::run_sections<SplitBackend>(d, "split", make);
        print_log(log);
    }
    catch (const std::exception& e) {
        std::cerr << "\n*** " << e.what() << "\n";
        return 1;
    }
    return 0;
}
