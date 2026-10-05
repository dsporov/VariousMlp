// gpt_split.cpp - every batch shared between the GPU and the CPU.
//
// gpt_cuda.cu and the blocked multiply each train the model alone. Here they
// train it together, by data parallelism: each step the batch's windows are
// divided between the two, each side runs forward and backward on its share
// with its own copy of the weights, and the two gradients become one before a
// single AdamW step. The combination is weighted by share, because each side's
// loss - and so its gradient - is a mean over its own windows only. With n of
// the B windows on the GPU,
//
//     g = (n / B) * g_gpu + ((B - n) / B) * g_cpu
//
// which is the gradient of the mean over all B, the one the reference
// computes. Section 1 of the bench checks exactly that.
//
// The CPU's copy of the weights is the master. AdamW runs there and the result
// goes to the GPU after every step, so there is one set of weights and one
// optimiser state. Were each side to run its own AdamW on the same gradient,
// the two copies would drift apart in the last bits - nvcc fuses multiply-adds
// that the host compiler does not - and nothing would notice. The choice is
// forced as much as chosen: gpt_cuda.h takes weights in and gives gradients
// out and does nothing else, and this target uses it unchanged so that
// gpt_cuda stays exactly what it was. The price is two full copies of the
// parameters across PCIe every step, through the pageable buffers load() and
// read_grads() allocate for themselves. The section this program adds after
// the bench's three says what that costs.
//
// The split adapts. Every step times both halves, and the GPU's share moves
// toward where the two finish together. It starts from the throughputs
// measured at the bench's width: 504 GFLOP/s on the GTX 1050 Ti against 319
// blocked on the i9, about 0.6 of the windows to the GPU. At the walkthrough's
// 32 channels expect nothing from it - the GPU is launch-bound there and
// already slower than the blocked multiply on its own.
//
// CMake-only, because it has its own main. Build the gpt_split target.

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

// The GPU's share of each batch before anything has been timed:
// 504 / (504 + 319), rounded.
constexpr double kInitialGpuShare = 0.6;

// How far one step's timing moves the running estimates. Low enough that one
// slow step - a burst of page faults, the OS scheduling something else - does
// not swing the split, high enough to settle within the few dozen steps
// section 3 times.
constexpr double kSmoothing = 0.25;

// The thread that drives the GPU is busy for the whole of its half: waiting on
// copies, which the CUDA runtime does by default by spinning, and then
// unpacking the gradients. The CPU half gives up a thread to it rather than
// oversubscribing the cores, which an OpenMP team, waiting at every barrier
// for its slowest member, takes badly.
constexpr int kGpuDriverThreads = 1;

// A running average that takes its first sample as it is.
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

// Sets the OpenMP team size for this thread's parallel regions for the length
// of one call, and puts back what was there. The bench runs the reference on
// this same thread, and it should keep every core.
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

// Where the most recent backend's steps went, smoothed. The bench owns the
// backends and destroys each at the end of its section, so they leave this
// behind for main to print once section 3 is over.
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
        // combine() walks the two sets of gradients side by side, so they
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

    // After forward_backward the CPU model's gradients are the combined ones.
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

        // The first step pays for allocation and first touch on both sides,
        // and would skew the split if it counted.
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
    // One side's part of a pass: the mean loss over its windows, and how long
    // it took.
    struct Half {
        real loss;
        double seconds;
    };

    // Both sides get at least one window whenever there are two to go round,
    // so that both are always being timed and the split can keep moving. A
    // single window cannot be split, and goes to the GPU.
    int gpu_rows(int B) const {
        if (B < 2) {
            return B;
        }
        const int n = static_cast<int>(std::lround(gpu_share_ * B));
        return std::clamp(n, 1, B - 1);
    }

    // The first n windows on the GPU, driven from a thread of its own, and the
    // rest on this one, at the same time. A side with no windows is skipped.
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

        // Should this half throw, gpu_half's destructor waits for the GPU's
        // before the exception leaves, so the thread never outlives the
        // buffers it writes.
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

    // Each side's mean, weighted by how many windows it saw. In double, since
    // the bench holds this to 1e-4 of the reference's.
    static real mean_loss(real gpu, real cpu, int n, int B) {
        const double sum = static_cast<double>(n) * static_cast<double>(gpu)
            + static_cast<double>(B - n) * static_cast<double>(cpu);
        return static_cast<real>(sum / static_cast<double>(B));
    }

    // Turns the CPU model's gradients, a mean over its own windows, into the
    // mean over the whole batch. With no windows on the GPU they already are.
    // Split by tensor, like AdamW, for one fork a step.
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

    // Kept between steps so that their capacity is too.
    std::vector<std::vector<float>> gpu_grads_;
    std::vector<std::vector<float>> staged_;
};

// A step is its slower half plus the two serial stages after it, so this says
// whether what is left to win is in the balance or in the exchange.
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
    // A CHECK that fires - a CUDA call that failed included, on either thread
    // - is reported and ends the program rather than being left to terminate().
    try {
        printf("gpt_split: each batch shared between CUDA and the blocked multiply\n");

        Data d;
        if (!bench::load_data(d)) {
            return 1;
        }

        // The CPU side's own initial weights are thrown away by load(), so
        // the generator it draws them from does not matter.
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
