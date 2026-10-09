// gpt_cuda_main.cpp - runs gpt_cuda.cu's model on the bench.
//
// gpt_cuda.h uses plain floats, so that nvcc never has to compile gpt_model.h.
// This file is the host code that sees both. CudaBackend is the adapter
// between them: it provides the members that gpt_bench.h needs from a
// candidate, and converts gpt::real to float on the way in and back on the
// way out. With the default real = float, the conversions are plain copies.
//
// CMake only, because it has its own main. Build the gpt_cuda target.

#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "gpt_bench.h"
#include "gpt_cuda.h"
#include "gpt_model.h"

using namespace gpt;

namespace {

class CudaBackend {
public:
    CudaBackend(const Config& cfg, int vocab)
        : model_(gpt_cuda::Shape{ cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.block,
            vocab, static_cast<float>(cfg.dropout) })
    {
    }

    void load(const std::vector<Param>& weights) {
        std::vector<std::vector<float>> w(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            w[i].assign(weights[i].data->begin(), weights[i].data->end());
        }
        model_.load(w);
    }

    void read_grads(std::vector<std::vector<real>>& out) {
        std::vector<std::vector<float>> g;
        model_.read_grads(g);
        out.resize(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            out[i].assign(g[i].begin(), g[i].end());
        }
    }

    void zero_grad() { model_.zero_grad(); }

    real forward_backward(const int* x, const int* y, int B, int T) {
        return static_cast<real>(model_.forward_backward(x, y, B, T));
    }

    real forward_loss(const int* x, const int* y, int B, int T) {
        return static_cast<real>(model_.forward_loss(x, y, B, T));
    }

    void set_training(bool training) { model_.set_training(training); }
    void adamw_step(real lr) { model_.adamw_step(static_cast<float>(lr)); }
    void sync() { model_.sync(); }
    std::string describe() const { return model_.describe(); }

private:
    gpt_cuda::Model model_;
};

}  // namespace

int main() {
    // A failed CHECK, including a failed CUDA call, is reported here and ends
    // the program. Without this, it would reach terminate().
    try {
        printf("gpt_cuda: the model as CUDA kernels\n");

        Data d;
        if (!bench::load_data(d)) {
            return 1;
        }

        const auto make = [](const Config& cfg, int vocab) {
            return std::make_unique<CudaBackend>(cfg, vocab);
        };
        {
            const Config probe_cfg{ kEmbed, kHeads, kLayers, kBlock, kBatch,
                kDropout, kLr };
            printf("  %s\n", make(probe_cfg, d.vocab)->describe().c_str());
        }
        bench::run_sections<CudaBackend>(d, "CUDA", make);
    }
    catch (const std::exception& e) {
        std::cerr << "\n*** " << e.what() << "\n";
        return 1;
    }
    return 0;
}
