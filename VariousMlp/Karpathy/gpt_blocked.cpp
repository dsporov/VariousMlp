// gpt_blocked.cpp - the walkthrough's model with its multiplies cache-blocked
// and vectorised.
//
// Not a second model. This is the same Gpt class gpt.cpp trains, with
// gpt::active_kernels() switched to kBlocked around every call, so that the
// Linear layers and the products inside attention go through gpt_gemm.h
// instead of the loops written beside them. The weights, the layer norms, the
// softmaxes and every backward pass are shared, which is the point: whatever
// differs between the two columns below is the multiply and nothing else.
//
// The bench in gpt_bench.h does the rest. Section 1 says whether the blocked
// multiply is right, section 2 whether it trains the same model, and section 3
// what it is worth at a width where there is finally something to block. At
// the walkthrough's 32 channels expect little from it - a 32-wide multiply fits
// in L1 already, and packing it costs about what it saves.
//
// CMake-only, because it has its own main. Build the gpt_blocked target.

#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <random>

#include "gpt_bench.h"
#include "gpt_model.h"

int main() {
    using namespace gpt;

    // The whole program is one long demonstration, so a CHECK that fires is
    // reported and ends it rather than being left to terminate().
    try {
        printf("gpt_blocked: the reference model with Kernels::kBlocked\n");
        printf("  micro-kernel: %s\n\n", blocked::kernel_name<real>());

        Data d;
        if (!bench::load_data(d)) {
            return 1;
        }

        // The candidate's own initial weights are thrown away by load(), so
        // the generator it draws them from does not matter.
        const auto make = [](const Config& cfg, int vocab) {
            std::mt19937 unused(0);
            return std::make_unique<bench::CpuModel>(cfg, vocab, Kernels::kBlocked,
                unused);
        };
        bench::run_sections<bench::CpuModel>(d, "blocked", make);
    }
    catch (const std::exception& e) {
        std::cerr << "\n*** " << e.what() << "\n";
        return 1;
    }
    return 0;
}
