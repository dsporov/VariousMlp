// gpt_blocked.cpp - the walkthrough's model with its multiplies cache-blocked
// and vectorised.
//
// This is not a second model. It is the same Gpt class that gpt.cpp trains,
// with gpt::active_kernels() set to kBlocked around every call. So the Linear
// layers and the products inside attention use gpt_gemm.h instead of the
// loops written next to them. The weights, the layer norms, the softmaxes and
// every backward pass are shared, and that is the point: any difference
// between the two columns below comes from the multiply and nothing else.
//
// The bench in gpt_bench.h does the rest. Section 1 shows whether the blocked
// multiply is correct, section 2 whether it trains the same model, and
// section 3 how much faster it is at a width where blocking makes a
// difference. At the walkthrough's 32 channels, expect little gain: a 32-wide
// multiply already fits in L1, and packing costs about as much as it saves.
//
// CMake only, because it has its own main. Build the gpt_blocked target.

#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <random>

#include "gpt_bench.h"
#include "gpt_model.h"

int main() {
    using namespace gpt;

    // The whole program is one long demonstration. So a failed CHECK is
    // reported here and ends the program. Without this, it would reach
    // terminate().
    try {
        printf("gpt_blocked: the reference model with Kernels::kBlocked\n");
        printf("  micro-kernel: %s\n\n", blocked::kernel_name<real>());

        Data d;
        if (!bench::load_data(d)) {
            return 1;
        }

        // load() replaces the candidate's own initial weights, so the
        // generator that creates them does not matter.
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
