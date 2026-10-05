// GptScaled's own copy of makemore/gpt_cuda.h, taken at commit 512eaaa and
// otherwise unchanged, so that the two diff cleanly. gpt_scaled compiles only
// the files in this directory, which keeps it independent of makemore/: a
// change on either side stays on that side. Where the comments below mention
// other files, they mean the ones in makemore/.
//
// gpt_cuda.h - the host side of the model in gpt_cuda.cu.
//
// Nothing here mentions CUDA. The model's device buffers and kernels all stay
// inside gpt_cuda.cu behind an opaque Impl, so gpt_cuda_main.cpp -
// and gpt_model.h, which it includes - are compiled by the ordinary host
// compiler and never pass through nvcc. The price is that this interface
// speaks plain floats rather than gpt::real, and gpt_cuda_main.cpp converts.
//
// Float only. The GTX 1050 Ti this was written for runs double at 1/32 of its
// float rate, and a double build of the reference still checks it to within
// float rounding, which is all a float model can be held to anyway.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace gpt_cuda {

// What gpt::Config says about the shape of the weights, plus the vocabulary
// and the dropout rate. Batch size is not in it: every call says how big its
// batch is, and the activation buffers grow to fit the largest seen so far.
struct Shape {
    int n_embd;
    int n_head;
    int n_layer;
    int block;
    int vocab;
    float dropout;
};

class Model {
public:
    explicit Model(const Shape& shape);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // The size of every tensor, in the order gpt::Gpt collects its parameters.
    const std::vector<size_t>& tensor_sizes() const;

    // Weights in, gradients out, in that same order.
    void load(const std::vector<std::vector<float>>& weights);
    void read_grads(std::vector<std::vector<float>>& out) const;

    void zero_grad();
    float forward_backward(const int* x, const int* y, int B, int T);
    float forward_loss(const int* x, const int* y, int B, int T);
    void set_training(bool training);
    void adamw_step(float lr);

    // Kernel launches return before the kernels finish; this waits for them.
    void sync();
    std::string describe() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gpt_cuda
