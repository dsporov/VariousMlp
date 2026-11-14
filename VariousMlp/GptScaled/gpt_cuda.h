// GptScaled's own copy of makemore/gpt_cuda.h, copied at commit 512eaaa. Later changes went into both copies, except for
// score_scale in Shape. That field exists only here, because only this directory's gpt_model.h can run the attention scale of the
// code that the video runs. gpt_scaled compiles only the files in this directory, which keeps it independent of makemore/: a change
// to one copy does not reach the other. Where the comments below mention other files, they mean the files in makemore/.
//
// gpt_cuda.h - the host side of the model in gpt_cuda.cu.
//
// Nothing here mentions CUDA. The model's device buffers and kernels all stay
// inside gpt_cuda.cu, behind an opaque Impl. So gpt_cuda_main.cpp, and the
// gpt_model.h that it includes, are compiled by the normal host compiler and
// never by nvcc. The cost is that this interface uses plain floats instead of
// gpt::real, so gpt_cuda_main.cpp has to convert.
//
// Float only. The GTX 1050 Ti that this was written for runs double at 1/32 of
// its float speed. A double build of the reference can still check this model
// to within float rounding, and a float model cannot be more exact than that
// anyway.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace gpt_cuda {

// The parts of gpt::Config that define the shape of the weights, plus the vocabulary size, the dropout rate and the scale factor
// for attention's scores. The batch size is not here: every call passes its own batch size, and the activation buffers grow to fit
// the largest batch so far.
//
// The scale factor is the number itself, not gpt::ScoreScale, because this file does not include gpt_model.h. Get it from
// gpt::score_scale_factor, so that both models multiply by the same float.
struct Shape {
    int n_embd;
    int n_head;
    int n_layer;
    int block;
    int vocab;
    float dropout;
    float score_scale;
};

class Model {
public:
    explicit Model(const Shape& shape);
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // The size of every tensor, in the order gpt::Gpt collects its parameters.
    const std::vector<size_t>& tensor_sizes() const;

    // Weights are copied in, and gradients copied out, in that same order.
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
