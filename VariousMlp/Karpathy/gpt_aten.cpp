// gpt_aten.cpp - the model on ATen, PyTorch's tensor library, with the backward
// passes still written by hand.
//
// ATen is the layer of PyTorch underneath torch.nn and autograd: one tensor
// type and a couple of thousand operations on it, each dispatched to whatever
// device the tensor lives on. Written against it, the model reads almost line
// for line like the video's Python - q.matmul(k.transpose(-2, -1)),
// masked_fill, softmax - and runs on the CPU or, with a CUDA build of libtorch,
// on the GPU, without a line of it changing.
//
// What it deliberately leaves alone is autograd. Every backward pass is
// written out below exactly as gpt_model.h writes it, only as tensor
// expressions instead of loops, so gpt_bench.h can check it against the
// reference like any other candidate. Then section 4 turns autograd on, runs
// the very same forward pass through it, and holds the hand-written gradients
// up against what PyTorch derives by itself: a third opinion on every
// derivative in the model, from something that never saw either of the other
// two.
//
// Needs libtorch, the C++ download from pytorch.org. Configure CMake with
// -DCMAKE_PREFIX_PATH=<unpacked libtorch> and the gpt_aten target appears.
// `gpt_aten cuda` runs on the GPU when libtorch was built with CUDA and
// supports the card; recent CUDA builds of libtorch have been dropping older
// cards such as the Pascal-generation GTX 1050 Ti, so check the release notes
// before downloading one for it. The CPU build always works.
//
// CMake builds this without OpenMP on purpose. libtorch brings its own OpenMP
// runtime, and a second one in the same process - MSVC's libomp140 next to
// its libiomp5, or clang's libomp next to its libgomp - either refuses to
// start or fights it for the cores. So the reference in this program runs on
// one thread, and its timings say so.

#include <torch/torch.h>

// c10's logging header defines a glog-style one-argument CHECK. Everything
// below uses the two-argument one from check.h, which gpt_bench.h is about to
// define in its place.
#ifdef CHECK
#undef CHECK
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "check.h"
#include "gpt_bench.h"
#include "gpt_model.h"

using namespace gpt;

namespace {

constexpr at::ScalarType kReal =
    std::is_same<real, float>::value ? at::kFloat : at::kDouble;

// nn.LayerNorm's default, and gpt_model.h's.
constexpr double kLnEps = 1e-5;

// gpt::AdamW's defaults, which are torch.optim.AdamW's.
constexpr double kBeta1 = 0.9;
constexpr double kBeta2 = 0.999;
constexpr double kAdamEps = 1e-8;
constexpr double kWeightDecay = 0.01;

// Where each of a block's tensors sits in the parameter list. The order is the
// one Gpt's constructor collects them in - it is what makes load() and
// read_grads() line up with the reference - and load() checks it by size.
struct BlockSlots {
    int ln1_g, ln1_b;
    int wk, wq, wv, wp, bp;
    int ln2_g, ln2_b;
    int w1, b1, w2, b2;
};

// What a layer norm keeps for its backward pass.
struct LnSaved {
    at::Tensor xhat;
    at::Tensor invstd;
};

// What a block keeps between its forward and backward passes: the same list
// gpt_model.h's layers keep, gathered in one place. q, k, v, p and pd are
// (B, heads, T, *); everything else is (B, T, C) or (B, T, 4C). A mask is left
// undefined when dropout did not run.
struct BlockSaved {
    at::Tensor h1, q, k, v, p, pd, attn_mask, o, resid_mask;
    at::Tensor h2, r, ffwd_mask;
    LnSaved ln1, ln2;
};

class AtenGpt {
public:
    AtenGpt(const Config& cfg, int vocab, at::Device device)
        : cfg_(cfg), vocab_(vocab), hs_(cfg.n_embd / cfg.n_head), device_(device)
    {
        CHECK(cfg.n_head > 0 && cfg.n_embd % cfg.n_head == 0,
            cfg.n_embd << " channels do not divide into " << cfg.n_head << " heads");
        const int64_t C = cfg.n_embd;
        const int64_t V = vocab;
        wte_ = add({ V, C });
        wpe_ = add({ cfg.block, C });
        for (int i = 0; i < cfg.n_layer; ++i) {
            BlockSlots s;
            s.ln1_g = add({ C });
            s.ln1_b = add({ C });
            s.wk = add({ C, C });
            s.wq = add({ C, C });
            s.wv = add({ C, C });
            s.wp = add({ C, C });
            s.bp = add({ C });
            s.ln2_g = add({ C });
            s.ln2_b = add({ C });
            s.w1 = add({ C, 4 * C });
            s.b1 = add({ 4 * C });
            s.w2 = add({ 4 * C, C });
            s.b2 = add({ C });
            blocks_.push_back(s);
        }
        lnf_g_ = add({ C });
        lnf_b_ = add({ C });
        wlm_ = add({ C, V });
        blm_ = add({ V });
        saved_.resize(blocks_.size());
    }

    void load(const std::vector<Param>& weights) {
        CHECK(weights.size() == p_.size(), weights.size()
            << " tensors for a model with " << p_.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            CHECK(static_cast<int64_t>(weights[i].data->size()) == p_[i].numel(),
                weights[i].name << " has " << weights[i].data->size()
                << " entries, this model's tensor " << p_[i].numel());
            // from_blob wraps the reference's buffer without copying it;
            // copy_ then moves the numbers onto the device and into storage
            // this model owns.
            p_[i].copy_(at::from_blob(weights[i].data->data(), p_[i].sizes(),
                at::TensorOptions().dtype(kReal)));
            m_[i].zero_();
            v_[i].zero_();
        }
        step_ = 0;
    }

    void read_grads(std::vector<std::vector<real>>& out) {
        out.resize(g_.size());
        for (size_t i = 0; i < g_.size(); ++i) {
            out[i] = to_host(g_[i]);
        }
    }

    void zero_grad() {
        for (at::Tensor& g : g_) {
            g.zero_();
        }
    }

    real forward_backward(const int* x, const int* y, int B, int T) {
        torch::NoGradGuard no_grad;
        const at::Tensor loss = forward(x, y, B, T);
        backward();
        return static_cast<real>(loss.item<double>());
    }

    real forward_loss(const int* x, const int* y, int B, int T) {
        torch::NoGradGuard no_grad;
        return static_cast<real>(forward(x, y, B, T).item<double>());
    }

    void set_training(bool training) { training_ = training; }

    // Python: torch.optim.AdamW(model.parameters(), lr=learning_rate).step()
    void adamw_step(real lr) {
        torch::NoGradGuard no_grad;
        ++step_;
        const double bc1 = 1.0 - std::pow(kBeta1, step_);
        const double bc2 = 1.0 - std::pow(kBeta2, step_);
        const double rate = static_cast<double>(lr);
        for (size_t i = 0; i < p_.size(); ++i) {
            m_[i].mul_(kBeta1).add_(g_[i], 1.0 - kBeta1);
            v_[i].mul_(kBeta2).addcmul_(g_[i], g_[i], 1.0 - kBeta2);
            p_[i].mul_(1.0 - rate * kWeightDecay);
            const at::Tensor denom = (v_[i] / bc2).sqrt_().add_(kAdamEps);
            p_[i].addcdiv_(m_[i], denom, -rate / bc1);
        }
    }

    void sync() {
        if (device_.is_cuda()) {
            torch::cuda::synchronize();
        }
    }

    std::string describe() const {
        std::string s = "ATen on " + device_.str();
        if (device_.is_cpu()) {
            s += ", " + std::to_string(at::get_num_threads()) + " threads";
        }
        return s;
    }

    // Section 4's opinion: the same forward pass with autograd recording it,
    // and autograd's gradients for every parameter, in the usual order.
    std::vector<std::vector<real>> autograd_grads(const int* x, const int* y,
        int B, int T) {
        for (at::Tensor& p : p_) {
            p.requires_grad_(true);
        }
        forward(x, y, B, T).backward();

        std::vector<std::vector<real>> out(p_.size());
        for (size_t i = 0; i < p_.size(); ++i) {
            out[i] = to_host(p_[i].grad());
            p_[i].mutable_grad() = at::Tensor();
            p_[i].requires_grad_(false);
        }
        return out;
    }

private:
    int add(std::vector<int64_t> shape) {
        const at::TensorOptions opts = at::TensorOptions().dtype(kReal).device(device_);
        p_.push_back(at::zeros(shape, opts));
        g_.push_back(at::zeros(shape, opts));
        m_.push_back(at::zeros(shape, opts));
        v_.push_back(at::zeros(shape, opts));
        return static_cast<int>(p_.size()) - 1;
    }

    static std::vector<real> to_host(const at::Tensor& t) {
        const at::Tensor cpu = t.to(at::kCPU).contiguous();
        const real* src = cpu.data_ptr<real>();
        return std::vector<real>(src, src + cpu.numel());
    }

    at::Tensor tokens(const int* ids, int B, int T) const {
        const at::Tensor host = at::from_blob(const_cast<int*>(ids),
            { static_cast<int64_t>(B), static_cast<int64_t>(T) }, at::kInt);
        return host.to(device_, at::kLong);
    }

    double scale() const {
        return 1.0 / std::sqrt(static_cast<double>(hs_));
    }

    // (B, T, C) -> (B, heads, T, hs): head h is channels [h*hs, (h+1)*hs), the
    // same split gpt_model.h's CausalSelfAttention makes.
    at::Tensor split_heads(const at::Tensor& t) const {
        return t.reshape({ t.size(0), t.size(1), static_cast<int64_t>(cfg_.n_head),
            static_cast<int64_t>(hs_) }).transpose(1, 2);
    }

    at::Tensor merge_heads(const at::Tensor& t) const {
        return t.transpose(1, 2).reshape({ t.size(0), t.size(2),
            static_cast<int64_t>(cfg_.n_embd) });
    }

    // True above the diagonal: the future a position may not attend to.
    const at::Tensor& future(int64_t T) {
        if (!future_.defined() || future_.size(0) != T) {
            future_ = at::ones({ T, T }, at::TensorOptions().dtype(at::kBool).device(device_))
                .triu(1);
        }
        return future_;
    }

    // Inverted dropout, as gpt_model.h's Dropout: the survivors are scaled up
    // by 1/(1-p) so that evaluation needs no correction. The mask carries the
    // scale, so the backward pass is one multiply.
    at::Tensor dropout(const at::Tensor& x, at::Tensor& mask) {
        if (!training_ || cfg_.dropout <= real(0)) {
            mask = at::Tensor();
            return x;
        }
        const double keep = 1.0 - static_cast<double>(cfg_.dropout);
        mask = (at::rand_like(x) < keep).to(kReal) / keep;
        return x * mask;
    }

    static at::Tensor undrop(const at::Tensor& d, const at::Tensor& mask) {
        return mask.defined() ? d * mask : d;
    }

    // Python: nn.LayerNorm(n_embd), over the last dimension.
    at::Tensor layer_norm(const at::Tensor& x, int g, int b, LnSaved& s) {
        const at::Tensor xc = x - x.mean({ -1 }, true);
        s.invstd = at::rsqrt((xc * xc).mean({ -1 }, true) + kLnEps);
        s.xhat = xc * s.invstd;
        return s.xhat * p_[static_cast<size_t>(g)] + p_[static_cast<size_t>(b)];
    }

    // gpt_model.h's formula, a row at a time over the channels:
    //   dx = invstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
    at::Tensor layer_norm_backward(const at::Tensor& dout, int g, int b,
        const LnSaved& s) {
        g_[static_cast<size_t>(g)].add_((dout * s.xhat).sum({ 0, 1 }));
        g_[static_cast<size_t>(b)].add_(dout.sum({ 0, 1 }));
        const at::Tensor dxhat = dout * p_[static_cast<size_t>(g)];
        return s.invstd * (dxhat - dxhat.mean({ -1 }, true)
            - s.xhat * (dxhat * s.xhat).mean({ -1 }, true));
    }

    // out = x W (+ b) over every row: dW += x^T dout, db the column sums,
    // dx = dout W^T. b < 0 for the three attention projections, which have no
    // bias.
    at::Tensor linear_backward(const at::Tensor& x, const at::Tensor& dout, int w,
        int b) {
        const at::Tensor x2 = x.reshape({ -1, x.size(-1) });
        const at::Tensor d2 = dout.reshape({ -1, dout.size(-1) });
        g_[static_cast<size_t>(w)].add_(x2.t().mm(d2));
        if (b >= 0) {
            g_[static_cast<size_t>(b)].add_(d2.sum({ 0 }));
        }
        return dout.matmul(p_[static_cast<size_t>(w)].t());
    }

    // Python:
    //   x = x + self.sa(self.ln1(x))
    //   x = x + self.ffwd(self.ln2(x))
    at::Tensor block_forward(const BlockSlots& s, BlockSaved& sv, const at::Tensor& x) {
        sv.h1 = layer_norm(x, s.ln1_g, s.ln1_b, sv.ln1);
        sv.q = split_heads(sv.h1.matmul(p_[static_cast<size_t>(s.wq)]));
        sv.k = split_heads(sv.h1.matmul(p_[static_cast<size_t>(s.wk)]));
        sv.v = split_heads(sv.h1.matmul(p_[static_cast<size_t>(s.wv)]));

        // wei = q @ k.transpose(-2, -1) * hs**-0.5, masked, softmaxed
        const at::Tensor scores = sv.q.matmul(sv.k.transpose(-2, -1)) * scale();
        sv.p = at::softmax(scores.masked_fill(future(x.size(1)),
            -std::numeric_limits<double>::infinity()), -1);
        sv.pd = dropout(sv.p, sv.attn_mask);
        sv.o = merge_heads(sv.pd.matmul(sv.v));

        const at::Tensor attended = sv.o.matmul(p_[static_cast<size_t>(s.wp)])
            + p_[static_cast<size_t>(s.bp)];
        const at::Tensor mid = x + dropout(attended, sv.resid_mask);

        sv.h2 = layer_norm(mid, s.ln2_g, s.ln2_b, sv.ln2);
        sv.r = at::relu(sv.h2.matmul(p_[static_cast<size_t>(s.w1)])
            + p_[static_cast<size_t>(s.b1)]);
        const at::Tensor f2 = sv.r.matmul(p_[static_cast<size_t>(s.w2)])
            + p_[static_cast<size_t>(s.b2)];
        return mid + dropout(f2, sv.ffwd_mask);
    }

    // The residual adds fan out in the forward pass, so the gradient arriving
    // at a block's output reaches mid both through the feedforward branch and
    // straight along the skip, and mid's reaches x the same way.
    at::Tensor block_backward(const BlockSlots& s, const BlockSaved& sv,
        const at::Tensor& dout) {
        // out = mid + drop(relu(h2 W1 + b1) W2 + b2)
        const at::Tensor dr = linear_backward(sv.r, undrop(dout, sv.ffwd_mask),
            s.w2, s.b2);
        const at::Tensor df1 = dr * (sv.r > 0).to(kReal);
        const at::Tensor dh2 = linear_backward(sv.h2, df1, s.w1, s.b1);
        const at::Tensor dmid = dout + layer_norm_backward(dh2, s.ln2_g, s.ln2_b, sv.ln2);

        // mid = x + drop(o Wp + bp), with o = pd @ v for every head at once
        const at::Tensor dO = split_heads(linear_backward(sv.o,
            undrop(dmid, sv.resid_mask), s.wp, s.bp));
        const at::Tensor dv = sv.pd.transpose(-2, -1).matmul(dO);
        const at::Tensor dp = undrop(dO.matmul(sv.v.transpose(-2, -1)), sv.attn_mask);

        // The softmax, a row at a time: ds = p * (dp - sum(p * dp)). p is
        // exactly zero above the diagonal, so the future gets nothing.
        const at::Tensor ds = sv.p * (dp - (sv.p * dp).sum({ -1 }, true));
        const at::Tensor dq = ds.matmul(sv.k) * scale();
        const at::Tensor dk = ds.transpose(-2, -1).matmul(sv.q) * scale();

        // h1 fed all three projections, so all three gradients come back to it.
        const at::Tensor dh1 = linear_backward(sv.h1, merge_heads(dq), s.wq, -1)
            + linear_backward(sv.h1, merge_heads(dk), s.wk, -1)
            + linear_backward(sv.h1, merge_heads(dv), s.wv, -1);
        return dmid + layer_norm_backward(dh1, s.ln1_g, s.ln1_b, sv.ln1);
    }

    // Python:
    //   tok_emb = self.token_embedding_table(idx)
    //   pos_emb = self.position_embedding_table(torch.arange(T))
    //   x = self.ln_f(self.blocks(tok_emb + pos_emb))
    //   logits = self.lm_head(x)
    //   loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    //
    // Returns the loss as a tensor, so that with autograd on it can be
    // backpropagated through; everything the hand-written backward pass needs
    // is kept on the way.
    at::Tensor forward(const int* x, const int* y, int B, int T) {
        CHECK(T > 0 && T <= cfg_.block, "a window of " << T
            << " positions, but the position table has " << cfg_.block << " rows");
        const int64_t C = cfg_.n_embd;
        idx_ = tokens(x, B, T);
        at::Tensor h = p_[static_cast<size_t>(wte_)].index_select(0, idx_.reshape({ -1 }))
            .reshape({ static_cast<int64_t>(B), static_cast<int64_t>(T), C })
            + p_[static_cast<size_t>(wpe_)].slice(0, 0, T);
        for (size_t i = 0; i < blocks_.size(); ++i) {
            h = block_forward(blocks_[i], saved_[i], h);
        }
        hf_ = layer_norm(h, lnf_g_, lnf_b_, lnf_saved_);
        const at::Tensor logits = hf_.matmul(p_[static_cast<size_t>(wlm_)])
            + p_[static_cast<size_t>(blm_)];

        targets_ = tokens(y, B, T).reshape({ -1 });
        logp_ = at::log_softmax(logits.reshape({ -1, static_cast<int64_t>(vocab_) }), -1);
        return -logp_.gather(1, targets_.unsqueeze(1)).mean();
    }

    void backward() {
        const int64_t N = logp_.size(0);
        const int64_t C = cfg_.n_embd;

        // Cross entropy's gradient is softmax minus the one-hot target, over
        // the N predictions the loss averaged.
        const at::Tensor dlogits = ((logp_.exp()
            - at::one_hot(targets_, vocab_).to(kReal)) / static_cast<double>(N))
            .reshape({ hf_.size(0), hf_.size(1), static_cast<int64_t>(vocab_) });
        at::Tensor d = linear_backward(hf_, dlogits, wlm_, blm_);
        d = layer_norm_backward(d, lnf_g_, lnf_b_, lnf_saved_);
        for (size_t i = blocks_.size(); i-- > 0; ) {
            d = block_backward(blocks_[i], saved_[i], d);
        }

        // The embedding tables scatter back by index. index_add_ sums the
        // repeats - a character used eight times in the batch contributes
        // eight times - and every window uses the same position rows.
        g_[static_cast<size_t>(wte_)].index_add_(0, idx_.reshape({ -1 }),
            d.reshape({ -1, C }));
        g_[static_cast<size_t>(wpe_)].slice(0, 0, d.size(1)).add_(d.sum({ 0 }));
    }

    Config cfg_;
    int vocab_;
    int hs_;
    at::Device device_;
    bool training_ = true;
    int step_ = 0;

    std::vector<at::Tensor> p_, g_, m_, v_;
    int wte_ = 0;
    int wpe_ = 0;
    std::vector<BlockSlots> blocks_;
    int lnf_g_ = 0;
    int lnf_b_ = 0;
    int wlm_ = 0;
    int blm_ = 0;

    std::vector<BlockSaved> saved_;
    LnSaved lnf_saved_;
    at::Tensor idx_, targets_, hf_, logp_, future_;
};

void autograd_opinion(const Data& d, at::Device device) {
    std::cout << "\n== 4. a third opinion: autograd ==\n";
    std::cout <<
        "  The same forward pass, recorded by autograd this time and differentiated\n"
        "  by it, against the backward passes written out in this file - and\n"
        "  against the reference's, which never saw a tensor library at all.\n";

    struct Case {
        const char* what;
        Config cfg;
        bool jittered;
    };
    const Case cases[] = {
        { "tiny, jittered", { 8, 2, 2, 5, 3, real(0), kLr }, true },
        { "the walkthrough's model",
            { kEmbed, kHeads, kLayers, kBlock, kBatch, real(0), kLr }, false },
    };

    sanity::Expect chk;
    for (const Case& c : cases) {
        std::mt19937 g(1337);
        bench::CpuModel ref(c.cfg, d.vocab, Kernels::kScalar, g);
        if (c.jittered) {
            bench::jitter(ref.model().params(), g);
        }
        AtenGpt model(c.cfg, d.vocab, device);
        model.load(ref.model().params());

        std::vector<int> x, y;
        sample_batch(d.train, c.cfg.batch, c.cfg.block, x, y, g);
        const int B = c.cfg.batch;
        const int T = c.cfg.block;

        std::vector<std::vector<real>> by_reference, by_hand;
        ref.zero_grad();
        ref.forward_backward(x.data(), y.data(), B, T);
        ref.read_grads(by_reference);
        model.zero_grad();
        model.forward_backward(x.data(), y.data(), B, T);
        model.read_grads(by_hand);
        const std::vector<std::vector<real>> by_autograd =
            model.autograd_grads(x.data(), y.data(), B, T);

        std::string hand_where, ref_where;
        const double hand = bench::worst_error(ref.model().params(), by_hand,
            by_autograd, hand_where);
        const double reference = bench::worst_error(ref.model().params(),
            by_reference, by_autograd, ref_where);
        printf("  %s\n    this file's backward passes against autograd  %.1e (%s)\n"
            "    the reference's against autograd              %.1e (%s)\n",
            c.what, hand, hand_where.c_str(), reference, ref_where.c_str());
        chk.that((std::string(c.what) + ": hand-written ATen agrees").c_str(),
            hand < bench::kGradTolerance);
        chk.that((std::string(c.what) + ": the reference agrees").c_str(),
            reference < bench::kGradTolerance);
    }
    chk.report();
}

}  // namespace

int main(int argc, char** argv) {
    // A CHECK that fires, or an error out of ATen, is reported and ends the
    // program rather than being left to terminate().
    try {
        at::Device device(at::kCPU);
        if (argc > 1 && std::string(argv[1]) == "cuda") {
            if (torch::cuda::is_available()) {
                device = at::Device(at::kCUDA);
            }
            else {
                printf("no CUDA device this libtorch can use - staying on the CPU\n");
            }
        }
        // Section 3's dropout masks come from torch's generator rather than
        // the reference's, so they get a seed of their own.
        torch::manual_seed(1337);

        printf("gpt_aten: the model on ATen tensors, backward passes by hand\n");
        printf("  device: %s\n\n", device.str().c_str());

        Data d;
        if (!bench::load_data(d)) {
            return 1;
        }

        const auto make = [device](const Config& cfg, int vocab) {
            return std::make_unique<AtenGpt>(cfg, vocab, device);
        };
        bench::run_sections<AtenGpt>(d, "ATen", make);
        autograd_opinion(d, device);
    }
    catch (const std::exception& e) {
        std::cerr << "\n*** " << e.what() << "\n";
        return 1;
    }
    return 0;
}
