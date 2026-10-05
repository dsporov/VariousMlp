// micrograd.cpp - a C++ port of Andrej Karpathy's micrograd
// ("The spelled-out intro to neural networks and backpropagation", video 1 of
//  Neural Networks: Zero to Hero)
//
// Build:  g++ -std=c++20 -O2 micrograd.cpp -o micrograd
// Run:    ./micrograd

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "check.h"

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

struct Node;
using NodePtr = std::shared_ptr<Node>;

struct Node {
    double data = 0.0;
    double grad = 0.0;
    std::string op;               // the operation that produced this node
    std::string label;            // optional, for the graphviz dump
    std::vector<NodePtr> prev;    // children in the expression DAG
    std::function<void()> backward = [] {};

    Node(double d, std::vector<NodePtr> children = {}, std::string o = "")
        : data(d), op(std::move(o)), prev(std::move(children)) {
    }
};

// Thin handle so we get Python-ish syntax. Copying a Value shares the node,
// which is exactly the aliasing behaviour Python gives you for free.
class Value {
    NodePtr n_;

public:
    Value(double d = 0.0) : n_(std::make_shared<Node>(d)) {}
    explicit Value(NodePtr n) : n_(std::move(n)) {}

    double  data() const { return n_->data; }
    double& data() { return n_->data; }
    double  grad() const { return n_->grad; }
    double& grad() { return n_->grad; }
    const NodePtr& node() const { return n_; }

    Value& label(const std::string& s) { n_->label = s; return *this; }

    void backward();
    void zero_grad();
};

static Value make(double data, std::vector<NodePtr> children, std::string op) {
    return Value(std::make_shared<Node>(data, std::move(children), std::move(op)));
}

// --- operators -------------------------------------------------------------
// Each closure captures the children by shared_ptr and the *output* by raw
// pointer. Capturing the output by shared_ptr would create a reference cycle
// (node -> closure -> node) and leak, since C++ has no GC to break it.

Value operator+(const Value& a, const Value& b) {
    Value out = make(a.data() + b.data(), { a.node(), b.node() }, "+");
    Node* o = out.node().get();
    NodePtr x = a.node(), y = b.node();
    out.node()->backward = [o, x, y] {
        x->grad += o->grad;   // += , not = : a node used twice accumulates
        y->grad += o->grad;
    };
    return out;
}

Value operator*(const Value& a, const Value& b) {
    Value out = make(a.data() * b.data(), { a.node(), b.node() }, "*");
    Node* o = out.node().get();
    NodePtr x = a.node(), y = b.node();
    out.node()->backward = [o, x, y] {
        x->grad += y->data * o->grad;
        y->grad += x->data * o->grad;
    };
    return out;
}

Value pow(const Value& a, double k) {
    // Both of these hand back inf/NaN silently, and one NaN poisons every
    // gradient downstream of it. Much easier to catch at the source than to
    // work backwards from a loss that printed "nan" 40 steps later.
    CHECK(!(a.data() == 0.0 && k < 0.0), "0 raised to " << k << " is not finite");
    CHECK(!(a.data() < 0.0 && k != std::floor(k)),
        "pow(" << a.data() << ", " << k << ") is not real: negative base, "
        "fractional exponent");

    Value out = make(std::pow(a.data(), k), { a.node() }, "^" + std::to_string(k));
    Node* o = out.node().get();
    NodePtr x = a.node();
    out.node()->backward = [o, x, k] {
        x->grad += k * std::pow(x->data, k - 1) * o->grad;
    };
    return out;
}

Value tanh(const Value& a) {
    double t = std::tanh(a.data());
    Value out = make(t, { a.node() }, "tanh");
    Node* o = out.node().get();
    NodePtr x = a.node();
    out.node()->backward = [o, x, t] { x->grad += (1 - t * t) * o->grad; };
    return out;
}

Value exp(const Value& a) {
    double e = std::exp(a.data());
    Value out = make(e, { a.node() }, "exp");
    Node* o = out.node().get();
    NodePtr x = a.node();
    out.node()->backward = [o, x, e] { x->grad += e * o->grad; };
    return out;
}

Value relu(const Value& a) {
    double r = a.data() < 0 ? 0.0 : a.data();
    Value out = make(r, { a.node() }, "relu");
    Node* o = out.node().get();
    NodePtr x = a.node();
    out.node()->backward = [o, x] { x->grad += (x->data > 0) * o->grad; };
    return out;
}

// Sugar. Python gets these from __neg__/__sub__/__truediv__/__radd__/__rmul__;
// in C++ free functions cover both argument orders at once.
Value operator-(const Value& a) { return a * Value(-1.0); }
Value operator-(const Value& a, const Value& b) { return a + (-b); }
Value operator/(const Value& a, const Value& b) {
    CHECK(b.data() != 0.0, "division by zero: " << a.data() << " / 0");
    return a * pow(b, -1.0);
}

// --- backward pass ---------------------------------------------------------

void Value::backward() {
    std::vector<Node*> topo;
    std::unordered_set<Node*> visited;
    std::function<void(Node*)> build = [&](Node* v) {
        if (!visited.insert(v).second) return;
        for (const auto& c : v->prev) build(c.get());
        topo.push_back(v);   // pushed only after all children -> topological
    };
    build(n_.get());

    n_->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) (*it)->backward();
}

void Value::zero_grad() {
    std::unordered_set<Node*> visited;
    std::function<void(Node*)> walk = [&](Node* v) {
        if (!visited.insert(v).second) return;
        v->grad = 0.0;
        for (const auto& c : v->prev) walk(c.get());
    };
    walk(n_.get());
}

// --- graphviz dump (the draw_dot() from the video) -------------------------

std::string to_dot(const Value& root) {
    std::ostringstream os;
    os << "digraph G {\n  rankdir=LR;\n";
    std::unordered_set<Node*> visited;
    std::function<void(Node*)> walk = [&](Node* v) {
        if (!visited.insert(v).second) return;
        os << "  \"" << v << "\" [shape=record,label=\"{" << v->label
            << " | data " << v->data << " | grad " << v->grad << "}\"];\n";
        if (!v->op.empty()) {
            os << "  \"" << v << v->op << "\" [label=\"" << v->op << "\"];\n";
            os << "  \"" << v << v->op << "\" -> \"" << v << "\";\n";
        }
        for (const auto& c : v->prev) {
            walk(c.get());
            os << "  \"" << c.get() << "\" -> \"" << v << v->op << "\";\n";
        }
    };
    walk(root.node().get());
    os << "}\n";
    return os.str();
}

// ---------------------------------------------------------------------------
// The neural net library
// ---------------------------------------------------------------------------

static std::mt19937 rng(1337);
static std::uniform_real_distribution<double> unif(-1.0, 1.0);

struct Neuron {
    std::vector<Value> w_;
    Value b_;

    explicit Neuron(int inputs)
        : b_(unif(rng))
    {
        CHECK(inputs > 0, "a neuron needs at least one input, got " << inputs);
        for (int i = 0; i < inputs; ++i)
            w_.emplace_back(unif(rng));
    }

    Value operator()(const std::vector<Value>& x) const {
        // Without this a short x reads past the end of the vector: not a wrong
        // answer, undefined behaviour.
        CHECK(x.size() == w_.size(),
            "this neuron takes " << w_.size() << " inputs, got " << x.size());

        Value act = b_;
        for (size_t i = 0; i < w_.size(); ++i)
            act = act + w_[i] * x[i];
        return tanh(act);
    }

    std::vector<Value> parameters() const {
        std::vector<Value> p = w_;
        p.push_back(b_);
        return p;
    }
};

struct Layer {
    std::vector<Neuron> neurons_;

    Layer(int inputs, int outputs) {
        // outputs = neurons count
        CHECK(outputs > 0, "a layer needs at least one neuron, got " << outputs);
        for (int i = 0; i < outputs; ++i)
            neurons_.emplace_back(inputs);
    }

    std::vector<Value> operator()(const std::vector<Value>& x) const {
        std::vector<Value> out;
        for (const auto& n : neurons_)
            out.push_back(n(x));
        return out;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> p;
        for (const auto& n : neurons_) {
            auto q = n.parameters();
            p.insert(p.end(), q.begin(), q.end());
        }
        return p;
    }
};

struct MLP {
    std::vector<Layer> layers_;
    int inputs_;

    MLP(int inputs, const std::vector<int>& outputs) : inputs_(inputs) {
        CHECK(!outputs.empty(), "an MLP needs at least one layer");
        int prev_inputs = inputs;
        for (int n : outputs) {
            layers_.emplace_back(prev_inputs, n);
            prev_inputs = n;
        }
    }

    // Width of the last layer, i.e. how many Values operator() hands back.
    size_t outputs() const { return layers_.back().neurons_.size(); }

    std::vector<Value> operator()(std::vector<Value> x) const {
        CHECK(static_cast<int>(x.size()) == inputs_,
            "this MLP takes " << inputs_ << " inputs, got " << x.size());
        for (const auto& l : layers_)
            x = l(x);
        return x;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> p;
        for (const auto& l : layers_) {
            auto q = l.parameters();
            p.insert(p.end(), q.begin(), q.end());
        }
        return p;
    }
};

// ---------------------------------------------------------------------------
// The three things the video actually walks through, plus a gradient check
// standing in for the video's "and here is what PyTorch says"
// ---------------------------------------------------------------------------

void example_manual_backprop() {
    std::cout << "== 1. L = (a*b + c) * f ==\n";
    Value a(2.0), b(-3.0), c(10.0), f(-2.0);
    a.label("a"); b.label("b"); c.label("c"); f.label("f");
    Value e = a * b;  e.label("e");
    Value d = e + c;  d.label("d");
    Value L = d * f;  L.label("L");
    L.backward();
    printf("  L = %.1f\n", L.data());
    printf("  dL/da=%.1f  dL/db=%.1f  dL/dc=%.1f  dL/df=%.1f  dL/de=%.1f\n",
        a.grad(), b.grad(), c.grad(), f.grad(), e.grad());

    // The values the video derives by hand, checked rather than eyeballed.
    // Every number here is exact in binary floating point, so the tolerance is
    // only there to keep the comparison from being a bare ==.
    sanity::Expect chk;
    chk.close("L", L.data(), -8.0, 1e-12);
    chk.close("dL/da", a.grad(), 6.0, 1e-12);
    chk.close("dL/db", b.grad(), -4.0, 1e-12);
    chk.close("dL/dc", c.grad(), -2.0, 1e-12);
    chk.close("dL/df", f.grad(), 4.0, 1e-12);
    chk.close("dL/de", e.grad(), -2.0, 1e-12);
    chk.report();
    std::cout << "\n";
}

void example_neuron() {
    std::cout << "== 2. single tanh neuron ==\n";
    Value x1(2.0), x2(0.0), w1(-3.0), w2(1.0), b(6.8813735870195432);
    Value n = x1 * w1 + x2 * w2 + b;
    Value o = tanh(n);
    o.backward();
    printf("  o = %.4f\n", o.data());
    printf("  dx1=%.4f dw1=%.4f dx2=%.4f dw2=%.4f\n",
        x1.grad(), w1.grad(), x2.grad(), w2.grad());

    // b is picked in the video so that o comes out at exactly 1/sqrt(2), which
    // makes 1 - o^2 = 1/2 and every gradient below a round number.
    sanity::Expect chk;
    chk.close("o", o.data(), std::sqrt(0.5), 1e-12);
    chk.close("dx1", x1.grad(), -1.5, 1e-12);
    chk.close("dw1", w1.grad(), 1.0, 1e-12);
    chk.close("dx2", x2.grad(), 0.5, 1e-12);
    chk.close("dw2", w2.grad(), 0.0, 1e-12);
    chk.report();
    std::cout << "\n";
}

// The video cross-checks micrograd against PyTorch. There is no torch here, so
// central differences play that role: for every leaf, compare the analytic
// gradient with (f(x+h) - f(x-h)) / 2h. This is the one check that would catch
// a wrong derivative in any of the operators above - and it is worth rerunning
// whenever a new operator is added.
void example_gradcheck() {
    std::cout << "== 3. gradients vs. finite differences ==\n";

    // One expression touching every operator, with x0 appearing twice so the
    // accumulating (+=) path in the backward closures is exercised too. The
    // point chosen keeps relu off its kink, where the derivative is undefined
    // and finite differences would disagree for a legitimate reason.
    const std::vector<double> x0 = { 0.7, -1.3, 0.4, 2.0 };

    auto leaves = [](const std::vector<double>& xs) {
        std::vector<Value> v;
        for (double d : xs) v.emplace_back(d);
        return v;
    };
    auto forward = [](const std::vector<Value>& v) {
        return tanh(v[0] * v[1] + v[2] / v[3]) * relu(v[0] - v[2])
            + exp(v[1] * 0.5) - pow(v[3], 3.0);
    };

    std::vector<Value> v = leaves(x0);
    Value out = forward(v);
    out.backward();
    printf("  f(x) = %.8f\n", out.data());

    const double h = 1e-6;
    sanity::Expect chk;
    for (size_t i = 0; i < x0.size(); ++i) {
        // Each perturbed evaluation needs a fresh graph: the nodes cache their
        // data, so poking a leaf does not re-run the ops above it.
        std::vector<double> hi = x0, lo = x0;
        hi[i] += h;
        lo[i] -= h;
        const double numeric = (forward(leaves(hi)).data() - forward(leaves(lo)).data()) / (2 * h);
        const double analytic = v[i].grad();

        char label[32];
        snprintf(label, sizeof label, "df/dx%zu", i);
        printf("  %s  analytic %+.8f   numeric %+.8f\n", label, analytic, numeric);
        // Central differences carry ~1e-9 of roundoff at this h, so the
        // tolerance is loose in absolute terms and relative for big gradients.
        chk.close(label, analytic, numeric, 1e-5 * std::max(1.0, std::fabs(analytic)));
    }
    chk.report();
    std::cout << "\n";
}

void example_training() {
    std::cout << "== 4. training a 3-layer MLP ==\n";
    constexpr int kInputs = 3;
    constexpr int kSteps = 100;
    MLP net(kInputs, { 4, 4, 1 });
    std::cout << "  parameters: " << net.parameters().size() << "\n";

    std::vector<std::vector<Value>> xs = {
        {Value(2.0),  Value(3.0),  Value(-1.0)},
        {Value(3.0),  Value(-1.0), Value(0.5)},
        {Value(0.5),  Value(1.0),  Value(1.0)},
        {Value(1.0),  Value(1.0),  Value(-1.0)},
    };
    std::vector<double> ys = { 1.0, -1.0, -1.0, 1.0 };

    // The mistake these catch: adding a row to xs and forgetting its target.
    // The loss loops over ys, so an extra input row would train on nothing and
    // never say so, and an extra target would read past the end of ypred.
    CHECK(!xs.empty(), "no training examples");
    CHECK(xs.size() == ys.size(),
        xs.size() << " input rows in xs but " << ys.size()
        << " targets in ys - one target per row");
    for (size_t i = 0; i < xs.size(); ++i)
        CHECK(xs[i].size() == kInputs,
            "xs[" << i << "] has " << xs[i].size() << " features, this net takes " << kInputs);
    CHECK(net.outputs() == 1, "the squared-error loss below wants one output per "
        "example, the net's last layer has " << net.outputs());

    double first_loss = 0.0, last_loss = 0.0;

    for (int step = 0; step < kSteps; ++step) {
        // forward: rebuild the graph each step, exactly like the Python version
        std::vector<Value> ypred;
        for (const auto& x : xs) {
            auto out = net(x);
            ypred.push_back(out[0]);
        }

        Value loss(0.0);
        for (size_t i = 0; i < ys.size(); ++i)
            loss = loss + pow(ypred[i] - Value(ys[i]), 2.0);

        // backward - zero first, or gradients from the last step leak in
        loss.zero_grad();
        loss.backward();

        // update
        for (auto& p : net.parameters())
            p.data() += -0.05 * p.grad();

        if (step == 0) first_loss = loss.data();
        last_loss = loss.data();

        if (step % 20 == 0 || step == kSteps - 1)
            printf("  step %3d  loss %.8f\n", step, loss.data());
    }

    std::vector<Value> final_pred;
    for (const auto& x : xs) final_pred.push_back(net(x)[0]);
    std::cout << "  predictions: ";
    for (const auto& p : final_pred) printf("%.4f ", p.data());
    std::cout << "\n  targets:     ";
    for (double y : ys) printf("%.4f ", y);
    std::cout << "\n";

    // Did the thing actually learn? Cheap to ask, and it is the check that
    // fails first if a gradient, the sign of the update, or the learning rate
    // is wrong - all of which still print a plausible-looking loss curve.
    // 100 steps at lr 0.05 lands around loss 0.0066 and predictions ~0.96; the
    // tolerances are the nearest round numbers above that, loose enough to
    // survive a compiler's floating-point reordering and tight enough that a
    // net which stopped learning cannot slip through.
    sanity::Expect chk;
    chk.that("loss went down", last_loss < first_loss);
    chk.close("final loss", last_loss, 0.0, 0.02);
    for (size_t i = 0; i < ys.size(); ++i)
        chk.close("prediction", final_pred[i].data(), ys[i], 0.1);
    chk.report();
    std::cout << "\n";
}

void test_micrograd() {
    example_manual_backprop();
    example_neuron();
    example_gradcheck();
    example_training();

    // dump a small graph for graphviz:  ./micrograd > out && dot -Tsvg ...
    Value a(2.0), b(-3.0), c(10.0);
    a.label("a"); b.label("b"); c.label("c");
    Value L = a * b + c;
    L.label("L");
    L.backward();
    std::cout << "== graphviz (pipe into `dot -Tsvg`) ==\n" << to_dot(L);
}
