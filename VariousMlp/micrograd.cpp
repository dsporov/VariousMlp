// micrograd.cpp — a C++ port of Andrej Karpathy's micrograd
// ("The spelled-out intro to neural networks and backpropagation", video 1 of
//  Neural Networks: Zero to Hero)
//
// Build:  g++ -std=c++17 -O2 micrograd.cpp -o micrograd
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
Value operator/(const Value& a, const Value& b) { return a * pow(b, -1.0); }

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
    std::vector<Value> w;
    Value b;

    explicit Neuron(int nin) : b(unif(rng)) {
        for (int i = 0; i < nin; ++i) w.emplace_back(unif(rng));
    }

    Value operator()(const std::vector<Value>& x) const {
        Value act = b;
        for (size_t i = 0; i < w.size(); ++i) act = act + w[i] * x[i];
        return tanh(act);
    }

    std::vector<Value> parameters() const {
        std::vector<Value> p = w;
        p.push_back(b);
        return p;
    }
};

struct Layer {
    std::vector<Neuron> neurons;

    Layer(int nin, int nout) {
        for (int i = 0; i < nout; ++i) neurons.emplace_back(nin);
    }

    std::vector<Value> operator()(const std::vector<Value>& x) const {
        std::vector<Value> out;
        for (const auto& n : neurons) out.push_back(n(x));
        return out;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> p;
        for (const auto& n : neurons) {
            auto q = n.parameters();
            p.insert(p.end(), q.begin(), q.end());
        }
        return p;
    }
};

struct MLP {
    std::vector<Layer> layers;

    MLP(int nin, const std::vector<int>& nouts) {
        int prev = nin;
        for (int n : nouts) { layers.emplace_back(prev, n); prev = n; }
    }

    std::vector<Value> operator()(std::vector<Value> x) const {
        for (const auto& l : layers) x = l(x);
        return x;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> p;
        for (const auto& l : layers) {
            auto q = l.parameters();
            p.insert(p.end(), q.begin(), q.end());
        }
        return p;
    }
};

// ---------------------------------------------------------------------------
// The three things the video actually walks through
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
    std::cout << "  (expected 6, -4, -2, 4, -2)\n\n";
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
    std::cout << "  (expected -1.5, 1.0, 0.5, 0.0)\n\n";
}

void example_training() {
    std::cout << "== 3. training a 3-layer MLP ==\n";
    MLP net(3, { 4, 4, 1 });
    std::cout << "  parameters: " << net.parameters().size() << "\n";

    std::vector<std::vector<Value>> xs = {
        {Value(2.0),  Value(3.0),  Value(-1.0)},
        {Value(3.0),  Value(-1.0), Value(0.5)},
        {Value(0.5),  Value(1.0),  Value(1.0)},
        {Value(1.0),  Value(1.0),  Value(-1.0)},
    };
    std::vector<double> ys = { 1.0, -1.0, -1.0, 1.0 };

    for (int step = 0; step < 100; ++step) {
        // forward: rebuild the graph each step, exactly like the Python version
        std::vector<Value> ypred;
        for (const auto& x : xs) ypred.push_back(net(x)[0]);

        Value loss(0.0);
        for (size_t i = 0; i < ys.size(); ++i)
            loss = loss + pow(ypred[i] - Value(ys[i]), 2.0);

        // backward — zero first, or gradients from the last step leak in
        loss.zero_grad();
        loss.backward();

        // update
        for (auto& p : net.parameters()) p.data() += -0.05 * p.grad();

        if (step % 20 == 0 || step == 99)
            printf("  step %3d  loss %.8f\n", step, loss.data());
    }

    std::vector<Value> final_pred;
    for (const auto& x : xs) final_pred.push_back(net(x)[0]);
    std::cout << "  predictions: ";
    for (const auto& p : final_pred) printf("%.4f ", p.data());
    std::cout << "\n  targets:     1.0000 -1.0000 -1.0000 1.0000\n\n";
}

void test_micrograd() {
    example_manual_backprop();
    example_neuron();
    example_training();

    // dump a small graph for graphviz:  ./micrograd > out && dot -Tsvg ...
    Value a(2.0), b(-3.0), c(10.0);
    a.label("a"); b.label("b"); c.label("c");
    Value L = a * b + c;
    L.label("L");
    L.backward();
    std::cout << "== graphviz (pipe into `dot -Tsvg`) ==\n" << to_dot(L);
}
