// gpt_chat.cpp - a prompt for the model gpt.cpp trains.
//
// Read this part before being disappointed by the output. The model was
// trained on exactly one objective: given the last few characters of its
// corpus, predict the next one. That is all it knows how to do, so what it
// does with a line you type is *continue* it, not reply to it. Ask it the
// capital of France and it will write you a stage direction.
//
// That is not a shortcoming of this file, and no amount of sampling code fixes
// it. Three separate things stand between a model like this and one that
// answers questions:
//
//   - scale. 10.8M parameters over a megabyte of text. GPT-2 small is 117M
//     over 40GB, and instruction-following does not appear below roughly a
//     billion parameters on hundreds of gigabytes.
//   - the objective. Predicting the next token of a corpus gives you a model
//     of that corpus. Question answering falls out of pretraining only when the
//     corpus is full of questions being answered. The internet is. Shakespeare
//     is not.
//   - the stages after pretraining. Even a 175B base model completes documents
//     rather than answering; supervised fine-tuning on instruction/response
//     pairs and then preference optimisation are what turn one into the other.
//     The video ends by saying exactly this: what it builds is stage one.
//
// So this is a text continuer, which is the real capability and is worth
// having. Point kCorpus at a corpus of questions and answers and it will
// produce well-formed questions and answers that mean nothing, which is its own
// kind of interesting: format is learned long before content.
//
// Run it with `VariousMlp.exe chat`. It needs a checkpoint, which section 6 of
// the walkthrough writes.

#include "gpt_model.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"

using namespace gpt;

namespace {

// The two configurations the walkthrough trains, narrowed to inference. batch,
// dropout and the learning rate are free to be anything here: checkpoint_name
// does not use them, and load_params validates only the five fields that
// change the shape of the weights.
std::vector<Config> inference_configs() {
    return {
        { kEmbed, kHeads, kLayers, kBlock, 1, real(0), kLr },
        { kBigEmbed, kBigHeads, kBigLayers, kBigBlock, 1, real(0), kBigLr },
    };
}

// Text to token ids. A character the corpus never contained has no row in the
// embedding table, so there is nothing the model could do with it; dropping it
// and saying so is better than substituting something and not saying so.
std::vector<int> encode(const Data& d, const std::string& s, size_t* dropped) {
    std::vector<int> ids;
    ids.reserve(s.size());
    *dropped = 0;
    for (char c : s) {
        const int id = d.stoi[static_cast<unsigned char>(c)];
        if (id < 0) {
            ++*dropped;
            continue;
        }
        ids.push_back(id);
    }
    return ids;
}

struct Options {
    int length = 400;
    real temperature = real(0.8);
};

void print_help(const Options& opt) {
    printf("  <text>     continue it\n"
        "  <blank>    continue from a blank line, which is where the\n"
        "             walkthrough's own samples start\n"
        "  :len N     characters to generate (now %d)\n"
        "  :temp X    sampling temperature (now %.2f). Below 1 sharpens the\n"
        "             distribution toward what the model is sure of; above 1\n"
        "             flattens it. 1.0 is what the video samples at.\n"
        "  :quit\n",
        opt.length, static_cast<double>(opt.temperature));
}

// True if the line was a command rather than a prompt. A prompt that genuinely
// begins with a colon is unreachable, which for this corpus is no loss.
bool command(const std::string& line, Options& opt, bool& quit) {
    if (line.empty() || line[0] != ':') {
        return false;
    }

    std::istringstream in(line);
    std::string verb;
    in >> verb;

    if (verb == ":quit" || verb == ":q") {
        quit = true;
        return true;
    }
    if (verb == ":help" || verb == ":h") {
        print_help(opt);
        return true;
    }
    if (verb == ":len") {
        int n = 0;
        if ((in >> n) && n > 0 && n <= 20000) {
            opt.length = n;
            printf("  length %d\n", opt.length);
        }
        else {
            printf("  :len wants a whole number between 1 and 20000\n");
        }
        return true;
    }
    if (verb == ":temp") {
        double t = 0.0;
        if ((in >> t) && t > 0.0 && t <= 5.0) {
            opt.temperature = static_cast<real>(t);
            printf("  temperature %.2f\n", static_cast<double>(opt.temperature));
        }
        else {
            printf("  :temp wants a number above 0 and at most 5\n");
        }
        return true;
    }

    printf("  no such command: %s   (:help for the list)\n", verb.c_str());
    return true;
}

// Looks for a checkpoint belonging to each configuration in turn. The file is
// probed before the model is built, because constructing the scaled-up one
// allocates 173MB of weights and gradients that would be thrown away.
std::unique_ptr<Gpt> load_any(const Data& d, std::string& from) {
    std::mt19937 init(1337);
    for (const Config& cfg : inference_configs()) {
        const std::string path = d.dir + checkpoint_name(cfg, d);
        {
            std::ifstream probe(path, std::ios::binary);
            if (!probe) {
                continue;
            }
        }
        std::unique_ptr<Gpt> candidate(new Gpt(cfg, d.vocab, init));
        if (load_params(candidate->params(), cfg, d, path)) {
            from = path;
            return candidate;
        }
    }
    return std::unique_ptr<Gpt>();
}

}  // namespace

void test_gpt_chat() {
    Data d;
    if (!load_corpus(d)) {
        printf("could not find %s - expected it next to gpt_chat.cpp in "
            "makemore/\n", kCorpus);
        return;
    }

    std::string from;
    std::unique_ptr<Gpt> model = load_any(d, from);
    if (!model) {
        printf("no checkpoint for %s beside it. Run the walkthrough first -\n"
            "section 6 trains a model and saves one.\n", kCorpus);
        return;
    }
    printf("  loaded %s\n  %zu parameters, %d characters of context\n",
        from.c_str(), model->parameter_count(), model->config().block);

    std::cout <<
        "\nThis model predicts the next character of its corpus and nothing\n"
        "else, so it continues what you type rather than answering it. A line\n"
        "of Shakespeare gets you more Shakespeare. :help for the options,\n"
        ":quit to leave.\n";

    std::mt19937 rng(1337);
    Options opt;
    std::string line;
    for (;;) {
        std::cout << "\n> " << std::flush;
        if (!std::getline(std::cin, line)) {
            break;                       // end of input: a pipe closing, or ^Z
        }

        bool quit = false;
        if (command(line, opt, quit)) {
            if (quit) {
                break;
            }
            continue;
        }

        size_t dropped = 0;
        const std::vector<int> prompt = encode(d, line, &dropped);
        if (dropped > 0) {
            printf("  (dropped %zu character(s) the corpus never contained)\n",
                dropped);
        }

        // The prompt is echoed with the continuation so the two read as one
        // piece of text, which is what they are.
        const std::vector<int> out =
            model->generate(prompt, opt.length, opt.temperature, rng);
        std::cout << line << d.decode(out) << "\n";
    }

    std::cout << "\n";
}
