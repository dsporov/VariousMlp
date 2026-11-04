// gpt_chat.cpp - an interactive prompt for the model that gpt.cpp trains.
//
// Read this part first, so that the output does not disappoint you. The model
// was trained for exactly one task: given the last few characters of its
// corpus, predict the next one. That is all it can do. So it *continues* the
// line you type; it does not reply to it. If you ask it for the capital of
// France, it will write a stage direction.
//
// This is not a weakness of this file, and no sampling code can fix it. Three
// separate things are missing between a model like this and a model that
// answers questions:
//
//   - scale. This model has 10.8M parameters, trained on one megabyte of
//     text. GPT-2 small has 117M parameters, trained on 40GB. The ability to
//     follow instructions does not appear below roughly a billion parameters
//     trained on hundreds of gigabytes.
//   - the objective. Predicting the next token of a corpus gives a model of
//     that corpus. Pretraining produces question answering only if the corpus
//     is full of questions with answers. The internet is. Shakespeare is not.
//   - the stages after pretraining. Even a 175B base model continues
//     documents instead of answering. Supervised fine-tuning on
//     instruction/response pairs, followed by preference optimisation, is
//     what turns a base model into an assistant. The video ends by saying
//     exactly this: what it builds is the first stage.
//
// So this is a tool that continues text. That is the model's real ability, and
// it is worth having. If you train it on a corpus of questions and answers, it
// will produce well-formed questions and answers that mean nothing. That is
// interesting in its own way: the format is learned long before the content.
//
// Run it with `VariousMlp.exe chat` to load the newest checkpoint beside the
// data, or with `VariousMlp.exe chat <file>` to load a given one. Section 6 of
// the walkthrough writes a checkpoint, and so does gpt_scaled. :load switches
// to another checkpoint without a restart. Each checkpoint holds its own shape
// and vocabulary, so this file never reads the corpus. A model trained on any
// text works here.

#include "gpt_model.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "check.h"

using namespace gpt;

namespace {

// Converts text to token ids. A character that is not in the model's
// vocabulary has no row in the embedding table, so the model cannot use it.
// It is better to drop such a character and report it than to replace it
// silently with something else.
std::vector<int> encode(const Vocab& v, const std::string& s, size_t* dropped) {
    std::vector<int> ids;
    ids.reserve(s.size());
    *dropped = 0;
    for (char c : s) {
        const int id = v.stoi[static_cast<unsigned char>(c)];
        if (id < 0) {
            ++*dropped;
            continue;
        }
        ids.push_back(id);
    }
    return ids;
}

// Every checkpoint in the directories that load_corpus searches, newest first.
// Checkpoints are saved next to input.txt, so that is where they are. Two of
// these paths can lead to the same directory, so each file is listed once, by
// its canonical path.
std::vector<std::string> find_checkpoints() {
    namespace fs = std::filesystem;
    std::vector<std::pair<fs::file_time_type, std::string>> found;
    std::set<fs::path> seen;
    for (const char* dir : kDataDirs) {
        const fs::path where = (dir[0] == '\0') ? fs::path(".") : fs::path(dir);
        std::error_code ec;
        for (fs::directory_iterator it(where, ec), end; !ec && it != end;
            it.increment(ec)) {
            const fs::path& p = it->path();
            if (p.extension() != ".ckpt") {
                continue;
            }
            std::error_code bad;
            fs::path key = fs::canonical(p, bad);
            if (bad) {
                key = p;
            }
            if (!seen.insert(key).second) {
                continue;
            }
            found.emplace_back(fs::last_write_time(p, bad),
                std::string(dir) + p.filename().string());
        }
    }

    std::sort(found.begin(), found.end(),
        [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
    std::vector<std::string> paths;
    for (const auto& f : found) {
        paths.push_back(f.second);
    }
    return paths;
}

// One line about a checkpoint, for :list. Only the header is read, so this is
// quick even for the scaled-up model.
std::string describe(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    CheckpointHeader h;
    const char* why = "it cannot be opened";
    if (f) {
        why = read_checkpoint_header(f, h);
    }
    char buf[256];
    if (why != nullptr) {
        std::snprintf(buf, sizeof(buf), "not usable: %s", why);
    }
    else {
        std::snprintf(buf, sizeof(buf),
            "%d blocks, %d channels, %d heads, context %d, %d characters",
            h.cfg.n_layer, h.cfg.n_embd, h.cfg.n_head, h.cfg.block,
            h.vocab.vocab);
    }
    return std::string(buf);
}

void list_checkpoints(const std::vector<std::string>& found) {
    if (found.empty()) {
        printf("  no checkpoints beside the data\n");
        return;
    }
    for (size_t i = 0; i < found.size(); ++i) {
        printf("  %2zu  %s\n      %s\n", i + 1, found[i].c_str(),
            describe(found[i]).c_str());
    }
}

// The model in use, and the vocabulary saved with it. They are kept together,
// because the model's token ids have a meaning only through that vocabulary.
struct Loaded {
    std::unique_ptr<Gpt> model;
    Vocab vocab;
    std::string path;
};

// Replaces `current` only if the file loads. So a failed :load keeps the model
// that was in use.
bool load(const std::string& path, Loaded& current) {
    Vocab vocab;
    std::unique_ptr<Gpt> model = load_model(path, vocab);
    if (!model) {
        return false;
    }
    current.model = std::move(model);
    current.vocab = vocab;
    current.path = path;
    printf("  loaded %s\n  %zu parameters, %d characters of context, "
        "%d characters in the vocabulary\n", path.c_str(),
        current.model->parameter_count(), current.model->config().block,
        current.vocab.vocab);
    return true;
}

struct Options {
    int length = 400;
    real temperature = real(0.8);
};

struct Session {
    Options opt;
    Loaded current;
    std::vector<std::string> found;      // as :list printed them last
    bool quit = false;
};

void print_help(const Options& opt) {
    printf("  <text>     continue it\n"
        "  <blank>    continue from a blank line, which is where the\n"
        "             walkthrough's own samples start\n"
        "  :len N     characters to generate (now %d)\n"
        "  :temp X    sampling temperature (now %.2f). Below 1 sharpens the\n"
        "             distribution toward what the model is sure of; above 1\n"
        "             flattens it. 1.0 is what the video samples at.\n"
        "  :list      the checkpoints beside the data, newest first\n"
        "  :load N    switch to entry N of :list. :load <file> opens any\n"
        "             checkpoint. If the file cannot be used, the current\n"
        "             model stays.\n"
        "  :quit\n",
        opt.length, static_cast<double>(opt.temperature));
}

// Returns true if the line was a command, not a prompt. So a prompt cannot
// start with a colon, which does not matter for this corpus.
bool command(const std::string& line, Session& s) {
    if (line.empty() || line[0] != ':') {
        return false;
    }

    std::istringstream in(line);
    std::string verb;
    in >> verb;

    if (verb == ":quit" || verb == ":q") {
        s.quit = true;
        return true;
    }
    if (verb == ":help" || verb == ":h") {
        print_help(s.opt);
        return true;
    }
    if (verb == ":len") {
        int n = 0;
        if ((in >> n) && n > 0 && n <= 20000) {
            s.opt.length = n;
            printf("  length %d\n", s.opt.length);
        }
        else {
            printf("  :len wants a whole number between 1 and 20000\n");
        }
        return true;
    }
    if (verb == ":temp") {
        double t = 0.0;
        if ((in >> t) && t > 0.0 && t <= 5.0) {
            s.opt.temperature = static_cast<real>(t);
            printf("  temperature %.2f\n", static_cast<double>(s.opt.temperature));
        }
        else {
            printf("  :temp wants a number above 0 and at most 5\n");
        }
        return true;
    }
    if (verb == ":list") {
        s.found = find_checkpoints();
        list_checkpoints(s.found);
        return true;
    }
    if (verb == ":load") {
        // The argument is the rest of the line, because a file name can
        // contain spaces.
        std::string arg;
        std::getline(in >> std::ws, arg);
        if (arg.empty()) {
            printf("  :load wants a number from :list, or a file name\n");
            return true;
        }
        // A number picks an entry of the list that :list printed last. The
        // list is not scanned again here, so that the number means what the
        // user saw.
        std::istringstream num(arg);
        size_t n = 0;
        if (arg.find_first_not_of("0123456789") == std::string::npos && (num >> n)) {
            if (n < 1 || n > s.found.size()) {
                printf("  the list has no entry %zu (:list prints it)\n", n);
                return true;
            }
            arg = s.found[n - 1];
        }
        load(arg, s.current);
        return true;
    }

    printf("  no such command: %s   (:help for the list)\n", verb.c_str());
    return true;
}

}  // namespace

void test_gpt_chat(const std::string& checkpoint) {
    Session s;
    if (!checkpoint.empty()) {
        if (!load(checkpoint, s.current)) {
            return;
        }
    }
    else {
        s.found = find_checkpoints();
        if (s.found.empty()) {
            printf("no checkpoint beside %s. Run the walkthrough first -\n"
                "section 6 trains a model and saves one. Or name a checkpoint:\n"
                "VariousMlp chat <file>\n", kCorpus);
            return;
        }
        list_checkpoints(s.found);

        // The newest checkpoint is usually the model that was trained last.
        // If it cannot be used, the next one is tried.
        for (const std::string& path : s.found) {
            if (load(path, s.current)) {
                break;
            }
        }
        if (!s.current.model) {
            printf("none of these checkpoints can be used. Training a model "
                "again replaces them.\n");
            return;
        }
    }

    std::cout <<
        "\nThis model predicts the next character of its corpus and nothing\n"
        "else, so it continues what you type rather than answering it. A line\n"
        "of Shakespeare gets you more Shakespeare. :help for the options,\n"
        ":quit to leave.\n";

    std::mt19937 rng(1337);
    std::string line;
    for (;;) {
        std::cout << "\n> " << std::flush;
        if (!std::getline(std::cin, line)) {
            break;                       // end of input: a pipe closing, or ^Z
        }

        if (command(line, s)) {
            if (s.quit) {
                break;
            }
            continue;
        }

        size_t dropped = 0;
        const std::vector<int> prompt = encode(s.current.vocab, line, &dropped);
        if (dropped > 0) {
            printf("  (dropped %zu character(s) that are not in the model's "
                "vocabulary)\n", dropped);
        }

        // The prompt is printed together with the continuation, so that the
        // two read as one piece of text, which is what they are.
        const std::vector<int> out = s.current.model->generate(prompt,
            s.opt.length, s.opt.temperature, rng);
        std::cout << line << s.current.vocab.decode(out) << "\n";
    }

    std::cout << "\n";
}
