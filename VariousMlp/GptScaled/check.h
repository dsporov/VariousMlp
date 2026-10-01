// GptScaled's own copy of Karpathy/check.h, taken at commit 512eaaa and
// otherwise unchanged, so that the two diff cleanly. gpt_scaled compiles only
// the files in this directory, which keeps it independent of Karpathy/: a
// change on either side stays on that side. Where the comments below mention
// other files, they mean the ones in Karpathy/.
//
// check.h - the sanity checks shared by the Karpathy ports.
//
// Two flavours, because the mistakes come in two flavours:
//
//   CHECK(cond, msg)  a hard check on structure: mismatched array lengths, an
//                     index outside the vocabulary, a division by zero. Throws,
//                     because whatever runs next is meaningless.
//
//                     Deliberately not assert(): assert compiles out under
//                     NDEBUG, and Release is exactly where these files are meant
//                     to be run (makemore's full-batch training is ~4x slower in
//                     Debug). No CHECK sits inside an inner arithmetic loop, and
//                     an interleaved A/B of makemore's 100 training steps with
//                     and without them came out inside the run-to-run noise, so
//                     they stay on in both configurations.
//
//   sanity::Expect    a soft check on numbers: "the video says 2.454, did we
//                     land there?". Reports and keeps going, because a number
//                     that drifts is information, not a broken program - and
//                     the expected values only hold for the video's dataset.
//
// The rule of thumb for which to use: CHECK when the program is wrong, Expect
// when the answer is.

#pragma once

#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sanity {

struct CheckFailed : std::logic_error {
    explicit CheckFailed(const std::string& what) : std::logic_error(what) {}
};

[[noreturn]] inline void fail(const char* file, int line, const char* expr,
    const std::string& detail) {
    std::ostringstream os;
    os << file << ":" << line << ": check failed: " << expr;
    if (!detail.empty()) os << "\n    " << detail;
    throw CheckFailed(os.str());
}

// Tallies soft checks so an example can end on a single summary line; only the
// failures print their numbers.
class Expect {
public:
    bool close(const char* what, double got, double want, double tol) {
        return record(what, std::fabs(got - want) <= tol, got, want, tol);
    }
    bool equal(const char* what, double got, double want) {
        return record(what, got == want, got, want, 0.0);
    }
    bool that(const char* what, bool ok) {
        if (ok) ++passed_;
        else { ++failed_; std::printf("    [FAIL] %s\n", what); }
        return ok;
    }
    void report() const {
        std::printf("  [%s] %d/%d checks\n", failed_ ? "FAIL" : "ok",
            passed_, passed_ + failed_);
    }
    int failed() const { return failed_; }

private:
    bool record(const char* what, bool ok, double got, double want, double tol) {
        if (ok) { ++passed_; return true; }
        ++failed_;
        std::printf("    [FAIL] %s: got %.6g, expected %.6g (tolerance %.3g)\n",
            what, got, want, tol);
        return false;
    }

    int passed_ = 0;
    int failed_ = 0;
};

}  // namespace sanity

// The message is streamed, so it can carry the offending numbers:
//   CHECK(xs.size() == ys.size(), xs.size() << " rows but " << ys.size() << " targets");
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::ostringstream sanity_os_;                                     \
            sanity_os_ << msg;                                                 \
            ::sanity::fail(__FILE__, __LINE__, #cond, sanity_os_.str());       \
        }                                                                      \
    } while (false)
