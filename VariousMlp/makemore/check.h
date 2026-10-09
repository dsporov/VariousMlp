// check.h - the sanity checks shared by the Karpathy ports.
//
// There are two kinds of check, because there are two kinds of mistake:
//
//   CHECK(cond, msg)  a hard check on structure: mismatched array lengths, an
//                     index outside the vocabulary, a division by zero. It
//                     throws, because nothing that runs after it would mean
//                     anything.
//
//                     It is not assert(), on purpose: assert is removed when
//                     NDEBUG is defined, and these files are meant to run in
//                     Release (makemore's full-batch training is ~4x slower in
//                     Debug). No CHECK is inside an inner arithmetic loop. An
//                     interleaved A/B test of makemore's 100 training steps,
//                     with and without the checks, showed no difference beyond
//                     the normal variation between runs. So the checks stay
//                     on in both configurations.
//
//   sanity::Expect    a soft check on numbers: "the video says 2.454; is our
//                     result close to that?". It reports and continues. A
//                     number that differs from the video is information, not
//                     a sign of a broken program, and the expected values only
//                     hold for the video's dataset.
//
// A simple rule for choosing: use CHECK when the program is wrong, and Expect
// when the result is wrong.

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

// Counts soft checks, so that an example can end with a single summary line.
// Only the failed checks print their numbers.
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

// The message is written to a stream, so it can include the wrong values:
//   CHECK(xs.size() == ys.size(), xs.size() << " rows but " << ys.size() << " targets");
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::ostringstream sanity_os_;                                     \
            sanity_os_ << msg;                                                 \
            ::sanity::fail(__FILE__, __LINE__, #cond, sanity_os_.str());       \
        }                                                                      \
    } while (false)
