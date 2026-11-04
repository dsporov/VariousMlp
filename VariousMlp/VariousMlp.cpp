#include <exception>
#include <iostream>
#include <string>

void test_micrograd();
void test_makemore();
void test_makemore_mlp();
void test_makemore_batchnorm();
void test_gpt();
void test_gpt_chat(const std::string& checkpoint);

int main(int argc, char** argv)
{
    // The hard checks (CHECK in makemore/check.h) throw. So they are caught
    // here, and the file:line is printed, instead of the runtime showing an
    // error dialog.
    try {
        // The walkthroughs take minutes and end by training a model. The
        // prompt that uses this model should start immediately, and
        // `VariousMlp.exe chat` goes directly to it, and
        // `VariousMlp.exe chat <file>` also names the checkpoint to load. In
        // Visual Studio, set this under Debugging -> Command Arguments.
        if (argc > 1 && std::string(argv[1]) == "chat") {
            test_gpt_chat(argc > 2 ? argv[2] : "");
            return 0;
        }

        test_micrograd();
        test_makemore();
        test_makemore_mlp();
        test_makemore_batchnorm();
        test_gpt();
    }
    catch (const std::exception& e) {
        std::cerr << "\n*** " << e.what() << "\n";
        return 1;
    }
}
