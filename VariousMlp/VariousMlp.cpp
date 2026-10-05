#include <exception>
#include <iostream>
#include <string>

void test_micrograd();
void test_makemore();
void test_makemore_mlp();
void test_makemore_batchnorm();
void test_gpt();
void test_gpt_chat();

int main(int argc, char** argv)
{
    // The hard checks (CHECK in makemore/check.h) throw, so catch them here and
    // print the file:line instead of letting the runtime pop a dialog.
    try {
        // The walkthroughs take minutes and end by training a model; the prompt
        // that model feeds wants to start immediately. `VariousMlp.exe chat`
        // skips straight to it. In Visual Studio: Debugging -> Command
        // Arguments.
        if (argc > 1 && std::string(argv[1]) == "chat") {
            test_gpt_chat();
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
