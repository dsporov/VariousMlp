#include <exception>
#include <iostream>

void test_micrograd();
void test_makemore();

int main()
{
    // The hard checks (CHECK in Karpathy/check.h) throw, so catch them here and
    // print the file:line instead of letting the runtime pop a dialog.
    try {
        test_micrograd();
        test_makemore();
    }
    catch (const std::exception& e) {
        std::cerr << "\n*** " << e.what() << "\n";
        return 1;
    }
}
