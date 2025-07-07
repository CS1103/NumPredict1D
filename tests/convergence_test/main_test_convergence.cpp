// =============================================
// tests/main_test_convergence.cpp
// =============================================
#include "test_convergence.h"

int main() {
    tests::TestConvergence test;
    test.run_tests();
    return (test.get_tests_passed() == test.get_tests_total()) ? 0 : 1;
}
