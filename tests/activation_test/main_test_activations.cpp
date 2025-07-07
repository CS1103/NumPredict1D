// =============================================
// tests/main_test_activations.cpp
// =============================================
#include "test_activations.h"

int main() {
    tests::TestActivations test;
    test.run_tests();
    return (test.get_tests_passed() == test.get_tests_total()) ? 0 : 1;
}
