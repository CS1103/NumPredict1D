// =============================================
// tests/main_test_dense_layer.cpp
// =============================================
#include "test_dense_layer.h"

int main() {
    tests::TestDenseLayer test;
    test.run_tests();
    return (test.get_tests_passed() == test.get_tests_total()) ? 0 : 1;
}
