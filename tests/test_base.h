#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <cassert>
#include <iomanip>

namespace tests {

    class TestBase {
    protected:
        int tests_passed = 0;
        int tests_total = 0;

        void print_test_header(const std::string& test_name) {
            std::cout << "\n========================================\n";
            std::cout << "EJECUTANDO: " << test_name << "\n";
            std::cout << "========================================\n";
        }

        void print_test_result(const std::string& test_name, bool passed) {
            tests_total++;
            if (passed) {
                tests_passed++;
                std::cout << "PASADO: " << test_name << "\n";
            } else {
                std::cout << "FALLIDO: " << test_name << "\n";
            }
        }

        bool is_close(float a, float b, float tolerance = 1e-5f) {
            return std::abs(a - b) < tolerance;
        }

    public:
        virtual ~TestBase() = default;
        virtual void run_tests() = 0;

        void print_summary(const std::string& test_suite_name) {
            std::cout << "\n==========================================\n";
            std::cout << "RESUMEN DE " << test_suite_name << "\n";
            std::cout << "==========================================\n";
            std::cout << "Tests ejecutados: " << tests_total << "\n";
            std::cout << "Tests exitosos: " << tests_passed << "\n";
            std::cout << "Tests fallidos: " << (tests_total - tests_passed) << "\n";
            std::cout << "Tasa de exito: " << std::fixed << std::setprecision(1)
                      << ((float)tests_passed / tests_total * 100.0f) << "%\n";
        }

        int get_tests_passed() const { return tests_passed; }
        int get_tests_total() const { return tests_total; }
    };

} // namespace tests
