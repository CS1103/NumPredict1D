#include "layer_test/test_dense_layer.h"
#include "activation_test/test_activations.h"
#include "convergence_test/test_convergence.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "========================================\n";
    std::cout << "EJECUTANDO TODOS LOS TESTS UNITARIOS\n";
    std::cout << "========================================\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int total_tests = 0;
    int total_passed = 0;
    
    // Ejecutar tests de capas densas
    {
        std::cout << "\nINICIANDO TESTS DE CAPAS DENSAS...\n";
        tests::TestDenseLayer test_dense;
        test_dense.run_tests();
        total_tests += test_dense.get_tests_total();
        total_passed += test_dense.get_tests_passed();
    }
    
    // Ejecutar tests de activaciones
    {
        std::cout << "\nINICIANDO TESTS DE ACTIVACIONES...\n";
        tests::TestActivations test_activations;
        test_activations.run_tests();
        total_tests += test_activations.get_tests_total();
        total_passed += test_activations.get_tests_passed();
    }
    
    // Ejecutar tests de convergencia
    {
        std::cout << "\nINICIANDO TESTS DE CONVERGENCIA...\n";
        tests::TestConvergence test_convergence;
        test_convergence.run_tests();
        total_tests += test_convergence.get_tests_total();
        total_passed += test_convergence.get_tests_passed();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Resumen general
    std::cout << "\n==========================================\n";
    std::cout << "RESUMEN GENERAL DE TODOS LOS TESTS\n";
    std::cout << "==========================================\n";
    std::cout << "Total de tests ejecutados: " << total_tests << "\n";
    std::cout << "Tests exitosos: " << total_passed << "\n";
    std::cout << "Tests fallidos: " << (total_tests - total_passed) << "\n";
    std::cout << "Tasa de exito general: " << std::fixed << std::setprecision(1)
              << ((float)total_passed / total_tests * 100.0f) << "%\n";
    std::cout << "Tiempo total de ejecucion: " << duration.count() << " ms\n";
    
    if (total_passed == total_tests) {
        std::cout << "\nTODOS LOS TESTS PASARON EXITOSAMENTE!\n";
        return 0;
    } else {
        std::cout << "\nALGUNOS TESTS FALLARON\n";
        return 1;
    }
}
