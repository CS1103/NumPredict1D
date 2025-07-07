cmake_minimum_required(VERSION 3.29)
project(projecto_final_port_8080_main)
set(CMAKE_CXX_STANDARD 20)

# Configurar directorios de include
include_directories(include/utec)

# ================================
# EJECUTABLE EXPERIMENT RUNNER
# ================================
add_executable(ExperimentRunner src/experiment_runner.cpp
        src/config.h
        src/trainer.h)

# ================================
# EJECUTABLES DE TESTS
# ================================
# Ejecutable para todos los tests
add_executable(run_all_tests
        tests/main_all_tests.cpp
        tests/test_base.h
        tests/layer_test/test_dense_layer.h
        tests/activation_test/test_activations.h
        tests/convergence_test/test_convergence.h
)

# Tests individuales
add_executable(test_dense_layer
        tests/layer_test/main_test_dense_layer.cpp
        tests/test_base.h
        tests/layer_test/test_dense_layer.h
)

add_executable(test_activations
        tests/activation_test/main_test_activations.cpp
        tests/test_base.h
        tests/activation_test/test_activations.h
)

add_executable(test_convergence
        tests/convergence_test/main_test_convergence.cpp
        tests/test_base.h
        tests/convergence_test/test_convergence.h
)

# ================================
# EJECUTABLE DE AYUDA/DOCUMENTACION
# ================================
add_executable(show_help docs/help_program.cpp)