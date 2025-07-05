cmake_minimum_required(VERSION 3.15)
project(pong_ai LANGUAGES CXX)

# --------------------------------------------------
# 1) Estándar de C++
# --------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------
# 2) Biblioteca 'core' con tu código fuente
#    (aquí sólo los .cpp que no son header-only)
# --------------------------------------------------
add_library(core
        src/utec/agent/PongAgent.cpp
        src/utec/agent/EnvGym.cpp
)

# Para que 'core' vea tus .h en include/…
target_include_directories(core
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# --------------------------------------------------
# 3) Tests
# --------------------------------------------------
enable_testing()

# Coge todos los tests .cpp
file(GLOB TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/tests/*.cpp
)

add_executable(tests_runner
        ${TEST_SOURCES}
)

# Indica dónde están tus headers y el propio catch.hpp
target_include_directories(tests_runner
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
)

# Linkea tu librería 'core' (contiene toda tu NN, tensor, agente…)
target_link_libraries(tests_runner
        PRIVATE
        core
)

add_test(NAME all_tests COMMAND tests_runner)
