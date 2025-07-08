#include "trainer.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>

class ExperimentRunner {
private:
    std::string data_path_train;
    std::string data_path_test;
    std::vector<utec::training::TrainingResult> results;
    std::vector<utec::config::TrainingConfig> configs_used;

public:
    ExperimentRunner(const std::string& train_path, const std::string& test_path)
        : data_path_train(train_path), data_path_test(test_path) {}

    void run_single_experiment(const std::string& config_input) {
        try {
            clear_results();

            std::string config_name = resolve_config_name(config_input);

            if (!config_exists(config_name)) {
                std::cerr << "Error: Configuracion '" << config_name << "' no encontrada.\n";
                std::cerr << "Configuraciones disponibles:\n";
                show_config_names();
                return;
            }

            auto config = utec::config::ConfigManager::get_config_by_name(config_name);

            std::cout << "========================================\n";
            std::cout << "EJECUTANDO EXPERIMENTO: " << config_name << "\n";
            std::cout << "========================================\n\n";

            utec::training::Trainer<float> trainer(data_path_train, data_path_test);
            trainer.run_training(config);

            auto result = trainer.get_last_result();
            results.push_back(result);
            configs_used.push_back(config);

            print_single_result(result);

        } catch (const std::exception& e) {
            std::cerr << "Error en experimento " << config_input << ": " << e.what() << "\n\n";
        }
    }

    void run_all_experiments() {
        clear_results();

        auto configs = utec::config::ConfigManager::get_all_configs();

        std::cout << "=== EJECUTANDO TODOS LOS EXPERIMENTOS ===\n";
        std::cout << "Total de configuraciones: " << configs.size() << "\n\n";

        for (const auto& config : configs) {
            try {
                std::cout << "========================================\n";
                std::cout << "EJECUTANDO EXPERIMENTO: " << config.name << "\n";
                std::cout << "========================================\n\n";

                utec::training::Trainer<float> trainer(data_path_train, data_path_test);
                trainer.run_training(config);

                auto result = trainer.get_last_result();
                results.push_back(result);
                configs_used.push_back(config);

                print_single_result(result);

                if (&config != &configs.back()) {
                    std::cout << "Presiona Enter para continuar con el siguiente experimento...\n";
                    std::cin.get();
                }

            } catch (const std::exception& e) {
                std::cerr << "Error en experimento " << config.name << ": " << e.what() << "\n\n";
            }
        }

        print_summary();
        save_results_to_csv();
    }

    void run_selected_experiments(const std::vector<std::string>& config_inputs) {
        clear_results();

        std::cout << "=== EJECUTANDO EXPERIMENTOS SELECCIONADOS ===\n";
        std::cout << "Configuraciones seleccionadas:\n";

        std::vector<std::string> valid_configs;
        for (const auto& input : config_inputs) {
            std::string config_name = resolve_config_name(input);
            if (config_exists(config_name)) {
                valid_configs.push_back(config_name);
                std::cout << "  - " << config_name << " (cargada)\n";
            } else {
                std::cout << "  - " << input << " -> " << config_name << " (no encontrada)\n";
            }
        }

        if (valid_configs.empty()) {
            std::cout << "No se encontraron configuraciones validas.\n";
            return;
        }

        std::cout << "\nEjecutando " << valid_configs.size() << " configuraciones validas...\n\n";

        for (size_t i = 0; i < valid_configs.size(); ++i) {
            try {
                auto config = utec::config::ConfigManager::get_config_by_name(valid_configs[i]);

                std::cout << "========================================\n";
                std::cout << "EJECUTANDO EXPERIMENTO: " << valid_configs[i] << "\n";
                std::cout << "========================================\n\n";

                utec::training::Trainer<float> trainer(data_path_train, data_path_test);
                trainer.run_training(config);

                auto result = trainer.get_last_result();
                results.push_back(result);
                configs_used.push_back(config);

                print_single_result(result);

                if (i < valid_configs.size() - 1) {
                    std::cout << "Presiona Enter para continuar...\n";
                    std::cin.get();
                }

            } catch (const std::exception& e) {
                std::cerr << "Error en experimento " << valid_configs[i] << ": " << e.what() << "\n\n";
            }
        }

        print_summary();
        save_results_to_csv();
    }

    void show_available_configs() {
        auto configs = utec::config::ConfigManager::get_all_configs();

        std::cout << "=== CONFIGURACIONES DISPONIBLES ===\n";
        for (size_t i = 0; i < configs.size(); ++i) {
            const auto& config = configs[i];
            std::cout << "[" << i + 1 << "] " << config.name << "\n";
            std::cout << "    Funcion de perdida: " << config.loss_function << "\n";
            std::cout << "    Optimizador: " << config.optimizer << "\n";
            std::cout << "    Epocas: " << config.epochs << "\n";
            std::cout << "    Batch size: " << config.batch_size << "\n";
            std::cout << "    Learning rate: " << std::setprecision(4) << config.learning_rate << "\n\n";
        }
    }

    void show_current_results() {
        if (results.empty()) {
            std::cout << "No hay resultados disponibles.\n";
            return;
        }
        print_summary();
    }

private:
    void clear_results() {
        results.clear();
        configs_used.clear();
    }

    bool config_exists(const std::string& config_name) {
        try {
            utec::config::ConfigManager::get_config_by_name(config_name);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    void show_config_names() {
        auto names = utec::config::ConfigManager::get_config_names();
        for (const auto& name : names) {
            std::cout << "  - " << name << "\n";
        }
    }

    void print_single_result(const utec::training::TrainingResult& result) {
        std::cout << "=== RESULTADO DEL EXPERIMENTO ===\n";
        std::cout << "Configuracion: " << result.config_name << "\n";
        std::cout << "Precision: " << std::fixed << std::setprecision(2) << result.accuracy << "%\n";
        std::cout << "Muestras correctas: " << result.correct_predictions << " / " << result.total_samples << "\n";
        std::cout << "Tiempo de carga: " << result.load_time_ms << " ms\n";
        std::cout << "Tiempo de entrenamiento: " << result.train_time_ms << " ms\n";
        std::cout << "Tiempo de evaluacion: " << result.eval_time_ms << " ms\n";
        std::cout << "Tiempo total: " << result.total_time_ms << " ms\n";
        std::cout << "========================================\n\n";
    }

    void print_summary() {
        if (results.empty()) {
            std::cout << "No hay resultados para mostrar.\n";
            return;
        }

        std::cout << "\n=== RESUMEN DE EXPERIMENTOS EJECUTADOS ===\n";
        std::cout << "Experimentos realizados: " << results.size() << "\n\n";
        std::cout << std::left;
        std::cout << std::setw(25) << "Configuracion"
                  << std::setw(8) << "Epocas"
                  << std::setw(12) << "Learn Rate"
                  << std::setw(12) << "Precision"
                  << std::setw(15) << "Entrenamiento"
                  << std::setw(12) << "Total(ms)" << "\n";
        std::cout << std::string(84, '-') << "\n";

        float best_accuracy = 0.0f;
        float worst_accuracy = 100.0f;
        std::string best_config;
        std::string worst_config;
        long long total_time = 0;

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            const auto& config = configs_used[i];

            std::stringstream precision_ss, epochs_ss, lr_ss, train_ss, total_ss;
            precision_ss << std::fixed << std::setprecision(2) << result.accuracy << "%";
            epochs_ss << config.epochs;
            lr_ss << std::fixed << std::setprecision(4) << config.learning_rate;
            train_ss << result.train_time_ms << "ms";
            total_ss << result.total_time_ms << "ms";

            std::cout << std::setw(25) << result.config_name
                      << std::setw(8) << epochs_ss.str()
                      << std::setw(12) << lr_ss.str()
                      << std::setw(12) << precision_ss.str()
                      << std::setw(15) << train_ss.str()
                      << std::setw(12) << total_ss.str() << "\n";

            if (result.accuracy > best_accuracy) {
                best_accuracy = result.accuracy;
                best_config = result.config_name;
            }
            if (result.accuracy < worst_accuracy) {
                worst_accuracy = result.accuracy;
                worst_config = result.config_name;
            }
            total_time += result.total_time_ms;
        }

        std::cout << std::string(84, '-') << "\n";
        std::cout << "MEJOR CONFIGURACION: " << best_config << " (" << best_accuracy << "%)\n";
        std::cout << "PEOR CONFIGURACION: " << worst_config << " (" << worst_accuracy << "%)\n";
        std::cout << "TIEMPO TOTAL DE EXPERIMENTACION: " << total_time << " ms\n";
        std::cout << "TIEMPO PROMEDIO POR EXPERIMENTO: " << total_time / results.size() << " ms\n\n";
    }

    void save_results_to_csv() {
        if (results.empty()) {
            std::cout << "No hay resultados para guardar.\n";
            return;
        }

        auto find_next_filename = [](const std::string& base_path, const std::string& fallback_path) -> std::string {
            std::string filename;
            int counter = 1;

            do {
                filename = base_path + "experiment_results_" + std::to_string(counter) + ".csv";
                counter++;
            } while (std::ifstream(filename).good());

            std::ofstream test_file(filename);
            if (!test_file.is_open()) {
                counter = 1;
                do {
                    filename = fallback_path + "experiment_results_" + std::to_string(counter) + ".csv";
                    counter++;
                } while (std::ifstream(filename).good());
            } else {
                test_file.close();
            }

            return filename;
        };

        std::string base_path = "../dataset/results/";
        std::string fallback_path = "./";
        std::string csv_path = find_next_filename(base_path, fallback_path);

        std::ofstream file(csv_path);
        if (!file.is_open()) {
            std::cerr << "No se pudo crear el archivo de resultados en: " << csv_path << "\n";
            return;
        }

        file << "Configuracion,Epocas,Learning_Rate,Precision,Correctas,Total,Tiempo_Carga,Tiempo_Entrenamiento,Tiempo_Evaluacion,Tiempo_Total\n";

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            const auto& config = configs_used[i];

            file << result.config_name << ","
                 << config.epochs << ","
                 << std::fixed << std::setprecision(4) << config.learning_rate << ","
                 << std::fixed << std::setprecision(2) << result.accuracy << ","
                 << result.correct_predictions << ","
                 << result.total_samples << ","
                 << result.load_time_ms << ","
                 << result.train_time_ms << ","
                 << result.eval_time_ms << ","
                 << result.total_time_ms << "\n";
        }

        file.close();
        std::cout << "Resultados guardados en: " << csv_path << "\n";
    }

    std::string resolve_config_name(const std::string& input) {
        try {
            int index = std::stoi(input);
            auto configs = utec::config::ConfigManager::get_all_configs();

            if (index >= 1 && index <= static_cast<int>(configs.size())) {
                return configs[index - 1].name;
            }
        } catch (const std::exception&) {
        }

        return input;
    }
};

int main() {
    try {
        std::string train_path = "../dataset/training/mnist8_train.csv";
        std::string test_path = "../dataset/training/mnist8_test.csv";

        ExperimentRunner runner(train_path, test_path);

        std::cout << "=== SISTEMA DE EXPERIMENTOS DE RED NEURONAL ===\n\n";

        while (true) {
            std::cout << "Selecciona una opcion:\n";
            std::cout << "1. Ver configuraciones disponibles\n";
            std::cout << "2. Ejecutar experimento especifico\n";
            std::cout << "3. Ejecutar todos los experimentos\n";
            std::cout << "4. Ejecutar experimentos seleccionados\n";
            std::cout << "5. Ver resultados actuales\n";
            std::cout << "6. Salir\n";
            std::cout << "Opcion: ";

            int option;
            if (!(std::cin >> option)) {
                std::cin.clear();
                std::cin.ignore(10000, '\n');
                std::cout << "Error: Ingrese un numero valido.\n\n";
                continue;
            }
            std::cin.ignore();

            switch (option) {
                case 1:
                    runner.show_available_configs();
                    break;

                case 2: {
                    std::cout << "\n";
                    runner.show_available_configs();
                    std::cout << "Ingresa el NUMERO (1-8) o el NOMBRE EXACTO de la configuracion: ";
                    std::string config_input;
                    std::getline(std::cin, config_input);

                    if (config_input.empty()) {
                        std::cout << "Error: No se ingresó ninguna configuracion.\n";
                        break;
                    }

                    runner.run_single_experiment(config_input);
                    break;
                }

                case 3:
                    runner.run_all_experiments();
                    break;

                case 4: {
                    std::cout << "\n";
                    runner.show_available_configs();
                    std::cout << "Ingresa NUMEROS (1-8) o NOMBRES EXACTOS separados por comas: ";
                    std::string input;
                    std::getline(std::cin, input);

                    if (input.empty()) {
                        std::cout << "Error: No se ingresó ninguna configuracion.\n";
                        break;
                    }

                    std::vector<std::string> config_inputs;
                    std::stringstream ss(input);
                    std::string config_input;

                    while (std::getline(ss, config_input, ',')) {
                        config_input.erase(0, config_input.find_first_not_of(" \t"));
                        config_input.erase(config_input.find_last_not_of(" \t") + 1);
                        if (!config_input.empty()) {
                            config_inputs.push_back(config_input);
                        }
                    }

                    if (config_inputs.empty()) {
                        std::cout << "Error: No se especificaron configuraciones validas.\n";
                    } else {
                        runner.run_selected_experiments(config_inputs);
                    }
                    break;
                }

                case 5:
                    runner.show_current_results();
                    break;

                case 6:
                    std::cout << "Hasta luego!\n";
                    return 0;

                default:
                    std::cout << "Opcion no valida. Intenta de nuevo.\n";
                    break;
            }
            
            std::cout << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
