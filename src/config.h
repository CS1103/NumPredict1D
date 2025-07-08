#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

namespace utec::config {

    struct TrainingConfig {
        std::string name;
        std::string loss_function;
        std::string optimizer;
        int epochs;
        int batch_size;
        float learning_rate;
        
        TrainingConfig(const std::string& n, const std::string& loss, const std::string& opt,
                      int e, int bs, float lr)
            : name(n), loss_function(loss), optimizer(opt), epochs(e), batch_size(bs), learning_rate(lr) {}
    };

    class ConfigManager {
    public:
        static std::vector<TrainingConfig> get_all_configs() {
            return {
                // Configuración 1: BCELoss + Adam
                TrainingConfig("BCELoss_Adam_Low", "BCELoss", "Adam", 30, 5, 0.1f),
                TrainingConfig("BCELoss_Adam_High", "BCELoss", "Adam", 10, 5, 0.001f),
                
                // Configuración 2: BCELoss + SGD
                TrainingConfig("BCELoss_SGD_Low", "BCELoss", "SGD", 10, 5, 0.001f),
                TrainingConfig("BCELoss_SGD_High", "BCELoss", "SGD", 30, 5, 0.1f),
                
                // Configuración 3: MSELoss + Adam
                TrainingConfig("MSELoss_Adam_Low", "MSELoss", "Adam", 30, 5, 0.1f),
                TrainingConfig("MSELoss_Adam_High", "MSELoss", "Adam", 10, 5, 0.001f),
                
                // Configuración 4: MSELoss + SGD
                TrainingConfig("MSELoss_SGD_Low", "MSELoss", "SGD", 10, 5, 0.001f),
                TrainingConfig("MSELoss_SGD_High", "MSELoss", "SGD", 30, 5, 0.1f)
            };
        }
        
        static TrainingConfig get_config_by_name(const std::string& name) {
            auto configs = get_all_configs();
            for (const auto& config : configs) {
                if (config.name == name) {
                    return config;
                }
            }
            throw std::runtime_error("Configuracion no encontrada: " + name);
        }
        
        static std::vector<std::string> get_config_names() {
            auto configs = get_all_configs();
            std::vector<std::string> names;
            for (const auto& config : configs) {
                names.push_back(config.name);
            }
            return names;
        }
    };

}

#endif // CONFIG_H
