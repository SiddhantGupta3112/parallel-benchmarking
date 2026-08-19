#include "cuda_metrics.h"
#include <nvml.h>

class NvmlSession {
public:
    NvmlSession(const NvmlSession&) = delete;           
    NvmlSession& operator=(const NvmlSession&) = delete;   

private:
    NvmlSession() { nvmlInit(); }
    ~NvmlSession() { nvmlShutdown(); }

    friend NvmlSession& get_nvml_session();
};

NvmlSession& get_nvml_session() {
    static NvmlSession instance;
    return instance;
}

void collect_gpu_utilization_metrics(BenchmarkResult& result) {
    get_nvml_session();

    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS) {
        return;  
    }

    nvmlUtilization_t utilization;
    if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
        result.gpu_utilization_pct = static_cast<double>(utilization.gpu);
    }

}

double read_current_gpu_memory_mb() {
    ensure_nvml_initialized();

    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS) {
        return 0.0;
    }

    nvmlMemory_t memory;
    if (nvmlDeviceGetMemoryInfo(device, &memory) != NVML_SUCCESS) {
        return 0.0;
    }

    return static_cast<double>(memory.used) / (1024.0 * 1024.0);
}

void ensure_nvml_initialized(){
    get_nvml_session();
}