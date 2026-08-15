#include "cuda_metrics.h"

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

    nvmlMemory_t memory;
    if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
        result.gpu_memory_used_mb = static_cast<double>(memory.used) / (1024.0 * 1024.0);
    }
}