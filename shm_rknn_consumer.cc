#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "flow_detection.h"
#include "shm_protocol.h"

namespace {
std::atomic<bool> g_running(true);

void on_signal(int) {
    g_running = false;
}

float max_value(const float* data, int n) {
    if (!data || n <= 0) {
        return 0.0f;
    }
    float max_v = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > max_v) {
            max_v = data[i];
        }
    }
    return max_v;
}

int run_consumer(const char* model_path, const char* shm_path, size_t shm_size) {
    RKNNModel model;
    if (load_rknn_model(&model, model_path) != 0) {
        std::cerr << "[C++] load_rknn_model failed." << std::endl;
        return -1;
    }

    if (model.io_num.n_input > AI_IDS_SHM_MAX_INPUTS) {
        std::cerr << "[C++] model input count exceeds protocol limit: " << model.io_num.n_input
                  << " > " << AI_IDS_SHM_MAX_INPUTS << std::endl;
        release_rknn_model(&model);
        return -1;
    }

    const size_t header_size = sizeof(SharedMemoryHeader);
    if (shm_size <= header_size) {
        std::cerr << "[C++] shm_size too small." << std::endl;
        release_rknn_model(&model);
        return -1;
    }

    int fd = -1;
    while (g_running && fd < 0) {
        fd = open(shm_path, O_RDONLY);
        if (fd < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
    if (fd < 0) {
        std::cerr << "[C++] aborted before shared memory became available." << std::endl;
        release_rknn_model(&model);
        return -1;
    }

    void* mapped = mmap(nullptr, shm_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        std::perror("[C++] mmap failed");
        close(fd);
        release_rknn_model(&model);
        return -1;
    }

    auto* header = reinterpret_cast<const SharedMemoryHeader*>(mapped);
    auto* payload = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(mapped) + header_size);

    std::vector<int> input_sizes(model.io_num.n_input);
    std::vector<std::vector<float>> input_storage(model.io_num.n_input);
    std::vector<const float*> input_ptrs(model.io_num.n_input);

    uint64_t last_seq = 0;
    std::cout << "[C++] waiting for features from " << shm_path << " ..." << std::endl;

    while (g_running) {
        if (header->magic != AI_IDS_SHM_MAGIC || header->version != AI_IDS_SHM_VERSION) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        const uint64_t seq = header->seq;
        if (seq == 0 || seq == last_seq) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        if (header->n_inputs != model.io_num.n_input) {
            std::cerr << "[C++] input count mismatch. shm=" << header->n_inputs
                      << " model=" << model.io_num.n_input << std::endl;
            last_seq = seq;
            continue;
        }

        bool shape_ok = true;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < model.io_num.n_input; ++i) {
            const uint32_t expected = model.input_attrs[i].n_elems;
            const uint32_t got = header->elem_counts[i];
            if (expected != got) {
                std::cerr << "[C++] input[" << i << "] elems mismatch. shm=" << got
                          << " model=" << expected << std::endl;
                shape_ok = false;
                break;
            }
            offset += got;
        }

        if (!shape_ok || offset != header->total_floats) {
            std::cerr << "[C++] invalid payload meta, skip seq=" << seq << std::endl;
            last_seq = seq;
            continue;
        }

        const size_t payload_bytes = static_cast<size_t>(header->total_floats) * sizeof(float);
        if (header_size + payload_bytes > shm_size) {
            std::cerr << "[C++] payload exceeds shared memory size, skip seq=" << seq << std::endl;
            last_seq = seq;
            continue;
        }

        uint32_t float_offset = 0;
        for (uint32_t i = 0; i < model.io_num.n_input; ++i) {
            const int n = static_cast<int>(header->elem_counts[i]);
            input_sizes[i] = n;
            input_storage[i].assign(payload + float_offset, payload + float_offset + n);
            input_ptrs[i] = input_storage[i].data();
            float_offset += header->elem_counts[i];
        }

        float** output_data = nullptr;
        int* output_sizes = nullptr;
        const int ret = rknn_inference(
            &model,
            input_ptrs.data(),
            input_sizes.data(),
            &output_data,
            &output_sizes);

        if (ret != 0) {
            std::cerr << "[C++] rknn_inference failed at seq=" << seq << std::endl;
            last_seq = seq;
            continue;
        }

        std::cout << "[C++] seq=" << seq << " inference ok";
        if (model.io_num.n_output > 0 && output_data && output_sizes && output_sizes[0] > 0) {
            std::cout << ", output0_max=" << max_value(output_data[0], output_sizes[0]);
        }
        std::cout << std::endl;

        if (output_data) {
            for (uint32_t i = 0; i < model.io_num.n_output; ++i) {
                delete[] output_data[i];
            }
            delete[] output_data;
        }
        delete[] output_sizes;

        last_seq = seq;
    }

    munmap(mapped, shm_size);
    close(fd);
    release_rknn_model(&model);
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    const char* model_path = "models/ids_transformer_model_new.rknn";
    const char* shm_path = "/dev/shm/ai_ids_feature_shm";
    size_t shm_size = 2 * 1024 * 1024;

    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        shm_path = argv[2];
    }
    if (argc > 3) {
        shm_size = static_cast<size_t>(strtoull(argv[3], nullptr, 10));
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    std::cout << "[C++] model: " << model_path << std::endl;
    std::cout << "[C++] shm path: " << shm_path << std::endl;
    std::cout << "[C++] shm size: " << shm_size << std::endl;

    return run_consumer(model_path, shm_path, shm_size);
}
