#pragma once

#include <stdint.h>

// Python/C++ shared memory protocol for AI-IDS feature tensors.
// Layout: [SharedMemoryHeader][float32 payload...]

static const uint32_t AI_IDS_SHM_MAGIC = 0x314D4853;  // "SHM1" little-endian
static const uint32_t AI_IDS_SHM_VERSION = 1;
static const uint32_t AI_IDS_SHM_MAX_INPUTS = 16;

#pragma pack(push, 1)
struct SharedMemoryHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t seq;
    uint32_t n_inputs;
    uint32_t total_floats;
    uint32_t elem_counts[AI_IDS_SHM_MAX_INPUTS];
};
#pragma pack(pop)
