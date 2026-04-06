#pragma once

// TurboQuant: Near-optimal KV cache compression for prompt cache.
// Based on arXiv:2504.19874 (ICLR 2026), Algorithm 1 (TurboQuant_MSE only).
//
// Compresses serialized KV cache state from llama_state_seq_get_data_ext()
// before storing in the prompt cache. Decompresses on cache hit before
// passing to llama_state_seq_set_data_ext().
//
// Self-contained: codebook tables are hardcoded, rotation matrix generated
// at runtime via Gram-Schmidt QR. No external dependencies.

#include <cstdint>
#include <cstddef>
#include <vector>

// Magic number for compressed cache entries: "TQKV"
static constexpr uint32_t TURBOQUANT_MAGIC   = 0x54514B56;
static constexpr uint32_t TURBOQUANT_VERSION = 1;

// Compress a serialized KV state buffer (from llama_state_seq_get_data_ext).
// Returns true on success. On failure (unsupported format, no KV data), returns false
// and dst is unchanged — caller should keep the original uncompressed data.
bool turboquant_compress(
    const uint8_t * src, size_t src_size,
    std::vector<uint8_t> & dst,
    uint8_t bits_k = 3, uint8_t bits_v = 4);

// Decompress a TurboQuant-compressed buffer back to the original format.
// Returns true on success.
bool turboquant_decompress(
    const uint8_t * src, size_t src_size,
    std::vector<uint8_t> & dst);

// Check if a buffer starts with the TurboQuant magic number.
bool turboquant_is_compressed(const uint8_t * data, size_t size);
