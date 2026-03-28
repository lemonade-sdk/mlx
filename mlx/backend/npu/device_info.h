// Copyright © 2026 Apple Inc. / AMD NPU Backend

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::npu {

MLX_API bool is_available();

/**
 * Get the number of available NPU devices.
 */
MLX_API int device_count();

/**
 * Get information about an NPU device.
 *
 * Returns a map of device properties:
 *   - device_name (string): Device name (e.g. "AMD XDNA NPU")
 *   - architecture (string): NPU architecture (e.g. "XDNA2", "XDNA3")
 *   - npu_type (string): NPU generation (e.g. "NPU1", "NPU2")
 *   - total_memory (size_t): Total device memory
 */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info(int device_index = 0);

} // namespace mlx::core::npu
