// Copyright © 2026 Apple Inc. / AMD NPU Backend

#include "mlx/backend/npu/device_info.h"
#include "mlx/backend/npu/device.h"

namespace mlx::core::npu {

bool is_available() {
  return Device::instance().is_available();
}

int device_count() {
  return Device::instance().device_count();
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  return Device::instance().device_info(device_index);
}

void init() {
  // Device singleton initializes on first access
  Device::instance();
}

void new_stream(Stream /* stream */) {
  // NPU streams are managed through the scheduler's thread pool.
  // No special stream creation needed for XRT - each execution
  // is submitted synchronously via the XRT kernel run API.
}

} // namespace mlx::core::npu
