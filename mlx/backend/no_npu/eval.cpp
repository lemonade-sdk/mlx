// Copyright © 2026 Apple Inc. / AMD NPU Backend

#include <stdexcept>

#include "mlx/backend/npu/device_info.h"
#include "mlx/backend/npu/eval.h"

namespace mlx::core::npu {

void init() {}

void new_stream(Stream) {}

void eval(array&) {
  throw std::runtime_error("[npu::eval] NPU backend is not available");
}

void finalize(Stream) {
  throw std::runtime_error("[npu::finalize] NPU backend is not available");
}

void synchronize(Stream) {
  throw std::runtime_error("[npu::synchronize] NPU backend is not available");
}

} // namespace mlx::core::npu
