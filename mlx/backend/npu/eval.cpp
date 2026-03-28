// Copyright © 2026 Apple Inc. / AMD NPU Backend

#include "mlx/backend/npu/eval.h"
#include "mlx/backend/npu/device.h"
#include "mlx/backend/npu/device_info.h"
#include "mlx/backend/cpu/eval.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::npu {

void init() {
  Device::instance();
}

void new_stream(Stream /* s */) {
  // NPU uses synchronous dispatch via XRT - no stream setup needed.
}

void eval(array& arr) {
  // For the initial implementation, NPU operations that are not yet
  // supported fall back to CPU evaluation. As IRON operator support
  // is added, specific primitives will be intercepted here and
  // dispatched to the NPU via pre-compiled xclbin artifacts.
  //
  // The dispatch strategy:
  // 1. Check if this primitive type has an NPU implementation
  // 2. Check if the shapes match a pre-compiled operator
  // 3. If both match, execute on NPU
  // 4. Otherwise, fall back to CPU

  auto& primitive = arr.primitive();
  auto outputs = arr.outputs();

  // TODO: Add NPU-accelerated dispatch for specific primitives:
  // - Matmul -> IRON GEMM/GEMV
  // - Softmax -> IRON Softmax
  // - RoPE -> IRON RoPE
  // - RMSNorm -> IRON RMSNorm
  // - SiLU -> IRON SiLU
  // - Multi-Head Attention -> IRON MHA/GQA
  //
  // For now, all operations fall back to CPU evaluation.
  // Each supported op will be added incrementally with shape-specific
  // pre-compiled IRON operator artifacts.

  primitive.eval_cpu(arr.inputs(), outputs);
}

void finalize(Stream /* s */) {
  // NPU operations are synchronous - nothing to finalize.
}

void synchronize(Stream /* s */) {
  // NPU operations complete synchronously via XRT run.wait().
  // No additional synchronization needed.
}

} // namespace mlx::core::npu
