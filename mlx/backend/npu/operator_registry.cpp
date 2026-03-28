// Copyright © 2026 Apple Inc. / AMD NPU Backend

#include "mlx/backend/npu/operator_registry.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace mlx::core::npu {

namespace fs = std::filesystem;

bool OperatorKey::operator==(const OperatorKey& other) const {
  return op_type == other.op_type && shapes == other.shapes &&
      dtypes == other.dtypes;
}

size_t OperatorKeyHash::operator()(const OperatorKey& key) const {
  size_t seed = std::hash<std::string>{}(key.op_type);
  for (const auto& shape : key.shapes) {
    for (int dim : shape) {
      seed ^= std::hash<int>{}(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
  }
  for (const auto& dtype : key.dtypes) {
    seed ^= std::hash<int>{}(static_cast<int>(dtype.val())) + 0x9e3779b9 +
        (seed << 6) + (seed >> 2);
  }
  return seed;
}

OperatorRegistry& OperatorRegistry::instance() {
  static OperatorRegistry registry;
  return registry;
}

void OperatorRegistry::register_operator(const OperatorDescriptor& desc) {
  std::lock_guard<std::mutex> lock(mutex_);
  operators_[desc.op_type].push_back(desc);
}

const OperatorDescriptor* OperatorRegistry::find(
    const std::string& op_type,
    const std::vector<std::vector<int>>& shapes,
    const std::vector<Dtype>& dtypes) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = operators_.find(op_type);
  if (it == operators_.end()) {
    return nullptr;
  }

  for (const auto& desc : it->second) {
    // Check dtype match
    if (desc.input_dtypes.size() != dtypes.size()) {
      continue;
    }
    bool dtype_match = true;
    for (size_t i = 0; i < dtypes.size(); i++) {
      if (desc.input_dtypes[i] != dtypes[i]) {
        dtype_match = false;
        break;
      }
    }
    if (!dtype_match) {
      continue;
    }

    // Check shape match
    if (desc.dynamic_shapes) {
      // Check within min/max bounds
      if (desc.min_shapes.size() != shapes.size() ||
          desc.max_shapes.size() != shapes.size()) {
        continue;
      }
      bool in_bounds = true;
      for (size_t i = 0; i < shapes.size() && in_bounds; i++) {
        if (shapes[i].size() != desc.min_shapes[i].size()) {
          in_bounds = false;
          break;
        }
        for (size_t j = 0; j < shapes[i].size(); j++) {
          if (shapes[i][j] < desc.min_shapes[i][j] ||
              shapes[i][j] > desc.max_shapes[i][j]) {
            in_bounds = false;
            break;
          }
        }
      }
      if (in_bounds) {
        return &desc;
      }
    } else {
      // Exact shape match
      if (desc.input_shapes == shapes) {
        return &desc;
      }
    }
  }

  return nullptr;
}

void OperatorRegistry::scan_artifacts_dir(const std::string& dir) {
  if (!fs::exists(dir)) {
    return;
  }

  // Scan for operator manifest files (JSON)
  // Each manifest describes a pre-compiled operator:
  // {
  //   "name": "gemm_256x64x512_bf16",
  //   "op_type": "matmul",
  //   "xclbin": "gemm_256x64x512_bf16.xclbin",
  //   "insts_bin": "gemm_256x64x512_bf16.bin",
  //   "kernel_name": "MLIR_AIE",
  //   "input_shapes": [[256, 64], [64, 512]],
  //   "output_shapes": [[256, 512]],
  //   "input_dtypes": ["bfloat16", "bfloat16"],
  //   "output_dtype": "bfloat16"
  // }
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".json") {
      register_from_manifest(entry.path().string());
    }
  }
}

void OperatorRegistry::register_from_manifest(
    const std::string& manifest_path) {
  // Simple JSON-like manifest parsing
  // In a production implementation, use nlohmann/json which is already
  // a dependency of MLX. For now, we register operators programmatically.
  //
  // TODO: Implement manifest parsing with nlohmann/json
  (void)manifest_path;
}

bool OperatorRegistry::has_operators_for(const std::string& op_type) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return operators_.find(op_type) != operators_.end();
}

std::vector<std::string> OperatorRegistry::registered_types() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> types;
  types.reserve(operators_.size());
  for (const auto& [type, _] : operators_) {
    types.push_back(type);
  }
  return types;
}

} // namespace mlx::core::npu
