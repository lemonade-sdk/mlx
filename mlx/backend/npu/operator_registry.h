// Copyright © 2026 Apple Inc. / AMD NPU Backend

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/npu/device.h"

namespace mlx::core::npu {

/**
 * Key for looking up a pre-compiled NPU operator.
 * Combines the operation type with input shapes and dtypes to find
 * a matching pre-compiled IRON operator artifact.
 */
struct OperatorKey {
  std::string op_type; // e.g. "matmul", "softmax", "rope"
  std::vector<std::vector<int>> shapes;
  std::vector<Dtype> dtypes;

  bool operator==(const OperatorKey& other) const;
};

struct OperatorKeyHash {
  size_t operator()(const OperatorKey& key) const;
};

/**
 * Descriptor for a pre-compiled IRON operator on disk.
 * Contains paths to the compiled artifacts and metadata about
 * what shapes/dtypes it supports.
 */
struct OperatorDescriptor {
  std::string name;
  std::string op_type;
  std::string xclbin_path;
  std::string insts_bin_path;
  std::string kernel_name;

  // Shape constraints
  std::vector<std::vector<int>> input_shapes;
  std::vector<std::vector<int>> output_shapes;
  std::vector<Dtype> input_dtypes;
  Dtype output_dtype;

  // Whether this operator supports dynamic shapes within bounds
  bool dynamic_shapes{false};
  std::vector<std::vector<int>> min_shapes;
  std::vector<std::vector<int>> max_shapes;
};

/**
 * Registry of pre-compiled IRON operators available for NPU execution.
 *
 * The registry is populated at startup by scanning the artifacts directory
 * for compiled xclbin/bin pairs. Each operator is indexed by its type and
 * shape signature for fast lookup during eval dispatch.
 *
 * Usage flow:
 * 1. At init, scan artifacts_dir for operator manifests
 * 2. During eval, look up (op_type, shapes) -> OperatorDescriptor
 * 3. If found, load the operator (cached) and execute on NPU
 * 4. If not found, fall back to CPU
 */
class OperatorRegistry {
 public:
  static OperatorRegistry& instance();

  /**
   * Register an operator descriptor.
   * Called during initialization or when new operators are compiled.
   */
  void register_operator(const OperatorDescriptor& desc);

  /**
   * Find a matching operator for the given type and shapes.
   * Returns nullptr if no matching operator exists.
   */
  const OperatorDescriptor* find(
      const std::string& op_type,
      const std::vector<std::vector<int>>& shapes,
      const std::vector<Dtype>& dtypes) const;

  /**
   * Scan the artifacts directory and register all found operators.
   * Looks for .json manifest files alongside xclbin/bin pairs.
   */
  void scan_artifacts_dir(const std::string& dir);

  /**
   * Register operators programmatically from IRON Python compilation.
   * This is called when operators are compiled on-demand.
   */
  void register_from_manifest(const std::string& manifest_path);

  /**
   * Check if any operators are registered for the given type.
   */
  bool has_operators_for(const std::string& op_type) const;

  /**
   * Get all registered operator types.
   */
  std::vector<std::string> registered_types() const;

 private:
  OperatorRegistry() = default;

  // Map from op_type to list of descriptors (multiple shape variants)
  std::unordered_map<std::string, std::vector<OperatorDescriptor>> operators_;
  mutable std::mutex mutex_;
};

} // namespace mlx::core::npu
