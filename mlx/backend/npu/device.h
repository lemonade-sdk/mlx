// Copyright © 2026 Apple Inc. / AMD NPU Backend

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/stream.h"

#ifdef MLX_BUILD_NPU
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_xclbin.h>
#endif

namespace mlx::core::npu {

/**
 * Represents a pre-compiled IRON operator artifact loaded on the NPU.
 *
 * Each compiled operator consists of an xclbin (device bitstream) and
 * an instruction binary (.bin) that together define how the NPU executes
 * a specific operation for specific tensor shapes.
 */
struct CompiledOperator {
  std::string name;
  std::string xclbin_path;
  std::string insts_bin_path;

#ifdef MLX_BUILD_NPU
  xrt::xclbin xclbin;
  xrt::hw_context context;
  xrt::kernel kernel;
  std::vector<uint32_t> instructions;
#endif

  // Shape constraints this operator was compiled for
  std::vector<std::vector<int>> input_shapes;
  std::vector<std::vector<int>> output_shapes;
};

/**
 * NPU Device manager - singleton that manages XRT device, contexts, and
 * operator loading/caching.
 */
class Device {
 public:
  static Device& instance();

  bool is_available() const;
  int device_count() const;

  const std::unordered_map<std::string, std::variant<std::string, size_t>>&
  device_info(int device_index = 0) const;

  /**
   * Load a pre-compiled IRON operator from xclbin + instruction binary.
   * Results are cached by xclbin_path.
   */
  CompiledOperator* load_operator(
      const std::string& name,
      const std::string& xclbin_path,
      const std::string& insts_bin_path,
      const std::string& kernel_name = "");

  /**
   * Find a cached operator by name.
   */
  CompiledOperator* find_operator(const std::string& name);

  /**
   * Execute a compiled operator with the given buffer arguments.
   * Buffers are synced to/from device automatically.
   *
   * @param op The compiled operator to execute
   * @param input_ptrs Pointers to input data
   * @param input_sizes Sizes of input data in bytes
   * @param output_ptrs Pointers to output data
   * @param output_sizes Sizes of output data in bytes
   */
  void execute(
      CompiledOperator* op,
      const std::vector<void*>& input_ptrs,
      const std::vector<size_t>& input_sizes,
      const std::vector<void*>& output_ptrs,
      const std::vector<size_t>& output_sizes);

  /**
   * Get the path to the IRON operator artifacts directory.
   * This is where pre-compiled xclbin/bin files are stored.
   */
  const std::string& artifacts_dir() const {
    return artifacts_dir_;
  }

  void set_artifacts_dir(const std::string& dir) {
    artifacts_dir_ = dir;
  }

 private:
  Device();
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  void init_device();

  bool initialized_{false};
  bool available_{false};
  std::string artifacts_dir_;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<CompiledOperator>>
      operator_cache_;

  mutable std::unordered_map<
      std::string,
      std::variant<std::string, size_t>>
      device_info_cache_;

#ifdef MLX_BUILD_NPU
  std::unique_ptr<xrt::device> xrt_device_;
#endif
};

} // namespace mlx::core::npu
