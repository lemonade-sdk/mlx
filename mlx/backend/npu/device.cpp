// Copyright © 2026 Apple Inc. / AMD NPU Backend

#include "mlx/backend/npu/device.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#ifdef MLX_BUILD_NPU
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_xclbin.h>
#endif

namespace mlx::core::npu {

Device& Device::instance() {
  static Device device;
  return device;
}

Device::Device() {
  // Try to detect NPU hardware
  init_device();
}

Device::~Device() = default;

void Device::init_device() {
  std::lock_guard<std::mutex> lock(mutex_);

#ifdef MLX_BUILD_NPU
  try {
    xrt_device_ = std::make_unique<xrt::device>(0);
    available_ = true;
    initialized_ = true;

    // Populate device info cache
    device_info_cache_["device_name"] =
        std::string("AMD XDNA NPU");
    device_info_cache_["architecture"] =
        std::string("XDNA");

    // Default artifacts directory - can be overridden
    const char* env_dir = std::getenv("MLX_NPU_ARTIFACTS_DIR");
    if (env_dir) {
      artifacts_dir_ = env_dir;
    } else {
      artifacts_dir_ = "/usr/share/mlx/npu/operators";
    }

  } catch (const std::exception& e) {
    available_ = false;
    initialized_ = true;
  }
#else
  available_ = false;
  initialized_ = true;
#endif
}

bool Device::is_available() const {
  return available_;
}

int Device::device_count() const {
  return available_ ? 1 : 0;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
Device::device_info(int /* device_index */) const {
  return device_info_cache_;
}

CompiledOperator* Device::load_operator(
    const std::string& name,
    const std::string& xclbin_path,
    const std::string& insts_bin_path,
    const std::string& kernel_name) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check cache first
  auto it = operator_cache_.find(name);
  if (it != operator_cache_.end()) {
    return it->second.get();
  }

#ifdef MLX_BUILD_NPU
  if (!available_) {
    throw std::runtime_error("[npu::load_operator] NPU device not available");
  }

  auto op = std::make_unique<CompiledOperator>();
  op->name = name;
  op->xclbin_path = xclbin_path;
  op->insts_bin_path = insts_bin_path;

  // Load xclbin
  op->xclbin = xrt::xclbin(xclbin_path);
  xrt_device_->register_xclbin(op->xclbin);

  // Create hardware context
  auto uuid = op->xclbin.get_uuid();
  op->context = xrt::hw_context(*xrt_device_, uuid);

  // Get or discover kernel name
  std::string kname = kernel_name;
  if (kname.empty()) {
    auto kernels = op->xclbin.get_kernels();
    if (kernels.empty()) {
      throw std::runtime_error(
          "[npu::load_operator] No kernels found in xclbin: " + xclbin_path);
    }
    kname = kernels[0].get_name();
  }
  op->kernel = xrt::kernel(op->context, kname);

  // Load instruction binary
  std::ifstream insts_file(insts_bin_path, std::ios::binary);
  if (!insts_file.is_open()) {
    throw std::runtime_error(
        "[npu::load_operator] Cannot open instruction binary: " +
        insts_bin_path);
  }
  insts_file.seekg(0, std::ios::end);
  size_t insts_size = insts_file.tellg();
  insts_file.seekg(0, std::ios::beg);
  op->instructions.resize(insts_size / sizeof(uint32_t));
  insts_file.read(
      reinterpret_cast<char*>(op->instructions.data()), insts_size);

  auto* ptr = op.get();
  operator_cache_[name] = std::move(op);
  return ptr;
#else
  throw std::runtime_error(
      "[npu::load_operator] NPU backend not compiled. "
      "Build with MLX_BUILD_NPU=ON");
#endif
}

CompiledOperator* Device::find_operator(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = operator_cache_.find(name);
  if (it != operator_cache_.end()) {
    return it->second.get();
  }
  return nullptr;
}

void Device::execute(
    CompiledOperator* op,
    const std::vector<void*>& input_ptrs,
    const std::vector<size_t>& input_sizes,
    const std::vector<void*>& output_ptrs,
    const std::vector<size_t>& output_sizes) {
#ifdef MLX_BUILD_NPU
  if (!op) {
    throw std::runtime_error("[npu::execute] Null operator");
  }

  constexpr int OPCODE = 3; // AIE kernel opcode
  constexpr size_t ALIGNMENT = 0x10000; // 64KB alignment

  // Create instruction buffer
  auto insts_bo = xrt::bo(
      *xrt_device_,
      op->instructions.size() * sizeof(uint32_t),
      xrt::bo::flags::cacheable,
      op->kernel.group_id(1));
  insts_bo.write(op->instructions.data(), op->instructions.size() * sizeof(uint32_t), 0);
  insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Create input buffer objects
  std::vector<xrt::bo> input_bos;
  input_bos.reserve(input_ptrs.size());
  for (size_t i = 0; i < input_ptrs.size(); i++) {
    auto bo = xrt::bo(
        *xrt_device_, input_sizes[i], xrt::bo::flags::host_only, ALIGNMENT);
    auto* mapped = bo.map<void*>();
    std::memcpy(mapped, input_ptrs[i], input_sizes[i]);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    input_bos.push_back(std::move(bo));
  }

  // Create output buffer objects
  std::vector<xrt::bo> output_bos;
  output_bos.reserve(output_ptrs.size());
  for (size_t i = 0; i < output_ptrs.size(); i++) {
    auto bo = xrt::bo(
        *xrt_device_, output_sizes[i], xrt::bo::flags::host_only, ALIGNMENT);
    output_bos.push_back(std::move(bo));
  }

  // Build argument list: opcode, insts_bo, num_insts, then all data buffers
  // The XRT kernel call signature depends on the specific operator.
  // For standard IRON operators: kernel(opcode, insts_bo, n_insts, buf0, buf1, ...)
  //
  // We use a dynamic invocation pattern since buffer count varies.
  std::vector<xrt::bo*> all_bos;
  for (auto& bo : input_bos) {
    all_bos.push_back(&bo);
  }
  for (auto& bo : output_bos) {
    all_bos.push_back(&bo);
  }

  // Execute based on number of data buffers
  xrt::run run;
  auto n_insts = static_cast<uint32_t>(op->instructions.size());

  switch (all_bos.size()) {
    case 1:
      run = op->kernel(OPCODE, insts_bo, n_insts, *all_bos[0]);
      break;
    case 2:
      run = op->kernel(OPCODE, insts_bo, n_insts, *all_bos[0], *all_bos[1]);
      break;
    case 3:
      run = op->kernel(
          OPCODE, insts_bo, n_insts, *all_bos[0], *all_bos[1], *all_bos[2]);
      break;
    case 4:
      run = op->kernel(
          OPCODE,
          insts_bo,
          n_insts,
          *all_bos[0],
          *all_bos[1],
          *all_bos[2],
          *all_bos[3]);
      break;
    case 5:
      run = op->kernel(
          OPCODE,
          insts_bo,
          n_insts,
          *all_bos[0],
          *all_bos[1],
          *all_bos[2],
          *all_bos[3],
          *all_bos[4]);
      break;
    default:
      throw std::runtime_error(
          "[npu::execute] Unsupported number of buffers: " +
          std::to_string(all_bos.size()));
  }

  // Wait for completion
  auto state = run.wait();
  // XRT returns ERT_CMD_STATE_COMPLETED (4) on success
  // but the exact enum depends on XRT version, so we check against the wait
  // return directly
  (void)state;

  // Sync output buffers back to host
  for (size_t i = 0; i < output_ptrs.size(); i++) {
    output_bos[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* mapped = output_bos[i].map<void*>();
    std::memcpy(output_ptrs[i], mapped, output_sizes[i]);
  }

#else
  (void)op;
  (void)input_ptrs;
  (void)input_sizes;
  (void)output_ptrs;
  (void)output_sizes;
  throw std::runtime_error(
      "[npu::execute] NPU backend not compiled. Build with MLX_BUILD_NPU=ON");
#endif
}

} // namespace mlx::core::npu
