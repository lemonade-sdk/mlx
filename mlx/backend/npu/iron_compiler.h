// Copyright © 2026 Apple Inc. / AMD NPU Backend

#pragma once

#include <string>
#include <vector>

#include "mlx/backend/npu/operator_registry.h"

namespace mlx::core::npu {

/**
 * Interface to the IRON compilation pipeline for on-demand operator
 * compilation. This invokes the IRON Python toolchain to compile
 * operators for specific shapes and dtypes, producing xclbin + bin
 * artifacts that can be loaded by the NPU device.
 *
 * Compilation is expensive (seconds to minutes) so results are cached
 * on disk in the artifacts directory.
 *
 * Requirements:
 * - Python3 with IRON package installed
 * - MLIR-AIE toolchain (aiecc compiler)
 * - XRT runtime libraries
 */
class IronCompiler {
 public:
  static IronCompiler& instance();

  /**
   * Check if the IRON compilation toolchain is available.
   */
  bool is_available() const;

  /**
   * Compile a GEMM operator for the given dimensions.
   * Returns an operator descriptor that can be registered.
   *
   * @param M Number of rows in output
   * @param K Inner dimension
   * @param N Number of columns in output
   * @param dtype Data type (typically bfloat16)
   * @param output_dir Directory to store compiled artifacts
   */
  OperatorDescriptor compile_gemm(
      int M,
      int K,
      int N,
      const std::string& dtype = "bfloat16",
      const std::string& output_dir = "");

  /**
   * Compile a GEMV operator for the given dimensions.
   */
  OperatorDescriptor compile_gemv(
      int M,
      int K,
      const std::string& dtype = "bfloat16",
      const std::string& output_dir = "");

  /**
   * Compile a Softmax operator.
   */
  OperatorDescriptor compile_softmax(
      int rows,
      int cols,
      const std::string& dtype = "bfloat16",
      const std::string& output_dir = "");

  /**
   * Compile a RoPE operator.
   */
  OperatorDescriptor compile_rope(
      int rows,
      int cols,
      const std::string& dtype = "bfloat16",
      const std::string& output_dir = "");

  /**
   * Compile an RMSNorm operator.
   */
  OperatorDescriptor compile_rms_norm(
      int size,
      int tile_size,
      bool weighted = true,
      const std::string& dtype = "bfloat16",
      const std::string& output_dir = "");

  /**
   * Compile a SiLU activation operator.
   */
  OperatorDescriptor compile_silu(
      int size,
      const std::string& dtype = "bfloat16",
      const std::string& output_dir = "");

  /**
   * Compile a Multi-Head Attention operator.
   */
  OperatorDescriptor compile_mha(
      int num_heads,
      int seq_len,
      int head_dim,
      int num_kv_heads = 0,
      const std::string& dtype = "bfloat16",
      const std::string& output_dir = "");

  /**
   * Set the path to the IRON installation.
   */
  void set_iron_path(const std::string& path);

  /**
   * Set the default output directory for compiled artifacts.
   */
  void set_output_dir(const std::string& dir);

 private:
  IronCompiler();

  /**
   * Run a Python script to compile an IRON operator.
   * Returns true if compilation succeeded.
   */
  bool run_compilation(
      const std::string& script,
      const std::string& output_dir);

  std::string iron_path_;
  std::string default_output_dir_;
  bool available_{false};
};

} // namespace mlx::core::npu
