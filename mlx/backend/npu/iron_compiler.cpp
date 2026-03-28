// Copyright © 2026 Apple Inc. / AMD NPU Backend

#include "mlx/backend/npu/iron_compiler.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace mlx::core::npu {

namespace fs = std::filesystem;

IronCompiler& IronCompiler::instance() {
  static IronCompiler compiler;
  return compiler;
}

IronCompiler::IronCompiler() {
  // Check for IRON installation
  const char* iron_env = std::getenv("IRON_PATH");
  if (iron_env) {
    iron_path_ = iron_env;
  }

  // Check for default output directory
  const char* output_env = std::getenv("MLX_NPU_ARTIFACTS_DIR");
  if (output_env) {
    default_output_dir_ = output_env;
  } else {
    // Use home directory cache
    const char* home = std::getenv("HOME");
    if (home) {
      default_output_dir_ = std::string(home) + "/.cache/mlx/npu/operators";
    }
  }

  // Check if IRON is importable
  int ret = std::system("python3 -c 'import iron' 2>/dev/null");
  available_ = (ret == 0);
}

bool IronCompiler::is_available() const {
  return available_;
}

void IronCompiler::set_iron_path(const std::string& path) {
  iron_path_ = path;
}

void IronCompiler::set_output_dir(const std::string& dir) {
  default_output_dir_ = dir;
}

bool IronCompiler::run_compilation(
    const std::string& script,
    const std::string& output_dir) {
  // Ensure output directory exists
  fs::create_directories(output_dir);

  // Write compilation script to temp file
  std::string script_path = output_dir + "/compile_op.py";
  {
    std::ofstream f(script_path);
    f << script;
  }

  // Run compilation
  std::string cmd = "python3 " + script_path + " 2>&1";
  int ret = std::system(cmd.c_str());

  // Clean up script
  fs::remove(script_path);

  return ret == 0;
}

OperatorDescriptor IronCompiler::compile_gemm(
    int M,
    int K,
    int N,
    const std::string& dtype,
    const std::string& output_dir_arg) {
  std::string output_dir =
      output_dir_arg.empty() ? default_output_dir_ : output_dir_arg;
  std::string name =
      "gemm_" + std::to_string(M) + "x" + std::to_string(K) + "x" +
      std::to_string(N) + "_" + dtype;

  std::string xclbin_path = output_dir + "/" + name + ".xclbin";
  std::string insts_path = output_dir + "/" + name + ".bin";

  // Check if already compiled
  if (fs::exists(xclbin_path) && fs::exists(insts_path)) {
    OperatorDescriptor desc;
    desc.name = name;
    desc.op_type = "matmul";
    desc.xclbin_path = xclbin_path;
    desc.insts_bin_path = insts_path;
    desc.input_shapes = {{M, K}, {K, N}};
    desc.output_shapes = {{M, N}};
    desc.input_dtypes = {bfloat16, bfloat16};
    desc.output_dtype = bfloat16;
    return desc;
  }

  // Generate compilation script
  std::ostringstream script;
  script << R"(
import sys
import os
sys.path.insert(0, ')" << iron_path_
         << R"(')
from iron.operators.gemm.op import AIEGEMM
from iron.common import AIEContext

ctx = AIEContext()
ctx.build_dir = ')" << output_dir
         << R"('

op = AIEGEMM(
    M=)" << M << R"(,
    K=)" << K << R"(,
    N=)" << N << R"(,
    context=ctx
)
op.compile()

# Copy artifacts to expected locations
import shutil
build_dir = ctx.build_dir
op_name = op.get_operator_name()
xclbin_src = os.path.join(build_dir, op_name, op_name + '.xclbin')
insts_src = os.path.join(build_dir, op_name, op_name + '.bin')
shutil.copy2(xclbin_src, ')" << xclbin_path
         << R"(')
shutil.copy2(insts_src, ')" << insts_path
         << R"(')
print('Compilation successful')
)";

  if (!run_compilation(script.str(), output_dir)) {
    throw std::runtime_error(
        "[IronCompiler::compile_gemm] Failed to compile GEMM operator for "
        "M=" +
        std::to_string(M) + " K=" + std::to_string(K) +
        " N=" + std::to_string(N));
  }

  OperatorDescriptor desc;
  desc.name = name;
  desc.op_type = "matmul";
  desc.xclbin_path = xclbin_path;
  desc.insts_bin_path = insts_path;
  desc.input_shapes = {{M, K}, {K, N}};
  desc.output_shapes = {{M, N}};
  desc.input_dtypes = {bfloat16, bfloat16};
  desc.output_dtype = bfloat16;
  return desc;
}

OperatorDescriptor IronCompiler::compile_gemv(
    int M,
    int K,
    const std::string& dtype,
    const std::string& output_dir_arg) {
  std::string output_dir =
      output_dir_arg.empty() ? default_output_dir_ : output_dir_arg;
  std::string name =
      "gemv_" + std::to_string(M) + "x" + std::to_string(K) + "_" + dtype;

  OperatorDescriptor desc;
  desc.name = name;
  desc.op_type = "gemv";
  desc.xclbin_path = output_dir + "/" + name + ".xclbin";
  desc.insts_bin_path = output_dir + "/" + name + ".bin";
  desc.input_shapes = {{M, K}, {K}};
  desc.output_shapes = {{M}};
  desc.input_dtypes = {bfloat16, bfloat16};
  desc.output_dtype = bfloat16;

  // TODO: Implement IRON compilation for GEMV
  return desc;
}

OperatorDescriptor IronCompiler::compile_softmax(
    int rows,
    int cols,
    const std::string& dtype,
    const std::string& output_dir_arg) {
  std::string output_dir =
      output_dir_arg.empty() ? default_output_dir_ : output_dir_arg;
  std::string name =
      "softmax_" + std::to_string(rows) + "x" + std::to_string(cols) + "_" +
      dtype;

  OperatorDescriptor desc;
  desc.name = name;
  desc.op_type = "softmax";
  desc.xclbin_path = output_dir + "/" + name + ".xclbin";
  desc.insts_bin_path = output_dir + "/" + name + ".bin";
  desc.input_shapes = {{rows * cols}};
  desc.output_shapes = {{rows * cols}};
  desc.input_dtypes = {bfloat16};
  desc.output_dtype = bfloat16;

  // TODO: Implement IRON compilation for Softmax
  return desc;
}

OperatorDescriptor IronCompiler::compile_rope(
    int rows,
    int cols,
    const std::string& dtype,
    const std::string& output_dir_arg) {
  std::string output_dir =
      output_dir_arg.empty() ? default_output_dir_ : output_dir_arg;
  std::string name =
      "rope_" + std::to_string(rows) + "x" + std::to_string(cols) + "_" +
      dtype;

  OperatorDescriptor desc;
  desc.name = name;
  desc.op_type = "rope";
  desc.xclbin_path = output_dir + "/" + name + ".xclbin";
  desc.insts_bin_path = output_dir + "/" + name + ".bin";
  desc.input_shapes = {{rows, cols}, {rows, cols}};
  desc.output_shapes = {{rows, cols}};
  desc.input_dtypes = {bfloat16, bfloat16};
  desc.output_dtype = bfloat16;

  // TODO: Implement IRON compilation for RoPE
  return desc;
}

OperatorDescriptor IronCompiler::compile_rms_norm(
    int size,
    int tile_size,
    bool weighted,
    const std::string& dtype,
    const std::string& output_dir_arg) {
  std::string output_dir =
      output_dir_arg.empty() ? default_output_dir_ : output_dir_arg;
  std::string name =
      "rms_norm_" + std::to_string(size) + "_tile" +
      std::to_string(tile_size) + (weighted ? "_weighted" : "") + "_" + dtype;

  OperatorDescriptor desc;
  desc.name = name;
  desc.op_type = "rms_norm";
  desc.xclbin_path = output_dir + "/" + name + ".xclbin";
  desc.insts_bin_path = output_dir + "/" + name + ".bin";
  if (weighted) {
    desc.input_shapes = {{size / tile_size, tile_size}, {tile_size}};
  } else {
    desc.input_shapes = {{size / tile_size, tile_size}};
  }
  desc.output_shapes = {{size / tile_size, tile_size}};
  desc.input_dtypes = weighted ? std::vector<Dtype>{bfloat16, bfloat16}
                               : std::vector<Dtype>{bfloat16};
  desc.output_dtype = bfloat16;

  // TODO: Implement IRON compilation for RMSNorm
  return desc;
}

OperatorDescriptor IronCompiler::compile_silu(
    int size,
    const std::string& dtype,
    const std::string& output_dir_arg) {
  std::string output_dir =
      output_dir_arg.empty() ? default_output_dir_ : output_dir_arg;
  std::string name = "silu_" + std::to_string(size) + "_" + dtype;

  OperatorDescriptor desc;
  desc.name = name;
  desc.op_type = "silu";
  desc.xclbin_path = output_dir + "/" + name + ".xclbin";
  desc.insts_bin_path = output_dir + "/" + name + ".bin";
  desc.input_shapes = {{size}};
  desc.output_shapes = {{size}};
  desc.input_dtypes = {bfloat16};
  desc.output_dtype = bfloat16;

  // TODO: Implement IRON compilation for SiLU
  return desc;
}

OperatorDescriptor IronCompiler::compile_mha(
    int num_heads,
    int seq_len,
    int head_dim,
    int num_kv_heads,
    const std::string& dtype,
    const std::string& output_dir_arg) {
  std::string output_dir =
      output_dir_arg.empty() ? default_output_dir_ : output_dir_arg;
  if (num_kv_heads == 0) {
    num_kv_heads = num_heads;
  }
  std::string name =
      "mha_h" + std::to_string(num_heads) + "_s" + std::to_string(seq_len) +
      "_d" + std::to_string(head_dim) + "_kv" +
      std::to_string(num_kv_heads) + "_" + dtype;

  OperatorDescriptor desc;
  desc.name = name;
  desc.op_type = "mha";
  desc.xclbin_path = output_dir + "/" + name + ".xclbin";
  desc.insts_bin_path = output_dir + "/" + name + ".bin";
  // MHA takes Q, K, V and produces O
  int q_size = num_heads * head_dim * seq_len;
  int kv_size = num_kv_heads * head_dim * seq_len;
  desc.input_shapes = {{q_size}, {kv_size}, {kv_size}};
  desc.output_shapes = {{q_size}};
  desc.input_dtypes = {bfloat16, bfloat16, bfloat16};
  desc.output_dtype = bfloat16;

  // TODO: Implement IRON compilation for MHA
  return desc;
}

} // namespace mlx::core::npu
