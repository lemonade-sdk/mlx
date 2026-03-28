# NPU (AMD XDNA) backend

target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/operator_registry.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/iron_compiler.cpp)

if(MLX_BUILD_NPU)
  # Find XRT (Xilinx Runtime) for NPU support
  find_path(
    XRT_INCLUDE_DIR
    NAMES xrt/xrt_device.h xrt.h
    PATHS /opt/xilinx/xrt/include /usr/include /usr/local/include
    PATH_SUFFIXES xrt)

  find_library(
    XRT_COREUTIL_LIB
    NAMES xrt_coreutil
    PATHS /opt/xilinx/xrt/lib /usr/lib /usr/lib64 /usr/local/lib)

  if(XRT_INCLUDE_DIR AND XRT_COREUTIL_LIB)
    message(STATUS "Found XRT: ${XRT_INCLUDE_DIR}")
    message(STATUS "Found XRT lib: ${XRT_COREUTIL_LIB}")
    target_include_directories(mlx PRIVATE ${XRT_INCLUDE_DIR})
    target_link_libraries(mlx PRIVATE ${XRT_COREUTIL_LIB})
    target_compile_definitions(mlx PRIVATE MLX_BUILD_NPU)
  else()
    message(
      WARNING
        "XRT (Xilinx Runtime) not found. NPU backend will compile "
        "without hardware support. Install XRT for AMD XDNA NPU "
        "acceleration.")
    # Still compile the NPU backend code but without XRT support.
    # The backend will report is_available() = false at runtime.
  endif()
endif()
