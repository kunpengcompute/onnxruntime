# Copyright (c) Huawei Technologies Co., Ltd. 2026.
# Licensed under the MIT License.

file(GLOB_RECURSE kdnn_provider_sources CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/kdnn/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/kdnn/*.h")
# Schema registration belongs to graph and must never depend on libkdnn.
list(FILTER kdnn_provider_sources EXCLUDE REGEX "/schema/")
onnxruntime_add_static_library(onnxruntime_providers_kdnn ${kdnn_provider_sources})
onnxruntime_add_include_to_target(onnxruntime_providers_kdnn
  onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB}
  flatbuffers::flatbuffers Boost::mp11 safeint_interface)
target_link_libraries(onnxruntime_providers_kdnn PRIVATE onnxruntime_kdnn)
add_dependencies(onnxruntime_providers_kdnn onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_providers_kdnn PROPERTIES FOLDER "ONNXRuntime" LINKER_LANGUAGE CXX)
# Keep the original two FP32 Sub/Mul operations in the mask adapter.
set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/kdnn/attention/attention_adapter.cc"
  PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")

# Cover the KDNN EP runtime and the schema compiled into onnxruntime_graph.
# Instrument the schema source without changing its owning target.
# LLVM source coverage is opt-in, mirroring the GCC route below
# (onnxruntime_KDNN_GCOV_COVERAGE). Keeping it OFF leaves a plain Clang +
# USE_KDNN build optimized and free of instrumentation.
option(onnxruntime_KDNN_LLVM_COVERAGE
  "Instrument the KDNN EP sources for LLVM source-based coverage (Clang only)" OFF)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND onnxruntime_KDNN_LLVM_COVERAGE)
  set(onnxruntime_kdnn_coverage_sources
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/kdnn_execution_provider.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/kdnn_provider_options.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/kdnn_provider_factory.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/kdnn_kernel_registry.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/fusion/attention_matcher.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/fusion/attention_capability.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/schema/kdnn_defs.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/attention/kdnn_fused_attention.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/attention/attention_adapter.cc"
  )
  foreach(source IN LISTS onnxruntime_kdnn_coverage_sources)
    # Append after the adapter settings above to preserve -fno-fast-math
    # and -ffp-contract=off.
    set_property(SOURCE "${source}" APPEND PROPERTY COMPILE_OPTIONS
      -fprofile-instr-generate -fcoverage-mapping -O0 -g)
  endforeach()
  foreach(target IN ITEMS onnxruntime_providers_kdnn onnxruntime_graph)
    target_link_options(${target} INTERFACE -fprofile-instr-generate)
  endforeach()
  set(onnxruntime_kdnn_coverage_files ${onnxruntime_kdnn_coverage_sources}
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/kdnn_execution_provider.h")
  list(JOIN onnxruntime_kdnn_coverage_files "\n" onnxruntime_kdnn_coverage_manifest)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/kdnn_coverage_sources.txt"
    "${onnxruntime_kdnn_coverage_manifest}\n")
else()
  file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/kdnn_coverage_sources.txt")
endif()

# Coverage/sanitizer instrumentation for the KDNN EP sources (the migrated
# fusion and attention kernels previously living in the optimizer and CPU
# contrib targets).
set(kdnn_ep_instrument_opts "")
if(onnxruntime_KDNN_GCOV_COVERAGE)
  list(APPEND kdnn_ep_instrument_opts "--coverage" "-g1")
endif()
if(onnxruntime_KDNN_ASAN_INSTRUMENT)
  list(APPEND kdnn_ep_instrument_opts "-fsanitize=address" "-fno-omit-frame-pointer" "-DNDEBUG_SANITIZER")
endif()
if(kdnn_ep_instrument_opts)
  set_property(TARGET onnxruntime_providers_kdnn APPEND PROPERTY COMPILE_OPTIONS "${kdnn_ep_instrument_opts}")
endif()

# GCC routes: tools/kdnn/collect_coverage.py reads the same coverage manifest
# as the Clang route, so mirror the manifest (identical file scope) for builds
# that turn on either the gcov or the ASan instrumentation option. UT and fuzz
# collection share this single scope; ASan alone emits no .gcda, so collecting
# a report from an ASan build additionally needs the gcov option or the env
# CFLAGS route.
if(onnxruntime_KDNN_GCOV_COVERAGE OR onnxruntime_KDNN_ASAN_INSTRUMENT)
  set(kdnn_coverage_manifest_files
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/kdnn_execution_provider.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/kdnn_provider_options.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/kdnn_provider_factory.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/kdnn_kernel_registry.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/fusion/attention_matcher.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/fusion/attention_capability.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/schema/kdnn_defs.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/attention/kdnn_fused_attention.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/attention/attention_adapter.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/kdnn/kdnn_execution_provider.h")
  list(JOIN kdnn_coverage_manifest_files "\n" kdnn_coverage_manifest)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/kdnn_coverage_sources.txt"
    "${kdnn_coverage_manifest}\n")
endif()

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_kdnn EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
install(FILES ${KDNN_RUNTIME_LIBRARIES} DESTINATION ${CMAKE_INSTALL_LIBDIR})
