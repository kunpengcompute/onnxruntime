// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "OnnxPrediction.h"
#include "onnxruntime_session_options_config_keys.h"

#include <vector>

namespace onnxruntime::fuzzing {
Ort::SessionOptions CreateSessionOptions(bool is_ort_format) {
  Ort::SessionOptions options;
  // Each fuzz input creates a Session. Bound worker creation so the campaign
  // spends its time executing models, including on machines with many cores.
  options.SetIntraOpNumThreads(1);
  options.SetInterOpNumThreads(1);
  if (is_ort_format) {
    options.AddConfigEntry(kOrtSessionOptionsConfigLoadModelFormat, "ORT");
  }
#ifdef USE_KDNN_EP
  // KDNN claims supported MHA subgraphs; ORT supplies CPU fallback for other nodes.
  // Empty provider options preserve the KDNN environment overrides and defaults.
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (is_ort_format) {
    // ORT-format runtime partitioning requires EP Compile for new subgraphs.
    // KDNN uses static kernels, so only claim existing fused nodes here.
    options.AppendExecutionProvider("KDNN", {{"enable_attention_fusion", "0"}});
  } else {
    options.AppendExecutionProvider("KDNN", {});
  }
#endif
  return options;
}
}  // namespace onnxruntime::fuzzing

// Uses the onnxruntime to load the model
// into a session.
//
#if defined(_WIN32) || defined(_WIN64)
OnnxPrediction::OnnxPrediction(std::wstring& onnx_model_file, Ort::Env& env)
    : raw_model{nullptr},
      ptr_session{std::make_unique<Ort::Session>(env, onnx_model_file.c_str(), session_options)},
      input_names(ptr_session->GetInputCount()),
      output_names(ptr_session->GetOutputCount()) {
  init();
}
#else
OnnxPrediction::OnnxPrediction(std::wstring& onnx_model_file, Ort::Env& env)
    : raw_model{nullptr},
      // Convert std::wstring to std::string
      ptr_session{std::make_unique<Ort::Session>(env, wstring_to_string(onnx_model_file).c_str(), session_options)},
      input_names(ptr_session->GetInputCount()),
      output_names(ptr_session->GetOutputCount()) {
  init();
}
#endif

// Uses the onnx to seri
//
OnnxPrediction::OnnxPrediction(onnx::ModelProto& onnx_model, Ort::Env& env) {
  raw_model = std::shared_ptr<void>(alloc.Alloc(onnx_model.ByteSizeLong()), [this](void* ptr) {
    this->GetAllocator().Free(ptr);
  });

  onnx_model.SerializeToArray(raw_model.get(), static_cast<int>(onnx_model.ByteSizeLong()));

  ptr_session = std::make_unique<Ort::Session>(env,
                                               static_cast<void*>(raw_model.get()),
                                               onnx_model.ByteSizeLong(),
                                               session_options);

  // TODO Use reserve instead of resize
  input_names.resize(ptr_session->GetInputCount());
  output_names.resize(ptr_session->GetOutputCount());

  init();
}

OnnxPrediction::OnnxPrediction(const std::vector<char>& model_data, Ort::Env& env)
    : session_options(onnxruntime::fuzzing::CreateSessionOptions(true)) {
  ptr_session = std::make_unique<Ort::Session>(env,
                                               model_data.data(),
                                               model_data.size(),
                                               session_options);

  // TODO Use reserve instead of resize
  input_names.resize(ptr_session->GetInputCount());
  output_names.resize(ptr_session->GetOutputCount());

  init();
}

// OnnxPrediction console output format
// prints the output data.
//
std::wostream& operator<<(std::wostream& out, OnnxPrediction& pred) {
  auto pretty_print = [&out](auto ptr, Ort::Value& val) {
    out << L"[";
    std::wstring msg = L"";
    for (int i = 0; i < static_cast<int>(val.GetTensorTypeAndShapeInfo().GetElementCount()); i++) {
      out << msg << ptr[i];
      msg = L", ";
    }
    out << L"]\n";
  };

  size_t index{0};
  for (auto& val : pred.output_values) {
    out << pred.output_names[index++] << L" = ";
    pred.ProcessOutputData(pretty_print, val);
  }

  return out;
}

// Used to Generate data for predict
//
void GenerateDataForInputTypeTensor(OnnxPrediction& predict,
                                    size_t input_index, const std::string& input_name,
                                    ONNXTensorElementDataType elem_type, size_t elem_count, size_t seed) {
  (void)input_name;
  (void)input_index;

  if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    auto raw_data = GenerateRandomData(0.0f, elem_count, seed);
    predict << std::move(raw_data);
  } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    int32_t initial = 0;
    auto raw_data = GenerateRandomData(initial, elem_count, seed);
    predict << std::move(raw_data);
  } else {
    throw std::runtime_error("only floats are implemented");
  }
}

// Run the prediction
//
void OnnxPrediction::RunInference() {
  Logger::testLog << L"inference starting " << Logger::endl;
  Logger::testLog.flush();

  try {
    ptr_session->Run(run_options,
                     input_names.data(), input_values.data(),
                     input_values.size(), output_names.data(), output_values.data(),
                     output_names.size());
  } catch (...) {
    Logger::testLog << L"Something went wrong in inference " << Logger::endl;
    Logger::testLog.flush();
    throw;
  }

  Logger::testLog << L"inference completed " << Logger::endl;
  Logger::testLog.flush();
}

// Print the output values of the prediction.
//
void OnnxPrediction::PrintOutputValues() {
  Logger::testLog << L"output data:\n";
  Logger::testLog << *this;
  Logger::testLog << Logger::endl;
}

// Common initilization amongst constructors
//
void OnnxPrediction::init() {
  // Initialize model input names
  //
  for (int i = 0; i < static_cast<int>(ptr_session->GetInputCount()); i++) {
    // TODO Use push_back on input_names instead of assignment
    input_names_ptrs.push_back(ptr_session->GetInputNameAllocated(i, alloc));
    input_names[i] = input_names_ptrs.back().get();
    input_values.emplace_back(nullptr);
  }

  // Initialize model output names
  //
  for (int i = 0; i < static_cast<int>(ptr_session->GetOutputCount()); i++) {
    // TODO Use push_back on output_names instead of assignment
    output_names_ptrs.push_back(ptr_session->GetOutputNameAllocated(i, alloc));
    output_names[i] = output_names_ptrs.back().get();
    output_values.emplace_back(nullptr);
  }
}

// Get the allocator used by the runtime
//
Ort::AllocatorWithDefaultOptions& OnnxPrediction::GetAllocator() {
  return alloc;
}

Ort::TensorTypeAndShapeInfo OnnxPrediction::GetConcreteInputTensorInfo(size_t input_index) const {
  auto input_type = ptr_session->GetInputTypeInfo(input_index);
  if (input_type.GetONNXType() != ONNX_TYPE_TENSOR) {
    throw std::runtime_error("Fuzz input must be a tensor");
  }

  auto tensor_info = input_type.GetTensorTypeAndShapeInfo();
  if (!tensor_info.HasShape()) {
    throw std::runtime_error("Fuzz input must have a known rank");
  }

  auto shape = tensor_info.GetShape();
  // Use the same deterministic value for symbolic and anonymous dimensions,
  // including shared symbols across inputs. Preserve concrete and zero dimensions.
  for (auto& dimension : shape) {
    if (dimension < 0) dimension = 1;
  }
  return Ort::TensorTypeAndShapeInfo(tensor_info.GetElementType(), shape);
}

void OnnxPrediction::SetupInput(
    InputGeneratorFunctionType GenerateData,
    size_t seed) {
  for (int i = 0; i < static_cast<int>(ptr_session->GetInputCount()); i++) {
    auto tensor_info = GetConcreteInputTensorInfo(i);
    auto elem_type = tensor_info.GetElementType();
    // Keep ORT's element-count overflow checks after resolving dynamic dimensions.
    auto elem_count = tensor_info.GetElementCount();

    // This can be any generic function to generate inputs.
    GenerateData(*this, i, std::string(input_names[i]), elem_type, elem_count, seed);

    // Update the seed in a predictable way to get other values for different inputs.
    seed++;
  }
  Logger::testLog << Logger::endl;
}
