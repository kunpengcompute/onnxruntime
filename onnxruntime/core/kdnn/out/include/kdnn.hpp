/*
   Copyright 2025 Huawei Technologies Co., Ltd.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

#ifndef KDNN_HPP
#define KDNN_HPP

#include "kdnn_config.h"
#include "service/kdnn_api.hpp"
#include "types/kdnn_data_type.hpp"
#include "types/kdnn_layout.hpp"
#include "types/kdnn_propagation_type.hpp"
#include "types/kdnn_shape.hpp"
#include "types/kdnn_types.hpp"
#include "types/kdnn_primitive_kind.hpp"

#include "service/kdnn_err_codes.hpp"
#include "service/kdnn_exception.hpp"
#include "service/kdnn_primitive_cache.hpp"
#include "service/kdnn_service.hpp"
#include "service/kdnn_threading.hpp"

#include "operations/kdnn_attributes.hpp"
#include "operations/kdnn_post_ops.hpp"
#include "operations/kdnn_batch_normalization.hpp"
#include "operations/kdnn_binary.hpp"
#include "operations/kdnn_convolution.hpp"
#include "operations/kdnn_deconvolution.hpp"
#include "operations/kdnn_eltwise.hpp"
#include "operations/kdnn_gemm.hpp"
#include "operations/kdnn_spblas.hpp"
#include "operations/kdnn_group_normalization.hpp"
#include "operations/kdnn_inner_product.hpp"
#include "operations/kdnn_layer_normalization.hpp"
#include "operations/kdnn_lrn.hpp"
#include "operations/kdnn_multihead_attention.hpp"
#include "operations/kdnn_pooling.hpp"
#include "operations/kdnn_prelu.hpp"
#include "operations/kdnn_reduction.hpp"
#include "operations/kdnn_reorder.hpp"
#include "operations/kdnn_shuffle.hpp"
#include "operations/kdnn_softmax.hpp"
#include "operations/kdnn_sum.hpp"
#include "operations/kdnn_resampling.hpp"
#include "operations/kdnn_concat.hpp"
#include "operations/kdnn_rnn.hpp"

#endif // KDNN_HPP
