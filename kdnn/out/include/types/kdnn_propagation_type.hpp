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

#ifndef KDNN_PROPAGATION_TYPE_HPP
#define KDNN_PROPAGATION_TYPE_HPP

namespace KDNN {

// Enumeration of common propagation types which are supported in KDNN.
enum class Propagation {
    UNDEFINED = 0,
    FORWARD_TRAINING,
    FORWARD_INFERENCE,
    BACKWARD_DATA,
    BACKWARD_WEIGHTS,
    BACKWARD_BIAS,
    BACKWARD,
    FORWARD = FORWARD_TRAINING,
};

} // KDNN

#endif // KDNN_PROPAGATION_TYPE_HPP
