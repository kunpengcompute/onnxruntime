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

#ifndef KDNN_CONFIG_HPP
#define KDNN_CONFIG_HPP

/// OpenMP runtime (Default)
#define KDNN_RUNTIME_OMP 2u

/// Threadpool runtime
#define KDNN_RUNTIME_THREADPOOL 8u

// KDNN CPU threading runtime
#define KDNN_CPU_THREADING_RUNTIME KDNN_RUNTIME_THREADPOOL

#endif
