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

#ifndef KDNN_EXCEPTION_HPP
#define KDNN_EXCEPTION_HPP

#include <exception>
#include <new>
#include <stdexcept>

#include "service/kdnn_api.hpp"

namespace KDNN {
namespace Service {

// Error handling in KDNN is performed via exceptions.
// KDNN has its own exceptions which are nested from std::exception standard class
struct KDNN_API BadAlloc : public std::bad_alloc {
    const char* what() const noexcept override
    {
        return "KDNN allocation failed";
    }
};

struct KDNN_API BadArrayNewLength : public std::bad_array_new_length {
    const char* what() const noexcept override
    {
        return "KDNN bad array new length";
    }
};

struct KDNN_API LogicError : public std::logic_error {
    explicit LogicError(const std::string& whatArg) : std::logic_error(whatArg) {}
    explicit LogicError(const char* whatArg) : std::logic_error(whatArg) {}
    LogicError(const LogicError& other) noexcept : std::logic_error(other) {}
    LogicError& operator=(const LogicError& other) noexcept = default;
};

struct KDNN_API Unsupported : public std::invalid_argument {
    explicit Unsupported(const std::string& whatArg) : std::invalid_argument(whatArg) {}
    explicit Unsupported(const char* whatArg) : std::invalid_argument(whatArg) {}
    Unsupported(const Unsupported& other) noexcept : std::invalid_argument(other) {}
    Unsupported& operator=(const Unsupported& other) noexcept = default;
};

} // Service
} // KDNN

#endif // KDNN_EXCEPTION_HPP
