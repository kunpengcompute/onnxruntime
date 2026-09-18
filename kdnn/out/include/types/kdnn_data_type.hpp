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

#ifndef KDNN_DATATYPE_HPP
#define KDNN_DATATYPE_HPP

#include <cstddef>
#include <cstdint>

#include "service/kdnn_api.hpp"
#include "service/kdnn_exception.hpp"

namespace KDNN {

using SizeType = std::size_t;
using IntType = std::int64_t;

namespace Element {

// enumeration of KDNN supported data types
enum class TypeT {
    UNDEFINED = 0,
    F32 = 0x1,
    F16 = 0x2,
    BF16 = 0x4,
    S32 = 0x8,
    S8 = 0x10,
    U8 = 0x20,
    S64 = 0x40
};

// This class is designed to provide runtime info about data type.
struct KDNN_API Type final {
public:
    using SizeType = ::KDNN::SizeType;
    Type(const TypeT& t) noexcept(false) : type{t}, size(0)
    {
        switch (type) {
            case TypeT::F32: {
                size = sizeof(float);
                break;
            }
            case TypeT::F16: {
                size = sizeof(__fp16);
                break;
            }
            case TypeT::BF16: {
                size = sizeof(__bf16);
                break;
            }
            case TypeT::S32: {
                size = sizeof(std::int32_t);
                break;
            }
            case TypeT::S8: {
                size = sizeof(std::int8_t);
                break;
            }
            case TypeT::U8: {
                size = sizeof(std::uint8_t);
                break;
            }
            case TypeT::S64: {
                size = sizeof(std::int64_t);
                break;
            }
            default: {}
        }
        if (type == TypeT::UNDEFINED) {
            throw Service::LogicError {"Type: unsupported data type"};
        }
    }
    SizeType GetSize() const noexcept
    {
        return size;
    }
    SizeType GetBitwidth() const noexcept
    {
        return (GetSize() * 8uLL);
    }
    operator TypeT() const noexcept
    {
        return type;
    }
    bool IsFP() const noexcept
    {
        if ((type == TypeT::F32) ||
            (type == TypeT::F16) ||
            (type == TypeT::BF16)) {
            return true;
        } else {
            return false;
        }
    }
    bool IsIntegral() const noexcept
    {
        return !IsFP();
    }
    bool IsSigned() const noexcept
    {
        if (type == TypeT::U8) {
            return false;
        } else {
            return true;
        }
    }
    bool IsUnsigned() const noexcept
    {
        return !IsSigned();
    }
private:
    TypeT type;
    SizeType size;
};

template <typename ElementType>
inline TypeT MatchType()
{
    return TypeT::UNDEFINED;
}

template <>
inline TypeT MatchType<float>()
{
    return TypeT::F32;
}

template <>
inline TypeT MatchType<__fp16>()
{
    return TypeT::F16;
}

template <>
inline TypeT MatchType<__bf16>()
{
    return TypeT::BF16;
}

template <>
inline TypeT MatchType<std::int32_t>()
{
    return TypeT::S32;
}

template <>
inline TypeT MatchType<std::int8_t>()
{
    return TypeT::S8;
}

template <>
inline TypeT MatchType<std::uint8_t>()
{
    return TypeT::U8;
}

inline bool operator==(const Type &lhs, const TypeT &rhs) noexcept
{
    return (static_cast<TypeT>(lhs) == rhs);
}

inline bool operator==(const TypeT &lhs, const Type &rhs) noexcept
{
    return (lhs == static_cast<TypeT>(rhs));
}

inline bool operator==(const Type &lhs, const Type &rhs) noexcept
{
    return (lhs.GetSize() == rhs.GetSize()) &&
           (static_cast<TypeT>(lhs) == (static_cast<TypeT>(rhs)));
}

inline bool operator!=(const Type &lhs, const TypeT &rhs) noexcept
{
    return !(lhs == rhs);
}
inline bool operator!=(const TypeT &lhs, const Type &rhs) noexcept
{
    return !(lhs == rhs);
}
inline bool operator!=(const Type &lhs, const Type &rhs) noexcept
{
    return !(lhs == rhs);
}

} // Element
} // KDNN

#endif // KDNN_DATATYPE_HPP
