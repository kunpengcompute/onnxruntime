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

#ifndef KDNN_SERVICE_HPP
#define KDNN_SERVICE_HPP

#include <type_traits>
#include <limits>
#include <cstddef>

#include "service/kdnn_api.hpp"
#include "types/kdnn_types.hpp"
#include "types/kdnn_data_type.hpp"
#include "types/kdnn_layout.hpp"
#include "service/kdnn_exception.hpp"

namespace KDNN {

struct TensorInfo;

namespace Service {

constexpr SizeType ALIGNMENT = 128;

KDNN_API SizeType GetIdxForUserLayout(Layout l, SizeType idx) noexcept(false);

KDNN_API Shape GetShapeAccordingToLayout(const Shape &sh, Layout parLayout) noexcept(false);

KDNN_API Shape FlushStrides(const Shape &dims, const Shape &strides) noexcept(false);

KDNN_API SizeType GetRequiredMemSize(const TensorInfo &tensorInfo) noexcept(false);

KDNN_API Layout GetStandardABXLayout(SizeType numDims) noexcept(false);

inline bool IsLayoutBlocked(Layout layout) noexcept
{
    switch (layout) {
        case Layout::BA4b:
        case Layout::BAVLHb:
        case Layout::BAVLWb:
        case Layout::ACB4c:
        case Layout::ACBVLHc:
        case Layout::ACBVLWc:
        case Layout::ABDC4d:
        case Layout::ABDCVLHd:
        case Layout::ABDCVLWd:
        case Layout::ABCED4e:
        case Layout::ABCEDVLHe:
        case Layout::ABCEDVLWe:
            return true;
        default:
            return false;
    }
}

inline Layout GetLayoutBlocked2D(Layout layout) noexcept(false)
{
    switch (layout) {
        case Layout::BA4b:
        case Layout::ACB4c:
        case Layout::ABDC4d:
        case Layout::ABCED4e:
            return Layout::BA4b;
        case Layout::BAVLHb:
        case Layout::ACBVLHc:
        case Layout::ABDCVLHd:
        case Layout::ABCEDVLHe:
            return Layout::BAVLHb;
        case Layout::BAVLWb:
        case Layout::ACBVLWc:
        case Layout::ABDCVLWd:
        case Layout::ABCEDVLWe:
            return Layout::BAVLWb;
        default:
            throw LogicError {"Service: layout is not blocked"};
    }
}

template<typename...>
struct IsAllSame : std::false_type {};

template<>
struct IsAllSame<> : std::true_type {};

template<typename T>
struct IsAllSame<T> : std::true_type {};

template<typename T>
struct IsAllSame<T, T> : std::true_type {};

template<typename T, typename... U>
struct IsAllSame<T, T, U...> : IsAllSame<T, U...> {};

template <typename IntType,
    typename std::enable_if<std::is_integral<typename std::remove_reference<IntType>::type>::value,
    bool>::type = true>
bool WillIntMultOverflow(IntType a, IntType b) noexcept
{
    if ((a == 0) || (b == 0)) {
        return false;
    } else {
        volatile IntType mul = a * b;
        return (mul / a) != b;
    }
}

template <typename IntType, typename ... Args,
    typename std::enable_if<IsAllSame<IntType, Args...>::value, bool>::type = true>
bool WillIntMultOverflow(IntType a, IntType b, Args ... args) noexcept
{
    if (WillIntMultOverflow(a, b)) {
        return true;
    } else {
        return WillIntMultOverflow(a * b, args ...);
    }
}

template <typename ForwardIt,
    typename std::enable_if<std::is_integral<typename std::iterator_traits<ForwardIt>::value_type>::value,
    bool>::type = true>
bool WillIntMultOverflow(ForwardIt begin, ForwardIt end) noexcept
{
    typename std::iterator_traits<ForwardIt>::value_type mult = 1;
    for (auto &&it = begin; it != end; ++it) {
        if (WillIntMultOverflow(mult, *it)) {
            return true;
        }
        mult *= *it;
    }
    return false;
}

// User may use allocation/deallocation functions directly or via allocator class
KDNN_API void *AlignedAlloc(SizeType n, SizeType alignment = ALIGNMENT) noexcept(false);
KDNN_API void Deallocate(void *p, SizeType n = 0) noexcept;
KDNN_API void Deallocate(const void *p, SizeType n = 0) noexcept;

// This class represents an allocator which user may use in order to obtain aligned memory
// and pass it inside KDNN functions. To be compatible with std::allocator all member names are in snake_case
template <typename T, SizeType alignment = ALIGNMENT>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    using size_type = ::KDNN::SizeType;
    using difference_type = std::ptrdiff_t;

    template <typename U> struct rebind {
        using other = AlignedAllocator<U, alignment>;
    };

    T *allocate(SizeType n) const noexcept(false)
    {
        if (n > std::numeric_limits<SizeType>::max() / sizeof(T)) {
            throw BadArrayNewLength();
        }
        return static_cast<T*>(AlignedAlloc(n * sizeof(T), alignment));
    }

    void deallocate(T *p, SizeType n = 0) const noexcept
    {
        ::KDNN::Service::Deallocate(p, n);
    }
    virtual ~AlignedAllocator() = default;
};

template <typename T>
struct Deallocator {
    void operator()(T *p, ::KDNN::SizeType n = 0) const noexcept
    {
        ::KDNN::Service::Deallocate(p, n);
    }
};

template <typename T, SizeType alignment_1, typename U, SizeType alignment_2>
constexpr bool operator == (const AlignedAllocator<T, alignment_1> &,
    const AlignedAllocator<U, alignment_2> &) noexcept
{
    return (alignment_1 == alignment_2) && std::is_same<T, U>::value;
}
template <typename T, SizeType alignment_1, typename U, SizeType alignment_2>
constexpr bool operator != (const AlignedAllocator<T, alignment_1> &lhs,
    const AlignedAllocator<U, alignment_2> &rhs) noexcept
{
    return !(lhs == rhs);
}

} // Service
} // KDNN

#endif // KDNN_SERVICE_HPP
