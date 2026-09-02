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

#ifndef KDNN_SHAPE_HPP
#define KDNN_SHAPE_HPP

#include <array>
#include <initializer_list>

#include "service/kdnn_api.hpp"
#include "service/kdnn_service.hpp"
#include "types/kdnn_types.hpp"
#include "service/kdnn_exception.hpp"

namespace KDNN {

constexpr SizeType MAX_DIMS = 5;

// This class designed to specify multidimensional data shape
struct KDNN_API Shape final {
    using SizeType = ::KDNN::SizeType;
    static constexpr SizeType NUM_MAX_DIMENSIONS = MAX_DIMS;

    Shape() noexcept : dimsArray{}, numDims{0}
    {
        for (SizeType i = 0; i < NUM_MAX_DIMENSIONS; ++i) {
            dimsArray[i] = 0;
        }
    }

    template <typename T>
    Shape(T *ptr, const SizeType size) noexcept(false) : numDims(size)
    {
        if (ptr == nullptr) {
            throw Service::LogicError("Shape: ptr is nullptr");
        }
        CheckNumDims(numDims);
        for (SizeType i = 0; i < numDims; ++i) {
            dimsArray[i] = ptr[i];
        }
        for (SizeType i = numDims; i < NUM_MAX_DIMENSIONS; ++i) {
            dimsArray[i] = 0;
        }
    }
    template <typename... Ts>
    Shape(Ts... dims) noexcept(false) : dimsArray{{static_cast<SizeType>(dims)...}}, numDims{sizeof...(dims)}
    {
        CheckNumDims(numDims);
    }
    Shape(const std::initializer_list<SizeType>& list) noexcept(false)
    {
        ResetShape(list);
    }

    template <typename... Ts>
    Shape& ResetShape(Ts... dims) noexcept(false)
    {
        CheckNumDims(sizeof...(dims));
        numDims = sizeof...(dims);
        dimsArray = {static_cast<SizeType>(dims)...};
        for (SizeType i = numDims; i < NUM_MAX_DIMENSIONS; ++i) {
            dimsArray[i] = 0;
        }
        return *this;
    }
    Shape& ResetShape(const std::initializer_list<SizeType>& list) noexcept(false)
    {
        CheckNumDims(list.size());
        numDims = list.size();
        for (SizeType i = 0; i < numDims; ++i) {
            dimsArray[i] = *(list.begin() + i);
        }
        for (SizeType i = numDims; i < NUM_MAX_DIMENSIONS; ++i) {
            dimsArray[i] = 0;
        }
        return *this;
    }
    template <typename T>
    Shape& ResetShape(T *ptr, const SizeType size) noexcept(false)
    {
        if (ptr == nullptr) {
            throw Service::LogicError("Shape: ptr is nullptr");
        }
        CheckNumDims(size);
        numDims = size;
        for (SizeType i = 0; i < numDims; ++i) {
            dimsArray[i] = ptr[i];
        }
        for (SizeType i = numDims; i < NUM_MAX_DIMENSIONS; ++i) {
            dimsArray[i] = 0;
        }
        return *this;
    }

    Shape& operator+=(const Shape &adder) noexcept(false)
    {
        if (adder.GetNumDims() !=  this->GetNumDims()) {
            throw Service::LogicError("Shape: different size of base and adder shapes");
        }
        for (SizeType i = 0; i < adder.GetNumDims(); ++i) {
            this->operator[](i) += adder[i];
        }
        return *this;
    }

    Shape operator+(const Shape &adder) const noexcept(false)
    {
        if (adder.GetNumDims() != this->GetNumDims()) {
            throw Service::LogicError("Shape: different size of base and adder shapes");
        }
        std::array<SizeType, NUM_MAX_DIMENSIONS> tmp;
        for (SizeType i = 0; i < adder.GetNumDims(); ++i) {
            tmp[i] = this->operator[](i) + adder[i];
        }
        return Shape(tmp.data(), this->GetNumDims());
    }

    SizeType operator[](SizeType id) const noexcept(false)
    {
        if (id >= numDims) {
            throw Service::LogicError("Shape: index >= num_dims");
        }
        return dimsArray[id];
    }

    SizeType& operator[](SizeType id) noexcept(false)
    {
        if (id >= numDims) {
            throw Service::LogicError("Shape: index >= num_dims");
        }
        return dimsArray[id];
    }

    SizeType GetNumDims() const noexcept
    {
        return numDims;
    }

    SizeType GetTotalDimsSize() const noexcept(false)
    {
        if (Service::WillIntMultOverflow(dimsArray.begin(), dimsArray.begin() + numDims)) {
            throw Service::LogicError("Shape: computing total size will cause overflow");
        }
        SizeType accum = 1;
        for (SizeType i = 0; i < numDims; ++i) {
            accum *= dimsArray[i];
        }
        return accum;
    }

private:
    void CheckNumDims(SizeType nDims) const noexcept(false)
    {
        if (nDims > NUM_MAX_DIMENSIONS) {
            throw Service::LogicError("Shape: dims is greater than NUM_MAX_DIMENSIONS");
        }
    }
    std::array<SizeType, NUM_MAX_DIMENSIONS> dimsArray;
    SizeType numDims;
};

inline bool operator==(const Shape &lhs, const Shape &rhs) noexcept
{
    if (lhs.GetNumDims() == rhs.GetNumDims()) {
        for (SizeType i = 0; i < lhs.GetNumDims(); ++i) {
            if (lhs[i] != rhs[i]) {
                return false;
            }
        }
        return true;
    }
    return false;
}

inline bool operator!=(const Shape &lhs, const Shape &rhs) noexcept
{
    return !(lhs == rhs);
}

} // KDNN

#endif // KDNN_SHAPE_HPP
