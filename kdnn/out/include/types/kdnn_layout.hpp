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

#ifndef KDNN_LAYOUT_HPP
#define KDNN_LAYOUT_HPP

namespace KDNN {

// Enumeration of common data types which are supported in KDNN.
enum class Layout {
    UNDEFINED = 0,
    A, // single row
    AB,
    BA,
    ABC,
    ACB,
    BAC,
    BCA,
    CAB,
    CBA,
    ABCD,
    ABDC,
    ACBD,
    ACDB,
    ADBC,
    ADCB,
    BACD,
    BCDA,
    CDAB,
    CDBA,
    DCAB,
    ABCDE,
    ABCED,
    ABDEC,
    ACBDE,
    ACDEB,
    ADECB,
    BACDE,
    BCDEA,
    CDEAB,
    CDEBA,
    DECAB,
    // GEMM prepacked layouts. The suffix names the innermost blocked logical
    // dimension; VLH/VLW use the runtime SVE vector length for half/word data.
    BA4b,
    BAVLHb,
    BAVLWb,
    ACB4c,
    ACBVLHc,
    ACBVLWc,
    ABDC4d,
    ABDCVLHd,
    ABDCVLWd,
    ABCED4e,
    ABCEDVLHe,
    ABCEDVLWe,
    ROW_MAJOR = AB,
    COL_MAJOR = BA,
    NCHW = ABCD,
    NHWC = ACDB,
    NCDHW = ABCDE,
    NDHWC = ACDEB,
    OIHW = ABCD,
    HWIO = CDBA,
    HWOI = CDAB,
    OHWI = ACDB,
    OHWO = BCDA,
    IOHW = BACD,
};

} // KDNN

#endif // KDNN_LAYOUT_HPP
