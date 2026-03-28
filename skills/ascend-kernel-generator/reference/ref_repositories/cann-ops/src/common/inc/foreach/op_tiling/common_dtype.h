/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common_dtype.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_COMMON_DTYPE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_COMMON_DTYPE_H_

namespace optiling {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_BLOCK_FOR_BF16 = 64; 
constexpr uint32_t BYTE_REPEAT = 256;
constexpr uint32_t BYTE_BASIC_BLOCK = 1024;

constexpr uint32_t BYTE_LEN_8 = 8;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;
constexpr uint32_t BYTE_LEN_1 = 1;

constexpr uint64_t TILING_KEY_HALF = 1;
constexpr uint64_t TILING_KEY_FLOAT = 2;
constexpr uint64_t TILING_KEY_INT = 3;
constexpr uint64_t TILING_KEY_BF16 = 4;
constexpr uint64_t TILING_KEY_INT16 = 5;
constexpr uint64_t TILING_KEY_UINT16 = 6;
constexpr uint64_t TILING_KEY_INT8 = 7;
constexpr uint64_t TILING_KEY_UINT8 = 8;
constexpr uint64_t TILING_KEY_UINT32 = 9;
constexpr uint64_t TILING_KEY_INT64 = 10;
constexpr uint64_t TILING_KEY_DOUBLE = 11;
constexpr uint64_t TILING_KEY_BOOL = 12;
/**
** function: GetDataTypeSize
*/
inline uint8_t GetDataTypeSize(ge::DataType dataType) {
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return BYTE_LEN_2;
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_BOOL:
            return BYTE_LEN_1;
        case ge::DT_INT64:
        case ge::DT_DOUBLE:
            return BYTE_LEN_8;
        default:
            return BYTE_LEN_4;
    }
}

/**
    ** function: GetTilingKeyByDtypeOnly
*/
inline uint64_t GetTilingKeyByDtypeOnly(ge::DataType dataType) {
    switch (dataType) {
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        case ge::DT_INT32:
            return TILING_KEY_INT;
        case ge::DT_BF16:
            return TILING_KEY_BF16;
        case ge::DT_INT8:
            return TILING_KEY_INT8;
        case ge::DT_UINT8:
            return TILING_KEY_UINT8;
        case ge::DT_INT16:
            return TILING_KEY_INT16;
        case ge::DT_UINT16:
            return TILING_KEY_UINT16;
        case ge::DT_UINT32:
            return TILING_KEY_UINT32;
        case ge::DT_INT64:
            return TILING_KEY_INT64;
        case ge::DT_DOUBLE:
            return TILING_KEY_DOUBLE;
        case ge::DT_BOOL:
            return TILING_KEY_BOOL;
        default:
            return 0;
    }
}
}

#endif
