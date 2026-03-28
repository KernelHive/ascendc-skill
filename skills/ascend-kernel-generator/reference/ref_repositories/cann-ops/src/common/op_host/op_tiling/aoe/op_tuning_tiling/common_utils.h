/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_TUNING_TILING_COMMON_UTILS_H
#define OP_TUNING_TILING_COMMON_UTILS_H

#include <iostream>
#include <memory>
#include <map>
#include <string>

namespace OpTuneSpace {
const uint32_t ASCEND_310_FLAG = 1; // Ascend310
const uint32_t ASCEND_910_FLAG = 2; // Ascend910A
const uint32_t HI3796CV300ES_FLAG = 3; // Hi3796CV300ES
const uint32_t ASCEND_310P3_FLAG = 4; // Ascend310P3
const uint32_t ASCEND_610_FLAG = 5; // Ascend610
const uint32_t ASCEND_910B_FLAG = 6; // Ascend910B
const uint32_t ASCEND_910PROA_FLAG = 7; // Ascend910ProA
const uint32_t HI3796CV300CS_FLAG = 8; // Hi3796CV300CS
const uint32_t BS9SX1AA_FLAG = 9; // BS9SX1AA
const uint32_t ASCEND_310P1_FLAG = 10; // Ascend310P1
const uint32_t ASCEND_910PROB_FLAG = 11; // Ascend910ProB
const uint32_t ASCEND_910PREMIUMA_FLAG = 12; // Ascend910PremiumA
const uint32_t ASCEND_910B2_FLAG = 13; // Ascend910B2
const uint32_t SD3403_FLAG = 14; // SD3403
const uint32_t ASCEND_310B1_FLAG = 15; // Ascend310B1
const uint32_t ASCEND_910B1_FLAG = 16; // Ascend910B1
const uint32_t ASCEND_910B3_FLAG = 17; // Ascend910B3
const uint32_t ASCEND_910B4_FLAG = 18; // Ascend910B4
const uint32_t ASCEND_031_FLAG = 19; // Ascend031
const uint32_t ASCEND_610_LITE_FLAG = 23; // Ascend610Lite
const uint32_t ASCEND_310M1_FLAG = 24; // Ascend310M1

enum class ReturnCode {
    TILING_SPACE_NOT_FULL = 0,
    TILING_SPACE_FULL = -1,
};

}
#endif // OP_TUNING_TILING_COMMON_UTILS_H_