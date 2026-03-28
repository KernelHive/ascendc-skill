/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_TUNING_TILING_TUNE_SPACE_H
#define OP_TUNING_TILING_TUNE_SPACE_H

#include <vector>
#include <nlohmann/json.hpp>

#include "exe_graph/runtime/tiling_context.h"
#include "common_utils.h"
#include "tune_space_log.h"

namespace OpTuneSpace {

/**
 * class analyze input tiling space
*/
class TuneSpace {
public:
    explicit TuneSpace(){};
    virtual ~TuneSpace(){};

    /**
     * analyze and split tiling space
     * param [in] args     input tiling args
     * return     spaces   tiling space
    */
   virtual Status GetTuneSpace(gert::TilingContext* op, std::vector<nlohmann::json> &jsonTuneSpace) = 0;
};
} // namespace OpTuneSpace
#endif // OP_TUNING_TILING_TUNE_SPACE_H_
