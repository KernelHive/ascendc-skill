/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TUNE_SPACE_MATMUL_TUNE_SPACE_H
#define TUNE_SPACE_MATMUL_TUNE_SPACE_H

#include "aoe/op_tuning_tiling/gemm_tuning_tiling.h"

#include "tune_space.h"
#include "tune_space_register.h"

namespace OpTuneSpace {

class MatMulTuneSpace : public TuneSpace {
public:
    explicit MatMulTuneSpace() = default;
    ~MatMulTuneSpace() override = default;

    Status GetTuneSpace(gert::TilingContext* op, std::vector<nlohmann::json> &jsonTuneSpace) override;
    virtual std::string GetOpType() = 0;
};

class BatchMatMulV2TuneSpace : public MatMulTuneSpace {
private:
    std::string GetOpType() override { return "BatchMatMulV2"; };
};

class MatMulV2TuneSpace : public MatMulTuneSpace {
private:
    std::string GetOpType() override { return "MatMulV2"; };
};

}
#endif // namespace OpTuneSpace