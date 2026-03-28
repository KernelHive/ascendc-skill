/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cube_util_l2.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> V100_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
    DataType::DT_FLOAT16};
static const std::initializer_list<DataType> V200_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
    DataType::DT_FLOAT16, DataType::DT_BF16};

// 根据dtype进行初步拦截，后续需要再和cubemathtype + 芯片再进行一次拦截
const std::initializer_list<DataType>& GetDtypeSupportListBySocVersion() {
    return (IsCubeSupportFp32()) ? V200_DTYPE_SUPPORT_LIST : V100_DTYPE_SUPPORT_LIST;
}

// 检查芯片是否支持输入的dtype allowFp32为True：芯片不允许输入为BF16   allowFp32为False：芯片不允许输入为BF16和FP32
static bool CheckSocSupportDtype(const op::DataType cubeTensorDtype, bool allowFp32) {
    bool dtypeValid = allowFp32 ? (cubeTensorDtype == DataType::DT_BF16) :
                                  (cubeTensorDtype == DataType::DT_FLOAT || cubeTensorDtype == DataType::DT_BF16);
    // 如果芯片不支持FP32 + dtype为FP32 / BF16，报错
    OP_CHECK(!(dtypeValid && !IsCubeSupportFp32()), OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "The soc version does not support bf16 / fp32 for calculations, please change the setting of "
            "cubeMathType or the Dtype of input tensor."), return false);
    return true;
}

// 校验cubeMathType的值是否符合预期
bool CheckCubeMathType(const op::DataType cubeTensorDtype, int8_t cubeMathType) {
    switch (cubeMathType) {
        case KEEP_DTYPE:
            OP_LOGD("The cubeMathType is KEEP_DTYPE.");
            return CheckSocSupportDtype(cubeTensorDtype, false);
        case ALLOW_FP32_DOWN_PRECISION:
            OP_LOGD("The cubeMathType is ALLOW_FP32_DOWN_PRECISION.");
            return CheckSocSupportDtype(cubeTensorDtype, true);
        case USE_FP16:
            OP_LOGD("The cubeMathType is USE_FP16.");
            return CheckSocSupportDtype(cubeTensorDtype, true);
        case USE_HF32:
            OP_LOGD("The cubeMathType is USE_HF32.");
            return CheckSocSupportDtype(cubeTensorDtype, false);
        default:
          OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                  "The value of cubeMathType only support {0: KEEP_DTYPE, 1: "
                  "ALLOW_FP32_DOWN_PRECISION, 2: USE_FP16, 3: USE_HF32}, but got %d",
                  cubeMathType);
          return false;
    }
}

// 校验mm算子cubeMathType的值是否符合预期
bool CheckCubeMathTypeForMm(const op::DataType cubeTensorDtype, int8_t cubeMathType) {
    if (cubeMathType == USE_FP16 && cubeTensorDtype == DataType::DT_BF16) {
        OP_LOGW("The cubeMathType is USE_FP16. For input BF16, it will not be enabled.");
    } else if (cubeMathType == USE_HF32 &&
               (cubeTensorDtype == DataType::DT_BF16 || cubeTensorDtype == DataType::DT_FLOAT16)) {
        OP_LOGW("The cubeMathType is USE_HF32. For input FP16/BF16, it will not be enabled.");
    }

    if (cubeMathType == -1) {
        OP_LOGD("The inner cubeMathType is FP16FP32_KEEP_DTYPE.");
        return CheckSocSupportDtype(cubeTensorDtype, false);
    } else {
        return CheckCubeMathType(cubeTensorDtype, cubeMathType);
    }
}

// 根据promote type + cubemathtype的组合算出最终算子应用的dtype
DataType CalcPromoteTypeCubemathtype(const DataType cubeTensorPromoteType, int8_t cubeMathType) {
    bool cubeSupportFp32Flag = IsCubeSupportFp32();
    // USE_FP16场景，如果promote type为bf16，提示不支持该选项
    if (cubeMathType == USE_FP16) {
        if (cubeTensorPromoteType == DataType::DT_BF16) {
            OP_LOGW("The cubeMathType cann't be set to USE_FP16 when the dtype is BF16.");
        }
        return DataType::DT_FLOAT16;
    }

    switch (cubeTensorPromoteType) {
        case DataType::DT_FLOAT16:
            return DataType::DT_FLOAT16;
        case DataType::DT_FLOAT:
            return cubeSupportFp32Flag ? DataType::DT_FLOAT: DataType::DT_FLOAT16;
        case DataType::DT_BF16:
            return cubeSupportFp32Flag ? DataType::DT_BF16: DataType::DT_FLOAT16;
        default:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Cube only support FP16, FP32, BF16, but got %s",
                op::ToString(cubeTensorPromoteType).GetString());
            return DataType::DT_UNDEFINED;
    }
}

// 根据promoteType + cubeMathType 判断是否要走HF32分支
bool NeedCubeGoHF32(const DataType cubeTensorPromoteType, int8_t cubeMathType) {
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();

    // USE_HF32场景，如果promoteType为BF16或FP16时，提示不支持该选项
    if (cubeMathType == USE_HF32) {
        if (cubeTensorPromoteType == DataType::DT_BF16) {
            OP_LOGW("The cubeMathType cann't be set to USE_HF32 when the dtype is BF16.");
        }
        if (cubeTensorPromoteType == DataType::DT_FLOAT16) {
            OP_LOGW("The cubeMathType cann't be set to USE_HF32 when the dtype is FP16.");
        }
    }

    if (IsCubeSupportHf32() && (cubeTensorPromoteType == DataType::DT_FLOAT) &&
        (cubeMathType == ALLOW_FP32_DOWN_PRECISION || cubeMathType == USE_HF32)) {
        return true;
    }
    return false;
}


#ifdef __cplusplus
}
#endif