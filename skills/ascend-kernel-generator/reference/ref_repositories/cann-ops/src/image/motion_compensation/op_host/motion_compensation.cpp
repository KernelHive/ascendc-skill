/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file motion_compensation.cpp
 */
#include <cmath>
#include "motion_compensation_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

namespace optiling {
const uint32_t BLOCK_SIZE = 256;
const uint32_t BUFFER_NUM = 2;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MotionCompensationTilingData tiling;
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize -= 8 * 1024;

    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    tiling.set_ndim(x1_shape->GetStorageShape().GetDim(0));
    tiling.set_N(x1_shape->GetStorageShape().GetDim(1));
    int64_t N = x1_shape->GetStorageShape().GetDim(1);
    int64_t ndim = x1_shape->GetStorageShape().GetDim(0);
    float f, theta, sin_theta;
    float trans[3];
    float qRel[4];
    int32_t  doRotation;
    float abs_d;
    float d_sign;
    
    auto attr = context->GetAttrs();
    int64_t tMin = *attr->GetInt(0);
    int64_t tMax = *attr->GetInt(1);
    int32_t tMaxLow32 = static_cast<int32_t>(tMax & 0xFFFFFFFF);
    
    f = (tMax != tMin) ? (1.0f / static_cast<float>(tMax - tMin)) : 0.0f;

    const float *transMin = attr->GetListFloat(2)->GetData();
    const float *transMax = attr->GetListFloat(3)->GetData();
    
    float transX = transMin[0] - transMax[0];
    float transY = transMin[1] - transMax[1];
    float transZ = transMin[2] - transMax[2];
    
    const float *qMin = attr->GetListFloat(4)->GetData();
    const float *qMax = attr->GetListFloat(5)->GetData();
    
    float qMaxW = qMax[0];
    float qMaxX = qMax[1];
    float qMaxY = qMax[2];
    float qMaxZ = qMax[3];

    float qMinW = qMin[0];
    float qMinX = qMin[1];
    float qMinY = qMin[2];
    float qMinZ = qMin[3];

    float qMaxConjW =  qMaxW;
    float qMaxConjX = -qMaxX;
    float qMaxConjY = -qMaxY;
    float qMaxConjZ = -qMaxZ;

    {
        float n = qMaxConjW*qMaxConjW + qMaxConjX*qMaxConjX +
                  qMaxConjY*qMaxConjY + qMaxConjZ*qMaxConjZ;
        float s = (n > 0.0f) ? (2.0f / n) : 0.0f;

        float wx = s * qMaxConjW * qMaxConjX;
        float wy = s * qMaxConjW * qMaxConjY;
        float wz = s * qMaxConjW * qMaxConjZ;
        float xx = s * qMaxConjX * qMaxConjX;
        float xy = s * qMaxConjX * qMaxConjY;
        float xz = s * qMaxConjX * qMaxConjZ;
        float yy = s * qMaxConjY * qMaxConjY;
        float yz = s * qMaxConjY * qMaxConjZ;
        float zz = s * qMaxConjZ * qMaxConjZ;

        trans[0] = (1.0f - (yy + zz)) * transX + (xy - wz) * transY + (xz + wy) * transZ;
        trans[1] = (xy + wz) * transX + (1.0f - (xx + zz)) * transY + (yz - wx) * transZ;
        trans[2] = (xz - wy) * transX + (yz + wx) * transY + (1.0f - (xx + yy)) * transZ;
    }

    qRel[0] = qMaxConjW*qMinW - qMaxConjX*qMinX - qMaxConjY*qMinY - qMaxConjZ*qMinZ;
    qRel[1] = qMaxConjW*qMinX + qMaxConjX*qMinW + qMaxConjY*qMinZ - qMaxConjZ*qMinY;
    qRel[2] = qMaxConjW*qMinY - qMaxConjX*qMinZ + qMaxConjY*qMinW + qMaxConjZ*qMinX;
    qRel[3] = qMaxConjW*qMinZ + qMaxConjX*qMinY - qMaxConjY*qMinX + qMaxConjZ*qMinW;

    float norm = sqrtf(qRel[0]*qRel[0] + qRel[1]*qRel[1] +
                      qRel[2]*qRel[2] + qRel[3]*qRel[3]);
    if(norm == 0){
        return ge::GRAPH_FAILED;
    }
    if (std::fabs(norm) > std::numeric_limits<float>::epsilon()) {
        float inv = 1.0f / norm;
        qRel[0] *= inv; qRel[1] *= inv; qRel[2] *= inv; qRel[3] *= inv;
    }

    float d = qRel[0];
    
    abs_d = std::fabs(d);
    doRotation = (abs_d < 1.0f - 1e-8f) ? 1 : 0;
    d_sign = (d >= 0.0f) ? 1.0f : -1.0f;
    theta = std::acos(abs_d);
    sin_theta = std::sin(theta);
    if(sin_theta == 0){
        return ge::GRAPH_FAILED;
    }
    if (std::fabs(sin_theta) > std::numeric_limits<float>::epsilon()){
        sin_theta = 1.0f / sin_theta;
    }

    tiling.set_f(f);
    tiling.set_theta(theta);
    tiling.set_r_sin_theta(sin_theta);
    tiling.set_d_sign(d_sign);
    tiling.set_doRotation(doRotation);
    tiling.set_trans(trans);
    tiling.set_qRel(qRel);
    tiling.set_tMaxLow32(tMaxLow32);
    tiling.set_tMax(static_cast<float>(tMax));

    int64_t inputLength = N * 4;
    int64_t inputBytes = 4;
    int64_t coreNum = 8;

    int64_t ubDataNumber = 16;

    int64_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
    int64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    int64_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    int64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    int64_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    
    int64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    int64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    int64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    int64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
    
    everyCoreInputBlockNum += 1;
    int64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    int64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    int64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    int64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
    
    tiling.set_smallCoreDataNum(smallCoreDataNum);
    tiling.set_bigCoreDataNum(bigCoreDataNum);
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_smallTailDataNum(smallTailDataNum);
    tiling.set_bigTailDataNum(bigTailDataNum);
    tiling.set_finalSmallTileNum(finalSmallTileNum);
    tiling.set_finalBigTileNum(finalBigTileNum);
    tiling.set_tailBlockNum(tailBlockNum);

    inputLength = (ndim - 3) * N * 4;
    ubDataNumber = 4;

    tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
    tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    
    smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
    
    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
    
    tiling.set_smallCoreDataNum2(smallCoreDataNum);
    tiling.set_bigCoreDataNum2(bigCoreDataNum);
    tiling.set_tileDataNum2(tileDataNum);
    tiling.set_smallTailDataNum2(smallTailDataNum);
    tiling.set_bigTailDataNum2(bigTailDataNum);
    tiling.set_finalSmallTileNum2(finalSmallTileNum);
    tiling.set_finalBigTileNum2(finalBigTileNum);
    tiling.set_tailBlockNum2(tailBlockNum);
    
    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class MotionCompensation : public OpDef {
public:
    explicit MotionCompensation(const char* name) : OpDef(name)
    {
        this->Input("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("timestamp")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("timestamp_min").Int();
        this->Attr("timestamp_max").Int();
        this->Attr("translation_min").ListFloat();
        this->Attr("translation_max").ListFloat();
        this->Attr("quaterniond_min").ListFloat();
        this->Attr("quaterniond_max").ListFloat();

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p").AddConfig("ascend910b");
    }
};
OP_ADD(MotionCompensation);
}
