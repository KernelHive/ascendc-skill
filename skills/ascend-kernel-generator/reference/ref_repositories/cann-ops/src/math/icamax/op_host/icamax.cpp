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
 * @file icamax.cpp
 */
#include "icamax_tiling.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"


namespace optiling {

// static variable
constexpr static int32_t MAXNUMF32ELEEACHCORE = 23040;
constexpr static uint32_t ELENUM_PER_REPEAT = 64;
constexpr static int32_t MAX_VECCORE_NUM = 40;

constexpr static int32_t GM_RESULT_LEN = 2;
constexpr static int32_t BYTE_LEN_4 = 4;
constexpr static uint64_t ELEMENTS_IN_BLOCK = 8;
constexpr static int32_t SYS_WORK_SPACE = 16 * 1024 * 1024;
constexpr static int32_t MAX_REPEATS = 255;
constexpr static int32_t DEAL_TIMES_EACH_CORE_REDUCE = 63;

static uint32_t CeilA2B(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
};

static uint32_t GetNeedVecCoreNum(uint32_t tensorLen, uint32_t elements)
{
    uint32_t needVecCoreNum = 1;
    if (tensorLen > MAX_VECCORE_NUM * elements) {
        needVecCoreNum = MAX_VECCORE_NUM;
    } else if (tensorLen > elements) {
        needVecCoreNum = tensorLen / elements;
    } else {
        needVecCoreNum = 1;
    }
    return needVecCoreNum;
}

// Calculate tiling data value
static void CalTilingData(uint32_t minEleRepeatsNumber, uint32_t minEleRepeatTail,
                                        uint32_t needVecCoreNum, uint32_t* allMem)
{
    uint32_t minEleRepeatsNumberEachCore = minEleRepeatsNumber / needVecCoreNum;
    uint32_t minEleRepeatsNumbeTail = minEleRepeatsNumber % needVecCoreNum;

    uint32_t *startOffset = allMem;
    uint32_t *endOffset = allMem + MAX_VECCORE_NUM;
    uint32_t *eleTotalEachCore = allMem + 2 * MAX_VECCORE_NUM ;
    uint32_t *dealLenEachTime = allMem + 3 * MAX_VECCORE_NUM;
    uint32_t *dealTimesEachCore = allMem + 4 * MAX_VECCORE_NUM;
    uint32_t *reduceMaxRstsLenEachCore = allMem + 5 * MAX_VECCORE_NUM;
    uint32_t *dealLenUpBlockEachTime = allMem + 6 * MAX_VECCORE_NUM;
    uint32_t *totalRptCntNor = allMem + 7 * MAX_VECCORE_NUM;
    uint32_t *totalRptCntNorRemainder = allMem + 8 * MAX_VECCORE_NUM;
    uint32_t *rptBatchCntNor = allMem + 9 * MAX_VECCORE_NUM;
    uint32_t *rptBatchCntNorRemainder = allMem + 10 * MAX_VECCORE_NUM;
    uint32_t *rmdRptLenNor = allMem + 11 * MAX_VECCORE_NUM;

    uint32_t eleLenEachCore = 0;
    uint32_t minEleEachCore = ELENUM_PER_REPEAT;
    for (uint32_t i = 0; i < needVecCoreNum; i++) {
        eleLenEachCore = minEleRepeatsNumberEachCore * minEleEachCore;
        if (i == 0) {
            startOffset[i] = 0;
        } else {
            startOffset[i] = endOffset[i - 1];
        }

        // 均分给所有核
        if (minEleRepeatsNumbeTail > 0) {
            eleLenEachCore += minEleEachCore;
            minEleRepeatsNumbeTail--;
        }
        dealTimesEachCore[i] = 0;
        dealLenEachTime[i] = eleLenEachCore;  // 不带尾块算
        if (eleLenEachCore > 0 && eleLenEachCore <= MAXNUMF32ELEEACHCORE) {
            dealTimesEachCore[i] = 1;
        } else if (eleLenEachCore > MAXNUMF32ELEEACHCORE) {
            dealTimesEachCore[i] = CeilA2B(eleLenEachCore, MAXNUMF32ELEEACHCORE);
            dealLenEachTime[i] = MAXNUMF32ELEEACHCORE;
        } else {
            dealTimesEachCore[i] = 0;
            dealLenEachTime[i] = 0;
        }

        uint32_t dealLenEachTimeAttachTail = dealLenEachTime[i];
        if (i == 0 && minEleRepeatTail != 0) {
            eleLenEachCore += minEleRepeatTail;  // 尾块全给第一个核
            if (dealTimesEachCore[i] == 0) {
                dealTimesEachCore[i] = 1;
            }
            dealLenEachTimeAttachTail += minEleRepeatTail;
        }
        endOffset[i] = startOffset[i] + eleLenEachCore;
        eleTotalEachCore[i] = eleLenEachCore;

        // 默认就申请这么大
        reduceMaxRstsLenEachCore[i] = DEAL_TIMES_EACH_CORE_REDUCE * ELEMENTS_IN_BLOCK + ELEMENTS_IN_BLOCK;
        dealLenUpBlockEachTime[i] = CeilA2B(dealLenEachTimeAttachTail, ELEMENTS_IN_BLOCK) * ELEMENTS_IN_BLOCK;

        totalRptCntNor[i] = dealLenEachTime[i] / ELENUM_PER_REPEAT;
        totalRptCntNorRemainder[i] = dealLenEachTime[i] % ELENUM_PER_REPEAT;  // should calc
        rptBatchCntNor[i] = totalRptCntNor[i] / MAX_REPEATS;                  // limit by L0 API, should calc
        rptBatchCntNorRemainder[i] = totalRptCntNor[i] % MAX_REPEATS;         // should calc
        rmdRptLenNor[i] = rptBatchCntNorRemainder[i] * ELENUM_PER_REPEAT;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto* attrs = context->GetAttrs();
    auto* elementNumPtr = attrs->GetAttrPointer<uint32_t>(0);
    auto* incxPtr = attrs->GetAttrPointer<uint32_t>(1);
    uint32_t elementNum = static_cast<uint32_t>(*elementNumPtr);
    uint32_t incx = static_cast<uint32_t>(*incxPtr);
    uint32_t dytpeFlag = 1; // complex64

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto vecCoreNum = ascendcPlatform.GetCoreNumAiv();

    size_t sizeNum = context->GetInputShape(0)->GetStorageShape().GetDimNum();
    uint32_t tensorLen = static_cast<uint32_t>(context->GetInputShape(0)->GetStorageShape().GetDim(sizeNum - 1));

    uint32_t complexNum = 2;
    if (dytpeFlag == 1) {
        tensorLen = tensorLen * complexNum;
        elementNum = elementNum * complexNum;
    }

    uint32_t minEleEachCore = ELENUM_PER_REPEAT;
    uint32_t tmpLen = (tensorLen < elementNum) ? tensorLen : elementNum;
    uint32_t needVecCoreNum = GetNeedVecCoreNum(tmpLen, minEleEachCore);
    if (needVecCoreNum == 0) {
        needVecCoreNum = 1;
    }
    uint32_t rstLenAllCoreBytes = needVecCoreNum * GM_RESULT_LEN * BYTE_LEN_4;
    uint32_t maxRepeatLen = MAX_REPEATS * ELENUM_PER_REPEAT;

    // 按repeat64均分，尽量保证每个核吃到整repeat的数据，尾块数据部分丢给头块核
    uint32_t minEleRepeatsNumber = tmpLen / minEleEachCore;
    uint32_t minEleRepeatTail = tmpLen % minEleEachCore;

    uint32_t allMem[12 * MAX_VECCORE_NUM] = {0};

    CalTilingData(minEleRepeatsNumber, minEleRepeatTail, needVecCoreNum, allMem);

    uint32_t *startOffset = allMem;
    uint32_t *eleTotalEachCore = allMem + 2 * MAX_VECCORE_NUM ;
    uint32_t *dealLenEachTime = allMem + 3 * MAX_VECCORE_NUM;
    uint32_t *dealTimesEachCore = allMem + 4 * MAX_VECCORE_NUM;
    uint32_t *reduceMaxRstsLenEachCore = allMem + 5 * MAX_VECCORE_NUM;
    uint32_t *dealLenUpBlockEachTime = allMem + 6 * MAX_VECCORE_NUM;
    uint32_t *totalRptCntNor = allMem + 7 * MAX_VECCORE_NUM;
    uint32_t *totalRptCntNorRemainder = allMem + 8 * MAX_VECCORE_NUM;
    uint32_t *rptBatchCntNor = allMem + 9 * MAX_VECCORE_NUM;
    uint32_t *rptBatchCntNorRemainder = allMem + 10 * MAX_VECCORE_NUM;
    uint32_t *rmdRptLenNor = allMem + 11 * MAX_VECCORE_NUM;

    IcamaxTilingData tiling;
    tiling.set_incx(incx);
    tiling.set_needVecCoreNum(needVecCoreNum);
    tiling.set_dytpeFlag(dytpeFlag);
    tiling.set_rstLenAllCoreBytes(rstLenAllCoreBytes);
    tiling.set_tailCount(minEleRepeatTail);
    tiling.set_maxRepeatLen(maxRepeatLen);
    tiling.set_startOffset(startOffset);
    tiling.set_eleTotalEachCore(eleTotalEachCore);
    tiling.set_dealTimesEachCore(dealTimesEachCore);
    tiling.set_dealLenEachTime(dealLenEachTime);
    tiling.set_reduceMaxRstsLenEachCore(reduceMaxRstsLenEachCore);
    tiling.set_dealLenUpBlockEachTime(dealLenUpBlockEachTime);
    tiling.set_totalRptCntNor(totalRptCntNor);
    tiling.set_totalRptCntNorRemainder(totalRptCntNorRemainder);
    tiling.set_rptBatchCntNor(rptBatchCntNor);
    tiling.set_rptBatchCntNorRemainder(rptBatchCntNorRemainder);
    tiling.set_rmdRptLenNor(rmdRptLenNor);

    context->SetTilingKey(0);
    context->SetBlockDim(vecCoreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = SYS_WORK_SPACE + vecCoreNum * GM_RESULT_LEN * BYTE_LEN_4;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Icamax : public OpDef {
public:
    explicit Icamax(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_COMPLEX64})
            .Format({ge::FORMAT_ND});
        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Attr("n").AttrType(OPTIONAL).Int(0);
        this->Attr("incx").AttrType(OPTIONAL).Int(1);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Icamax);
} // namespace ops
