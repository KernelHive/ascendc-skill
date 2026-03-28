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
 * @file reflection_pad3d_grad.cpp
 */
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "reflection_pad3d_grad_tiling.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace std;

namespace {
constexpr size_t INDEX_X = 0;
constexpr size_t INDEX_PADDINGS = 1;
constexpr size_t INDEX_Y = 0;
constexpr size_t INDEX_PADDINGS_CONTIGUOUS = 1;
constexpr size_t PAIR = 2;

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                        \
        if (cond) {                               \
            log_func;                               \
            expr;                                   \
        }                                         \
    } while (0)

template <typename T>
inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
    return context->GetCompiledInfo<T>();
}

static inline const std::string &GetOpInfo(const std::string &str)
{
    return str;
}
} // namespace

namespace optiling {
constexpr size_t MODE_INDEX = 0;
constexpr bool PADDINGS_CONTIGUOUS_INDEX = 1;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr int32_t X_INPUT_INDEX = 0;
constexpr int32_t PAD_INPUT_INDEX = 1;
constexpr int32_t FLOAT_BYTES = 4;
constexpr int32_t FLOAT16_BYTES = 2;
constexpr int32_t BFLOAT16_BYTES = 2;
constexpr size_t CHECK_DIM_NUM = 5;
constexpr int DIVIDE_UB_NUM = 5;
constexpr int DIVIDE_UB_NUM_CAST = 8;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t DIM_INDEX2 = 2;
constexpr uint32_t DIM_INDEX3 = 3;
constexpr uint32_t DIM_INDEX4 = 4;
constexpr uint32_t PADDING_NUM_INDEX4 = 4;
constexpr uint32_t PADDING_NUM_INDEX5 = 5;
constexpr uint32_t PADDING_NUM_INDEX6 = 6;
constexpr uint32_t PADDING_NUM_INDEX7 = 7;
constexpr uint32_t PADDING_NUM_INDEX8 = 8;
constexpr uint32_t PADDING_NUM_INDEX9 = 9;
constexpr uint32_t RESERVED_UB = static_cast<uint32_t>(16 * 1024);
constexpr uint32_t ALIGN_256_BYTES = 256;
constexpr uint32_t ALIGN_16 = 16;
constexpr uint32_t WORK_SPACE_PART = 32;
constexpr uint32_t MIN_ALIGN_HEIGHT = 32;
constexpr uint32_t MIN_ALIGN_WIDTH  = 32;
constexpr uint32_t FLOAT_SMALL_REFLECTION = 100;
constexpr uint32_t FLOAT_MID_REFLECTION = 101;
constexpr uint32_t FLOAT_FLAT_REFLECTION = 102;
constexpr uint32_t FLOAT_BIG_REFLECTION = 103;
constexpr uint32_t FLOAT16_SMALL_REFLECTION = 200;
constexpr uint32_t FLOAT16_FLAT_REFLECTION = 202;
constexpr uint32_t FLOAT16_BIG_REFLECTION = 203;
constexpr uint32_t BF16_SMALL_REFLECTION = 300;
constexpr uint32_t BF16_FLAT_REFLECTION = 302;
constexpr uint32_t BF16_BIG_REFLECTION = 303;
const static uint32_t MAX_LINE = 16;

static std::map<ge::DataType, int32_t> DATATYPE_LEN_MAP = {{ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}, {ge::DT_FLOAT, 4}};
static std::map<ge::DataType, uint32_t> SMALL_TILING_MAP = {{ge::DT_FLOAT, FLOAT_SMALL_REFLECTION}, {ge::DT_FLOAT16, FLOAT16_SMALL_REFLECTION}, {ge::DT_BF16, BF16_SMALL_REFLECTION}};
static std::map<ge::DataType, uint32_t> BIG_TILING_MAP = {{ge::DT_FLOAT, FLOAT_BIG_REFLECTION}, {ge::DT_FLOAT16, FLOAT16_BIG_REFLECTION}, {ge::DT_BF16, BF16_BIG_REFLECTION}};
static std::map<ge::DataType, uint32_t> FLAT_TILING_MAP = {{ge::DT_FLOAT, FLOAT_FLAT_REFLECTION}, {ge::DT_FLOAT16, FLOAT16_FLAT_REFLECTION}, {ge::DT_BF16, BF16_FLAT_REFLECTION}};

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b) {
    if (b <= 0) {
        return 0;
    }

    return (a + b - 1) / b * b;
}
template <typename T1, typename T2>
inline T1 FloorAlign(T1 a, T2 b) {
    if (b <= 0) {
        return 0;
    }
    return (a) / b * b;
}

template <typename T>
inline T Mymax(T a, T b) {
    if (a > b) {
        return a;
    }
    return b;
}
template <typename TilingData, int32_t dataTypeLen>
class PadV3GradV2Tiling {
public:
    explicit PadV3GradV2Tiling(InputParamsInfo& param, const uint32_t inputCoreNum, const uint32_t inputUbSize) {
        this->batch = param.batch;
        this->channel = param.channel;
        this->depth = param.depth;
        this->height = param.height;
        this->width = param.width;
        this->alignHeight = param.alignHeight;
        this->alignWidth = param.alignWidth;
        this->outDepth = param.outDepth;
        this->outHeight = param.outHeight;
        this->outWidth = param.outWidth;
        this->dPad1 = param.dPad1;
        this->dPad2 = param.dPad2;
        this->hPad1 = param.hPad1;
        this->hPad2 = param.hPad2;
        this->wPad1 = param.wPad1;
        this->wPad2 = param.wPad2;
        this->tilingKey = param.tilingKey;
        this->ubSize = FloorAlign(inputUbSize, BYTE_BLOCK);
        this->dataTypeSize = dataTypeLen;
        this->elementsPerBlock = BYTE_BLOCK / dataTypeSize;
        this->coreNum = inputCoreNum;
        return;
    }

void GetTiling(TilingData* tilingData, bool isCast);

private:
    void GetUsedCore();
    void SplitUb(bool isCast);
    void FillTilingData(TilingData* tilingData);

private:
    uint32_t batch = 0;
    uint32_t channel = 0;
    uint32_t depth = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t alignHeight = 0;
    uint32_t alignWidth = 0;
    uint32_t outDepth = 0;
    uint32_t outHeight = 0;
    uint32_t outWidth = 0;
    int32_t dPad1 = 0;
    int32_t dPad2 = 0;
    int32_t hPad1 = 0;
    int32_t hPad2 = 0;
    int32_t wPad1 = 0;
    int32_t wPad2 = 0;
    uint32_t ubSize = 0;
    uint32_t usedCoreNum = 0;
    uint32_t coreNum = 0;
    uint32_t ncPerCore = 1;
    uint32_t tailNC = 0;
    uint32_t ubFactorElement = 0;
    uint32_t tilingKey = 0;
    uint8_t dataTypeSize = 0;
    uint8_t elementsPerBlock = 0;
};

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::GetUsedCore() {
    uint64_t nMulC = batch * channel;
    if (!dPad1 && !dPad2) {
        nMulC *= depth;
    }
    if (nMulC <= coreNum) {
        ncPerCore = 1;
        usedCoreNum = nMulC;
        tailNC = 0;
        return;
    }
    ncPerCore = nMulC / coreNum;
    tailNC = nMulC % coreNum;
    usedCoreNum = coreNum;
    return;
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::SplitUb(bool isCast) {
    uint32_t tilingDataSize = CeilAlign(sizeof(TilingData), BYTE_BLOCK);
    uint32_t canUseUbSize = FloorAlign(ubSize - tilingDataSize, BYTE_BLOCK);
    ubFactorElement = FloorAlign(canUseUbSize / DIVIDE_UB_NUM, ALIGN_256_BYTES) / dataTypeLen;
    if (isCast) {
       ubFactorElement = FloorAlign(canUseUbSize / DIVIDE_UB_NUM_CAST, ALIGN_256_BYTES) / dataTypeLen;
    }
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::FillTilingData(TilingData* tilingData) {
    tilingData->set_batch(batch);
    tilingData->set_channel(channel);
    tilingData->set_depth(depth);
    tilingData->set_height(height);
    tilingData->set_width(width);
    tilingData->set_alignHeight(alignHeight);
    tilingData->set_alignWidth(alignWidth);
    tilingData->set_outDepth(outDepth);
    tilingData->set_outHeight(outHeight);
    tilingData->set_outWidth(outWidth);
    tilingData->set_dPad1(dPad1);
    tilingData->set_dPad2(dPad2);
    tilingData->set_hPad1(hPad1);
    tilingData->set_hPad2(hPad2);
    tilingData->set_wPad1(wPad1);
    tilingData->set_wPad2(wPad2);
    tilingData->set_blockNum(usedCoreNum);
    tilingData->set_ubFactorElement(ubFactorElement);
    tilingData->set_ncPerCore(ncPerCore);
    tilingData->set_tailNC(tailNC);
    tilingData->set_tilingKey(tilingKey);
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::GetTiling(TilingData* tilingData, bool isCast) {
    GetUsedCore();
    SplitUb(isCast);
    FillTilingData(tilingData);
}

template <typename TilingData, int32_t dataTypeLen>
void GetPadV3GradV2Tiling(TilingData* tilingData, InputParamsInfo& params, uint32_t coreNum, uint32_t ubSize, bool isCast) {
    class PadV3GradV2Tiling<TilingData, dataTypeLen> tilingObj(params, coreNum, ubSize);
    tilingObj.GetTiling(tilingData, isCast);
}

template <typename T>
static ge::graphStatus GetInputInfo(gert::TilingContext* tilingContext, InputParamsInfo& params) {
    const gert::StorageShape* xShape = tilingContext->GetInputShape(X_INPUT_INDEX);
    OP_TILING_CHECK(
        xShape->GetStorageShape().GetDimNum() != CHECK_DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "input dim is not 5, please check input."),
        return ge::GRAPH_FAILED);
    params.batch = xShape->GetStorageShape().GetDim(DIM_INDEX0);
    params.channel = xShape->GetStorageShape().GetDim(DIM_INDEX1);
    params.depth = xShape->GetStorageShape().GetDim(DIM_INDEX2);
    params.height = xShape->GetStorageShape().GetDim(DIM_INDEX3);
    params.width = xShape->GetStorageShape().GetDim(DIM_INDEX4);

    const gert::StorageShape* paddingShape = tilingContext->GetInputShape(PAD_INPUT_INDEX);
    OP_TILING_CHECK((size_t)(2 * xShape->GetStorageShape().GetDimNum()) != (size_t)paddingShape->GetStorageShape().GetDim(0),
                    VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Please check input or padding shape"),
                    return ge::GRAPH_FAILED);
    const gert::Tensor* paddings_tensor = tilingContext->GetInputTensor(PAD_INPUT_INDEX);
    const T* paddingsValue = paddings_tensor->GetData<T>();
    params.dPad1 = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX4]);
    params.dPad2 = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX5]);
    params.hPad1 = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX6]);
    params.hPad2 = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX7]);
    params.wPad1 = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX8]);
    params.wPad2 = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX9]);
    const gert::StorageShape* outShape = tilingContext->GetOutputShape(0);
    params.outDepth = outShape->GetStorageShape().GetDim(DIM_INDEX2);;
    params.outHeight = outShape->GetStorageShape().GetDim(DIM_INDEX3);;
    params.outWidth = outShape->GetStorageShape().GetDim(DIM_INDEX4);;
    OP_TILING_CHECK(params.outHeight != (params.height - params.hPad1 - params.hPad2) ||
                    params.outWidth != (params.width - params.wPad1 - params.wPad2) ||
                    params.outDepth != (params.depth - params.dPad1 - params.dPad2),
                    VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Please check input or output shape"),
                    return ge::GRAPH_FAILED);

    params.alignHeight = CeilAlign(params.height, ALIGN_16);
    params.alignWidth = CeilAlign(params.width, ALIGN_16);
  
    return ge::GRAPH_SUCCESS;
}

static void FillTilingKey(ReflectionPad3dGradTilingData* tilingData, ge::DataType inputDatatype) {
    int64_t alignHeight = tilingData->get_alignHeight();
    int64_t alignWidth= tilingData->get_alignWidth();
    int64_t height= tilingData->get_height();
    int64_t width= tilingData->get_width();
    if (alignHeight * alignWidth <= tilingData->get_ubFactorElement()) {
        tilingData->set_tilingKey(SMALL_TILING_MAP[inputDatatype]);
    } else if (width <= MAX_LINE + MAX_LINE || height <= MAX_LINE + MAX_LINE){
        tilingData->set_tilingKey(FLAT_TILING_MAP[inputDatatype]);
    } else if (MAX_LINE * Mymax(alignHeight, alignWidth) <= tilingData->get_ubFactorElement() && inputDatatype == ge::DT_FLOAT){
        tilingData->set_tilingKey(FLOAT_MID_REFLECTION);
    } else {
        tilingData->set_tilingKey(BIG_TILING_MAP[inputDatatype]);
    }
}

static ge::graphStatus Tiling4PadV3GradV2(gert::TilingContext* tilingContext) {
    auto compileInfo = reinterpret_cast<const Tiling4PadV3GradV2CompileInfo*>(tilingContext->GetCompileInfo());
    uint64_t ubSizePlatForm = compileInfo->ubSizePlatForm;
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);
    uint32_t availableUb = ubSize - RESERVED_UB;
    uint32_t coreNum = compileInfo->coreNum; 
    uint32_t sysWorkspaceSize  = compileInfo->sysWorkspaceSize;
    ge::DataType inputDatatype = tilingContext->GetInputDesc(0)->GetDataType();
    ge::DataType paddingDatatype = tilingContext->GetInputDesc(1)->GetDataType();
    InputParamsInfo params;
    if (paddingDatatype == ge::DT_INT32) {
        GetInputInfo<int32_t>(tilingContext, params);
        OP_TILING_CHECK(GetInputInfo<int32_t>(tilingContext, params) != ge::GRAPH_SUCCESS,
                        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "get op inputs failed."),
                        return ge::GRAPH_FAILED);
    } else if (paddingDatatype == ge::DT_INT64) {
        GetInputInfo<int64_t>(tilingContext, params);
        OP_TILING_CHECK(GetInputInfo<int64_t>(tilingContext, params) != ge::GRAPH_SUCCESS,
                        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "get op inputs failed."),
                        return ge::GRAPH_FAILED);
    }
    ReflectionPad3dGradTilingData tilingData;
    if (inputDatatype == ge::DT_FLOAT) {
        GetPadV3GradV2Tiling<ReflectionPad3dGradTilingData, FLOAT_BYTES>(&tilingData, params, coreNum, availableUb, false);
    } else if (inputDatatype == ge::DT_FLOAT16) {
        GetPadV3GradV2Tiling<ReflectionPad3dGradTilingData, FLOAT16_BYTES>(&tilingData, params, coreNum, availableUb, true);
    } else {
        GetPadV3GradV2Tiling<ReflectionPad3dGradTilingData, BFLOAT16_BYTES>(&tilingData, params, coreNum, availableUb, true);
    }
    OP_TILING_CHECK(
        tilingData.get_ubFactorElement() <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "ub space is not enough, please check input."),
        return ge::GRAPH_FAILED); 
    FillTilingKey(&tilingData, inputDatatype);
    tilingContext->SetTilingKey(tilingData.get_tilingKey());
    tilingContext->SetNeedAtomic(true);
    tilingContext->SetBlockDim(tilingData.get_blockNum());
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    size_t usrWorkspace = Mymax(tilingData.get_alignHeight(), tilingData.get_alignWidth()) 
              * WORK_SPACE_PART * tilingData.get_blockNum() * DATATYPE_LEN_MAP[inputDatatype];
    if (inputDatatype != ge::DT_FLOAT) {
        usrWorkspace = tilingData.get_alignHeight() * tilingData.get_alignWidth() * tilingData.get_blockNum() * DATATYPE_LEN_MAP[ge::DT_FLOAT];
    }
    workspaces[0] = usrWorkspace + sysWorkspaceSize;
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                          tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4PadV3GradV2(gert::TilingParseContext* context) {
    auto compileInfo = GetCompileInfoPtr<Tiling4PadV3GradV2CompileInfo>(context);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK((compileInfo->coreNum <= 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get core num."),
                    return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_TILING_CHECK(compileInfo->ubSizePlatForm <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get ub size."),
                    return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ReflectionPad3dGrad)
    .Tiling(Tiling4PadV3GradV2)
    .TilingParse<Tiling4PadV3GradV2CompileInfo>(TilingPrepare4PadV3GradV2)
    .TilingInputsDataDependency({1});
}  // namespace optiling