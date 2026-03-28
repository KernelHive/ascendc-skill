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
 * @file pad_v4_grad.cpp
 */
#include <map>
#include <string>
#include <vector>

#include "pad_v4_grad_tilingdata.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace ops {
#define OP_TILING_CHECK(cond, log_func, expr)   \
  do {                                          \
    if (cond) {                                 \
        std::printf(log_func);                  \
        expr;                                   \
    }                                           \
  } while (0)

#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                           \
    if ((ptr) == nullptr) {                                                                          \
        const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
        std::printf(name, "%s is nullptr!", #ptr);                                                   \
        return ret;                                                                                  \
    }

#define OP_LOGD(opname, ...)
#define OP_LOGI(opname, ...)
}//namespace ops

template <typename T>
inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
    return context->GetCompiledInfo<T>();
}

namespace optiling {

constexpr uint32_t BYTE_BLOCK = 32;
constexpr size_t MODE_INDEX = 0;
constexpr int32_t X_INPUT_INDEX = 0;
constexpr int32_t PAD_INPUT_INDEX = 1;
constexpr int32_t FLOAT_BYTES = 4;
constexpr int32_t FLOAT16_BYTES = 2;
constexpr size_t CHECK_DIM_NUM = 4;
constexpr uint32_t FLOAT_MINI_SHAPE_TILING_KEY = 1000;
constexpr uint32_t FLOAT_SMALL_H_LARGE_W_TILING_KEY = 1100;
constexpr uint32_t FLOAT_LARGE_H_SMALL_W_TILING_KEY = 1010;
constexpr uint32_t FLOAT_NO_W_PAD_TILING_KEY = 1110;
constexpr uint32_t FLOAT_NO_H_PAD_TILING_KEY = 1101;
constexpr uint32_t FLOAT_H_W_PAD_TILING_KEY = 1111;
constexpr uint32_t FLOAT16_MINI_SHAPE_TILING_KEY = 2000;
constexpr uint32_t FLOAT16_SMALL_H_LARGE_W_TILING_KEY = 2100;
constexpr uint32_t FLOAT16_LARGE_H_SMALL_W_TILING_KEY = 2010;
constexpr uint32_t FLOAT16_NO_W_PAD_TILING_KEY = 2110;
constexpr uint32_t FLOAT16_NO_H_PAD_TILING_KEY = 2101;
constexpr uint32_t FLOAT16_H_W_PAD_TILING_KEY = 2111;
constexpr uint32_t BFLOAT16_MINI_SHAPE_TILING_KEY = 3000;
constexpr uint32_t BFLOAT16_SMALL_H_LARGE_W_TILING_KEY = 3100;
constexpr uint32_t BFLOAT16_LARGE_H_SMALL_W_TILING_KEY = 3010;
constexpr uint32_t BFLOAT16_NO_W_PAD_TILING_KEY = 3110;
constexpr uint32_t BFLOAT16_NO_H_PAD_TILING_KEY = 3101;
constexpr uint32_t BFLOAT16_H_W_PAD_TILING_KEY = 3111;
constexpr uint32_t FLOAT_EDGE_TILING_KEY = 12;
constexpr uint32_t FLOAT16_EDGE_TILING_KEY = 22;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t DIM_INDEX2 = 2;
constexpr uint32_t DIM_INDEX3 = 3;
constexpr uint32_t PADDING_NUM_INDEX4 = 4;
constexpr uint32_t PADDING_NUM_INDEX5 = 5;
constexpr uint32_t PADDING_NUM_INDEX6 = 6;
constexpr uint32_t PADDING_NUM_INDEX7 = 7;
constexpr uint32_t RESERVED_UB = 11 * 1024;
constexpr uint32_t ALIGN_256_BYTES = 256;
constexpr uint32_t ALIGN_16 = 16;
constexpr uint32_t TRANSPOSE_LINES = 16;
constexpr uint32_t CAL_COUNT = 64;
constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t WORK_SPACE_PART = 64;
constexpr int32_t MINI_SHAPE_MAX_HEIGHT = 128;
constexpr int32_t FLOAT16_MINI_SHAPE_MAX_WIDTH = 224;
constexpr int32_t FLOAT_MINI_SHAPE_MAX_WIDTH = 112;
constexpr int32_t BFLOAT16_MINI_SHAPE_MAX_WIDTH = 112;
constexpr int32_t SMALL_W_LIMIT = 128;
constexpr int32_t SMALL_H_LIMIT = 64;
constexpr uint32_t CONST_VALUE_2 = 2;
constexpr uint32_t CONST_VALUE_3 = 3;
constexpr uint32_t CONST_VALUE_4 = 4;
constexpr uint32_t CONST_VALUE_5 = 5;
constexpr uint32_t CONST_VALUE_6 = 6;
constexpr uint32_t CONST_VALUE_8 = 8;
constexpr uint32_t CONST_VALUE_12 = 12;
constexpr uint32_t REFLECTION_MODE = 0;
constexpr uint32_t EDGE_MODE = 1;
constexpr uint32_t FLOAT_DTYPE = 1;
constexpr uint32_t FLOAT16_DTYPE = 2;
constexpr uint32_t BF16_DTYPE = 3;
constexpr int32_t W_PAD_LOWER_LIMIT = 16;
static std::map<std::string, int> PADDING_MODE_MAP = {{"reflect", 0}, {"edge", 1}, {"constant", 2}};
static std::map<ge::DataType, uint32_t> DTYPE_MAP = {{ge::DT_FLOAT, 1}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 3}};
static std::map<ge::DataType, int32_t> DATATYPE_LEN_MAP = {{ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}, {ge::DT_FLOAT, 4}};

template <typename TilingData, int32_t dataTypeLen>
class PadV3GradV2Tiling {
public:
    explicit PadV3GradV2Tiling(InputParamsInfo &param, const uint32_t inputCoreNum, const uint32_t inputUbSize) {
        this->batch = param.batch;
        this->channel = param.channel;
        this->height = param.height;
        this->width = param.width;
        this->alignHeight = param.alignHeight;
        this->alignWidth = param.alignWidth;
        this->outHeight = param.outHeight;
        this->outWidth = param.outWidth;
        this->alignOutHeight = param.alignOutHeight;
        this->alignOutWidth = param.alignOutWidth;
        this->hPad1 = param.hPad1;
        this->hPad2 = param.hPad2;
        this->wPad1 = param.wPad1;
        this->wPad2 = param.wPad2;
        this->mode = param.mode;
        this->dtype = param.dtype;
        this->ubSize = FloorAlign(inputUbSize, BYTE_BLOCK);
        this->dataTypeSize = dataTypeLen;
        this->elementsPerBlock = BYTE_BLOCK / dataTypeSize;
        this->coreNum = inputCoreNum;
        return;
    }

    void GetTiling(TilingData *tilingData);

private:
    void GetTilingKey();
    void GetUsedCore();
    void SplitUb();
    void FillTilingData(TilingData *tilingData);
    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b) {
        return (a + b - 1) / b;
    }
    template <typename T1, typename T2>
    inline T1 FloorDiv(T1 a, T2 b) {
        return (a) / (b);
    }
    template <typename T1, typename T2>
    inline T1 CeilAlign(T1 a, T2 b) {
        return (a + b - 1) / b * b;
    }
    template <typename T1, typename T2>
    inline T1 FloorAlign(T1 a, T2 b) {
        return (a) / b * b;
    }

private:
    uint32_t batch = 0;
    uint32_t channel = 0;
    uint32_t height = 0;
    uint32_t width = 0;
    uint32_t alignHeight = 0;
    uint32_t alignWidth = 0;
    uint32_t outHeight = 0;
    uint32_t outWidth = 0;
    uint32_t alignOutHeight = 0;
    uint32_t alignOutWidth = 0;
    int32_t hPad1 = 0;
    int32_t hPad2 = 0;
    int32_t wPad1 = 0;
    int32_t wPad2 = 0;
    uint32_t mode = 0;
    uint32_t ubSize = 0;
    uint32_t usedCoreNum = 0;
    uint32_t coreNum = 0;
    uint32_t ncPerCore = 1;
    uint32_t tailNC = 0;
    uint32_t ubFactorElement = 0;
    uint32_t tilingKey = 0;
    uint32_t dtype = 1;
    uint8_t dataTypeSize = 0;
    uint8_t elementsPerBlock = 0;
    uint32_t divideUbNum = 1;
    uint32_t wPadCopyCount = 0;
    uint64_t workspacePerCore = 0;
};

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::GetTilingKey() {
    if (dtype == FLOAT_DTYPE && mode == REFLECTION_MODE) {
        if (height <= MINI_SHAPE_MAX_HEIGHT && width <= FLOAT_MINI_SHAPE_MAX_WIDTH) {
            tilingKey = FLOAT_MINI_SHAPE_TILING_KEY;  // float, mini shape
            divideUbNum = CONST_VALUE_3;
        } else if (wPad1 == 0 && wPad2 == 0 && (hPad1 != 0 || hPad2 != 0) && height > SMALL_H_LIMIT) {
            tilingKey = FLOAT_NO_W_PAD_TILING_KEY;  // float, w dim no pad
            divideUbNum = CONST_VALUE_8;
        } else if (hPad1 == 0 && hPad2 == 0 && (wPad1 != 0 || wPad2 != 0) && width > SMALL_W_LIMIT) {
            tilingKey = FLOAT_NO_H_PAD_TILING_KEY;  // float, reflect, h dim no pad
            divideUbNum = CONST_VALUE_4;
        } else if (height <= SMALL_H_LIMIT && width > FLOAT_MINI_SHAPE_MAX_WIDTH) {
            tilingKey = FLOAT_SMALL_H_LARGE_W_TILING_KEY;  // float, mini h dim
            divideUbNum = CONST_VALUE_3;
        } else if (height > MINI_SHAPE_MAX_HEIGHT && width <= SMALL_W_LIMIT) {
            tilingKey = FLOAT_LARGE_H_SMALL_W_TILING_KEY;  // float, mini w dim
            divideUbNum = CONST_VALUE_3;
        } else {
            tilingKey = FLOAT_H_W_PAD_TILING_KEY;  // float32, big shape, impl h_dim and w_dim pad
            divideUbNum = CONST_VALUE_5;
        }
    } else if (dtype == FLOAT_DTYPE && mode == EDGE_MODE) {
        tilingKey = FLOAT_EDGE_TILING_KEY;  // mode2: float, edge
    } else if (dtype == FLOAT16_DTYPE && mode == REFLECTION_MODE) {
        if (height <= MINI_SHAPE_MAX_HEIGHT && width <= FLOAT16_MINI_SHAPE_MAX_WIDTH) {
            tilingKey = FLOAT16_MINI_SHAPE_TILING_KEY;  // float16, mini shape
            divideUbNum = CONST_VALUE_3;
        } else if (wPad1 == 0 && wPad2 == 0 && (hPad1 != 0 || hPad2 != 0) && height > SMALL_H_LIMIT) {
            tilingKey = FLOAT16_NO_W_PAD_TILING_KEY;  // w dim no pad
            divideUbNum = CONST_VALUE_8;
        } else if (hPad1 == 0 && hPad2 == 0 && (wPad1 != 0 || wPad2 != 0) && width > SMALL_W_LIMIT) {
            tilingKey = FLOAT16_NO_H_PAD_TILING_KEY;  // h dim no pad
            divideUbNum = CONST_VALUE_4;
        } else if (height <= SMALL_H_LIMIT && width > FLOAT16_MINI_SHAPE_MAX_WIDTH) {
            tilingKey = FLOAT16_SMALL_H_LARGE_W_TILING_KEY;  // float, mini h dim
            divideUbNum = CONST_VALUE_3;
        } else if (height > MINI_SHAPE_MAX_HEIGHT && width <= SMALL_W_LIMIT) {
            tilingKey = FLOAT16_LARGE_H_SMALL_W_TILING_KEY;  // float, mini w dim
            divideUbNum = CONST_VALUE_3;
        } else {
            tilingKey = FLOAT16_H_W_PAD_TILING_KEY;  // float16, big shape, impl h_dim and w_dim pad
            divideUbNum = CONST_VALUE_5;
        }
    } else if (dtype == FLOAT16_DTYPE && mode == EDGE_MODE) {
        tilingKey = FLOAT16_EDGE_TILING_KEY;  // mode2: float16, edge
    } else if (dtype == BF16_DTYPE && mode == REFLECTION_MODE) {
        if (height <= MINI_SHAPE_MAX_HEIGHT && width <= BFLOAT16_MINI_SHAPE_MAX_WIDTH) {
            tilingKey = BFLOAT16_MINI_SHAPE_TILING_KEY;  // bfloat16, mini shape
            divideUbNum = CONST_VALUE_6;
        } else if (wPad1 == 0 && wPad2 == 0 && (hPad1 != 0 || hPad2 != 0) && height > SMALL_H_LIMIT) {
            tilingKey = BFLOAT16_NO_W_PAD_TILING_KEY;  // w dim no pad
            divideUbNum = CONST_VALUE_12;
        } else if (hPad1 == 0 && hPad2 == 0 && (wPad1 != 0 || wPad2 != 0) && width > SMALL_W_LIMIT) {
            tilingKey = BFLOAT16_NO_H_PAD_TILING_KEY;  // h dim no pad
            divideUbNum = CONST_VALUE_6;
        } else if (height <= SMALL_H_LIMIT && width > BFLOAT16_MINI_SHAPE_MAX_WIDTH) {
            tilingKey = BFLOAT16_SMALL_H_LARGE_W_TILING_KEY;  // bfloat16, mini h dim
            divideUbNum = CONST_VALUE_6;
        } else if (height > MINI_SHAPE_MAX_HEIGHT && width <= SMALL_W_LIMIT) {
            tilingKey = BFLOAT16_LARGE_H_SMALL_W_TILING_KEY;  // bfloat16, mini w dim
            divideUbNum = CONST_VALUE_6;
        } else {
            tilingKey = BFLOAT16_H_W_PAD_TILING_KEY;  // bfloat16, big shape, impl h_dim and w_dim pad
            divideUbNum = CONST_VALUE_8;
        }
    }
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::GetUsedCore() {
    uint64_t nMulC = batch * channel;
    if (tilingKey == FLOAT_NO_H_PAD_TILING_KEY || tilingKey == FLOAT16_NO_H_PAD_TILING_KEY ||
        tilingKey == BFLOAT16_NO_H_PAD_TILING_KEY) {
        nMulC = nMulC * height;
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
void PadV3GradV2Tiling<TilingData, dataTypeLen>::SplitUb() {
    uint32_t tilingDataSize = CeilAlign(sizeof(TilingData), BYTE_BLOCK);
    uint32_t canUseUbSize = FloorAlign(ubSize - tilingDataSize, BYTE_BLOCK);
    if (tilingKey == FLOAT_H_W_PAD_TILING_KEY || tilingKey == FLOAT16_H_W_PAD_TILING_KEY ||
        tilingKey == BFLOAT16_H_W_PAD_TILING_KEY) {
        ubFactorElement =
            FloorAlign(FloorAlign(canUseUbSize / divideUbNum / TRANSPOSE_LINES, BYTE_BLOCK) / dataTypeLen, ALIGN_16);
    } else if (tilingKey == FLOAT_MINI_SHAPE_TILING_KEY || tilingKey == FLOAT16_MINI_SHAPE_TILING_KEY ||
            tilingKey == BFLOAT16_MINI_SHAPE_TILING_KEY) {
        ubFactorElement = FloorAlign(
            FloorAlign(canUseUbSize / divideUbNum / MINI_SHAPE_MAX_HEIGHT, BYTE_BLOCK) / dataTypeLen, ALIGN_16);
    } else if (tilingKey == FLOAT_SMALL_H_LARGE_W_TILING_KEY || tilingKey == FLOAT16_SMALL_H_LARGE_W_TILING_KEY ||
            tilingKey == BFLOAT16_SMALL_H_LARGE_W_TILING_KEY) {
        ubFactorElement =
            FloorAlign(FloorAlign(canUseUbSize / divideUbNum / SMALL_H_LIMIT, BYTE_BLOCK) / dataTypeLen, ALIGN_16);
    } else if (tilingKey == FLOAT_LARGE_H_SMALL_W_TILING_KEY || tilingKey == FLOAT16_LARGE_H_SMALL_W_TILING_KEY ||
            tilingKey == BFLOAT16_LARGE_H_SMALL_W_TILING_KEY) {
        ubFactorElement =
            FloorAlign(FloorAlign(canUseUbSize / divideUbNum / SMALL_W_LIMIT, BYTE_BLOCK) / dataTypeLen, ALIGN_16);
    } else {
        ubFactorElement = FloorAlign(canUseUbSize / divideUbNum, ALIGN_256_BYTES) / dataTypeLen;
    }
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::FillTilingData(TilingData *tilingData) {
    tilingData->set_batch(batch);
    tilingData->set_channel(channel);
    tilingData->set_height(height);
    tilingData->set_width(width);
    tilingData->set_alignHeight(alignHeight);
    tilingData->set_alignWidth(alignWidth);
    tilingData->set_outHeight(outHeight);
    tilingData->set_outWidth(outWidth);
    tilingData->set_alignOutHeight(alignOutHeight);
    tilingData->set_alignOutWidth(alignOutWidth);
    tilingData->set_hPad1(hPad1);
    tilingData->set_hPad2(hPad2);
    tilingData->set_wPad1(wPad1);
    tilingData->set_wPad2(wPad2);
    tilingData->set_blockNum(usedCoreNum);
    tilingData->set_ubFactorElement(ubFactorElement);
    tilingData->set_ncPerCore(ncPerCore);
    tilingData->set_tailNC(tailNC);
    tilingData->set_tilingKey(tilingKey);
    if (tilingKey == FLOAT_NO_W_PAD_TILING_KEY || tilingKey == FLOAT16_NO_W_PAD_TILING_KEY ||
        tilingKey == BFLOAT16_NO_W_PAD_TILING_KEY || tilingKey == FLOAT_MINI_SHAPE_TILING_KEY ||
        tilingKey == FLOAT16_MINI_SHAPE_TILING_KEY || tilingKey == BFLOAT16_MINI_SHAPE_TILING_KEY) {
        workspacePerCore = 0;
    } else if (tilingKey == FLOAT_NO_H_PAD_TILING_KEY || tilingKey == FLOAT16_NO_H_PAD_TILING_KEY ||
            tilingKey == BFLOAT16_NO_H_PAD_TILING_KEY) {
        wPadCopyCount = CeilAlign(CONST_VALUE_2 * std::max(wPad1, wPad2) + 1, elementsPerBlock);
        int32_t partWs = wPad1 < W_PAD_LOWER_LIMIT && wPad2 < W_PAD_LOWER_LIMIT ? CAL_COUNT : wPadCopyCount;
        workspacePerCore = CONST_VALUE_2 * partWs * dataTypeSize;
    } else {
        workspacePerCore = alignWidth * WORK_SPACE_PART * dataTypeSize;
    }
    tilingData->set_wPadCopyCount(wPadCopyCount);
    tilingData->set_workspacePerCore(workspacePerCore);
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradV2Tiling<TilingData, dataTypeLen>::GetTiling(TilingData *tilingData) {
    GetTilingKey();
    GetUsedCore();
    SplitUb();
    FillTilingData(tilingData);
}

template <typename TilingData, int32_t dataTypeLen>
void GetPadV3GradV2Tiling(TilingData *tilingData, InputParamsInfo &params, uint32_t coreNum, uint32_t ubSize) {
    class PadV3GradV2Tiling<TilingData, dataTypeLen> tilingObj(params, coreNum, ubSize);
    tilingObj.GetTiling(tilingData);
}

static void PrintTilingData(gert::TilingContext *tilingContext, PadV4GradTilingData &tilingData,
                            const size_t usrWorkspace) {
    OP_LOGD(tilingContext->GetNodeName(), "Start printing");
    OP_LOGD(tilingContext->GetNodeName(), "batch is %u.", tilingData.get_batch());
    OP_LOGD(tilingContext->GetNodeName(), "channel is %u.", tilingData.get_channel());
    OP_LOGD(tilingContext->GetNodeName(), "height is %u.", tilingData.get_height());
    OP_LOGD(tilingContext->GetNodeName(), "width is %u.", tilingData.get_width());
    OP_LOGD(tilingContext->GetNodeName(), "alignHeight is %u.", tilingData.get_alignHeight());
    OP_LOGD(tilingContext->GetNodeName(), "alignWidth is %u.", tilingData.get_alignWidth());
    OP_LOGD(tilingContext->GetNodeName(), "outHeight is %u.", tilingData.get_outHeight());
    OP_LOGD(tilingContext->GetNodeName(), "outWidth is %u.", tilingData.get_outWidth());
    OP_LOGD(tilingContext->GetNodeName(), "alignOutHeight is %u.", tilingData.get_alignOutHeight());
    OP_LOGD(tilingContext->GetNodeName(), "alignOutWidth is %u.", tilingData.get_alignOutWidth());
    OP_LOGD(tilingContext->GetNodeName(), "hPad1 is %u.", tilingData.get_hPad1());
    OP_LOGD(tilingContext->GetNodeName(), "hPad2 is %u.", tilingData.get_hPad2());
    OP_LOGD(tilingContext->GetNodeName(), "wPad1 is %u.", tilingData.get_wPad1());
    OP_LOGD(tilingContext->GetNodeName(), "wPad2 is %u.", tilingData.get_wPad2());
    OP_LOGD(tilingContext->GetNodeName(), "blockNum is %u.", tilingData.get_blockNum());
    OP_LOGD(tilingContext->GetNodeName(), "ubFactorElement is %u.", tilingData.get_ubFactorElement());
    OP_LOGD(tilingContext->GetNodeName(), "ncPerCore is %u.", tilingData.get_ncPerCore());
    OP_LOGD(tilingContext->GetNodeName(), "tailNC is %u.", tilingData.get_tailNC());
    OP_LOGD(tilingContext->GetNodeName(), "tilingKey is %u.", tilingData.get_tilingKey());
    OP_LOGD(tilingContext->GetNodeName(), "usrWorkspace is %lu.", usrWorkspace);
    OP_LOGD(tilingContext->GetNodeName(), "End printing");
}

template <typename T1, typename T2>
static ge::graphStatus CeilAlign(T1 a, T2 b) {
    return (a + b - 1) / b * b;
}

template <typename T>
static ge::graphStatus GetInputInfo(gert::TilingContext *tilingContext, InputParamsInfo &params) {
    OP_LOGI(tilingContext->GetNodeName(), "strat to get input dims");
    const gert::StorageShape *xShape = tilingContext->GetInputShape(X_INPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(tilingContext, xShape, ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        xShape->GetStorageShape().GetDimNum() != CHECK_DIM_NUM,
        "input dim is not 4, please check input.", return ge::GRAPH_FAILED);
    const gert::StorageShape *paddingShape = tilingContext->GetInputShape(PAD_INPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(tilingContext, paddingShape, ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        static_cast<int32_t>(xShape->GetStorageShape().GetDimNum() * 2) !=
            static_cast<int32_t>(paddingShape->GetStorageShape().GetDim(0)),
        "Please check input or padding shape", return ge::GRAPH_FAILED);
    const gert::Tensor *paddings_tensor = tilingContext->GetInputTensor(PAD_INPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(tilingContext, paddings_tensor, ge::GRAPH_FAILED);

    const T *paddingsValue = paddings_tensor->GetData<T>();

    params.hPad1 = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX4]);
    params.hPad2 = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX5]);
    params.wPad1 = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX6]);
    params.wPad2 = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX7]);

    const gert::StorageShape *outShape = tilingContext->GetOutputShape(0);
    uint32_t outHeight = outShape->GetStorageShape().GetDim(DIM_INDEX2);
    uint32_t outWidth = outShape->GetStorageShape().GetDim(DIM_INDEX3);
    params.batch = xShape->GetStorageShape().GetDim(DIM_INDEX0);
    params.channel = xShape->GetStorageShape().GetDim(DIM_INDEX1);
    params.height = xShape->GetStorageShape().GetDim(DIM_INDEX2);
    params.width = xShape->GetStorageShape().GetDim(DIM_INDEX3);
    params.outHeight = outHeight;
    params.outWidth = outWidth;

    OP_TILING_CHECK(
        (outHeight != (params.height - params.hPad1 - params.hPad2)) ||
            (outWidth != (params.width - params.wPad1 - params.wPad2)),
        "Please check input or output shape", return ge::GRAPH_FAILED);

    params.alignHeight = CeilAlign(params.height, ALIGN_16);
    params.alignWidth = CeilAlign(params.width, ALIGN_16);
    params.alignOutHeight = CeilAlign(params.outHeight, ALIGN_16);
    params.alignOutWidth = CeilAlign(params.outWidth, ALIGN_16);

    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
                    "Get attrs Failed.", return ge::GRAPH_FAILED);
    const std::string mode = std::string(attrs->GetAttrPointer<char>(MODE_INDEX));
    OP_TILING_CHECK(mode != "reflect" && mode != "edge" && mode != "constant",
                    " is not supported", return ge::GRAPH_FAILED);

    params.mode = PADDING_MODE_MAP[mode];
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4PadV4Grad(gert::TilingContext *tilingContext) {
    OP_LOGI(tilingContext->GetNodeName(), "PadV3GradV2 tiling starts running");
    auto compileInfo = reinterpret_cast<const Tiling4PadV4GradCompileInfo *>(tilingContext->GetCompileInfo());
    OPS_CHECK_NULL_WITH_CONTEXT_RET(tilingContext, compileInfo, ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm = compileInfo->ubSizePlatForm;
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);
    uint32_t availableUb = ubSize - RESERVED_UB;
    uint32_t coreNum = compileInfo->coreNum;
    OP_LOGI(tilingContext->GetNodeName(), "ubSizePlatForm:%lu, coreNum:%u", ubSizePlatForm, coreNum);
    uint32_t sysWorkspaceSize = compileInfo->sysWorkspaceSize;
    OP_TILING_CHECK(coreNum <= 0,
                    "Failed to get core num.", return ge::GRAPH_FAILED);

    OP_TILING_CHECK(ubSizePlatForm <= 0,
                    "Failed to get ub size.", return ge::GRAPH_FAILED);

    ge::DataType inputDatatype = tilingContext->GetInputDesc(0)->GetDataType();
    OP_TILING_CHECK(inputDatatype != ge::DT_FLOAT && inputDatatype != ge::DT_FLOAT16 && inputDatatype != ge::DT_BF16,
                    "the current x dtype is not in dtype support list [bfloat16, float16, float].",
                    return ge::GRAPH_FAILED);

    ge::DataType paddingDatatype = tilingContext->GetInputDesc(1)->GetDataType();
    OP_TILING_CHECK(
        paddingDatatype != ge::DT_INT32 && paddingDatatype != ge::DT_INT64,
        "the current padding dtype is not in dtype support list [int32, int64].",
        return ge::GRAPH_FAILED);
    InputParamsInfo params;
    params.dtype = DTYPE_MAP[inputDatatype];

    if (paddingDatatype == ge::DT_INT32) {
        OP_TILING_CHECK(GetInputInfo<int32_t>(tilingContext, params) != ge::GRAPH_SUCCESS,
                        "get op inputs failed.", return ge::GRAPH_FAILED);
    } else if (paddingDatatype == ge::DT_INT64) {
        OP_TILING_CHECK(GetInputInfo<int64_t>(tilingContext, params) != ge::GRAPH_SUCCESS,
                        "get op inputs failed.", return ge::GRAPH_FAILED);
    }

    PadV4GradTilingData tilingData;
    if (inputDatatype == ge::DT_FLOAT) {
        GetPadV3GradV2Tiling<PadV4GradTilingData, FLOAT_BYTES>(&tilingData, params, coreNum, availableUb);
    } else {
        GetPadV3GradV2Tiling<PadV4GradTilingData, FLOAT16_BYTES>(&tilingData, params, coreNum, availableUb);
    }

    OP_TILING_CHECK(
        tilingData.get_ubFactorElement() <= 0,
        "ub space is not enough, please check input.", return ge::GRAPH_FAILED);
    // set tilingdata
    uint64_t workspacePerCore = tilingData.get_workspacePerCore();
    uint32_t tilingKey = tilingData.get_tilingKey();
    uint32_t blockNum = tilingData.get_blockNum();
    size_t usrWorkspace = workspacePerCore * blockNum;
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(blockNum);
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);

    workspaces[0] = usrWorkspace + sysWorkspaceSize;
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    PrintTilingData(tilingContext, tilingData, usrWorkspace);
    OP_LOGI(tilingContext->GetNodeName(), "PadV3GradV2 tiling end running");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4PadV4Grad(gert::TilingParseContext *context) {
    OP_LOGI(context->GetNodeName(), "TilingPrepare4PadV4Grad start.");
    auto compileInfo = GetCompileInfoPtr<Tiling4PadV4GradCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, compileInfo, ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT_RET(context, platformInfo, ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK((compileInfo->coreNum <= 0),
                    "Failed to get core num.", return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_TILING_CHECK(ubSizePlatForm <= 0,
                    "Failed to get ub size.", return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGI(context->GetNodeName(), "TilingPrepare4PadV4Grad end.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PadV4Grad)
    .Tiling(Tiling4PadV4Grad)
    .TilingParse<Tiling4PadV4GradCompileInfo>(TilingPrepare4PadV4Grad)
    .TilingInputsDataDependency({1});
}  // namespace optiling
