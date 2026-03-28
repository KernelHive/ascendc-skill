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
 * @file pad_v3_grad_replicate.cpp
 */
#include <map>
#include <vector>
#include <string>
#include "pad_v3_grad_replicate_tiling.h"
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
static inline const std::string &GetOpInfo(const std::string &str)
{
    return str;
}
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

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
} // namespace

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
constexpr uint32_t FLOAT_H_W_ONE_TILING_KEY = 11111;
constexpr uint32_t FLOAT16_MINI_SHAPE_TILING_KEY = 2000;
constexpr uint32_t FLOAT16_SMALL_H_LARGE_W_TILING_KEY = 2100;
constexpr uint32_t FLOAT16_LARGE_H_SMALL_W_TILING_KEY = 2010;
constexpr uint32_t FLOAT16_NO_W_PAD_TILING_KEY = 2110;
constexpr uint32_t FLOAT16_NO_H_PAD_TILING_KEY = 2101;
constexpr uint32_t FLOAT16_H_W_PAD_TILING_KEY = 2111;
constexpr uint32_t FLOAT16_H_W_ONE_TILING_KEY = 22222;
constexpr uint32_t BFLOAT16_MINI_SHAPE_TILING_KEY = 3000;
constexpr uint32_t BFLOAT16_SMALL_H_LARGE_W_TILING_KEY = 3100;
constexpr uint32_t BFLOAT16_LARGE_H_SMALL_W_TILING_KEY = 3010;
constexpr uint32_t BFLOAT16_NO_W_PAD_TILING_KEY = 3110;
constexpr uint32_t BFLOAT16_NO_H_PAD_TILING_KEY = 3101;
constexpr uint32_t BFLOAT16_H_W_PAD_TILING_KEY = 3111;
constexpr uint32_t BFLOAT16_H_W_ONE_TILING_KEY = 33333;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t DIM_INDEX2 = 2;
constexpr uint32_t DIM_INDEX3 = 3;
constexpr uint32_t PADDING_NUM_INDEX4 = 4;
constexpr uint32_t PADDING_NUM_INDEX5 = 5;
constexpr uint32_t PADDING_NUM_INDEX6 = 6;
constexpr uint32_t PADDING_NUM_INDEX7 = 7;
constexpr uint32_t RESERVED_UB = 32 * 1024;
constexpr uint32_t ALIGN_256_BYTES = 256;
constexpr uint32_t ALIGN_16 = 16;
constexpr uint32_t TRANSPOSE_LINES = 16;
constexpr uint32_t CAL_COUNT = 64;
constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t WORK_SPACE_PART = 64;
constexpr uint32_t SMALL_W_LIMIT = 64;
constexpr uint32_t SMALL_H_LIMIT = 64;
constexpr uint32_t CONST_VALUE_2 = 2;       //CONST_VALUE都是作用域ub切分的个数
constexpr uint32_t CONST_VALUE_3 = 3;
constexpr uint32_t CONST_VALUE_4 = 4;
constexpr uint32_t CONST_VALUE_5 = 5;
constexpr uint32_t CONST_VALUE_6 = 6;
constexpr uint32_t CONST_VALUE_8 = 8;
constexpr uint32_t CONST_VALUE_12 = 12;
constexpr uint32_t EDGE_MODE = 1;
constexpr uint32_t FLOAT_DTYPE = 1;
constexpr uint32_t FLOAT16_DTYPE = 2;
constexpr uint32_t BF16_DTYPE = 3;
static std::map<std::string, int> PADDING_MODE_MAP = {{"reflect", 0}, {"edge", 1}, {"constant", 2}};
static std::map<ge::DataType, uint32_t> DTYPE_MAP = {{ge::DT_FLOAT, 1}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 3}};
static std::map<ge::DataType, int32_t> DATATYPE_LEN_MAP = {{ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}};

template <typename TilingData, int32_t dataTypeLen>
class PadV3GradReplicateTiling {
public:
    explicit PadV3GradReplicateTiling(InputParamsInfo& param, const uint32_t inputCoreNum,
                                        const uint32_t inputUbSize) {
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
        this->padTop = param.padTop;
        this->padBottom = param.padBottom;
        this->padLeft = param.padLeft;
        this->padRight = param.padRight;
        this->mode = param.mode;
        this->dtype = param.dtype;
        this->ubSize = FloorAlign(inputUbSize, BYTE_BLOCK);
        this->dataTypeSize = dataTypeLen;
        this->elementsPerBlock = BYTE_BLOCK / dataTypeSize;
        this->coreNum = inputCoreNum;
        this->wCalCount = CeilAlign(std::max(param.padLeft, param.padRight) + 1, BYTE_BLOCK);
        return;
    }

    void GetTiling(TilingData* tilingData);

private:
    void GetTilingKey();
    void GetUsedCore();
    void SplitUb();
    void FillTilingData(TilingData* tilingData);
    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b) {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    }
    template <typename T1, typename T2>
    inline T1 FloorDiv(T1 a, T2 b) {
        if (b == 0) {
            return a;
        }
        return a / b;
    }
    template <typename T1, typename T2>
    inline T1 CeilAlign(T1 a, T2 b) {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b * b;
    }
    template <typename T1, typename T2>
    inline T1 FloorAlign(T1 a, T2 b) {
        if (b == 0) {
            return a;
        }
        return a / b * b;
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
    int32_t padTop = 0;
    int32_t padBottom = 0;
    int32_t padLeft = 0;
    int32_t padRight = 0;
    uint32_t mode = 1;
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
    uint64_t workspacePerCore = 0;
    uint32_t wCalCount = 0;
};

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradReplicateTiling<TilingData, dataTypeLen>::GetTilingKey() {
    if (dtype == FLOAT_DTYPE && mode == EDGE_MODE) {
        if (padLeft == 0 && padRight == 0 && (padTop != 0 || padBottom != 0)) {
            tilingKey = FLOAT_NO_W_PAD_TILING_KEY;  // mode1: float, replicate, w dim no pad
            divideUbNum = CONST_VALUE_8;
        } else if (padTop == 0 && padBottom == 0 && (padLeft != 0 || padRight != 0)) {
            tilingKey = FLOAT_NO_H_PAD_TILING_KEY;  // mode1: float, replicate, h dim no pad
            divideUbNum = CONST_VALUE_2;
        } else if (height <= SMALL_H_LIMIT && width <= SMALL_W_LIMIT) {
            tilingKey = FLOAT_MINI_SHAPE_TILING_KEY;  // mode1: float, replicate, h and w dim pad, small shape
            divideUbNum = CONST_VALUE_4;
        } else if (height <= SMALL_H_LIMIT && width > SMALL_W_LIMIT) {
            tilingKey = FLOAT_SMALL_H_LARGE_W_TILING_KEY;  // float, mini h dim
            divideUbNum = CONST_VALUE_3;
        } else if (height > SMALL_H_LIMIT && width <= SMALL_W_LIMIT) {
            tilingKey = FLOAT_LARGE_H_SMALL_W_TILING_KEY;  // float, mini w dim
            divideUbNum = CONST_VALUE_3;
        } else if (outHeight == 1) {
            tilingKey = FLOAT_H_W_ONE_TILING_KEY;  // mode1: float, replicate, h and w dim pad, outHeight == 1
            divideUbNum = CONST_VALUE_4;
        } else {
            tilingKey = FLOAT_H_W_PAD_TILING_KEY;  // mode1: float, replicate, h and w dim pad, big shape
            divideUbNum = CONST_VALUE_4;
        }
    } else if (dtype == FLOAT16_DTYPE && mode == EDGE_MODE) {
        if (padLeft == 0 && padRight == 0 && (padTop != 0 || padBottom != 0)) {
            tilingKey = FLOAT16_NO_W_PAD_TILING_KEY;  // mode1: float16, replicate, w dim no pad
            divideUbNum = CONST_VALUE_8;
        } else if (padTop == 0 && padBottom == 0 && (padLeft != 0 || padRight != 0)) {
            tilingKey = FLOAT16_NO_H_PAD_TILING_KEY;  // mode1: float16, replicate, h dim no pad
            divideUbNum = CONST_VALUE_2;
        } else if (height <= SMALL_H_LIMIT && width <= SMALL_W_LIMIT) {
            tilingKey = FLOAT16_MINI_SHAPE_TILING_KEY;  // mode1: float16, replicate, h and w dim pad, small shape
            divideUbNum = CONST_VALUE_4;
        } else if (height <= SMALL_H_LIMIT && width > SMALL_W_LIMIT) {
            tilingKey = FLOAT16_SMALL_H_LARGE_W_TILING_KEY;  // float, mini h dim
            divideUbNum = CONST_VALUE_3;
        } else if (height > SMALL_H_LIMIT && width <= SMALL_W_LIMIT) {
            tilingKey = FLOAT16_LARGE_H_SMALL_W_TILING_KEY;  // float, mini w dim
            divideUbNum = CONST_VALUE_3;
        } else if (outHeight == 1) {
            tilingKey = FLOAT16_H_W_ONE_TILING_KEY;  // mode1: float16, replicate, h and w dim pad, outHeight == 1
            divideUbNum = CONST_VALUE_4;
        } else {
            tilingKey = FLOAT16_H_W_PAD_TILING_KEY;  // mode1: float16, replicate, h and w dim pad, big shape
            divideUbNum = CONST_VALUE_4;
        }
    } else if (dtype == BF16_DTYPE && mode == EDGE_MODE) {
        if (padLeft == 0 && padRight == 0 && (padTop != 0 || padBottom != 0)) {
            tilingKey = BFLOAT16_NO_W_PAD_TILING_KEY;  // mode1: bfloat16, replicate, w dim no pad
            divideUbNum = CONST_VALUE_12;
        } else if (padTop == 0 && padBottom == 0 && (padLeft != 0 || padRight != 0)) {
            tilingKey = BFLOAT16_NO_H_PAD_TILING_KEY;  // mode1: bfloat16, replicate, h dim no pad
            divideUbNum = CONST_VALUE_6;
        } else if (height <= SMALL_H_LIMIT && width <= SMALL_W_LIMIT) {
            tilingKey = BFLOAT16_MINI_SHAPE_TILING_KEY;  // mode1: bfloat16, replicate, h and w dim pad, small shape
            divideUbNum = CONST_VALUE_8;
        } else if (height <= SMALL_H_LIMIT && width > SMALL_W_LIMIT) {
            tilingKey = BFLOAT16_SMALL_H_LARGE_W_TILING_KEY;  // bfloat16, mini h dim
            divideUbNum = CONST_VALUE_6;
        } else if (height > SMALL_H_LIMIT && width <= SMALL_W_LIMIT) {
            tilingKey = BFLOAT16_LARGE_H_SMALL_W_TILING_KEY;  // bfloat16, mini w dim
            divideUbNum = CONST_VALUE_6;
        } else if (outHeight == 1) {
            tilingKey = BFLOAT16_H_W_ONE_TILING_KEY;  // mode1: bfloat16, replicate, h and w dim pad, outHeight == 1
            divideUbNum = CONST_VALUE_8;
        } else {
            tilingKey = BFLOAT16_H_W_PAD_TILING_KEY;  // mode1: bfloat16, replicate, h and w dim pad, big shape
            divideUbNum = CONST_VALUE_8;
        }
    }
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradReplicateTiling<TilingData, dataTypeLen>::GetUsedCore() {
    uint64_t nMulC = batch * channel;
    if (tilingKey == FLOAT_NO_H_PAD_TILING_KEY || tilingKey == FLOAT16_NO_H_PAD_TILING_KEY ||
        tilingKey == BFLOAT16_NO_H_PAD_TILING_KEY) {
        nMulC = nMulC * height;
    }
    if (nMulC <= coreNum) {  // 总行数不超过核心数，一行一核
        ncPerCore = 1;
        usedCoreNum = nMulC;
        tailNC = 0;
        return;
    }
    ncPerCore = nMulC / coreNum;  // 总行数大于核心数，按照nc分核
    tailNC = nMulC % coreNum;
    usedCoreNum = coreNum;
    return;
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradReplicateTiling<TilingData, dataTypeLen>::SplitUb() {
    uint32_t tilingDataSize = CeilAlign(sizeof(TilingData), BYTE_BLOCK);
    uint32_t canUseUbSize = FloorAlign(ubSize - tilingDataSize, BYTE_BLOCK);
    if (tilingKey == FLOAT_H_W_PAD_TILING_KEY || tilingKey == FLOAT16_H_W_PAD_TILING_KEY ||
        tilingKey == BFLOAT16_H_W_PAD_TILING_KEY || tilingKey == FLOAT_H_W_ONE_TILING_KEY ||
        tilingKey == FLOAT16_H_W_ONE_TILING_KEY || tilingKey == BFLOAT16_H_W_ONE_TILING_KEY) {
        ubFactorElement = FloorAlign(canUseUbSize / divideUbNum / TRANSPOSE_LINES, ALIGN_256_BYTES) / dataTypeLen;
    } else if (tilingKey == FLOAT_MINI_SHAPE_TILING_KEY || tilingKey == FLOAT16_MINI_SHAPE_TILING_KEY ||
               tilingKey == BFLOAT16_MINI_SHAPE_TILING_KEY) {
        ubFactorElement = FloorAlign(canUseUbSize / divideUbNum / SMALL_H_LIMIT, ALIGN_256_BYTES) / dataTypeLen;
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
void PadV3GradReplicateTiling<TilingData, dataTypeLen>::FillTilingData(TilingData* tilingData) {
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
    tilingData->set_padTop(padTop);
    tilingData->set_padBottom(padBottom);
    tilingData->set_padLeft(padLeft);
    tilingData->set_padRight(padRight);
    tilingData->set_blockNum(usedCoreNum);
    tilingData->set_ubFactorElement(ubFactorElement);
    tilingData->set_ncPerCore(ncPerCore);
    tilingData->set_tailNC(tailNC);
    tilingData->set_tilingKey(tilingKey);
    tilingData->set_wCalCount(wCalCount);
    if (tilingKey == FLOAT_NO_W_PAD_TILING_KEY || tilingKey == FLOAT16_NO_W_PAD_TILING_KEY ||
        tilingKey == BFLOAT16_NO_W_PAD_TILING_KEY) {
        workspacePerCore = 0;
    } else if (tilingKey == FLOAT_NO_H_PAD_TILING_KEY || tilingKey == FLOAT16_NO_H_PAD_TILING_KEY ||
               tilingKey == BFLOAT16_NO_H_PAD_TILING_KEY) {
        workspacePerCore = CONST_VALUE_2 * wCalCount * dataTypeSize;
    } else {
        workspacePerCore = std::max(alignHeight, alignWidth) * WORK_SPACE_PART * dataTypeSize;
    }
    tilingData->set_workspacePerCore(workspacePerCore);
}

template <typename TilingData, int32_t dataTypeLen>
void PadV3GradReplicateTiling<TilingData, dataTypeLen>::GetTiling(TilingData* tilingData) {
    GetTilingKey();
    GetUsedCore();
    SplitUb();
    FillTilingData(tilingData);
}

template <typename TilingData, int32_t dataTypeLen>
void GetPadV3GradReplicateTiling(TilingData* tilingData, InputParamsInfo& params, uint32_t coreNum, uint32_t ubSize) {
    class PadV3GradReplicateTiling<TilingData, dataTypeLen> tilingObj(params, coreNum, ubSize);
    tilingObj.GetTiling(tilingData);
}

static void PrintTilingData(gert::TilingContext* tilingContext, PadV3GradReplicateTilingData& tilingData, 
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
    OP_LOGD(tilingContext->GetNodeName(), "padTop is %d.", tilingData.get_padTop());
    OP_LOGD(tilingContext->GetNodeName(), "padBottom is %d.", tilingData.get_padBottom());
    OP_LOGD(tilingContext->GetNodeName(), "padLeft is %d.", tilingData.get_padLeft());
    OP_LOGD(tilingContext->GetNodeName(), "padRight is %d.", tilingData.get_padRight());
    OP_LOGD(tilingContext->GetNodeName(), "blockNum is %u.", tilingData.get_blockNum());
    OP_LOGD(tilingContext->GetNodeName(), "ubFactorElement is %u.", tilingData.get_ubFactorElement());
    OP_LOGD(tilingContext->GetNodeName(), "ncPerCore is %u.", tilingData.get_ncPerCore());
    OP_LOGD(tilingContext->GetNodeName(), "tailNC is %u.", tilingData.get_tailNC());
    OP_LOGD(tilingContext->GetNodeName(), "tilingKey is %u.", tilingData.get_tilingKey());
    OP_LOGD(tilingContext->GetNodeName(), "wCalCount is %u.", tilingData.get_wCalCount());
    OP_LOGD(tilingContext->GetNodeName(), "usrWorkspace is %lu.", usrWorkspace);
    OP_LOGD(tilingContext->GetNodeName(), "End printing");
}

template <typename T1, typename T2>
static ge::graphStatus CeilAlign(T1 a, T2 b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
static ge::graphStatus GetInputInfo(gert::TilingContext* tilingContext, InputParamsInfo& params) {
    const gert::StorageShape* xShape = tilingContext->GetInputShape(X_INPUT_INDEX);
    OP_TILING_CHECK(
        xShape->GetStorageShape().GetDimNum() != CHECK_DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "input dim is not 4, please check input."),
        return ge::GRAPH_FAILED);
    const gert::StorageShape *paddingShape = tilingContext->GetInputShape(PAD_INPUT_INDEX);
    OP_TILING_CHECK(
        static_cast<int32_t>(xShape->GetStorageShape().GetDimNum() * 2) !=
            static_cast<int32_t>(paddingShape->GetStorageShape().GetDim(0)),
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Please check input or padding shape"),
        return ge::GRAPH_FAILED);
    const gert::Tensor* paddings_tensor = tilingContext->GetInputTensor(PAD_INPUT_INDEX);

    const T* paddingsValue = paddings_tensor->GetData<T>();

    params.padTop = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX4]);
    params.padBottom = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX5]);
    params.padLeft = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX6]);
    params.padRight = static_cast<int32_t>(paddingsValue[PADDING_NUM_INDEX7]);

    const gert::StorageShape* outShape = tilingContext->GetOutputShape(0);
    uint32_t outHeight = outShape->GetStorageShape().GetDim(DIM_INDEX2);
    uint32_t outWidth = outShape->GetStorageShape().GetDim(DIM_INDEX3);

    params.batch = xShape->GetStorageShape().GetDim(DIM_INDEX0);
    params.channel = xShape->GetStorageShape().GetDim(DIM_INDEX1);
    params.height = xShape->GetStorageShape().GetDim(DIM_INDEX2);
    params.width = xShape->GetStorageShape().GetDim(DIM_INDEX3);
    params.outHeight = outHeight;
    params.outWidth = outWidth;

    OP_TILING_CHECK(
        (outHeight != (params.height - params.padTop - params.padBottom)) ||
            (outWidth != (params.width - params.padLeft - params.padRight)),
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Please check input or output shape"),
        return ge::GRAPH_FAILED);

    params.alignHeight = CeilAlign(params.height, ALIGN_16);
    params.alignWidth = CeilAlign(params.width, ALIGN_16);
    params.alignOutHeight = CeilAlign(params.outHeight, ALIGN_16);
    params.alignOutWidth = CeilAlign(params.outWidth, ALIGN_16);

    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Get attrs Failed."),
                    return ge::GRAPH_FAILED);
    const std::string mode = std::string(attrs->GetAttrPointer<char>(MODE_INDEX));
    OP_TILING_CHECK(mode != "reflect" && mode != "edge",
                    VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "%s is not supported", mode.c_str()),
                    return ge::GRAPH_FAILED);
    params.mode = PADDING_MODE_MAP[mode];
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4PadV3GradReplicate(gert::TilingContext* tilingContext) {
    auto compileInfo = reinterpret_cast<const Tiling4PadV3GradReplicateCompileInfo *>(tilingContext->GetCompileInfo());
    uint64_t ubSizePlatForm = compileInfo->ubSizePlatForm;
    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);
    uint32_t availableUb = ubSize - RESERVED_UB;
    uint32_t coreNum = compileInfo->coreNum;
    uint32_t sysWorkspaceSize = compileInfo->sysWorkspaceSize;
    OP_TILING_CHECK(coreNum <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Failed to get core num."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(ubSizePlatForm <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Failed to get ub size."),
                    return ge::GRAPH_FAILED);

    ge::DataType inputDatatype = tilingContext->GetInputDesc(0)->GetDataType();
    OP_TILING_CHECK(inputDatatype != ge::DT_FLOAT && inputDatatype != ge::DT_FLOAT16 && inputDatatype != ge::DT_BF16,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        tilingContext->GetNodeName(),
                        "the current x dtype is not in dtype support list [bfloat16, float16, float]."),
                    return ge::GRAPH_FAILED);
    
    ge::DataType paddingDatatype = tilingContext->GetInputDesc(1)->GetDataType();
    OP_TILING_CHECK(
        paddingDatatype != ge::DT_INT32 && paddingDatatype != ge::DT_INT64,
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(),
                                        "the current padding dtype is not in dtype support list [int32, int64]."),
        return ge::GRAPH_FAILED);
    InputParamsInfo params;
    params.dtype = DTYPE_MAP[inputDatatype];

    if (paddingDatatype == ge::DT_INT32) {
        OP_TILING_CHECK(GetInputInfo<int32_t>(tilingContext, params) != ge::GRAPH_SUCCESS,
                        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "get op inputs failed."),
                        return ge::GRAPH_FAILED);
    } else if (paddingDatatype == ge::DT_INT64) {
        OP_TILING_CHECK(GetInputInfo<int64_t>(tilingContext, params) != ge::GRAPH_SUCCESS,
                        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "get op inputs failed."),
                        return ge::GRAPH_FAILED);
    }

    PadV3GradReplicateTilingData tilingData;
    if (inputDatatype == ge::DT_FLOAT) {
        GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, FLOAT_BYTES>(&tilingData, params, 
                                                                               coreNum, availableUb);
    } else {
        GetPadV3GradReplicateTiling<PadV3GradReplicateTilingData, FLOAT16_BYTES>(&tilingData, params, 
                                                                                 coreNum, availableUb);
    }

    OP_TILING_CHECK(
        tilingData.get_ubFactorElement() <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "ub space is not enough, please check input."),
        return ge::GRAPH_FAILED);
    // set tilingdata
    uint64_t workspacePerCore = tilingData.get_workspacePerCore();
    uint32_t tilingKey = tilingData.get_tilingKey();
    uint32_t blockNum = tilingData.get_blockNum();
    size_t usrWorkspace = workspacePerCore * blockNum;
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(blockNum);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = usrWorkspace + sysWorkspaceSize;
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    PrintTilingData(tilingContext, tilingData, usrWorkspace);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4PadV3GradReplicate(gert::TilingParseContext* context) {
    auto compileInfo = GetCompileInfoPtr<Tiling4PadV3GradReplicateCompileInfo>(context);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_TILING_CHECK((compileInfo->coreNum <= 0),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get core num."),
                    return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_TILING_CHECK(ubSizePlatForm <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get ub size."),
                    return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PadV3GradReplicate)
    .Tiling(Tiling4PadV3GradReplicate)
    .TilingParse<Tiling4PadV3GradReplicateCompileInfo>(TilingPrepare4PadV3GradReplicate)
    .TilingInputsDataDependency({1});
}  // namespace optiling