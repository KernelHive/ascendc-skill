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
 * \file reduce_sum_v2_common.h
 * \brief tiling for reduce sum
 */
#ifndef REDUCE_SUM_V2_COMMON_H
#define REDUCE_SUM_V2_COMMON_H

#include <numeric>
#include <algorithm>

constexpr int64_t MAX_DIM = 8;
constexpr int64_t NUM_TWO = 2;

#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)
#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                        \
  if ((ptr) == nullptr) {                                                                         \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();  \
    std::printf(name, "is nullptr!");                                                             \
    REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                            \
    return ret;                                                                                   \
  }
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                  \
  if ((ptr) == nullptr) {                                                                          \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();   \
    std::printf(name, "is nullptr!");                                                              \
    REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                             \
    return ge::GRAPH_FAILED;                                                                       \
  }

template <typename T>
inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
  return context->GetCompiledInfo<T>();
}
}

namespace optiling {
namespace ReduceCommon {
template <typename T>
ge::graphStatus GetAxesData(gert::TilingContext* context, int32_t idx, std::vector<uint64_t>& axes)
{
    auto axesInput = context->GetInputTensor(idx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, axesInput);
    auto size = axesInput->GetShapeSize();
    OP_TILING_CHECK(
        (size > static_cast<int64_t>(MAX_DIM)),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "dim size:%ld is over max dim, cannot support.", size),
        return ge::GRAPH_FAILED);

    if (size == 0) {
        return ge::GRAPH_SUCCESS;
    }
    axes.resize(size);
    auto axesData = axesInput->GetData<T>();
    OP_TILING_CHECK((axesData == nullptr), VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "GetData failed"),
                    return ge::GRAPH_FAILED;);
    for (size_t i = 0; i < size; i++) {
        axes[i] = static_cast<uint64_t>(axesData[i]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetInputDtype(gert::TilingContext* context, int32_t idx, ge::DataType& dtype)
{
    auto inputDesc = context->GetInputDesc(idx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dtype = inputDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

inline void DoFuseAxes(const std::vector<uint64_t> &xShape, std::vector<uint64_t> &axes, std::vector<uint64_t> &xFusedShape, uint64_t &reduceCnt)
{
    std::sort(axes.begin(), axes.end());
    size_t j = 0;
    bool isLastR = true;
    if (!axes[0]) {
        xFusedShape.emplace_back(1);
        isLastR = false;
    }
    
    for (size_t i = 0; i < xShape.size(); i++) {
        uint64_t dim = xShape[i];
        if (j < axes.size() && i == axes[j]) {
            if (isLastR) {
                xFusedShape[xFusedShape.size() - 1] *= dim;
            } else {
                reduceCnt++;
                xFusedShape.emplace_back(dim);
                isLastR = true;
            }
            j++;
        } else {
            if (isLastR) {
                xFusedShape.emplace_back(dim);
                isLastR = false;
            } else {
                xFusedShape[xFusedShape.size() - 1] *= dim;
            }
        }
    }
    if ((xFusedShape.size() % NUM_TWO) && (xFusedShape.back() == 1)) {
        xFusedShape.pop_back();
    }
}

}  // namespace ReduceCommon
}  // namespace optiling
#endif