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
 * \file group_quant_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_GROUP_QUANT_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_GROUP_QUANT_H
#include <cstdint>
#include <vector>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "tiling/tiling_type.h"

#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")

namespace ops {

const gert::Shape g_vec_1_shape = {1};

template <typename T1, typename T2>
inline T1 CeilDiv(T1 a, T2 b)
{
    return b == 0 ? a : (a + b - 1) / b;
}

inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

}

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                  \
  if ((ptr) == nullptr) {                                                                          \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();   \
    std::printf(name, "is nullptr!");                                                              \
    REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                             \
    return ge::GRAPH_FAILED;                                                                       \
  }
#define unlikely(x) __builtin_expect((x), 0)
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)

}

namespace optiling {

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

struct GroupQuantCompileInfo {
  int64_t coreNum = 0;
  uint64_t ubSizePlatForm = 0;
};

BEGIN_TILING_DATA_DEF(GroupQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, dimS);
TILING_DATA_FIELD_DEF(int64_t, dimE);
TILING_DATA_FIELD_DEF(int64_t, dimH);
TILING_DATA_FIELD_DEF(int64_t, hasOffset);
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, preCoreNum);
TILING_DATA_FIELD_DEF(int64_t, xRowNumPreCore);
TILING_DATA_FIELD_DEF(int64_t, xRowNumPostCore);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupQuant, GroupQuantTilingData)

class GroupQuantTiling : public TilingBaseClass {
 public:
  explicit GroupQuantTiling(gert::TilingContext* context) : TilingBaseClass(context) {
  }

 protected:
  bool IsCapable() override;
  // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
  ge::graphStatus GetPlatformInfo() override;
  // 2、获取INPUT/OUTPUT/ATTR信息
  ge::graphStatus GetShapeAttrsInfo() override;
  // 3、计算数据切分TilingData
  ge::graphStatus DoOpTiling() override;
  // 4、计算高阶API的TilingData
  ge::graphStatus DoLibApiTiling() override;
  // 5、计算TilingKey
  uint64_t GetTilingKey() const override;
  // 6、计算Workspace 大小
  ge::graphStatus GetWorkspaceSize() override;
  // 7、保存Tiling数据
  ge::graphStatus PostTiling() override;

 private:
  int64_t coreNumVar{0};
  int64_t dimS{0};
  int64_t dimE{0};
  int64_t dimH{0};
  int64_t hasOffset{0};
  int64_t needCoreNum{0};
  int64_t preCoreNum{0};
  int64_t xRowNumPreCore{0};
  int64_t xRowNumPostCore{0};
  GroupQuantTilingData tilingData;
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_GROUP_QUANT_H
