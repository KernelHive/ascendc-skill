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
 * \file multi_scale_deformable_attn_function_def.cpp
 * \brief
 */

 #include <cstdint>
 #include "register/op_def_registry.h"
 
 namespace ops {
 class MultiScaleDeformableAttnFunction : public OpDef {
 public:
     explicit MultiScaleDeformableAttnFunction(const char *name) : OpDef(name)
     {
         this->Input("value")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT})
             .Format({ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND})
             .AutoContiguous();
         this->Input("value_spatial_shapes")
             .ParamType(REQUIRED)
             .DataType({ge::DT_INT32})
             .Format({ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND})
             .AutoContiguous();
         this->Input("value_level_start_index")
             .ParamType(REQUIRED)
             .DataType({ge::DT_INT32})
             .Format({ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND})
             .AutoContiguous();
         this->Input("sampling_locations")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT})
             .Format({ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND})
             .AutoContiguous();
         this->Input("attention_weights")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT})
             .Format({ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND})
             .AutoContiguous();
         this->Output("output")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT})
             .Format({ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND});
 
         OpAICoreConfig aiConfig;
         aiConfig.ExtendCfgInfo("enableVectorCore.flag", "false");
         aiConfig.DynamicCompileStaticFlag(true);
         this->AICore().AddConfig("ascend910b");
     }
 };
     OP_ADD(MultiScaleDeformableAttnFunction);
 }