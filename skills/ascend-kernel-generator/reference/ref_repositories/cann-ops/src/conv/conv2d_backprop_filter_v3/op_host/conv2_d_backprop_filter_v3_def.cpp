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
 * \file conv2_d_backprop_filter_v3_def.cpp
 * \brief Conv2DBackpropFilterV3 ascendc impl
 */

 #include "register/op_def_registry.h"

 namespace ops {
 class Conv2DBackpropFilterV3 : public OpDef {
 public:
     explicit Conv2DBackpropFilterV3(const char *name) : OpDef(name)
     {
         this->Input("x")
             .ParamType(REQUIRED)
             .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
             .Format({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
             .UnknownShapeFormat({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0});
         this->Input("filter_size")
             .ParamType(REQUIRED)
             .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
             .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
         this->Input("out_backprop")
             .ParamType(REQUIRED)
             .DataType({ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16})
             .Format({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
             .UnknownShapeFormat({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0});
         this->Output("y")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
             .Format({ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z})
             .UnknownShapeFormat({ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z});
 
         this->Attr("strides").AttrType(REQUIRED).ListInt();
         this->Attr("pads").AttrType(REQUIRED).ListInt();
         this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1});
         this->Attr("groups").AttrType(OPTIONAL).Int(1);
         this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");
 
         OpAICoreConfig aicore_config;
         aicore_config.DynamicCompileStaticFlag(true)
             .DynamicFormatFlag(true)
             .DynamicRankSupportFlag(true)
             .DynamicShapeSupportFlag(true)
             .NeedCheckSupportFlag(false)
             .PrecisionReduceFlag(true)
             .ExtendCfgInfo("opFile.value", "conv2d_backprop_filter_v3")
             .ExtendCfgInfo("opInterface.value", "conv2d_backprop_filter_v3")
             .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
 
         this->AICore().AddConfig("ascend910b", aicore_config);
         this->AICore().AddConfig("ascend910_93", aicore_config);
     }
 };
 
 OP_ADD(Conv2DBackpropFilterV3);
 }  // namespace ops