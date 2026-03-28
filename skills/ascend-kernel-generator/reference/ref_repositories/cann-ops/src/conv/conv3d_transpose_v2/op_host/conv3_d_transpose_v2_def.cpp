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
 * \file conv3d_transpose_v2.cpp
 * \brief
 */

 #include "register/op_def_registry.h"

 namespace ops {
 class Conv3DTransposeV2 : public OpDef {
 public:
     explicit Conv3DTransposeV2(const char *name) : OpDef(name)
     {
         this->Input("input_size")
             .ParamType(REQUIRED)
             .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64})
             .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
         this->Input("x")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT})
             .Format({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0})
             .UnknownShapeFormat({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0});
         this->Input("filter")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT})
             .Format(
                 {ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D})
             .UnknownShapeFormat(
                 {ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D});
         this->Input("bias")
             .ParamType(OPTIONAL)
             .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
             .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
         this->Input("offset_w")
             .ParamType(OPTIONAL)
             .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
             .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
             .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
         this->Output("y")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT})
             .Format({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0})
             .UnknownShapeFormat({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0});
 
         this->Attr("strides").AttrType(REQUIRED).ListInt();
         this->Attr("pads").AttrType(REQUIRED).ListInt();
         this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1, 1});
         this->Attr("groups").AttrType(OPTIONAL).Int(1);
         this->Attr("data_format").AttrType(OPTIONAL).String("NDHWC");
         this->Attr("output_padding").AttrType(OPTIONAL).ListInt({0, 0, 0, 0, 0});
         this->Attr("offset_x").AttrType(OPTIONAL).Int(0);
 
         OpAICoreConfig aicore_config;
         aicore_config.DynamicCompileStaticFlag(true)
             .DynamicFormatFlag(true)
             .DynamicRankSupportFlag(true)
             .DynamicShapeSupportFlag(true)
             .NeedCheckSupportFlag(false)
             .PrecisionReduceFlag(true)
             .ExtendCfgInfo("opFile.value", "conv3d_transpose_v2")
             .ExtendCfgInfo("opInterface.value", "conv3d_transpose_v2")
             .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
 
         this->AICore().AddConfig("ascend910b", aicore_config);
         this->AICore().AddConfig("ascend910_93", aicore_config);
     }
 };
 
 OP_ADD(Conv3DTransposeV2);
 }  // namespace ops