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
 * @file add_sigmoid_mul_reduce_sum_d.cpp
 */
#include "add_sigmoid_mul_reduce_sum_d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace ge
{
    static ge::graphStatus CheckInputDimAndInit(gert::TilingContext *context,
                            int32_t addInput0Dim0Length,
                            int32_t addInput0Dim1Length,
                            int32_t addInput0Dim2Length,
                            int32_t &addInput0Dim3Length,
                            int32_t &addInput0Dim4Length,
                            int32_t &addInput0Dim1234Length,
                            int32_t &addInput0Dim14Length,
                            int32_t &addInput0Dim23Length,
                            int32_t &addInput0Dim234Length)
    {
        int32_t num_2 = 2;
        int32_t dim_3 = 3;
        int32_t dim_4 = 4;

        if (addInput0Dim0Length != context->GetInputShape(dim_3)->GetStorageShape().GetDim(0) || addInput0Dim0Length != context->GetInputShape(dim_4)->GetStorageShape().GetDim(0) || addInput0Dim1Length != context->GetInputShape(dim_4)->GetStorageShape().GetDim(1) || addInput0Dim2Length != context->GetInputShape(dim_4)->GetStorageShape().GetDim(num_2)) {
            return ge::GRAPH_FAILED;
        }

        if (context->GetInputDesc(0)->GetOriginFormat() == 1 || context->GetInputDesc(0)->GetOriginFormat() == num_2) {
            context->SetTilingKey(num_2);
            if (context->GetInputShape(0)->GetStorageShape().GetDimNum() == dim_4) {
                addInput0Dim3Length = context->GetInputShape(0)->GetStorageShape().GetDim(dim_3);
                if (addInput0Dim3Length != context->GetInputShape(dim_4)->GetStorageShape().GetDim(dim_3)) {
                    return ge::GRAPH_FAILED;
                }
                addInput0Dim1234Length = addInput0Dim1Length * addInput0Dim2Length * addInput0Dim3Length;
                addInput0Dim14Length = addInput0Dim2Length * addInput0Dim3Length;
                addInput0Dim234Length = addInput0Dim1Length * addInput0Dim2Length * addInput0Dim3Length;
            } else {
                addInput0Dim1234Length = addInput0Dim1Length * addInput0Dim2Length;
                addInput0Dim234Length = addInput0Dim1Length * addInput0Dim2Length;
            }
        } else {
            context->SetTilingKey(1);
            addInput0Dim3Length = context->GetInputShape(0)->GetStorageShape().GetDim(dim_3);
            addInput0Dim4Length = context->GetInputShape(0)->GetStorageShape().GetDim(dim_4);
            if (addInput0Dim3Length != context->GetInputShape(dim_4)->GetStorageShape().GetDim(dim_3) || addInput0Dim4Length != context->GetInputShape(dim_4)->GetStorageShape().GetDim(dim_4)) {
                return ge::GRAPH_FAILED;
            }
            addInput0Dim1234Length = addInput0Dim1Length * addInput0Dim2Length * addInput0Dim3Length * addInput0Dim4Length;
            addInput0Dim14Length = addInput0Dim1Length * addInput0Dim4Length;
            addInput0Dim23Length = addInput0Dim2Length * addInput0Dim3Length;
            addInput0Dim234Length = addInput0Dim2Length * addInput0Dim3Length * addInput0Dim4Length;
        }

        if (addInput0Dim14Length != context->GetInputShape(1)->GetStorageShape().GetDim(num_2) || addInput0Dim23Length != context->GetInputShape(dim_3)->GetStorageShape().GetDim(1)) {
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus ComputeTilingData(gert::TilingContext *context,
                                int32_t aivNum,
                                uint64_t ubSize,
                                int32_t addInput0Dim0Length,
                                int32_t addInput0Dim1234Length,
                                int32_t addInput0Dim14Length,
                                int32_t addInput0Dim23Length,
                                int32_t &formerCoreNum,
                                int32_t &formerCoreLength,
                                int32_t &formerTileNum,
                                int32_t &formerTileLength,
                                int32_t &formerLastTileLength,
                                int32_t &tailCoreNum,
                                int32_t &tailCoreLength,
                                int32_t &tailTileNum,
                                int32_t &tailTileLength,
                                int32_t &tailLastTileLength)
    {
        if (aivNum == 0) {
            return ge::GRAPH_FAILED;
        }

        int32_t num_2 = 2;
        ubSize = ubSize - addInput0Dim14Length * num_2;
        int32_t tileLength = ubSize / ((addInput0Dim1234Length * 5 + addInput0Dim1234Length + addInput0Dim14Length + addInput0Dim23Length) * num_2);
        if (addInput0Dim0Length <= aivNum) {
            context->SetBlockDim(addInput0Dim0Length);
            formerCoreNum = addInput0Dim0Length;
            formerCoreLength = 1;
            formerTileNum = 1;
            formerTileLength = 1;
            formerLastTileLength = 1;
        } else {
            context->SetBlockDim(aivNum);
            formerCoreNum = addInput0Dim0Length % aivNum;
            if (formerCoreNum == 0) {
                formerCoreNum = aivNum;
                formerCoreLength = (addInput0Dim0Length + aivNum - 1) / aivNum;
                formerTileNum = (formerCoreLength + tileLength - 1) / tileLength;
                formerTileLength = tileLength;
                formerLastTileLength = formerCoreLength - (formerTileNum - 1) * tileLength;
            } else {
                formerCoreLength = (addInput0Dim0Length + aivNum - 1) / aivNum;
                formerTileNum = (formerCoreLength + tileLength - 1) / tileLength;
                formerTileLength = tileLength;
                formerLastTileLength = formerCoreLength - (formerTileNum - 1) * tileLength;
                tailCoreNum = aivNum - formerCoreNum;
                tailCoreLength = addInput0Dim0Length / aivNum;
                tailTileNum = (tailCoreLength + tileLength - 1) / tileLength;
                tailTileLength = tileLength;
                tailLastTileLength = tailCoreLength - (tailTileNum - 1) * tileLength;
            }
        }

        return ge::GRAPH_SUCCESS;
    }
    
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        int32_t num_2 = 2;
        int32_t dim_3 = 3;
        int32_t dim_4 = 4;
        const gert::Shape *input_shape_0 = context->GetInputShape(0);
        gert::Shape *output_shape = context->GetOutputShape(0);
        int32_t addInput0Dim0Length = input_shape_0->GetDim(0);
        int32_t addInput0Dim1Length = input_shape_0->GetDim(1);
        int32_t addInput0Dim2Length = input_shape_0->GetDim(num_2);
        if (addInput0Dim0Length != context->GetInputShape(dim_3)->GetDim(0) || addInput0Dim0Length != context->GetInputShape(dim_4)->GetDim(0) || addInput0Dim1Length != context->GetInputShape(dim_4)->GetDim(1) || addInput0Dim2Length != context->GetInputShape(dim_4)->GetDim(num_2)) {
            return ge::GRAPH_FAILED;
        }
        output_shape->SetDimNum(num_2);
        output_shape->SetDim(0, addInput0Dim0Length);
        int32_t addInput1Dim2Length = context->GetInputShape(1)->GetDim(num_2);
        int32_t mul1Input0Dim1Length = context->GetInputShape(dim_3)->GetDim(1);
        if (context->GetInputDesc(0)->GetOriginFormat() == 1 || context->GetInputDesc(0)->GetOriginFormat() == num_2) {
            if (context->GetInputShape(0)->GetDimNum() == dim_4) {
                int32_t addInput0Dim3Length = input_shape_0->GetDim(dim_3);
                if (addInput0Dim1Length != mul1Input0Dim1Length || addInput0Dim2Length * addInput0Dim3Length != addInput1Dim2Length || addInput0Dim3Length != context->GetInputShape(dim_4)->GetDim(dim_3)) {
                    return ge::GRAPH_FAILED;
                }
                output_shape->SetDim(1, addInput0Dim2Length * addInput0Dim3Length);
            } else {
                if (addInput0Dim1Length != mul1Input0Dim1Length || addInput0Dim2Length != addInput1Dim2Length) {
                    return ge::GRAPH_FAILED;
                }
                output_shape->SetDim(1, addInput0Dim2Length); 
            }
        } else {
            int32_t addInput0Dim3Length = input_shape_0->GetDim(dim_3);
            int32_t addInput0Dim4Length = input_shape_0->GetDim(dim_4);
            if (addInput0Dim2Length * addInput0Dim3Length != mul1Input0Dim1Length || addInput0Dim1Length * addInput0Dim4Length != addInput1Dim2Length || addInput0Dim3Length != context->GetInputShape(dim_4)->GetDim(dim_3) || addInput0Dim4Length != context->GetInputShape(dim_4)->GetDim(dim_4)) {
                return ge::GRAPH_FAILED;
            }
            output_shape->SetDim(1, addInput0Dim1Length * addInput0Dim4Length);
        }
        
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        auto data_type = context->GetInputDataType(0);
        context->SetOutputDataType(0, data_type);
        return GRAPH_SUCCESS;
    }
}

namespace optiling
{
   static ge::graphStatus TilingFunc(gert::TilingContext *context)
   {
       AddSigmoidMulReduceSumDTilingData tiling;

       const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
       int32_t aivNum = ascendcPlatform.GetCoreNumAiv();
       uint64_t ubSize = 0;
       ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

       int32_t addInput0Dim0Length = context->GetInputShape(0)->GetStorageShape().GetDim(0);
       int32_t addInput0Dim1Length = context->GetInputShape(0)->GetStorageShape().GetDim(1);
       int32_t addInput0Dim2Length = context->GetInputShape(0)->GetStorageShape().GetDim(2);
       int32_t addInput0Dim3Length = 0;
       int32_t addInput0Dim4Length = 0;
       int32_t addInput0Dim1234Length = 0;
       int32_t addInput0Dim14Length = addInput0Dim2Length;
       int32_t addInput0Dim23Length = addInput0Dim1Length;
       int32_t addInput0Dim234Length = 0;

       int32_t formerCoreNum = 0;
       int32_t formerCoreLength = 0;
       int32_t formerTileNum = 0;
       int32_t formerTileLength = 0;
       int32_t formerLastTileLength = 0;

       int32_t tailCoreNum = 0;
       int32_t tailCoreLength = 0;
       int32_t tailTileNum = 0;
       int32_t tailTileLength = 0;
       int32_t tailLastTileLength = 0; 
       
       ge::CheckInputDimAndInit(context, addInput0Dim0Length, addInput0Dim1Length, 
           addInput0Dim2Length, addInput0Dim3Length, addInput0Dim4Length, 
           addInput0Dim1234Length, addInput0Dim14Length, addInput0Dim23Length, addInput0Dim234Length);

       ge::ComputeTilingData(context, aivNum, ubSize, addInput0Dim0Length, addInput0Dim1234Length, 
           addInput0Dim14Length, addInput0Dim23Length, formerCoreNum, formerCoreLength, 
           formerTileNum, formerTileLength, formerLastTileLength, tailCoreNum, tailCoreLength, 
           tailTileNum, tailTileLength, tailLastTileLength);

       tiling.set_formerCoreNum(formerCoreNum);
       tiling.set_formerCoreLength(formerCoreLength);
       tiling.set_formerTileNum(formerTileNum);
       tiling.set_formerTileLength(formerTileLength);
       tiling.set_formerLastTileLength(formerLastTileLength);

       tiling.set_tailCoreNum(tailCoreNum);
       tiling.set_tailCoreLength(tailCoreLength);
       tiling.set_tailTileNum(tailTileNum);
       tiling.set_tailTileLength(tailTileLength);
       tiling.set_tailLastTileLength(tailLastTileLength);

       tiling.set_addInput0Dim1234Length(addInput0Dim1234Length);
       tiling.set_addInput0Dim14Length(addInput0Dim14Length);
       tiling.set_addInput0Dim23Length(addInput0Dim23Length);
       tiling.set_addInput0Dim1Length(addInput0Dim1Length);
       tiling.set_addInput0Dim234Length(addInput0Dim234Length);

       uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
       size_t *currentWorkspace = context->GetWorkspaceSizes(1);

       size_t usrSize = 0;
       currentWorkspace[0] = usrSize + sysWorkspaceSize;

       tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
       context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

       return ge::GRAPH_SUCCESS;
   }
}

namespace ops
{
    class AddSigmoidMulReduceSumD : public OpDef
    {
    public:
        explicit AddSigmoidMulReduceSumD(const char *name) : OpDef(name)
        {
            this->Input("add_0_input0")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, ge::FORMAT_ND});
            this->Input("add_0_input1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_ND});
            this->Input("mul_0_input1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_ND});
            this->Input("mult_1_input1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_ND});
            this->Input("mult_2_input1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NHWC, ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape);
            this->SetInferDataType(ge::InferDataType);

            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(AddSigmoidMulReduceSumD);
}