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
 * @file eye_fp64.cpp
 */
#include "eye_fp64_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context){
        EyeFp64TilingData tiling;
        const auto runtime_attrs=context->GetAttrs();
        const auto numRows=*(runtime_attrs->GetInt(0));
        auto numColumns=*(runtime_attrs->GetInt(1));
        const auto *batchShape = runtime_attrs->GetListInt(2);
        const auto dtype=*(runtime_attrs->GetInt(3));
        if(numColumns==0){
            numColumns=numRows;
        }
        uint16_t totalMatrixNum=context->GetInputTensor(0)->GetShapeSize()/numColumns/numRows;
        uint64_t mask0=0;
        uint64_t mask1=0;
        uint64_t mask_remain0=0;
        uint64_t mask_remain1=0;
        auto dt = context->GetInputTensor(0)->GetDataType(); //  ge::DT_FLOAT
        int DataTypeSize=0;
        if(dt==ge::DT_DOUBLE){
            DataTypeSize=8;
            uint64_t tmp=1<<1;
            for(uint32_t k=0;k<8;k++){
                mask0+=tmp;
                tmp<<=8;
            }
            //64个元素就只会用mask[0]
            tmp=1<<7;
            for(uint32_t k=0;k<8;k++){
                mask_remain0+=tmp;
                tmp<<=8;
            }
        }else if(dt==ge::DT_FLOAT || dt==ge::DT_INT32){
            DataTypeSize=4;
            //64个元素就只会用mask[0]
            uint64_t tmp=1;
            for(uint32_t k=0;k<8;k++){
                mask0+=tmp;
                tmp<<=8;
            }
            tmp=1<<7;
            for(uint32_t k=0;k<8;k++){
                mask_remain0+=tmp;
                tmp<<=8;
            }
        }else if(dt==ge::DT_FLOAT16){
            DataTypeSize=2;
            uint64_t tmp=1;
            for(uint32_t k=0;k<4;k++){
                mask0+=tmp;
                mask1+=tmp;
                tmp<<=16;
            }
            tmp=1<<15;
            for(uint32_t k=0;k<4;k++){
                mask_remain0+=tmp;
                mask_remain1+=tmp;
                tmp<<=16;
            }
        }
        tiling.set_totalMatrixNum(totalMatrixNum);
        tiling.set_numColumns(numColumns);
        tiling.set_numRows(numRows);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t  aivNum;
        if(DataTypeSize==8 && numColumns%4==0 && numRows>numColumns && 8*numColumns*numColumns*totalMatrixNum<=40*160000 && numColumns*numColumns<255*32){
            context->SetTilingKey(3);
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
            aivNum=8*numColumns*numColumns*totalMatrixNum/160000;
        }else if(DataTypeSize*(numColumns+1)>64){
            context->SetTilingKey(2);
            EyeFp64TilingData_slice tiling_slice;
            tiling_slice.set_totalMatrixNum(totalMatrixNum);
            tiling_slice.set_numColumns(numColumns);
            tiling_slice.set_numRows(numRows);
            tiling_slice.set_mask0(mask0);
            tiling_slice.set_mask1(mask1);
            tiling_slice.set_mask_remain0(mask_remain0);
            tiling_slice.set_mask_remain1(mask_remain1);
            tiling_slice.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling_slice.GetDataSize());
            aivNum=std::min((uint16_t)40,totalMatrixNum);
        }else{
            context->SetTilingKey(1);
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
            aivNum = std::min((int64_t)40,(totalMatrixNum*numColumns*numRows+127)/128);
        }
        context->SetBlockDim(aivNum);
        return ge::GRAPH_SUCCESS;
    }
}
namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        const gert::Shape* x1_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
}
namespace ops {
    class EyeFp64 : public OpDef {
    public:
        explicit EyeFp64(const char* name) : OpDef(name)
        {
            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_DOUBLE})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_DOUBLE})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("num_rows").Int();
            this->Attr("num_columns").AttrType(OPTIONAL).Int(0);
            this->Attr("batch_shape").ListInt();
            this->Attr("dtype").AttrType(OPTIONAL).Int(0);
            this->SetInferShape(ge::InferShape);
            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };
    OP_ADD(EyeFp64);
}