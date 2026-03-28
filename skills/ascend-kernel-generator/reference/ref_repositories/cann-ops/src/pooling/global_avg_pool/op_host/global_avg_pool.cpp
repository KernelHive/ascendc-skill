/*
* @file global_avg_pool.cpp
*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
* This program is free software; you can redistribute it and/or modify
* it under the terms of the Apache License Version 2.0.
* You may not use this file except in compliance with the License.
* 
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "global_avg_pool_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
    const uint32_t BLOCK_SIZE = 32;
    const uint32_t HALF_SIZE = 2;
    const uint32_t FLOAT_SIZE = 4;
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        GlobalAvgPoolTilingData tiling;
        uint32_t sizeofdatatype;
        uint32_t totalLengthAligned;

        // 1. 获取平台信息
        uint64_t ub_size;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto socVersion = ascendcPlatform.GetSocVersion();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        auto aivNum = ascendcPlatform.GetCoreNumAiv();  // vector core num  1

        // 2. 获取数据信息
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        auto shape = context->GetInputTensor(0)->GetOriginShape();
        const uint32_t dim0 = shape.GetDim(0);
        const uint32_t dim1 = shape.GetDim(1);
        const uint32_t outDim = dim0 * dim1;
        const int32_t dimLength = totalLength / outDim;
        tiling.set_outDim(outDim);
        tiling.set_dimLength(dimLength);

        auto dt = context->GetInputTensor(0)->GetDataType();
        uint32_t typeKey; // typeKey 处理不同shape的标志
        if (dt == ge::DT_FLOAT16) { // half
            sizeofdatatype = HALF_SIZE;
        }else if (dt == ge::DT_FLOAT) { // float
            sizeofdatatype = FLOAT_SIZE;
        }
        uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
        uint32_t dimAligned = ((outDim + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
        uint32_t stride;
        uint32_t block_dim = 1;
        context->SetBlockDim(block_dim);

        if(dimLength == 1){ // 处理输入特征图等于输出的情况
            typeKey = 0;
            tiling.set_typeKey(typeKey);
            uint32_t dataNum = 2;

            // 3. 填满UB大小
            uint32_t ub_block_num = ub_size / BLOCK_SIZE / dataNum;
            if (ub_block_num % 2 != 0) {
                ub_block_num = ub_block_num - 1;
            }

            // 4. 输入向量满足32字节对齐
            totalLengthAligned = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

            // 5. Tiling参数计算
            uint32_t blockLength = 0;
            uint32_t tileLength = 0;
            uint32_t lasttileLength = 0;
            blockLength = totalLengthAligned / block_dim;
            uint32_t tile_num = blockLength / ALIGN_NUM / ub_block_num;

            if (tile_num == 0) {    // 不足一个ub的情况
                tile_num = 1;
                tileLength = blockLength;
                lasttileLength = blockLength;
            } else if((blockLength / ALIGN_NUM) % ub_block_num == 0){   // 核内能均分
                tileLength = ub_block_num * ALIGN_NUM;
                lasttileLength = tileLength;
            }else{  // 核内不能均分
                tile_num = tile_num + 1;
                tileLength = ub_block_num * ALIGN_NUM;
                lasttileLength = blockLength - (tile_num - 1) * tileLength;
            }

            tiling.set_blockLength(blockLength);
            tiling.set_tileNum(tile_num);
            tiling.set_tileLength(tileLength);
            tiling.set_lasttileLength(lasttileLength);
        }else if(sizeofdatatype * dimLength <= 256 && dimLength % ALIGN_NUM == 0 &&
               (totalLength + dimAligned * 2) * sizeofdatatype <= ub_size && dt == 0) {
            // 处理所有数据可以在一个 ub 放下地情况和数据类型为 float 的情况
            typeKey = 3;
            stride = dimLength / ALIGN_NUM;
            tiling.set_stride(stride);
            tiling.set_blockLength(totalLength);
            tiling.set_typeKey(typeKey);
            tiling.set_tileLength(dimAligned);
        }else{ // 处理一般输入特征图的情况
            if (dt == ge::DT_FLOAT16) {
                typeKey = 1;
            }else if (dt == ge::DT_FLOAT) {
                typeKey = 2;
            }
            tiling.set_typeKey(typeKey);

            // 3. 确定ub的block数量
            uint32_t ub_block_num = ub_size / BLOCK_SIZE * 29 / 30;
            uint32_t tile_num;
            if (ub_block_num % 2 != 0) {
                ub_block_num = ub_block_num - 1;
            }

            // 4. 数据对齐
            totalLength = dimLength;
            totalLengthAligned = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
            context->SetBlockDim(1);

            // 5. 计算tiling参数
            uint32_t blockLength = 0;
            uint32_t tileLength = 0;
            uint32_t lasttileLength = 0;
            uint32_t actLastLen = 0;
            blockLength = totalLengthAligned / block_dim;
            tile_num = blockLength / ALIGN_NUM / ub_block_num;

            if (tile_num == 0) {    // 不足一个ub的情况
                tile_num = 1;
                tileLength = blockLength;
                lasttileLength = tileLength;
            } else if((blockLength / ALIGN_NUM) % ub_block_num == 0){ // 核内能均分
                tileLength = ub_block_num * ALIGN_NUM;
                lasttileLength = tileLength;
            }else{  // 核内不能均分
                tile_num = tile_num + 1;
                tileLength = ub_block_num * ALIGN_NUM;
                lasttileLength = blockLength - (tile_num - 1) * tileLength;
            }
            actLastLen = dimLength - (tile_num - 1) * tileLength;

            uint32_t workLength;
            int elementsPerRepeat = 256 / sizeofdatatype;
            int firstMaxRepeat = tileLength / elementsPerRepeat;
            if(firstMaxRepeat == 0){
                firstMaxRepeat = 1;
            }
            int iter1OutputCount = firstMaxRepeat * 2;  // 第一轮操作产生的元素个数
            workLength = ((iter1OutputCount + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

            tiling.set_tileNum(tile_num);
            tiling.set_tileLength(tileLength);
            tiling.set_lasttileLength(lasttileLength);
            tiling.set_workLength(workLength);
            tiling.set_actLastLen(actLastLen);
        }

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}


namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context) {
        const gert::Shape* x_shape = context->GetInputShape(0);        
        gert::Shape* y_shape = context->GetOutputShape(0);
        y_shape->SetDimNum(x_shape->GetDimNum());
        y_shape->SetDim(0, x_shape->GetDim(0));
        y_shape->SetDim(1, x_shape->GetDim(1));
        for (int i = 2; i < x_shape->GetDimNum(); i++){
            y_shape->SetDim(i, 1);
        }    
        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
        auto data_type = context->GetInputDataType(0);
        context->SetOutputDataType(0, data_type);
        return GRAPH_SUCCESS;
    }    
}


namespace ops {
class GlobalAvgPool : public OpDef {
public:
    explicit GlobalAvgPool(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b")
                      .AddConfig("ascend910b");
    }
};

OP_ADD(GlobalAvgPool);
}
