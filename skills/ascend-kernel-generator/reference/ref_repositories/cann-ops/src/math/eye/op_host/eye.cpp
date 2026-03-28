/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "eye_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
    const uint32_t BLOCK_SIZE = 32;
    constexpr int BATCH_SIZE_DIM_THRESHOLD = 2;
    constexpr uint32_t DATATYPE_SIZE_2 = 2;
    constexpr uint32_t DATATYPE_SIZE_4 = 4;
    constexpr uint32_t TYPE_KEY_0 = 0;
    constexpr uint32_t TYPE_KEY_1 = 1;
    constexpr uint32_t DATA_NUM = 4;
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        EyeTilingData tiling;
        uint32_t sizeofdatatype;
        uint32_t totalLengthAligned;

        // 1. 获取平台信息
        uint64_t ub_size;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 2. 获取数据信息
        uint32_t dataNum;
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        uint32_t tanhxLength = context->GetOptionalInputTensor(0)->GetShapeSize();
        const int64_t* pnum_rows = context->GetAttrs()->GetInt(0);
        const int64_t* pnum_columns = context->GetAttrs()->GetInt(1);
        const int64_t* pdtype = context->GetAttrs()->GetInt(2);
        int32_t num_rows = *pnum_rows;
        int32_t num_columns = *pnum_columns;
        int32_t dtype = *pdtype;
        if(num_columns == 0){
            num_columns = num_rows;
        }

        int32_t mark = 0;
        int32_t batchNum = 1;
        int32_t batchSize = 0;
        auto shape = context->GetInputTensor(0)->GetOriginShape();
        const uint32_t dimNum = shape.GetDimNum();
        if(dimNum > BATCH_SIZE_DIM_THRESHOLD){
            mark = 1;
            batchSize = num_rows * num_columns;
            batchNum = totalLength / batchSize;
        }

        tiling.set_num_columns(num_columns);
        tiling.set_num_rows(num_rows);
        tiling.set_dtype(dtype);
        tiling.set_mark(mark);
        tiling.set_batchNum(batchNum);
        tiling.set_batchSize(batchSize);

        auto dt = context->GetInputTensor(0)->GetDataType();
        uint32_t typeKey;
        if (dt == 1) {
            sizeofdatatype = DATATYPE_SIZE_2;
            dataNum = DATA_NUM;
            typeKey = TYPE_KEY_0;
        }else{
            sizeofdatatype = DATATYPE_SIZE_4;
            dataNum = DATA_NUM;
            typeKey = TYPE_KEY_1;
        }

        tiling.set_typeKey(typeKey);

        // 3. 填满UB大小
        uint32_t ub_block_num = ub_size / BLOCK_SIZE / dataNum - 256;
	uint32_t DOUBLE_BUFFER = 2;
        if (ub_block_num % DOUBLE_BUFFER != 0) {
            ub_block_num = ub_block_num - 1;
        }

        // 4. 输入向量满足32字节对齐
        uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
        if (totalLength % ALIGN_NUM != 0) {  //不对齐，先32位对齐
            totalLengthAligned = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
        } else {
            totalLengthAligned = totalLength;
        }

        // 5. Tiling参数计算
        uint32_t tile_num, block_dim = 1;
        context->SetBlockDim(block_dim);
        uint32_t blockLength = 0;
        uint32_t tileLength = 0;
        uint32_t lasttileLength = 0;
        blockLength = totalLengthAligned / block_dim;
        tile_num = blockLength / ALIGN_NUM / ub_block_num;

        if (tile_num == 0) { // 不足一个ub的情况
            tile_num = 1;
            tileLength = ((blockLength / ALIGN_NUM) + 1) / 2 * 2 * ALIGN_NUM; // 计算块的长度，使其对齐到ALIGN_NUM的倍数
            lasttileLength = tileLength;
        } else if((blockLength / ALIGN_NUM) % ub_block_num == 0){ // 核内能均分
            tileLength = ub_block_num * ALIGN_NUM;
            lasttileLength = tileLength;
        }else{ // 核内不能均分
            tile_num = tile_num + 1;    // 加一个小包的数量
            tileLength = ub_block_num * ALIGN_NUM;
            lasttileLength = blockLength - (tile_num - 1) * tileLength;
            lasttileLength = ((lasttileLength / ALIGN_NUM) + 1) / 2 * 2 * ALIGN_NUM; // 计算最后一个块的长度，使其对齐到ALIGN_NUM的倍数
        }

        tiling.set_blockLength(blockLength);
        tiling.set_tileNum(tile_num);
        tiling.set_tileLength(tileLength);
        tiling.set_lasttileLength(lasttileLength);
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
        const gert::Shape* x1_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
}


namespace ops {
class Eye : public OpDef {
public:
    explicit Eye(const char* name) : OpDef(name) {
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_rows").Int();
        this->Attr("num_columns").AttrType(OPTIONAL).Int(0);
        this->Attr("batch_shape").AttrType(OPTIONAL).ListInt({});
        this->Attr("dtype").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b")
                      .AddConfig("ascend910b");
    }
};

OP_ADD(Eye);
}

