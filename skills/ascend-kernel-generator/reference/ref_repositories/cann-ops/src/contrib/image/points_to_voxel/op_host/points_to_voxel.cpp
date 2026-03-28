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
 * @file points_to_voxel.cpp
 */
#include <cmath>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "points_to_voxel_tiling.h"


namespace optiling {
const uint32_t BUFFER_NUM = 1;
const uint32_t BLOCK_SIZE = 32;
const int32_t DATATYPE1 = 2;
const int32_t DATATYPE2 = 4;
const int32_t BYTESIZE = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    PointsToVoxelTilingData tiling;
    int32_t NUM = 20;
    uint32_t sizeofdatatype;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();
    int32_t ndimlength =  context->GetInputShape(0)->GetStorageShape().GetDim(0);
    int32_t Nlength =  context->GetInputShape(0)->GetStorageShape().GetDim(1);
    auto dt = context->GetInputTensor(0)->GetDataType();
    if (dt == ge::DT_FLOAT)
    {
        sizeofdatatype = DATATYPE2;
    }
    uint32_t ALIGN_NUM = BLOCK_SIZE / sizeofdatatype;
    uint32_t tiling_size = ((ub_size) / BLOCK_SIZE / 2) / NUM;
    tiling_size = tiling_size <= BYTESIZE ? tiling_size : tiling_size / BYTESIZE * BYTESIZE;
    uint32_t block_size = tiling_size * ALIGN_NUM;
    aivNum = (aivNum < Nlength / block_size) ? aivNum : (Nlength / block_size);
    aivNum = aivNum >= 1 ? aivNum : 1;
    uint32_t core_size = 0;
    uint32_t ALIGN_REPEAT = ALIGN_NUM * BYTESIZE;
    if(ALIGN_REPEAT != 0 && aivNum != 0) {
        core_size = (Nlength / aivNum) / ALIGN_REPEAT * ALIGN_REPEAT;
    }    
    uint32_t core_remain = Nlength - aivNum * core_size;

    tiling.set_ALIGN_NUM(ALIGN_NUM);
    tiling.set_tiling_size(tiling_size);
    tiling.set_block_size(block_size);
    tiling.set_aivNum(aivNum);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);

    auto *pvoxel_sizecv = context->GetAttrs()->GetListFloat(0);
    const float *pvoxel_size = pvoxel_sizecv->GetData();
    float voxel_size_x = *pvoxel_size;
    float voxel_size_y = *(pvoxel_size + 1);
    float voxel_size_z = *(pvoxel_size + 2);
    auto *pcoors_rangecv = context->GetAttrs()->GetListFloat(1);
    const float *pcoors_range = pcoors_rangecv->GetData();
    float coors_range_xL = *pcoors_range;
    float coors_range_yL = *(pcoors_range + 1);
    float coors_range_zL = *(pcoors_range + 2);
    float coors_range_xH = *(pcoors_range + 3);
    float coors_range_yH = *(pcoors_range + 4);
    float coors_range_zH = *(pcoors_range + 5);
    const int64_t *pmax_points = context->GetAttrs()->GetInt(2);
    const bool *preverse_index = context->GetAttrs()->GetBool(3);
    const int64_t *pmax_voxels = context->GetAttrs()->GetInt(4);
    int32_t max_points = *pmax_points;
    bool reverse_index = *preverse_index;
    int32_t max_voxels = *pmax_voxels;

    float voxelmap_shape_x = (coors_range_xH - coors_range_xL) / voxel_size_x;
    float voxelmap_shape_y = (coors_range_yH - coors_range_yL) / voxel_size_y;
    float voxelmap_shape_z = (coors_range_zH - coors_range_zL) / voxel_size_z;
    int32_t voxelmap_shape_xR = round(voxelmap_shape_x);
    int32_t voxelmap_shape_yR = round(voxelmap_shape_y);
    int32_t voxelmap_shape_zR = round(voxelmap_shape_z);
    int32_t voxelmap_length = voxelmap_shape_xR * voxelmap_shape_yR * voxelmap_shape_zR;

    tiling.set_ndimlength(ndimlength);
    tiling.set_Nlength(Nlength);
    tiling.set_voxel_size_x(voxel_size_x);
    tiling.set_voxel_size_y(voxel_size_y);
    tiling.set_voxel_size_z(voxel_size_z);
    tiling.set_coors_range_xL(coors_range_xL);
    tiling.set_coors_range_yL(coors_range_yL);
    tiling.set_coors_range_zL(coors_range_zL);
    tiling.set_voxelmap_shape_xR(voxelmap_shape_xR);
    tiling.set_voxelmap_shape_yR(voxelmap_shape_yR);
    tiling.set_voxelmap_shape_zR(voxelmap_shape_zR);
    tiling.set_max_points(max_points);
    tiling.set_reverse_index(reverse_index);
    tiling.set_max_voxels(max_voxels);

    context->SetBlockDim(aivNum);

    size_t usrSize = (((Nlength + ALIGN_NUM) * 5) + voxelmap_length + 512) * 4;
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    int32_t ndimlength = context->GetInputShape(0)->GetDim(0);
    const int64_t *pmax_points = context->GetAttrs()->GetInt(2);
    const int64_t *pmax_voxels = context->GetAttrs()->GetInt(4);
    int32_t max_points = *pmax_points;
    int32_t max_voxels = *pmax_voxels;
    gert::Shape* voxels_out_shape = context->GetOutputShape(0);
    voxels_out_shape->SetDimNum(3);
    voxels_out_shape->SetDim(0, max_voxels);
    voxels_out_shape->SetDim(1, max_points);
    voxels_out_shape->SetDim(2, ndimlength);
    gert::Shape* coors_out_shape = context->GetOutputShape(1);
    coors_out_shape->SetDimNum(2);
    coors_out_shape->SetDim(0, max_voxels);
    coors_out_shape->SetDim(0, 3);
    gert::Shape* num_points_per_voxel_shape = context->GetOutputShape(2);
    num_points_per_voxel_shape->SetDimNum(1);
    num_points_per_voxel_shape->SetDim(0, max_voxels);
    gert::Shape* voxel_num_shape = context->GetOutputShape(3);
    voxel_num_shape->SetDimNum(1);
    voxel_num_shape->SetDim(0, 1);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(2, ge::DT_INT32);
    context->SetOutputDataType(3, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class PointsToVoxel : public OpDef {
public:
    explicit PointsToVoxel(const char* name) : OpDef(name)
    {
        this->Input("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("voxels_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("coors_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("num_points_per_voxel")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("voxel_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("voxel_size").AttrType(OPTIONAL).ListFloat({0.0, 0.0, 0.0});
        this->Attr("coors_range").AttrType(OPTIONAL).ListFloat({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        this->Attr("max_points").AttrType(OPTIONAL).Int(0);
        this->Attr("reverse_index").AttrType(OPTIONAL).Bool(false);
        this->Attr("max_voxels").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p").AddConfig("ascend910b");
    }
};

OP_ADD(PointsToVoxel);
}
