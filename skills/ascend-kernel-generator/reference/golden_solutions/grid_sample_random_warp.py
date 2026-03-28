project_json_src='''
[
    {
        "op": "GridSampleRandomWarpCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "grid",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "y",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GridSampleRandomWarpCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, inHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GridSampleRandomWarpCustom, GridSampleRandomWarpCustomTilingData)
}
"""

host_operator_src="""
#include "grid_sample_random_warp_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* gridShape = context->GetInputShape(1);
    if (xShape == nullptr || gridShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& xStorage = xShape->GetStorageShape();
    const auto& gridStorage = gridShape->GetStorageShape();
    if (xStorage.GetDimNum() != 4 || gridStorage.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    if (gridStorage.GetDim(3) != 2) {
        return ge::GRAPH_FAILED;
    }
    if (xStorage.GetDim(0) != gridStorage.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }

    GridSampleRandomWarpCustomTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(xStorage.GetDim(0)));
    tiling.set_channels(static_cast<uint32_t>(xStorage.GetDim(1)));
    tiling.set_inHeight(static_cast<uint32_t>(xStorage.GetDim(2)));
    tiling.set_inWidth(static_cast<uint32_t>(xStorage.GetDim(3)));
    tiling.set_outHeight(static_cast<uint32_t>(gridStorage.GetDim(1)));
    tiling.set_outWidth(static_cast<uint32_t>(gridStorage.GetDim(2)));

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* gridShape = context->GetInputShape(1);
    if (xShape == nullptr || gridShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 4 || gridShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }

    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(4);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, xShape->GetDim(1));
    yShape->SetDim(2, gridShape->GetDim(1));
    yShape->SetDim(3, gridShape->GetDim(2));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class GridSampleRandomWarpCustom : public OpDef {
public:
    explicit GridSampleRandomWarpCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GridSampleRandomWarpCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

class KernelGridSampleRandomWarp {
public:
    __aicore__ inline KernelGridSampleRandomWarp() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR grid,
        GM_ADDR y,
        uint32_t batch,
        uint32_t channels,
        uint32_t inHeight,
        uint32_t inWidth,
        uint32_t outHeight,
        uint32_t outWidth)
    {
        this->batch = static_cast<int32_t>(batch);
        this->channels = static_cast<int32_t>(channels);
        this->inHeight = static_cast<int32_t>(inHeight);
        this->inWidth = static_cast<int32_t>(inWidth);
        this->outHeight = static_cast<int32_t>(outHeight);
        this->outWidth = static_cast<int32_t>(outWidth);
        this->inputPlane = this->inHeight * this->inWidth;
        this->outputPlane = this->outHeight * this->outWidth;
        this->inputBatchStride = this->channels * this->inputPlane;
        this->outputBatchStride = this->channels * this->outputPlane;
        this->gridBatchStride = this->outHeight * this->outWidth * 2;

        xGm.SetGlobalBuffer((__gm__ float*)x, static_cast<uint64_t>(this->batch) * static_cast<uint64_t>(this->inputBatchStride));
        gridGm.SetGlobalBuffer((__gm__ float*)grid, static_cast<uint64_t>(this->batch) * static_cast<uint64_t>(this->gridBatchStride));
        yGm.SetGlobalBuffer((__gm__ float*)y, static_cast<uint64_t>(this->batch) * static_cast<uint64_t>(this->outputBatchStride));
    }

    __aicore__ inline void Process()
    {
        const float inWidthFp = static_cast<float>(inWidth);
        const float inHeightFp = static_cast<float>(inHeight);

        for (int32_t n = 0; n < batch; ++n) {
            const int64_t inputBatchBase = static_cast<int64_t>(n) * inputBatchStride;
            const int64_t outputBatchBase = static_cast<int64_t>(n) * outputBatchStride;
            const int64_t gridBatchBase = static_cast<int64_t>(n) * gridBatchStride;
            for (int32_t oh = 0; oh < outHeight; ++oh) {
                for (int32_t ow = 0; ow < outWidth; ++ow) {
                    const int64_t gridOffset = gridBatchBase + static_cast<int64_t>(oh * outWidth + ow) * 2;
                    const float gridX = gridGm.GetValue(gridOffset);
                    const float gridY = gridGm.GetValue(gridOffset + 1);

                    float ix = ((gridX + 1.0f) * inWidthFp - 1.0f) * 0.5f;
                    float iy = ((gridY + 1.0f) * inHeightFp - 1.0f) * 0.5f;

                    int32_t x0 = FloorToInt(ix);
                    int32_t y0 = FloorToInt(iy);
                    int32_t x1 = x0 + 1;
                    int32_t y1 = y0 + 1;

                    const float wx1 = ix - static_cast<float>(x0);
                    const float wy1 = iy - static_cast<float>(y0);
                    const float wx0 = 1.0f - wx1;
                    const float wy0 = 1.0f - wy1;

                    for (int32_t c = 0; c < channels; ++c) {
                        const int64_t inputChannelBase = inputBatchBase + static_cast<int64_t>(c) * inputPlane;
                        const int64_t outputOffset =
                            outputBatchBase + static_cast<int64_t>(c) * outputPlane + oh * outWidth + ow;

                        const float v00 = ReadWithZeroPadding(inputChannelBase, y0, x0);
                        const float v01 = ReadWithZeroPadding(inputChannelBase, y0, x1);
                        const float v10 = ReadWithZeroPadding(inputChannelBase, y1, x0);
                        const float v11 = ReadWithZeroPadding(inputChannelBase, y1, x1);

                        const float top = v00 * wx0 + v01 * wx1;
                        const float bottom = v10 * wx0 + v11 * wx1;
                        yGm.SetValue(outputOffset, top * wy0 + bottom * wy1);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline int32_t FloorToInt(float value) const
    {
        int32_t ivalue = static_cast<int32_t>(value);
        if (static_cast<float>(ivalue) > value) {
            ivalue = ivalue - 1;
        }
        return ivalue;
    }

    __aicore__ inline float ReadWithZeroPadding(int64_t channelBase, int32_t h, int32_t w) const
    {
        if (h < 0 || w < 0) {
            return 0.0f;
        }
        if (h >= static_cast<int32_t>(inHeight) || w >= static_cast<int32_t>(inWidth)) {
            return 0.0f;
        }

        const int64_t offset = channelBase + static_cast<int64_t>(h) * inWidth + w;
        return xGm.GetValue(offset);
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> gridGm;
    AscendC::GlobalTensor<float> yGm;
    int32_t batch;
    int32_t channels;
    int32_t inHeight;
    int32_t inWidth;
    int32_t outHeight;
    int32_t outWidth;
    int32_t inputPlane;
    int32_t outputPlane;
    int32_t inputBatchStride;
    int32_t outputBatchStride;
    int32_t gridBatchStride;
};

extern "C" __global__ __aicore__ void grid_sample_random_warp_custom(
    GM_ADDR x,
    GM_ADDR grid,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelGridSampleRandomWarp op;
    op.Init(
        x,
        grid,
        y,
        tiling_data.batch,
        tiling_data.channels,
        tiling_data.inHeight,
        tiling_data.inWidth,
        tiling_data.outHeight,
        tiling_data.outWidth);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor grid_sample_random_warp_custom_impl_npu(const at::Tensor& x, const at::Tensor& grid)
{
    auto gridSizes = grid.sizes();
    at::Tensor result = at::empty({x.size(0), x.size(1), gridSizes[1], gridSizes[2]}, x.options());
    EXEC_NPU_CMD(aclnnGridSampleRandomWarpCustom, x, grid, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("grid_sample_random_warp_custom", &grid_sample_random_warp_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_sample_random_warp_custom", &grid_sample_random_warp_custom_impl_npu, "grid_sample_random_warp_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, grid):
        return custom_ops_lib.grid_sample_random_warp_custom(x, grid)
'''
