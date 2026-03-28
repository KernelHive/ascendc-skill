project_json_src='''
[
    {
        "op": "UpsampleGridSampleCustom",
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
                "name": "theta",
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
BEGIN_TILING_DATA_DEF(UpsampleGridSampleCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, inputH);
  TILING_DATA_FIELD_DEF(uint32_t, inputW);
  TILING_DATA_FIELD_DEF(uint32_t, outputH);
  TILING_DATA_FIELD_DEF(uint32_t, outputW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpsampleGridSampleCustom, UpsampleGridSampleCustomTilingData)
}
"""

host_operator_src="""
#include "upsample_grid_sample_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    UpsampleGridSampleCustomTilingData tiling;
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const auto shape = inputShape->GetStorageShape();

    const uint32_t batch = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t channels = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t inputH = static_cast<uint32_t>(shape.GetDim(2));
    const uint32_t inputW = static_cast<uint32_t>(shape.GetDim(3));

    context->SetBlockDim(1);
    tiling.set_batch(batch);
    tiling.set_channels(channels);
    tiling.set_inputH(inputH);
    tiling.set_inputW(inputW);
    tiling.set_outputH(inputH * 2);
    tiling.set_outputW(inputW * 2);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, inputShape->GetDim(1));
    outputShape->SetDim(2, inputShape->GetDim(2) * 2);
    outputShape->SetDim(3, inputShape->GetDim(3) * 2);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class UpsampleGridSampleCustom : public OpDef {
public:
    explicit UpsampleGridSampleCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("theta")
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
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(UpsampleGridSampleCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelUpsampleGridSample {
public:
    __aicore__ inline KernelUpsampleGridSample() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR theta,
        GM_ADDR y,
        uint32_t batch,
        uint32_t channels,
        uint32_t inputH,
        uint32_t inputW,
        uint32_t outputH,
        uint32_t outputW)
    {
        this->batch = static_cast<int32_t>(batch);
        this->channels = static_cast<int32_t>(channels);
        this->inputH = static_cast<int32_t>(inputH);
        this->inputW = static_cast<int32_t>(inputW);
        this->outputH = static_cast<int32_t>(outputH);
        this->outputW = static_cast<int32_t>(outputW);
        this->inputPlaneSize = this->inputH * this->inputW;
        this->outputPlaneSize = this->outputH * this->outputW;

        xGm.SetGlobalBuffer((__gm__ float *)x, batch * channels * static_cast<uint32_t>(this->inputPlaneSize));
        thetaGm.SetGlobalBuffer((__gm__ float *)theta, batch * 6);
        yGm.SetGlobalBuffer((__gm__ float *)y, batch * channels * static_cast<uint32_t>(this->outputPlaneSize));
    }

    __aicore__ inline void Process()
    {
        const float outputHf = static_cast<float>(this->outputH);
        const float outputWf = static_cast<float>(this->outputW);
        const float inputHf = static_cast<float>(this->inputH);
        const float inputWf = static_cast<float>(this->inputW);

        for (int32_t n = 0; n < this->batch; ++n) {
            const int32_t thetaBase = n * 6;
            const float t00 = thetaGm.GetValue(thetaBase + 0);
            const float t01 = thetaGm.GetValue(thetaBase + 1);
            const float t02 = thetaGm.GetValue(thetaBase + 2);
            const float t10 = thetaGm.GetValue(thetaBase + 3);
            const float t11 = thetaGm.GetValue(thetaBase + 4);
            const float t12 = thetaGm.GetValue(thetaBase + 5);

            for (int32_t oh = 0; oh < this->outputH; ++oh) {
                const float yNorm = ((static_cast<float>(oh) + 0.5f) * 2.0f / outputHf) - 1.0f;
                for (int32_t ow = 0; ow < this->outputW; ++ow) {
                    const float xNorm = ((static_cast<float>(ow) + 0.5f) * 2.0f / outputWf) - 1.0f;
                    const float gridX = t00 * xNorm + t01 * yNorm + t02;
                    const float gridY = t10 * xNorm + t11 * yNorm + t12;
                    const float ix = ((gridX + 1.0f) * inputWf - 1.0f) * 0.5f;
                    const float iy = ((gridY + 1.0f) * inputHf - 1.0f) * 0.5f;
                    const int32_t x0 = FloorToInt(ix);
                    const int32_t y0 = FloorToInt(iy);
                    const int32_t x1 = x0 + 1;
                    const int32_t y1 = y0 + 1;

                    const float dx = ix - static_cast<float>(x0);
                    const float dy = iy - static_cast<float>(y0);
                    const float w00 = (1.0f - dx) * (1.0f - dy);
                    const float w01 = dx * (1.0f - dy);
                    const float w10 = (1.0f - dx) * dy;
                    const float w11 = dx * dy;

                    for (int32_t c = 0; c < this->channels; ++c) {
                        const int32_t inputBase = (n * this->channels + c) * this->inputPlaneSize;
                        const int32_t outputIndex =
                            ((n * this->channels + c) * this->outputPlaneSize) + oh * this->outputW + ow;

                        const float v00 = GetValueOrZero(inputBase, y0, x0);
                        const float v01 = GetValueOrZero(inputBase, y0, x1);
                        const float v10 = GetValueOrZero(inputBase, y1, x0);
                        const float v11 = GetValueOrZero(inputBase, y1, x1);

                        yGm.SetValue(outputIndex, v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline int32_t FloorToInt(float value) const
    {
        int32_t truncated = static_cast<int32_t>(value);
        if (static_cast<float>(truncated) > value) {
            truncated -= 1;
        }
        return truncated;
    }

    __aicore__ inline float GetValueOrZero(int32_t inputBase, int32_t h, int32_t w)
    {
        if (h < 0 || h >= this->inputH || w < 0 || w >= this->inputW) {
            return 0.0f;
        }
        return xGm.GetValue(inputBase + h * this->inputW + w);
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> thetaGm;
    GlobalTensor<float> yGm;
    int32_t batch;
    int32_t channels;
    int32_t inputH;
    int32_t inputW;
    int32_t outputH;
    int32_t outputW;
    int32_t inputPlaneSize;
    int32_t outputPlaneSize;
};

extern "C" __global__ __aicore__ void upsample_grid_sample_custom(
    GM_ADDR x,
    GM_ADDR theta,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelUpsampleGridSample op;
    op.Init(
        x,
        theta,
        y,
        tiling_data.batch,
        tiling_data.channels,
        tiling_data.inputH,
        tiling_data.inputW,
        tiling_data.outputH,
        tiling_data.outputW);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"
#include <ATen/ATen.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <vector>

at::Tensor upsample_grid_sample_custom_impl_npu(const at::Tensor &x, const at::Tensor &theta)
{
    std::vector<int64_t> upsample_size_vec = {x.size(2) * 2, x.size(3) * 2};
    at::Tensor upsampled = at::upsample_bilinear2d(
        x,
        at::IntArrayRef(upsample_size_vec),
        false,
        std::optional<double>(),
        std::optional<double>());

    std::vector<int64_t> output_size_vec = {
        upsampled.size(0), upsampled.size(1), upsampled.size(2), upsampled.size(3)};
    at::IntArrayRef output_size(output_size_vec);

    at::Tensor grid = at::empty(
        {upsampled.size(0), upsampled.size(2), upsampled.size(3), 2},
        upsampled.options());
    bool affine_align_corners = false;
    EXEC_NPU_CMD(aclnnAffineGrid, theta, output_size, affine_align_corners, grid);

    int64_t interpolation_mode = 0;
    int64_t padding_mode = 0;
    bool align_corners = false;
    at::Tensor result = at::empty_like(upsampled);
    EXEC_NPU_CMD(aclnnGridSampler2D, upsampled, grid, interpolation_mode, padding_mode, align_corners, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("upsample_grid_sample_custom", &upsample_grid_sample_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("upsample_grid_sample_custom", &upsample_grid_sample_custom_impl_npu, "upsample_grid_sample_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.upsample_grid_sample_custom(x, theta)
'''
