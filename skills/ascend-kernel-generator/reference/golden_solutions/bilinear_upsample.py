project_json_src='''
[
    {
        "op": "BilinearUpsampleCustom",
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
BEGIN_TILING_DATA_DEF(BilinearUpsampleCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, inHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outWidth);
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BilinearUpsampleCustom, BilinearUpsampleCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "bilinear_upsample_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batch = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t channels = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t inHeight = static_cast<uint32_t>(shape.GetDim(2));
    const uint32_t inWidth = static_cast<uint32_t>(shape.GetDim(3));
    if (inHeight == 0 || inWidth == 0) {
        return ge::GRAPH_FAILED;
    }

    BilinearUpsampleCustomTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim = ascendcPlatform.GetCoreNumAiv();
    if (blockDim == 0) {
        blockDim = 1;
    }

    const uint32_t totalRows = batch * channels * inHeight * 2;
    if (totalRows < blockDim) {
        blockDim = totalRows;
    }
    if (blockDim == 0) {
        blockDim = 1;
    }

    tiling.set_blockDim(blockDim);
    tiling.set_batch(batch);
    tiling.set_channels(channels);
    tiling.set_inHeight(inHeight);
    tiling.set_inWidth(inWidth);
    tiling.set_outHeight(inHeight * 2);
    tiling.set_outWidth(inWidth * 2);
    tiling.set_totalRows(totalRows);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr || inputShape->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, inputShape->GetDim(1));
    outputShape->SetDim(2, inputShape->GetDim(2) * 2);
    outputShape->SetDim(3, inputShape->GetDim(3) * 2);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class BilinearUpsampleCustom : public OpDef {
public:
    explicit BilinearUpsampleCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
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

OP_ADD(BilinearUpsampleCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

class KernelBilinearUpsample {
public:
    __aicore__ inline KernelBilinearUpsample() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batch,
        uint32_t channels,
        uint32_t inHeight,
        uint32_t inWidth,
        uint32_t outHeight,
        uint32_t outWidth,
        uint32_t blockDim,
        uint32_t totalRows)
    {
        this->batch = batch;
        this->channels = channels;
        this->inHeight = inHeight;
        this->inWidth = inWidth;
        this->outHeight = outHeight;
        this->outWidth = outWidth;
        this->blockDim = blockDim;
        this->totalRows = totalRows;
        this->inputPlane = inHeight * inWidth;
        this->outputPlane = outHeight * outWidth;

        xGm.SetGlobalBuffer((__gm__ float*)x, batch * channels * inputPlane);
        yGm.SetGlobalBuffer((__gm__ float*)y, batch * channels * outputPlane);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();

        for (uint32_t rowIdx = blockIdx; rowIdx < totalRows; rowIdx += blockDim) {
            ProcessRow(rowIdx);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t rowIdx)
    {
        const uint32_t ncIdx = rowIdx / outHeight;
        const uint32_t outY = rowIdx % outHeight;
        const uint32_t inputBase = ncIdx * inputPlane;
        const uint32_t outputBase = ncIdx * outputPlane + outY * outWidth;

        int32_t y0 = 0;
        int32_t y1 = 0;
        float wy0 = 1.0f;
        float wy1 = 0.0f;
        ComputeSourceIndex(outY, inHeight, y0, y1, wy0, wy1);

        for (uint32_t outX = 0; outX < outWidth; ++outX) {
            int32_t x0 = 0;
            int32_t x1 = 0;
            float wx0 = 1.0f;
            float wx1 = 0.0f;
            ComputeSourceIndex(outX, inWidth, x0, x1, wx0, wx1);

            const uint32_t topLeftIdx = inputBase + static_cast<uint32_t>(y0) * inWidth + static_cast<uint32_t>(x0);
            const uint32_t topRightIdx = inputBase + static_cast<uint32_t>(y0) * inWidth + static_cast<uint32_t>(x1);
            const uint32_t bottomLeftIdx = inputBase + static_cast<uint32_t>(y1) * inWidth + static_cast<uint32_t>(x0);
            const uint32_t bottomRightIdx = inputBase + static_cast<uint32_t>(y1) * inWidth + static_cast<uint32_t>(x1);

            const float topLeft = xGm.GetValue(topLeftIdx);
            const float topRight = xGm.GetValue(topRightIdx);
            const float bottomLeft = xGm.GetValue(bottomLeftIdx);
            const float bottomRight = xGm.GetValue(bottomRightIdx);

            const float top = topLeft * wx0 + topRight * wx1;
            const float bottom = bottomLeft * wx0 + bottomRight * wx1;
            yGm.SetValue(outputBase + outX, top * wy0 + bottom * wy1);
        }
    }

    __aicore__ inline void ComputeSourceIndex(
        uint32_t outIdx,
        uint32_t inputSize,
        int32_t& idx0,
        int32_t& idx1,
        float& w0,
        float& w1) const
    {
        const int32_t maxIndex = static_cast<int32_t>(inputSize) - 1;
        if (outIdx == 0 || maxIndex <= 0) {
            idx0 = 0;
            idx1 = 0;
            w0 = 1.0f;
            w1 = 0.0f;
            return;
        }

        if ((outIdx & 1U) == 1U) {
            idx0 = static_cast<int32_t>(outIdx >> 1);
            if (idx0 >= maxIndex) {
                idx0 = maxIndex;
                idx1 = maxIndex;
                w0 = 1.0f;
                w1 = 0.0f;
                return;
            }
            idx1 = idx0 + 1;
            w0 = 0.75f;
            w1 = 0.25f;
            return;
        }

        idx0 = static_cast<int32_t>(outIdx >> 1) - 1;
        if (idx0 < 0) {
            idx0 = 0;
        }
        if (idx0 >= maxIndex) {
            idx0 = maxIndex;
            idx1 = maxIndex;
            w0 = 1.0f;
            w1 = 0.0f;
            return;
        }
        idx1 = idx0 + 1;
        w0 = 0.25f;
        w1 = 0.75f;
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batch;
    uint32_t channels;
    uint32_t inHeight;
    uint32_t inWidth;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t blockDim;
    uint32_t totalRows;
    uint32_t inputPlane;
    uint32_t outputPlane;
};

extern "C" __global__ __aicore__ void bilinear_upsample_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelBilinearUpsample op;
    op.Init(
        x,
        y,
        tilingData.batch,
        tilingData.channels,
        tilingData.inHeight,
        tilingData.inWidth,
        tilingData.outHeight,
        tilingData.outWidth,
        tilingData.blockDim,
        tilingData.totalRows);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor bilinear_upsample_custom_impl_npu(const at::Tensor& self)
{
    auto outputShape = self.sizes().vec();
    outputShape[2] *= 2;
    outputShape[3] *= 2;
    at::Tensor result = at::empty(outputShape, self.options());
    EXEC_NPU_CMD(aclnnBilinearUpsampleCustom, self, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("bilinear_upsample_custom", &bilinear_upsample_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bilinear_upsample_custom", &bilinear_upsample_custom_impl_npu, "bilinear_upsample_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.bilinear_upsample_custom(x)
'''
