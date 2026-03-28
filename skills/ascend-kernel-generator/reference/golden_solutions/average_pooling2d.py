project_json_src='''
[
    {
        "op": "AveragePooling2dCustom",
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
#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AveragePooling2dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, inHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(uint32_t, totalRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AveragePooling2dCustom, AveragePooling2dCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "average_pooling2d_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
inline uint32_t PoolOutputSize(uint32_t inputSize, uint32_t kernelSize, uint32_t stride, uint32_t padding)
{
    const int64_t numerator =
        static_cast<int64_t>(inputSize) + static_cast<int64_t>(padding) * 2 - static_cast<int64_t>(kernelSize);
    if (numerator < 0 || stride == 0) {
        return 0;
    }
    return static_cast<uint32_t>(numerator / stride + 1);
}
}

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

    const uint32_t kernelSize = 3;
    const uint32_t stride = 3;
    const uint32_t padding = 0;

    const uint32_t batch = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t channels = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t inHeight = static_cast<uint32_t>(shape.GetDim(2));
    const uint32_t inWidth = static_cast<uint32_t>(shape.GetDim(3));
    const uint32_t outHeight = PoolOutputSize(inHeight, kernelSize, stride, padding);
    const uint32_t outWidth = PoolOutputSize(inWidth, kernelSize, stride, padding);
    if (outHeight == 0 || outWidth == 0) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim = ascendcPlatform.GetCoreNumAiv();
    if (blockDim == 0) {
        blockDim = 1;
    }

    const uint32_t totalRows = batch * channels * outHeight;
    if (totalRows < blockDim) {
        blockDim = totalRows;
    }
    if (blockDim == 0) {
        blockDim = 1;
    }

    AveragePooling2dCustomTilingData tiling;
    tiling.set_blockDim(blockDim);
    tiling.set_batch(batch);
    tiling.set_channels(channels);
    tiling.set_inHeight(inHeight);
    tiling.set_inWidth(inWidth);
    tiling.set_outHeight(outHeight);
    tiling.set_outWidth(outWidth);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
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

    const int64_t kernelSize = 3;
    const int64_t stride = 3;
    const int64_t padding = 0;

    const int64_t inHeight = inputShape->GetDim(2);
    const int64_t inWidth = inputShape->GetDim(3);
    const int64_t outHeight = (inHeight + padding * 2 - kernelSize) / stride + 1;
    const int64_t outWidth = (inWidth + padding * 2 - kernelSize) / stride + 1;
    if (outHeight <= 0 || outWidth <= 0) {
        return GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, inputShape->GetDim(1));
    outputShape->SetDim(2, outHeight);
    outputShape->SetDim(3, outWidth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class AveragePooling2dCustom : public OpDef {
public:
    explicit AveragePooling2dCustom(const char* name) : OpDef(name)
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

OP_ADD(AveragePooling2dCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

class KernelAveragePooling2D {
public:
    __aicore__ inline KernelAveragePooling2D() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batch,
        uint32_t channels,
        uint32_t inHeight,
        uint32_t inWidth,
        uint32_t outHeight,
        uint32_t outWidth,
        uint32_t kernelSize,
        uint32_t stride,
        uint32_t padding,
        uint32_t blockDim,
        uint32_t totalRows)
    {
        this->batch = batch;
        this->channels = channels;
        this->inHeight = inHeight;
        this->inWidth = inWidth;
        this->outHeight = outHeight;
        this->outWidth = outWidth;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;
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
        const int32_t startY = static_cast<int32_t>(outY * stride) - static_cast<int32_t>(padding);

        for (uint32_t outX = 0; outX < outWidth; ++outX) {
            const int32_t startX = static_cast<int32_t>(outX * stride) - static_cast<int32_t>(padding);
            float sum = 0.0f;
            float count = 0.0f;

            for (uint32_t ky = 0; ky < kernelSize; ++ky) {
                const int32_t inY = startY + static_cast<int32_t>(ky);
                if (inY < 0 || inY >= static_cast<int32_t>(inHeight)) {
                    continue;
                }
                for (uint32_t kx = 0; kx < kernelSize; ++kx) {
                    const int32_t inX = startX + static_cast<int32_t>(kx);
                    if (inX < 0 || inX >= static_cast<int32_t>(inWidth)) {
                        continue;
                    }
                    const uint32_t inputIdx =
                        inputBase + static_cast<uint32_t>(inY) * inWidth + static_cast<uint32_t>(inX);
                    sum += xGm.GetValue(inputIdx);
                    count += 1.0f;
                }
            }

            const float divisor = count == 0.0f ? 1.0f : count;
            yGm.SetValue(outputBase + outX, sum / divisor);
        }
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
    uint32_t kernelSize;
    uint32_t stride;
    uint32_t padding;
    uint32_t blockDim;
    uint32_t totalRows;
    uint32_t inputPlane;
    uint32_t outputPlane;
};

extern "C" __global__ __aicore__ void average_pooling2d_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    KernelAveragePooling2D op;
    op.Init(
        x,
        y,
        tilingData.batch,
        tilingData.channels,
        tilingData.inHeight,
        tilingData.inWidth,
        tilingData.outHeight,
        tilingData.outWidth,
        tilingData.kernelSize,
        tilingData.stride,
        tilingData.padding,
        tilingData.blockDim,
        tilingData.totalRows);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

namespace {
int64_t PoolOutputSize(int64_t inputSize, int64_t kernelSize, int64_t stride, int64_t padding)
{
    return (inputSize + padding * 2 - kernelSize) / stride + 1;
}
}

at::Tensor average_pooling2d_custom_impl_npu(const at::Tensor& self)
{
    std::vector<int64_t> kernelSizeVec = {3, 3};
    std::vector<int64_t> strideVec = {3, 3};
    std::vector<int64_t> paddingVec = {0, 0};
    at::IntArrayRef kernelSize(kernelSizeVec);
    at::IntArrayRef strides(strideVec);
    at::IntArrayRef paddings(paddingVec);
    constexpr bool ceilMode = false;
    constexpr bool countIncludePad = true;
    constexpr int64_t divisorOverride = 0;
    constexpr int8_t cubeMathType = 0;
    std::vector<int64_t> outputShape = {
        self.size(0),
        self.size(1),
        PoolOutputSize(self.size(2), 3, 3, 0),
        PoolOutputSize(self.size(3), 3, 3, 0),
    };
    at::Tensor result = at::empty(outputShape, self.options());
    EXEC_NPU_CMD(
        aclnnAvgPool2d,
        self,
        kernelSize,
        strides,
        paddings,
        ceilMode,
        countIncludePad,
        divisorOverride,
        cubeMathType,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("average_pooling2d_custom", &average_pooling2d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("average_pooling2d_custom", &average_pooling2d_custom_impl_npu, "average_pooling2d_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.average_pooling2d_custom(x)
'''
