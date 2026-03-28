project_json_src='''
[
    {
        "op": "MaxPooling1dCustom",
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
BEGIN_TILING_DATA_DEF(MaxPooling1dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, channelSize);
    TILING_DATA_FIELD_DEF(uint32_t, inputLength);
    TILING_DATA_FIELD_DEF(uint32_t, outputLength);
    TILING_DATA_FIELD_DEF(uint32_t, totalPlanes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPooling1dCustom, MaxPooling1dCustomTilingData)
}
"""

host_operator_src="""
#include "max_pooling1d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kKernelSize = 4;
constexpr uint32_t kStride = 2;
constexpr uint32_t kPadding = 2;
constexpr uint32_t kDilation = 3;
constexpr uint32_t kBlockDim = 1;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batchSize = shape.GetDim(0);
    const int64_t channelSize = shape.GetDim(1);
    const int64_t inputLength = shape.GetDim(2);
    if (batchSize <= 0 || channelSize <= 0 || inputLength <= 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t effectiveKernel = static_cast<int64_t>(kDilation) * (static_cast<int64_t>(kKernelSize) - 1) + 1;
    const int64_t numerator = inputLength + 2 * static_cast<int64_t>(kPadding) - effectiveKernel;
    if (numerator < 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t outputLength = numerator / static_cast<int64_t>(kStride) + 1;
    if (outputLength <= 0) {
        return ge::GRAPH_FAILED;
    }

    MaxPooling1dCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(batchSize));
    tiling.set_channelSize(static_cast<uint32_t>(channelSize));
    tiling.set_inputLength(static_cast<uint32_t>(inputLength));
    tiling.set_outputLength(static_cast<uint32_t>(outputLength));
    tiling.set_totalPlanes(static_cast<uint32_t>(batchSize * channelSize));

    context->SetBlockDim(kBlockDim);
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
    if (inputShape == nullptr || inputShape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }

    const int64_t inputLength = inputShape->GetDim(2);
    const int64_t effectiveKernel = static_cast<int64_t>(kDilation) * (static_cast<int64_t>(kKernelSize) - 1) + 1;
    const int64_t numerator = inputLength + 2 * static_cast<int64_t>(kPadding) - effectiveKernel;
    if (numerator < 0) {
        return GRAPH_FAILED;
    }

    const int64_t outputLength = numerator / static_cast<int64_t>(kStride) + 1;
    if (outputLength <= 0) {
        return GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(3);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, inputShape->GetDim(1));
    outputShape->SetDim(2, outputLength);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class MaxPooling1dCustom : public OpDef {
public:
    explicit MaxPooling1dCustom(const char* name) : OpDef(name)
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

OP_ADD(MaxPooling1dCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

namespace {
constexpr uint32_t kKernelSize = 4;
constexpr uint32_t kStride = 2;
constexpr int32_t kPadding = 2;
constexpr uint32_t kDilation = 3;
}

__aicore__ inline float MaxF32(float lhs, float rhs)
{
    return lhs > rhs ? lhs : rhs;
}

class KernelMaxPooling1d {
public:
    __aicore__ inline KernelMaxPooling1d() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t channelSize,
        uint32_t inputLength,
        uint32_t outputLength,
        uint32_t totalPlanes)
    {
        this->batchSize = batchSize;
        this->channelSize = channelSize;
        this->inputLength = inputLength;
        this->outputLength = outputLength;
        this->totalPlanes = totalPlanes;
        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * channelSize * inputLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * channelSize * outputLength);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t planeIdx = 0; planeIdx < totalPlanes; ++planeIdx) {
            ProcessPlane(planeIdx);
        }
    }

private:
    __aicore__ inline void ProcessPlane(uint32_t planeIdx)
    {
        const uint32_t inputBase = planeIdx * inputLength;
        const uint32_t outputBase = planeIdx * outputLength;
        for (uint32_t outIdx = 0; outIdx < outputLength; ++outIdx) {
            const int32_t start = static_cast<int32_t>(outIdx * kStride) - kPadding;
            bool hasValue = false;
            float maxValue = 0.0f;
            for (uint32_t kernelIdx = 0; kernelIdx < kKernelSize; ++kernelIdx) {
                const int32_t inputIdx = start + static_cast<int32_t>(kernelIdx * kDilation);
                if (inputIdx < 0 || inputIdx >= static_cast<int32_t>(inputLength)) {
                    continue;
                }
                const float value = xGm.GetValue(inputBase + static_cast<uint32_t>(inputIdx));
                if (!hasValue) {
                    maxValue = value;
                    hasValue = true;
                } else {
                    maxValue = MaxF32(maxValue, value);
                }
            }
            yGm.SetValue(outputBase + outIdx, hasValue ? maxValue : 0.0f);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t channelSize;
    uint32_t inputLength;
    uint32_t outputLength;
    uint32_t totalPlanes;
};

extern "C" __global__ __aicore__ void max_pooling1d_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaxPooling1d op;
    op.Init(
        x,
        y,
        tiling_data.batchSize,
        tiling_data.channelSize,
        tiling_data.inputLength,
        tiling_data.outputLength,
        tiling_data.totalPlanes);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor max_pooling1d_custom_impl_npu(const at::Tensor& input)
{
    const int64_t outputLength =
        (input.size(2) + 2 * 2 - static_cast<int64_t>(3) * (static_cast<int64_t>(4) - 1) - 1) / 2 + 1;
    auto outputShape = std::vector<int64_t>{input.size(0), input.size(1), outputLength};
    at::Tensor result = at::empty(outputShape, input.options());
    EXEC_NPU_CMD(aclnnMaxPooling1dCustom, input, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("max_pooling1d_custom", &max_pooling1d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pooling1d_custom", &max_pooling1d_custom_impl_npu, "max_pooling1d_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
    ) -> None:
        super().__init__()
        stride = kernel_size if stride is None else stride
        if kernel_size != 4 or stride != 2 or padding != 2 or dilation != 3 or return_indices:
            raise ValueError(
                "This AscendC implementation currently supports "
                "kernel_size=4, stride=2, padding=2, dilation=3, return_indices=False only."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.max_pooling1d_custom(x)
'''
