project_json_src='''
[
    {
        "op": "AveragePooling1dCustom",
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
        ],
        "attr": [
            {
                "name": "kernel_size",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "stride",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding",
                "param_type": "required",
                "type": "int"
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AveragePooling1dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, channelCount);
    TILING_DATA_FIELD_DEF(uint32_t, inputLength);
    TILING_DATA_FIELD_DEF(uint32_t, outputLength);
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(float, invKernelSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AveragePooling1dCustom, AveragePooling1dCustomTilingData)
}
"""

host_operator_src="""
#include "average_pooling1d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    AveragePooling1dCustomTilingData tiling;
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const auto shape = inputShape->GetStorageShape();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();

    const int64_t *kernelSizePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(2);

    const uint32_t batchSize = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t channelCount = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t inputLength = static_cast<uint32_t>(shape.GetDim(2));
    const uint32_t kernelSize = static_cast<uint32_t>(*kernelSizePtr);
    const uint32_t stride = static_cast<uint32_t>(*stridePtr);
    const uint32_t padding = static_cast<uint32_t>(*paddingPtr);
    const uint32_t paddedLength = inputLength + padding * 2;
    const uint32_t outputLength =
        paddedLength < kernelSize ? 0 : ((paddedLength - kernelSize) / stride + 1);

    context->SetBlockDim(batchSize == 0 ? 1 : batchSize);
    tiling.set_batchSize(batchSize);
    tiling.set_channelCount(channelCount);
    tiling.set_inputLength(inputLength);
    tiling.set_outputLength(outputLength);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_invKernelSize(kernelSize == 0 ? 0.0f : 1.0f / static_cast<float>(kernelSize));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *kernelSizePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(2);

    const int64_t inputLength = inputShape->GetDim(2);
    const int64_t paddedLength = inputLength + (*paddingPtr) * 2;
    const int64_t outputLength =
        paddedLength < *kernelSizePtr ? 0 : ((paddedLength - *kernelSizePtr) / *stridePtr + 1);

    *outputShape = {
        inputShape->GetDim(0),
        inputShape->GetDim(1),
        outputLength
    };
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class AveragePooling1dCustom : public OpDef {
public:
    explicit AveragePooling1dCustom(const char *name) : OpDef(name)
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
        this->Attr("kernel_size").AttrType(REQUIRED).Int();
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->Attr("padding").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(AveragePooling1dCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelAveragePooling1d {
public:
    __aicore__ inline KernelAveragePooling1d() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t channelCount,
        uint32_t inputLength,
        uint32_t outputLength,
        uint32_t kernelSize,
        uint32_t stride,
        uint32_t padding,
        float invKernelSize)
    {
        this->batchSize = batchSize;
        this->channelCount = channelCount;
        this->inputLength = inputLength;
        this->outputLength = outputLength;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;
        this->invKernelSize = invKernelSize;

        this->blockIdx = GetBlockIdx();
        this->elementsPerBatch = channelCount * inputLength;
        this->outputElementsPerBatch = channelCount * outputLength;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockIdx * this->elementsPerBatch, this->elementsPerBatch);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockIdx * this->outputElementsPerBatch, this->outputElementsPerBatch);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }
        if (this->outputLength == 0 || this->inputLength == 0) {
            return;
        }

        for (uint32_t channelIdx = 0; channelIdx < this->channelCount; ++channelIdx) {
            const uint32_t inputBase = channelIdx * this->inputLength;
            const uint32_t outputBase = channelIdx * this->outputLength;
            for (uint32_t outIdx = 0; outIdx < this->outputLength; ++outIdx) {
                const int32_t start = static_cast<int32_t>(outIdx * this->stride) - static_cast<int32_t>(this->padding);
                const int32_t end = start + static_cast<int32_t>(this->kernelSize);
                float sum = 0.0f;
                for (int32_t inIdx = start; inIdx < end; ++inIdx) {
                    if (inIdx >= 0 && inIdx < static_cast<int32_t>(this->inputLength)) {
                        sum += xGm.GetValue(inputBase + static_cast<uint32_t>(inIdx));
                    }
                }
                yGm.SetValue(outputBase + outIdx, sum * this->invKernelSize);
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t channelCount;
    uint32_t inputLength;
    uint32_t outputLength;
    uint32_t kernelSize;
    uint32_t stride;
    uint32_t padding;
    float invKernelSize;
    uint32_t blockIdx;
    uint32_t elementsPerBatch;
    uint32_t outputElementsPerBatch;
};

extern "C" __global__ __aicore__ void average_pooling1d_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelAveragePooling1d op;
    op.Init(
        x,
        y,
        tiling_data.batchSize,
        tiling_data.channelCount,
        tiling_data.inputLength,
        tiling_data.outputLength,
        tiling_data.kernelSize,
        tiling_data.stride,
        tiling_data.padding,
        tiling_data.invKernelSize);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor average_pooling1d_custom_impl_npu(
    const at::Tensor &x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding)
{
    TORCH_CHECK(x.dim() == 3, "average_pooling1d_custom expects a 3D tensor");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    const int64_t paddedLength = x.size(2) + padding * 2;
    const int64_t outputLength = paddedLength < kernel_size ? 0 : ((paddedLength - kernel_size) / stride + 1);
    at::Tensor result = at::empty({x.size(0), x.size(1), outputLength}, x.options());
    EXEC_NPU_CMD(aclnnAveragePooling1dCustom, x, kernel_size, stride, padding, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("average_pooling1d_custom", &average_pooling1d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "average_pooling1d_custom",
        &average_pooling1d_custom_impl_npu,
        "AveragePool1d custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.average_pooling1d_custom(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
        )
'''
