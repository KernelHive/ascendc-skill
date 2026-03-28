project_json_src='''
[
    {
        "op": "MaxPooling3dCustom",
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
            },
            {
                "name": "dilation",
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
BEGIN_TILING_DATA_DEF(MaxPooling3dCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, channelCount);
    TILING_DATA_FIELD_DEF(uint32_t, inputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(uint32_t, dilation);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPooling3dCustom, MaxPooling3dCustomTilingData)
}
"""

host_operator_src="""
#include "max_pooling3d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeOutputDim(int64_t input, int64_t kernelSize, int64_t stride, int64_t padding, int64_t dilation)
{
    const int64_t effectiveKernel = dilation * (kernelSize - 1) + 1;
    const int64_t numerator = input + padding * 2 - effectiveKernel;
    if (numerator < 0) {
        return 0;
    }
    return static_cast<uint32_t>(numerator / stride + 1);
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    MaxPooling3dCustomTilingData tiling;
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const auto shape = inputShape->GetStorageShape();
    const gert::RuntimeAttrs *attrs = context->GetAttrs();

    const int64_t *kernelSizePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *dilationPtr = attrs->GetAttrPointer<int64_t>(3);

    const uint32_t batchSize = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t channelCount = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t inputDepth = static_cast<uint32_t>(shape.GetDim(2));
    const uint32_t inputHeight = static_cast<uint32_t>(shape.GetDim(3));
    const uint32_t inputWidth = static_cast<uint32_t>(shape.GetDim(4));
    const uint32_t kernelSize = static_cast<uint32_t>(*kernelSizePtr);
    const uint32_t stride = static_cast<uint32_t>(*stridePtr);
    const uint32_t padding = static_cast<uint32_t>(*paddingPtr);
    const uint32_t dilation = static_cast<uint32_t>(*dilationPtr);

    const uint32_t outputDepth = ComputeOutputDim(inputDepth, kernelSize, stride, padding, dilation);
    const uint32_t outputHeight = ComputeOutputDim(inputHeight, kernelSize, stride, padding, dilation);
    const uint32_t outputWidth = ComputeOutputDim(inputWidth, kernelSize, stride, padding, dilation);

    context->SetBlockDim(batchSize == 0 ? 1 : batchSize);
    tiling.set_batchSize(batchSize);
    tiling.set_channelCount(channelCount);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_outputDepth(outputDepth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_dilation(dilation);
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
    const int64_t *dilationPtr = attrs->GetAttrPointer<int64_t>(3);

    const uint32_t outputDepth =
        ComputeOutputDim(inputShape->GetDim(2), *kernelSizePtr, *stridePtr, *paddingPtr, *dilationPtr);
    const uint32_t outputHeight =
        ComputeOutputDim(inputShape->GetDim(3), *kernelSizePtr, *stridePtr, *paddingPtr, *dilationPtr);
    const uint32_t outputWidth =
        ComputeOutputDim(inputShape->GetDim(4), *kernelSizePtr, *stridePtr, *paddingPtr, *dilationPtr);

    *outputShape = {
        inputShape->GetDim(0),
        inputShape->GetDim(1),
        outputDepth,
        outputHeight,
        outputWidth
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
class MaxPooling3dCustom : public OpDef {
public:
    explicit MaxPooling3dCustom(const char *name) : OpDef(name)
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
        this->Attr("dilation").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(MaxPooling3dCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelMaxPooling3d {
public:
    __aicore__ inline KernelMaxPooling3d() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t channelCount,
        uint32_t inputDepth,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t outputDepth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t kernelSize,
        uint32_t stride,
        uint32_t padding,
        uint32_t dilation)
    {
        this->batchSize = batchSize;
        this->channelCount = channelCount;
        this->inputDepth = inputDepth;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->outputDepth = outputDepth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;

        this->blockIdx = GetBlockIdx();
        this->inputCubeSize = inputDepth * inputHeight * inputWidth;
        this->outputCubeSize = outputDepth * outputHeight * outputWidth;
        this->elementsPerBatch = channelCount * this->inputCubeSize;
        this->outputElementsPerBatch = channelCount * this->outputCubeSize;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockIdx * this->elementsPerBatch, this->elementsPerBatch);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockIdx * this->outputElementsPerBatch, this->outputElementsPerBatch);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }
        if (this->outputDepth == 0 || this->outputHeight == 0 || this->outputWidth == 0) {
            return;
        }
        if (this->inputDepth == 0 || this->inputHeight == 0 || this->inputWidth == 0) {
            return;
        }

        const int32_t inputDepth = static_cast<int32_t>(this->inputDepth);
        const int32_t inputHeight = static_cast<int32_t>(this->inputHeight);
        const int32_t inputWidth = static_cast<int32_t>(this->inputWidth);
        const int32_t kernelSize = static_cast<int32_t>(this->kernelSize);
        const int32_t stride = static_cast<int32_t>(this->stride);
        const int32_t padding = static_cast<int32_t>(this->padding);
        const int32_t dilation = static_cast<int32_t>(this->dilation);

        for (uint32_t channelIdx = 0; channelIdx < this->channelCount; ++channelIdx) {
            const uint32_t inputBase = channelIdx * this->inputCubeSize;
            const uint32_t outputBase = channelIdx * this->outputCubeSize;
            for (uint32_t outD = 0; outD < this->outputDepth; ++outD) {
                const int32_t startD = static_cast<int32_t>(outD) * stride - padding;
                for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                    const int32_t startH = static_cast<int32_t>(outH) * stride - padding;
                    for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                        const int32_t startW = static_cast<int32_t>(outW) * stride - padding;
                        float maxValue = -3.40282347e+38f;
                        bool hasValid = false;
                        for (int32_t kernelD = 0; kernelD < kernelSize; ++kernelD) {
                            const int32_t inD = startD + kernelD * dilation;
                            if (inD < 0 || inD >= inputDepth) {
                                continue;
                            }
                            for (int32_t kernelH = 0; kernelH < kernelSize; ++kernelH) {
                                const int32_t inH = startH + kernelH * dilation;
                                if (inH < 0 || inH >= inputHeight) {
                                    continue;
                                }
                                for (int32_t kernelW = 0; kernelW < kernelSize; ++kernelW) {
                                    const int32_t inW = startW + kernelW * dilation;
                                    if (inW < 0 || inW >= inputWidth) {
                                        continue;
                                    }
                                    const uint32_t inputOffset =
                                        inputBase +
                                        static_cast<uint32_t>(inD) * this->inputHeight * this->inputWidth +
                                        static_cast<uint32_t>(inH) * this->inputWidth +
                                        static_cast<uint32_t>(inW);
                                    const float value = xGm.GetValue(inputOffset);
                                    if (!hasValid || value > maxValue) {
                                        maxValue = value;
                                        hasValid = true;
                                    }
                                }
                            }
                        }
                        const uint32_t outputOffset =
                            outputBase +
                            outD * this->outputHeight * this->outputWidth +
                            outH * this->outputWidth +
                            outW;
                        yGm.SetValue(outputOffset, hasValid ? maxValue : -3.40282347e+38f);
                    }
                }
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t channelCount;
    uint32_t inputDepth;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t outputDepth;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t kernelSize;
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
    uint32_t blockIdx;
    uint32_t inputCubeSize;
    uint32_t outputCubeSize;
    uint32_t elementsPerBatch;
    uint32_t outputElementsPerBatch;
};

extern "C" __global__ __aicore__ void max_pooling3d_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaxPooling3d op;
    op.Init(
        x,
        y,
        tiling_data.batchSize,
        tiling_data.channelCount,
        tiling_data.inputDepth,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.outputDepth,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.kernelSize,
        tiling_data.stride,
        tiling_data.padding,
        tiling_data.dilation);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

namespace {
int64_t ComputeOutputDim(int64_t input, int64_t kernelSize, int64_t stride, int64_t padding, int64_t dilation)
{
    const int64_t effectiveKernel = dilation * (kernelSize - 1) + 1;
    const int64_t numerator = input + padding * 2 - effectiveKernel;
    if (numerator < 0) {
        return 0;
    }
    return numerator / stride + 1;
}
}

at::Tensor max_pooling3d_custom_impl_npu(
    const at::Tensor &x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation)
{
    TORCH_CHECK(x.dim() == 5, "max_pooling3d_custom expects a 5D NCDHW tensor");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation > 0, "dilation must be positive");

    const int64_t outputDepth = ComputeOutputDim(x.size(2), kernel_size, stride, padding, dilation);
    const int64_t outputHeight = ComputeOutputDim(x.size(3), kernel_size, stride, padding, dilation);
    const int64_t outputWidth = ComputeOutputDim(x.size(4), kernel_size, stride, padding, dilation);
    at::Tensor result = at::empty({x.size(0), x.size(1), outputDepth, outputHeight, outputWidth}, x.options());
    EXEC_NPU_CMD(aclnnMaxPooling3dCustom, x, kernel_size, stride, padding, dilation, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("max_pooling3d_custom", &max_pooling3d_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "max_pooling3d_custom",
        &max_pooling3d_custom_impl_npu,
        "MaxPool3d custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.max_pooling3d_custom(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
'''
