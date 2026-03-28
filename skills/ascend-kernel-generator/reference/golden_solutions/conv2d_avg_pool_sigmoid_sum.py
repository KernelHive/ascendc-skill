project_json_src='''
[
    {
        "op": "Conv2dAvgPoolSigmoidSumCustom",
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
                "name": "weight",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "bias",
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
                "name": "pool_kernel_size",
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
BEGIN_TILING_DATA_DEF(Conv2dAvgPoolSigmoidSumCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, convOutputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, convOutputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, poolKernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, poolOutputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, poolOutputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, pooledHw);
    TILING_DATA_FIELD_DEF(uint32_t, alignedPooledHw);
    TILING_DATA_FIELD_DEF(float, poolReciprocal);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Conv2dAvgPoolSigmoidSumCustom,
    Conv2dAvgPoolSigmoidSumCustomTilingData)
}
"""

host_operator_src="""
#include "conv2d_avg_pool_sigmoid_sum_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t AlignUp8(uint32_t value)
{
    if (value == 0) {
        return 8;
    }
    return ((value + 7U) / 8U) * 8U;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *biasShape = context->GetInputShape(2);
    if (inputShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    const auto bShape = biasShape->GetStorageShape();
    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 4 || bShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *poolKernelPtr = attrs->GetAttrPointer<int64_t>(0);
    if (poolKernelPtr == nullptr || *poolKernelPtr <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t weightInChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t biasLength = static_cast<uint32_t>(bShape.GetDim(0));
    const uint32_t poolKernelSize = static_cast<uint32_t>(*poolKernelPtr);

    if (inChannels != weightInChannels || biasLength != outChannels) {
        return ge::GRAPH_FAILED;
    }
    if (inputHeight < kernelHeight || inputWidth < kernelWidth) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t convOutputHeight = inputHeight - kernelHeight + 1;
    const uint32_t convOutputWidth = inputWidth - kernelWidth + 1;
    if (convOutputHeight < poolKernelSize || convOutputWidth < poolKernelSize) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t poolOutputHeight = (convOutputHeight - poolKernelSize) / poolKernelSize + 1;
    const uint32_t poolOutputWidth = (convOutputWidth - poolKernelSize) / poolKernelSize + 1;
    const uint32_t pooledHw = poolOutputHeight * poolOutputWidth;

    Conv2dAvgPoolSigmoidSumCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_convOutputHeight(convOutputHeight);
    tiling.set_convOutputWidth(convOutputWidth);
    tiling.set_poolKernelSize(poolKernelSize);
    tiling.set_poolOutputHeight(poolOutputHeight);
    tiling.set_poolOutputWidth(poolOutputWidth);
    tiling.set_pooledHw(pooledHw);
    tiling.set_alignedPooledHw(AlignUp8(pooledHw));
    tiling.set_poolReciprocal(1.0f / static_cast<float>(poolKernelSize * poolKernelSize));

    context->SetBlockDim(1);
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
    const gert::Shape *weightShape = context->GetInputShape(1);
    const gert::Shape *biasShape = context->GetInputShape(2);
    if (inputShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDim(1) != weightShape->GetDim(1) || weightShape->GetDim(0) != biasShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *poolKernelPtr = attrs->GetAttrPointer<int64_t>(0);
    if (poolKernelPtr == nullptr || *poolKernelPtr <= 0) {
        return GRAPH_FAILED;
    }

    const int64_t convOutputHeight = inputShape->GetDim(2) - weightShape->GetDim(2) + 1;
    const int64_t convOutputWidth = inputShape->GetDim(3) - weightShape->GetDim(3) + 1;
    if (convOutputHeight < *poolKernelPtr || convOutputWidth < *poolKernelPtr) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(1);
    outputShape->SetDim(0, inputShape->GetDim(0));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv2dAvgPoolSigmoidSumCustom : public OpDef {
public:
    explicit Conv2dAvgPoolSigmoidSumCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("pool_kernel_size").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dAvgPoolSigmoidSumCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConv2dAvgPoolSigmoidSum {
public:
    __aicore__ inline KernelConv2dAvgPoolSigmoidSum() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelHeight,
        uint32_t kernelWidth,
        uint32_t convOutputHeight,
        uint32_t convOutputWidth,
        uint32_t poolKernelSize,
        uint32_t poolOutputHeight,
        uint32_t poolOutputWidth,
        uint32_t pooledHw,
        uint32_t alignedPooledHw,
        float poolReciprocal)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->convOutputHeight = convOutputHeight;
        this->convOutputWidth = convOutputWidth;
        this->poolKernelSize = poolKernelSize;
        this->poolOutputHeight = poolOutputHeight;
        this->poolOutputWidth = poolOutputWidth;
        this->pooledHw = pooledHw;
        this->alignedPooledHw = alignedPooledHw;
        this->inputChannelStride = inputHeight * inputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->weightOutStride = inChannels * kernelHeight * kernelWidth;
        this->weightInStride = kernelHeight * kernelWidth;
        this->poolScale = poolReciprocal;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * this->weightOutStride);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize);
        pipe.InitBuffer(calcBuf, this->alignedPooledHw * 2 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> poolLocal = calcBuf.Get<float>();
        LocalTensor<float> sigLocal = calcBuf.Get<float>()[this->alignedPooledHw];

        for (uint32_t batchIdx = 0; batchIdx < this->batchSize; ++batchIdx) {
            if (this->pooledHw == 0) {
                yGm.SetValue(batchIdx, 0.0f);
                continue;
            }

            float batchSum = 0.0f;
            for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                FillPooledChannel(poolLocal, batchIdx, outChannel);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Sigmoid(sigLocal, poolLocal, this->pooledHw);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                for (uint32_t idx = 0; idx < this->pooledHw; ++idx) {
                    batchSum += sigLocal.GetValue(idx);
                }
            }
            yGm.SetValue(batchIdx, batchSum);
        }
    }

private:
    __aicore__ inline float ComputeConvAt(
        uint32_t batchIdx,
        uint32_t outChannel,
        uint32_t convH,
        uint32_t convW) const
    {
        const uint32_t inputBatchBase = batchIdx * this->inputBatchStride;
        const uint32_t weightBase = outChannel * this->weightOutStride;
        float sum = biasGm.GetValue(outChannel);
        for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
            const uint32_t inputChannelBase = inputBatchBase + inChannel * this->inputChannelStride;
            const uint32_t weightChannelBase = weightBase + inChannel * this->weightInStride;
            for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                const uint32_t inputRowBase = inputChannelBase + (convH + kernelH) * this->inputWidth;
                const uint32_t weightRowBase = weightChannelBase + kernelH * this->kernelWidth;
                for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                    const uint32_t xOffset = inputRowBase + convW + kernelW;
                    const uint32_t wOffset = weightRowBase + kernelW;
                    sum += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                }
            }
        }
        return sum;
    }

    __aicore__ inline void FillPooledChannel(
        LocalTensor<float> &poolLocal,
        uint32_t batchIdx,
        uint32_t outChannel)
    {
        uint32_t pooledIdx = 0;
        for (uint32_t poolH = 0; poolH < this->poolOutputHeight; ++poolH) {
            const uint32_t convStartH = poolH * this->poolKernelSize;
            for (uint32_t poolW = 0; poolW < this->poolOutputWidth; ++poolW) {
                const uint32_t convStartW = poolW * this->poolKernelSize;
                float pooledValue = 0.0f;
                for (uint32_t windowH = 0; windowH < this->poolKernelSize; ++windowH) {
                    for (uint32_t windowW = 0; windowW < this->poolKernelSize; ++windowW) {
                        pooledValue += ComputeConvAt(
                            batchIdx,
                            outChannel,
                            convStartH + windowH,
                            convStartW + windowW);
                    }
                }
                poolLocal.SetValue(pooledIdx, pooledValue * this->poolScale);
                ++pooledIdx;
            }
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> calcBuf;
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t convOutputHeight;
    uint32_t convOutputWidth;
    uint32_t poolKernelSize;
    uint32_t poolOutputHeight;
    uint32_t poolOutputWidth;
    uint32_t pooledHw;
    uint32_t alignedPooledHw;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t weightOutStride;
    uint32_t weightInStride;
    float poolScale;
};

extern "C" __global__ __aicore__ void conv2d_avg_pool_sigmoid_sum_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dAvgPoolSigmoidSum op;
    op.Init(
        x,
        weight,
        bias,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.kernelHeight,
        tiling_data.kernelWidth,
        tiling_data.convOutputHeight,
        tiling_data.convOutputWidth,
        tiling_data.poolKernelSize,
        tiling_data.poolOutputHeight,
        tiling_data.poolOutputWidth,
        tiling_data.pooledHw,
        tiling_data.alignedPooledHw,
        tiling_data.poolReciprocal);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv2d_avg_pool_sigmoid_sum_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    int64_t poolKernelSize)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(poolKernelSize > 0, "pool_kernel_size must be positive");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == bias.size(0), "bias length must match output channels");
    TORCH_CHECK(x.size(2) >= weight.size(2) && x.size(3) >= weight.size(3), "invalid conv kernel size");

    const int64_t convOutputHeight = x.size(2) - weight.size(2) + 1;
    const int64_t convOutputWidth = x.size(3) - weight.size(3) + 1;
    TORCH_CHECK(
        convOutputHeight >= poolKernelSize && convOutputWidth >= poolKernelSize,
        "pool kernel is larger than conv output");

    at::Tensor result = at::empty({x.size(0)}, x.options());
    EXEC_NPU_CMD(aclnnConv2dAvgPoolSigmoidSumCustom, x, weight, bias, poolKernelSize, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_avg_pool_sigmoid_sum_custom", &conv2d_avg_pool_sigmoid_sum_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv2d_avg_pool_sigmoid_sum_custom",
        &conv2d_avg_pool_sigmoid_sum_impl_npu,
        "conv2d_avg_pool_sigmoid_sum_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_kernel_size: int,
    ) -> None:
        super().__init__()
        self.pool_kernel_size = pool_kernel_size
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv2d_avg_pool_sigmoid_sum_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.pool_kernel_size,
        )
'''
