project_json_src='''
[
    {
        "op": "Conv3dSoftmaxMaxPoolMaxPoolCustom",
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
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dSoftmaxMaxPoolMaxPoolCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelDepth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, convDepth);
    TILING_DATA_FIELD_DEF(uint32_t, convHeight);
    TILING_DATA_FIELD_DEF(uint32_t, convWidth);
    TILING_DATA_FIELD_DEF(uint32_t, poolKernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, fusedPoolKernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, fusedPoolStride);
    TILING_DATA_FIELD_DEF(uint32_t, outputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, alignedChannels);
    TILING_DATA_FIELD_DEF(uint32_t, tmpSize);
    TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Conv3dSoftmaxMaxPoolMaxPoolCustom,
    Conv3dSoftmaxMaxPoolMaxPoolCustomTilingData)
}
"""

host_operator_src="""
#include "conv3d_softmax_max_pool_max_pool_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
uint32_t AlignUp8(uint32_t value)
{
    return (value + 7U) / 8U * 8U;
}

uint32_t ComputeConvOutputDim(int64_t input, int64_t kernelSize)
{
    const int64_t output = input - kernelSize + 1;
    if (output <= 0) {
        return 0;
    }
    return static_cast<uint32_t>(output);
}

uint32_t ComputePoolOutputDim(int64_t input, int64_t kernelSize, int64_t stride)
{
    if (kernelSize <= 0 || stride <= 0) {
        return 0;
    }
    const int64_t numerator = input - kernelSize;
    if (numerator < 0) {
        return 0;
    }
    return static_cast<uint32_t>(numerator / stride + 1);
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
    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5 || bShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *poolKernelSizePtr = attrs->GetAttrPointer<int64_t>(0);
    if (poolKernelSizePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputDepth = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(4));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t weightInChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelDepth = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(4));
    const uint32_t poolKernelSize = static_cast<uint32_t>(*poolKernelSizePtr);

    if (poolKernelSize == 0 || inChannels != weightInChannels || bShape.GetDim(0) != wShape.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t convDepth = ComputeConvOutputDim(inputDepth, kernelDepth);
    const uint32_t convHeight = ComputeConvOutputDim(inputHeight, kernelHeight);
    const uint32_t convWidth = ComputeConvOutputDim(inputWidth, kernelWidth);
    if (convDepth == 0 || convHeight == 0 || convWidth == 0 || outChannels == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t fusedPoolKernelSize = poolKernelSize * poolKernelSize;
    const uint32_t fusedPoolStride = fusedPoolKernelSize;
    const uint32_t outputDepth = ComputePoolOutputDim(convDepth, fusedPoolKernelSize, fusedPoolStride);
    const uint32_t outputHeight = ComputePoolOutputDim(convHeight, fusedPoolKernelSize, fusedPoolStride);
    const uint32_t outputWidth = ComputePoolOutputDim(convWidth, fusedPoolKernelSize, fusedPoolStride);
    if (outputDepth == 0 || outputHeight == 0 || outputWidth == 0) {
        return ge::GRAPH_FAILED;
    }

    Conv3dSoftmaxMaxPoolMaxPoolCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelDepth(kernelDepth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_convDepth(convDepth);
    tiling.set_convHeight(convHeight);
    tiling.set_convWidth(convWidth);
    tiling.set_poolKernelSize(poolKernelSize);
    tiling.set_fusedPoolKernelSize(fusedPoolKernelSize);
    tiling.set_fusedPoolStride(fusedPoolStride);
    tiling.set_outputDepth(outputDepth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_alignedChannels(AlignUp8(outChannels));

    ge::Shape srcShape({1, static_cast<int64_t>(outChannels)});
    const uint32_t localWorkSpaceSize = AscendC::GetSoftMaxMinTmpSize(srcShape, sizeof(float), false);
    tiling.set_tmpSize(localWorkSpaceSize);
    AscendC::SoftMaxTilingFunc(srcShape, sizeof(float), localWorkSpaceSize, tiling.softmaxTilingData);

    context->SetBlockDim(batchSize == 0 ? 1U : batchSize);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *workspaceSize = context->GetWorkspaceSizes(1);
    workspaceSize[0] = localWorkSpaceSize + sysWorkspaceSize;
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
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDim(1) != weightShape->GetDim(1) || weightShape->GetDim(0) != biasShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *poolKernelSizePtr = attrs->GetAttrPointer<int64_t>(0);
    if (poolKernelSizePtr == nullptr || *poolKernelSizePtr <= 0) {
        return GRAPH_FAILED;
    }

    const int64_t convDepth = inputShape->GetDim(2) - weightShape->GetDim(2) + 1;
    const int64_t convHeight = inputShape->GetDim(3) - weightShape->GetDim(3) + 1;
    const int64_t convWidth = inputShape->GetDim(4) - weightShape->GetDim(4) + 1;
    const int64_t fusedPoolKernelSize = (*poolKernelSizePtr) * (*poolKernelSizePtr);
    if (convDepth <= 0 || convHeight <= 0 || convWidth <= 0) {
        return GRAPH_FAILED;
    }

    const int64_t outputDepth = convDepth >= fusedPoolKernelSize ? ((convDepth - fusedPoolKernelSize) / fusedPoolKernelSize + 1) : 0;
    const int64_t outputHeight = convHeight >= fusedPoolKernelSize ? ((convHeight - fusedPoolKernelSize) / fusedPoolKernelSize + 1) : 0;
    const int64_t outputWidth = convWidth >= fusedPoolKernelSize ? ((convWidth - fusedPoolKernelSize) / fusedPoolKernelSize + 1) : 0;
    if (outputDepth <= 0 || outputHeight <= 0 || outputWidth <= 0) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
    outputShape->SetDim(2, outputDepth);
    outputShape->SetDim(3, outputHeight);
    outputShape->SetDim(4, outputWidth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv3dSoftmaxMaxPoolMaxPoolCustom : public OpDef {
public:
    explicit Conv3dSoftmaxMaxPoolMaxPoolCustom(const char *name) : OpDef(name)
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

OP_ADD(Conv3dSoftmaxMaxPoolMaxPoolCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"
#include <cfloat>

using namespace AscendC;

class KernelConv3dSoftmaxMaxPoolMaxPool {
public:
    __aicore__ inline KernelConv3dSoftmaxMaxPoolMaxPool() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputDepth,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelDepth,
        uint32_t kernelHeight,
        uint32_t kernelWidth,
        uint32_t convDepth,
        uint32_t convHeight,
        uint32_t convWidth,
        uint32_t fusedPoolKernelSize,
        uint32_t fusedPoolStride,
        uint32_t outputDepth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t alignedChannels,
        uint32_t tmpSize,
        const SoftMaxTiling &softmaxTiling)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputDepth = inputDepth;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelDepth = kernelDepth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->convDepth = convDepth;
        this->convHeight = convHeight;
        this->convWidth = convWidth;
        this->fusedPoolKernelSize = fusedPoolKernelSize;
        this->fusedPoolStride = fusedPoolStride;
        this->outputDepth = outputDepth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->alignedChannels = alignedChannels;
        this->softmaxTiling = softmaxTiling;
        this->blockIdx = GetBlockIdx();

        this->inputChannelStride = inputDepth * inputHeight * inputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->weightOutStride = inChannels * kernelDepth * kernelHeight * kernelWidth;
        this->weightInStride = kernelDepth * kernelHeight * kernelWidth;
        this->outputChannelStride = outputDepth * outputHeight * outputWidth;
        this->outputBatchStride = outChannels * this->outputChannelStride;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * this->weightOutStride);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);

        pipe.InitBuffer(logitsBuf, alignedChannels * sizeof(float));
        pipe.InitBuffer(probsBuf, alignedChannels * sizeof(float));
        pipe.InitBuffer(poolBuf, alignedChannels * sizeof(float));
        pipe.InitBuffer(sumBuf, 8 * sizeof(float));
        pipe.InitBuffer(maxBuf, 8 * sizeof(float));
        pipe.InitBuffer(tmpBuf, tmpSize);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        LocalTensor<float> logitsLocal = logitsBuf.Get<float>();
        LocalTensor<float> probsLocal = probsBuf.Get<float>();
        LocalTensor<float> poolLocal = poolBuf.Get<float>();
        LocalTensor<float> sumLocal = sumBuf.Get<float>();
        LocalTensor<float> maxLocal = maxBuf.Get<float>();
        LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();

        const uint32_t batchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t outBatchBase = this->blockIdx * this->outputBatchStride;

        for (uint32_t outD = 0; outD < this->outputDepth; ++outD) {
            const uint32_t convStartD = outD * this->fusedPoolStride;
            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                const uint32_t convStartH = outH * this->fusedPoolStride;
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    const uint32_t convStartW = outW * this->fusedPoolStride;

                    for (uint32_t channel = 0; channel < this->alignedChannels; ++channel) {
                        logitsLocal.SetValue(channel, 0.0f);
                        probsLocal.SetValue(channel, 0.0f);
                        poolLocal.SetValue(channel, -FLT_MAX);
                    }

                    for (uint32_t poolD = 0; poolD < this->fusedPoolKernelSize; ++poolD) {
                        const uint32_t convD = convStartD + poolD;
                        for (uint32_t poolH = 0; poolH < this->fusedPoolKernelSize; ++poolH) {
                            const uint32_t convH = convStartH + poolH;
                            for (uint32_t poolW = 0; poolW < this->fusedPoolKernelSize; ++poolW) {
                                const uint32_t convW = convStartW + poolW;

                                for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                                    const uint32_t weightBase = outChannel * this->weightOutStride;
                                    float value = biasGm.GetValue(outChannel);
                                    for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                                        const uint32_t xChannelBase = batchBase + inChannel * this->inputChannelStride;
                                        const uint32_t wChannelBase = weightBase + inChannel * this->weightInStride;
                                        for (uint32_t kernelDIdx = 0; kernelDIdx < this->kernelDepth; ++kernelDIdx) {
                                            const uint32_t inputD = convD + kernelDIdx;
                                            const uint32_t inputDepthBase =
                                                xChannelBase + inputD * this->inputHeight * this->inputWidth;
                                            const uint32_t weightDepthBase =
                                                wChannelBase + kernelDIdx * this->kernelHeight * this->kernelWidth;
                                            for (uint32_t kernelHIdx = 0; kernelHIdx < this->kernelHeight; ++kernelHIdx) {
                                                const uint32_t inputH = convH + kernelHIdx;
                                                const uint32_t inputRowBase =
                                                    inputDepthBase + inputH * this->inputWidth;
                                                const uint32_t weightRowBase =
                                                    weightDepthBase + kernelHIdx * this->kernelWidth;
                                                for (uint32_t kernelWIdx = 0; kernelWIdx < this->kernelWidth; ++kernelWIdx) {
                                                    const uint32_t inputW = convW + kernelWIdx;
                                                    value +=
                                                        xGm.GetValue(inputRowBase + inputW) *
                                                        weightGm.GetValue(weightRowBase + kernelWIdx);
                                                }
                                            }
                                        }
                                    }
                                    logitsLocal.SetValue(outChannel, value);
                                }

                                SetFlag<HardEvent::S_V>(EVENT_ID0);
                                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                                SoftMaxShapeInfo srcShape = {1, this->outChannels, 1, this->outChannels};
                                SoftMax<float>(
                                    probsLocal,
                                    sumLocal,
                                    maxLocal,
                                    logitsLocal,
                                    tmpLocal,
                                    softmaxTiling,
                                    srcShape);
                                PipeBarrier<PIPE_V>();
                                SetFlag<HardEvent::V_S>(EVENT_ID0);
                                WaitFlag<HardEvent::V_S>(EVENT_ID0);

                                for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                                    const float probValue = probsLocal.GetValue(outChannel);
                                    if (probValue > poolLocal.GetValue(outChannel)) {
                                        poolLocal.SetValue(outChannel, probValue);
                                    }
                                }
                            }
                        }
                    }

                    const uint32_t outputOffsetBase =
                        outBatchBase + outD * this->outputHeight * this->outputWidth + outH * this->outputWidth + outW;
                    for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                        const uint32_t outputOffset = outChannel * this->outputChannelStride + outputOffsetBase;
                        yGm.SetValue(outputOffset, poolLocal.GetValue(outChannel));
                    }
                }
            }
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> logitsBuf;
    TBuf<TPosition::VECCALC> probsBuf;
    TBuf<TPosition::VECCALC> poolBuf;
    TBuf<TPosition::VECCALC> sumBuf;
    TBuf<TPosition::VECCALC> maxBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    SoftMaxTiling softmaxTiling;
    uint32_t batchSize = 0;
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    uint32_t inputDepth = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelDepth = 0;
    uint32_t kernelHeight = 0;
    uint32_t kernelWidth = 0;
    uint32_t convDepth = 0;
    uint32_t convHeight = 0;
    uint32_t convWidth = 0;
    uint32_t fusedPoolKernelSize = 0;
    uint32_t fusedPoolStride = 0;
    uint32_t outputDepth = 0;
    uint32_t outputHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t alignedChannels = 0;
    uint32_t blockIdx = 0;
    uint32_t inputChannelStride = 0;
    uint32_t inputBatchStride = 0;
    uint32_t weightOutStride = 0;
    uint32_t weightInStride = 0;
    uint32_t outputChannelStride = 0;
    uint32_t outputBatchStride = 0;
};

extern "C" __global__ __aicore__ void conv3d_softmax_max_pool_max_pool_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dSoftmaxMaxPoolMaxPool op;
    op.Init(
        x,
        weight,
        bias,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputDepth,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.kernelDepth,
        tiling_data.kernelHeight,
        tiling_data.kernelWidth,
        tiling_data.convDepth,
        tiling_data.convHeight,
        tiling_data.convWidth,
        tiling_data.fusedPoolKernelSize,
        tiling_data.fusedPoolStride,
        tiling_data.outputDepth,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.alignedChannels,
        tiling_data.tmpSize,
        tiling_data.softmaxTilingData);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

namespace {
int64_t ComputeConvOutputDim(int64_t input, int64_t kernelSize)
{
    return input - kernelSize + 1;
}

int64_t ComputePoolOutputDim(int64_t input, int64_t kernelSize, int64_t stride)
{
    const int64_t numerator = input - kernelSize;
    if (numerator < 0) {
        return 0;
    }
    return numerator / stride + 1;
}
}

at::Tensor conv3d_softmax_max_pool_max_pool_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    int64_t pool_kernel_size)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D OIDHW tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(pool_kernel_size > 0, "pool_kernel_size must be positive");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == bias.size(0), "bias length must match output channels");

    const int64_t convDepth = ComputeConvOutputDim(x.size(2), weight.size(2));
    const int64_t convHeight = ComputeConvOutputDim(x.size(3), weight.size(3));
    const int64_t convWidth = ComputeConvOutputDim(x.size(4), weight.size(4));
    TORCH_CHECK(convDepth > 0 && convHeight > 0 && convWidth > 0, "invalid conv3d output shape");

    const int64_t fusedPoolKernelSize = pool_kernel_size * pool_kernel_size;
    const int64_t outputDepth = ComputePoolOutputDim(convDepth, fusedPoolKernelSize, fusedPoolKernelSize);
    const int64_t outputHeight = ComputePoolOutputDim(convHeight, fusedPoolKernelSize, fusedPoolKernelSize);
    const int64_t outputWidth = ComputePoolOutputDim(convWidth, fusedPoolKernelSize, fusedPoolKernelSize);
    TORCH_CHECK(outputDepth > 0 && outputHeight > 0 && outputWidth > 0, "invalid pooled output shape");

    at::Tensor result = at::empty(
        {x.size(0), weight.size(0), outputDepth, outputHeight, outputWidth},
        x.options());
    EXEC_NPU_CMD(
        aclnnConv3dSoftmaxMaxPoolMaxPoolCustom,
        x,
        weight,
        bias,
        pool_kernel_size,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_softmax_max_pool_max_pool_custom", &conv3d_softmax_max_pool_max_pool_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv3d_softmax_max_pool_max_pool_custom",
        &conv3d_softmax_max_pool_max_pool_impl_npu,
        "conv3d_softmax_max_pool_max_pool_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv3d_softmax_max_pool_max_pool_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.pool_kernel_size,
        )
'''
