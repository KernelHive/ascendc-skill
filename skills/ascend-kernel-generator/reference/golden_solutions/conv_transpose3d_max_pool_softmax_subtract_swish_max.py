project_json_src='''
[
    {
        "op": "ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "weight",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "conv_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "subtract_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ],
        "output_desc": [
            {
                "name": "y",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            }
        ],
        "attr": [
            {
                "name": "stride_d",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "stride_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "stride_w",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding_d",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "padding_w",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "output_padding_d",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "output_padding_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "output_padding_w",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "pool_kernel_size",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "pool_stride",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "pool_padding",
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
BEGIN_TILING_DATA_DEF(ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelDepth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, convOutDepth);
    TILING_DATA_FIELD_DEF(uint32_t, convOutHeight);
    TILING_DATA_FIELD_DEF(uint32_t, convOutWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, strideD);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, paddingD);
    TILING_DATA_FIELD_DEF(uint32_t, paddingH);
    TILING_DATA_FIELD_DEF(uint32_t, paddingW);
    TILING_DATA_FIELD_DEF(uint32_t, outputPaddingD);
    TILING_DATA_FIELD_DEF(uint32_t, outputPaddingH);
    TILING_DATA_FIELD_DEF(uint32_t, outputPaddingW);
    TILING_DATA_FIELD_DEF(uint32_t, poolKernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, poolStride);
    TILING_DATA_FIELD_DEF(uint32_t, poolPadding);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom,
    ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose3d_max_pool_softmax_subtract_swish_max_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeTransposedOutputDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t outputPadding)
{
    if (stride <= 0) {
        return 0;
    }
    const int64_t output = (input - 1) * stride - 2 * padding + kernel + outputPadding;
    return output > 0 ? static_cast<uint32_t>(output) : 0;
}

uint32_t ComputePoolOutputDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding)
{
    if (stride <= 0) {
        return 0;
    }
    const int64_t numerator = input + padding * 2 - kernel;
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
    const gert::StorageShape *convBiasShape = context->GetInputShape(2);
    const gert::StorageShape *subtractBiasShape = context->GetInputShape(3);
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr || subtractBiasShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    const auto cbShape = convBiasShape->GetStorageShape();
    const auto sbShape = subtractBiasShape->GetStorageShape();
    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5 || cbShape.GetDimNum() != 1 || sbShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *strideDPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *strideHPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *strideWPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *paddingDPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *outputPaddingDPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t *outputPaddingHPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t *outputPaddingWPtr = attrs->GetAttrPointer<int64_t>(8);
    const int64_t *poolKernelPtr = attrs->GetAttrPointer<int64_t>(9);
    const int64_t *poolStridePtr = attrs->GetAttrPointer<int64_t>(10);
    const int64_t *poolPaddingPtr = attrs->GetAttrPointer<int64_t>(11);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        outputPaddingDPtr == nullptr || outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr ||
        poolKernelPtr == nullptr || poolStridePtr == nullptr || poolPaddingPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputDepth = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(4));
    const uint32_t weightInputChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelDepth = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(4));

    if (inChannels != weightInputChannels) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(cbShape.GetDim(0)) != outChannels || static_cast<uint32_t>(sbShape.GetDim(0)) != outChannels) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t convOutDepth = ComputeTransposedOutputDim(inputDepth, kernelDepth, *strideDPtr, *paddingDPtr, *outputPaddingDPtr);
    const uint32_t convOutHeight = ComputeTransposedOutputDim(inputHeight, kernelHeight, *strideHPtr, *paddingHPtr, *outputPaddingHPtr);
    const uint32_t convOutWidth = ComputeTransposedOutputDim(inputWidth, kernelWidth, *strideWPtr, *paddingWPtr, *outputPaddingWPtr);
    const uint32_t outputDepth = ComputePoolOutputDim(convOutDepth, *poolKernelPtr, *poolStridePtr, *poolPaddingPtr);
    const uint32_t outputHeight = ComputePoolOutputDim(convOutHeight, *poolKernelPtr, *poolStridePtr, *poolPaddingPtr);
    const uint32_t outputWidth = ComputePoolOutputDim(convOutWidth, *poolKernelPtr, *poolStridePtr, *poolPaddingPtr);

    ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelDepth(kernelDepth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_convOutDepth(convOutDepth);
    tiling.set_convOutHeight(convOutHeight);
    tiling.set_convOutWidth(convOutWidth);
    tiling.set_outputDepth(outputDepth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_strideD(static_cast<uint32_t>(*strideDPtr));
    tiling.set_strideH(static_cast<uint32_t>(*strideHPtr));
    tiling.set_strideW(static_cast<uint32_t>(*strideWPtr));
    tiling.set_paddingD(static_cast<uint32_t>(*paddingDPtr));
    tiling.set_paddingH(static_cast<uint32_t>(*paddingHPtr));
    tiling.set_paddingW(static_cast<uint32_t>(*paddingWPtr));
    tiling.set_outputPaddingD(static_cast<uint32_t>(*outputPaddingDPtr));
    tiling.set_outputPaddingH(static_cast<uint32_t>(*outputPaddingHPtr));
    tiling.set_outputPaddingW(static_cast<uint32_t>(*outputPaddingWPtr));
    tiling.set_poolKernelSize(static_cast<uint32_t>(*poolKernelPtr));
    tiling.set_poolStride(static_cast<uint32_t>(*poolStridePtr));
    tiling.set_poolPadding(static_cast<uint32_t>(*poolPaddingPtr));

    context->SetBlockDim(batchSize > 0 ? batchSize : 1);
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
    const gert::Shape *convBiasShape = context->GetInputShape(2);
    const gert::Shape *subtractBiasShape = context->GetInputShape(3);
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr || subtractBiasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5 || convBiasShape->GetDimNum() != 1 || subtractBiasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }
    const int64_t *strideDPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *strideHPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *strideWPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *paddingDPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *outputPaddingDPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t *outputPaddingHPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t *outputPaddingWPtr = attrs->GetAttrPointer<int64_t>(8);
    const int64_t *poolKernelPtr = attrs->GetAttrPointer<int64_t>(9);
    const int64_t *poolStridePtr = attrs->GetAttrPointer<int64_t>(10);
    const int64_t *poolPaddingPtr = attrs->GetAttrPointer<int64_t>(11);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        outputPaddingDPtr == nullptr || outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr ||
        poolKernelPtr == nullptr || poolStridePtr == nullptr || poolPaddingPtr == nullptr) {
        return GRAPH_FAILED;
    }

    const uint32_t convOutDepth = ComputeTransposedOutputDim(inputShape->GetDim(2), weightShape->GetDim(2), *strideDPtr, *paddingDPtr, *outputPaddingDPtr);
    const uint32_t convOutHeight = ComputeTransposedOutputDim(inputShape->GetDim(3), weightShape->GetDim(3), *strideHPtr, *paddingHPtr, *outputPaddingHPtr);
    const uint32_t convOutWidth = ComputeTransposedOutputDim(inputShape->GetDim(4), weightShape->GetDim(4), *strideWPtr, *paddingWPtr, *outputPaddingWPtr);
    const uint32_t outputDepth = ComputePoolOutputDim(convOutDepth, *poolKernelPtr, *poolStridePtr, *poolPaddingPtr);
    const uint32_t outputHeight = ComputePoolOutputDim(convOutHeight, *poolKernelPtr, *poolStridePtr, *poolPaddingPtr);
    const uint32_t outputWidth = ComputePoolOutputDim(convOutWidth, *poolKernelPtr, *poolStridePtr, *poolPaddingPtr);

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, outputDepth);
    outputShape->SetDim(2, outputHeight);
    outputShape->SetDim(3, outputWidth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom : public OpDef {
public:
    explicit ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom(const char *name) : OpDef(name)
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
        this->Input("conv_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("subtract_bias")
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

OP_ADD(ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr float kNegInf = -3.40282347e+38f;
constexpr float kLn2 = 0.69314718056f;
constexpr uint32_t kMaxSupportedChannels = 256;
}

class KernelConvTranspose3dMaxPoolSoftmaxSubtractSwishMax {
public:
    __aicore__ inline KernelConvTranspose3dMaxPoolSoftmaxSubtractSwishMax() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR convBias,
        GM_ADDR subtractBias,
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
        uint32_t convOutDepth,
        uint32_t convOutHeight,
        uint32_t convOutWidth,
        uint32_t outputDepth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t strideD,
        uint32_t strideH,
        uint32_t strideW,
        uint32_t paddingD,
        uint32_t paddingH,
        uint32_t paddingW,
        uint32_t poolKernelSize,
        uint32_t poolStride,
        uint32_t poolPadding)
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
        this->convOutDepth = convOutDepth;
        this->convOutHeight = convOutHeight;
        this->convOutWidth = convOutWidth;
        this->outputDepth = outputDepth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->strideD = strideD;
        this->strideH = strideH;
        this->strideW = strideW;
        this->paddingD = paddingD;
        this->paddingH = paddingH;
        this->paddingW = paddingW;
        this->poolKernelSize = poolKernelSize;
        this->poolStride = poolStride;
        this->poolPadding = poolPadding;
        this->blockIdx = GetBlockIdx();

        this->inputPlaneStride = inputHeight * inputWidth;
        this->inputChannelStride = inputDepth * this->inputPlaneStride;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->weightOutputStride = kernelDepth * kernelHeight * kernelWidth;
        this->weightInputStride = outChannels * this->weightOutputStride;
        this->outputPlaneStride = outputHeight * outputWidth;
        this->outputBatchStride = outputDepth * this->outputPlaneStride;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inChannels * this->weightInputStride);
        convBiasGm.SetGlobalBuffer((__gm__ float *)convBias, outChannels);
        subtractBiasGm.SetGlobalBuffer((__gm__ float *)subtractBias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }
        if (this->outChannels == 0 || this->outChannels > kMaxSupportedChannels) {
            return;
        }

        float pooledValues[kMaxSupportedChannels];
        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->outputBatchStride;

        for (uint32_t outPd = 0; outPd < this->outputDepth; ++outPd) {
            for (uint32_t outPh = 0; outPh < this->outputHeight; ++outPh) {
                for (uint32_t outPw = 0; outPw < this->outputWidth; ++outPw) {
                    float maxValue = kNegInf;
                    for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                        const float pooledValue = ComputePooledValue(xBatchBase, outChannel, outPd, outPh, outPw);
                        pooledValues[outChannel] = pooledValue;
                        if (pooledValue > maxValue) {
                            maxValue = pooledValue;
                        }
                    }

                    float sumExp = 0.0f;
                    for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                        sumExp += FastExp(pooledValues[outChannel] - maxValue);
                    }

                    float reduced = kNegInf;
                    for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                        const float softmaxValue = FastExp(pooledValues[outChannel] - maxValue) / sumExp;
                        const float shifted = softmaxValue - subtractBiasGm.GetValue(outChannel);
                        const float swishValue = shifted * Sigmoid(shifted);
                        if (swishValue > reduced) {
                            reduced = swishValue;
                        }
                    }

                    const uint32_t outOffset =
                        yBatchBase +
                        outPd * this->outputPlaneStride +
                        outPh * this->outputWidth +
                        outPw;
                    yGm.SetValue(outOffset, reduced);
                }
            }
        }
    }

private:
    __aicore__ inline float ComputePooledValue(
        uint32_t xBatchBase,
        uint32_t outChannel,
        uint32_t outPd,
        uint32_t outPh,
        uint32_t outPw) const
    {
        const int32_t convBaseD = static_cast<int32_t>(outPd * this->poolStride) - static_cast<int32_t>(this->poolPadding);
        const int32_t convBaseH = static_cast<int32_t>(outPh * this->poolStride) - static_cast<int32_t>(this->poolPadding);
        const int32_t convBaseW = static_cast<int32_t>(outPw * this->poolStride) - static_cast<int32_t>(this->poolPadding);
        float pooledMax = kNegInf;

        for (uint32_t poolKd = 0; poolKd < this->poolKernelSize; ++poolKd) {
            const int32_t convD = convBaseD + static_cast<int32_t>(poolKd);
            if (convD < 0 || convD >= static_cast<int32_t>(this->convOutDepth)) {
                continue;
            }
            for (uint32_t poolKh = 0; poolKh < this->poolKernelSize; ++poolKh) {
                const int32_t convH = convBaseH + static_cast<int32_t>(poolKh);
                if (convH < 0 || convH >= static_cast<int32_t>(this->convOutHeight)) {
                    continue;
                }
                for (uint32_t poolKw = 0; poolKw < this->poolKernelSize; ++poolKw) {
                    const int32_t convW = convBaseW + static_cast<int32_t>(poolKw);
                    if (convW < 0 || convW >= static_cast<int32_t>(this->convOutWidth)) {
                        continue;
                    }
                    const float value = ComputeConvTransposeValue(
                        xBatchBase,
                        outChannel,
                        static_cast<uint32_t>(convD),
                        static_cast<uint32_t>(convH),
                        static_cast<uint32_t>(convW));
                    if (value > pooledMax) {
                        pooledMax = value;
                    }
                }
            }
        }

        return pooledMax;
    }

    __aicore__ inline float ComputeConvTransposeValue(
        uint32_t xBatchBase,
        uint32_t outChannel,
        uint32_t outD,
        uint32_t outH,
        uint32_t outW) const
    {
        float sum = convBiasGm.GetValue(outChannel);
        for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
            const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
            const uint32_t wChannelBase = inChannel * this->weightInputStride + outChannel * this->weightOutputStride;
            for (uint32_t kernelD = 0; kernelD < this->kernelDepth; ++kernelD) {
                const int32_t numerD = static_cast<int32_t>(outD) + static_cast<int32_t>(this->paddingD) - static_cast<int32_t>(kernelD);
                if (numerD < 0 || numerD % static_cast<int32_t>(this->strideD) != 0) {
                    continue;
                }
                const int32_t inD = numerD / static_cast<int32_t>(this->strideD);
                if (inD < 0 || inD >= static_cast<int32_t>(this->inputDepth)) {
                    continue;
                }
                for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                    const int32_t numerH = static_cast<int32_t>(outH) + static_cast<int32_t>(this->paddingH) - static_cast<int32_t>(kernelH);
                    if (numerH < 0 || numerH % static_cast<int32_t>(this->strideH) != 0) {
                        continue;
                    }
                    const int32_t inH = numerH / static_cast<int32_t>(this->strideH);
                    if (inH < 0 || inH >= static_cast<int32_t>(this->inputHeight)) {
                        continue;
                    }
                    for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                        const int32_t numerW = static_cast<int32_t>(outW) + static_cast<int32_t>(this->paddingW) - static_cast<int32_t>(kernelW);
                        if (numerW < 0 || numerW % static_cast<int32_t>(this->strideW) != 0) {
                            continue;
                        }
                        const int32_t inW = numerW / static_cast<int32_t>(this->strideW);
                        if (inW < 0 || inW >= static_cast<int32_t>(this->inputWidth)) {
                            continue;
                        }

                        const uint32_t xOffset =
                            xChannelBase +
                            static_cast<uint32_t>(inD) * this->inputPlaneStride +
                            static_cast<uint32_t>(inH) * this->inputWidth +
                            static_cast<uint32_t>(inW);
                        const uint32_t wOffset =
                            wChannelBase +
                            kernelD * this->kernelHeight * this->kernelWidth +
                            kernelH * this->kernelWidth +
                            kernelW;
                        sum += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                    }
                }
            }
        }
        return sum;
    }

    __aicore__ inline float Sigmoid(float x) const
    {
        if (x >= 0.0f) {
            const float expNeg = FastExp(-x);
            return 1.0f / (1.0f + expNeg);
        }
        const float expPos = FastExp(x);
        return expPos / (1.0f + expPos);
    }

    __aicore__ inline float FastExp(float x) const
    {
        if (x < -20.0f) {
            return 0.0f;
        }
        int32_t k = 0;
        while (x > 0.5f * kLn2) {
            x -= kLn2;
            ++k;
        }
        while (x < -0.5f * kLn2) {
            x += kLn2;
            --k;
        }

        const float x2 = x * x;
        const float x3 = x2 * x;
        const float x4 = x3 * x;
        const float x5 = x4 * x;
        float result = 1.0f + x + 0.5f * x2 + 0.16666667f * x3 + 0.04166667f * x4 + 0.0083333333f * x5;
        if (k > 0) {
            for (int32_t i = 0; i < k; ++i) {
                result *= 2.0f;
            }
        } else {
            for (int32_t i = 0; i < -k; ++i) {
                result *= 0.5f;
            }
        }
        return result;
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> convBiasGm;
    GlobalTensor<float> subtractBiasGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t inputDepth;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelDepth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t convOutDepth;
    uint32_t convOutHeight;
    uint32_t convOutWidth;
    uint32_t outputDepth;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t strideD;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingD;
    uint32_t paddingH;
    uint32_t paddingW;
    uint32_t poolKernelSize;
    uint32_t poolStride;
    uint32_t poolPadding;
    uint32_t blockIdx;
    uint32_t inputPlaneStride;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t weightInputStride;
    uint32_t weightOutputStride;
    uint32_t outputPlaneStride;
    uint32_t outputBatchStride;
};

extern "C" __global__ __aicore__ void conv_transpose3d_max_pool_softmax_subtract_swish_max_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR convBias,
    GM_ADDR subtractBias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dMaxPoolSoftmaxSubtractSwishMax op;
    op.Init(
        x,
        weight,
        convBias,
        subtractBias,
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
        tiling_data.convOutDepth,
        tiling_data.convOutHeight,
        tiling_data.convOutWidth,
        tiling_data.outputDepth,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.strideD,
        tiling_data.strideH,
        tiling_data.strideW,
        tiling_data.paddingD,
        tiling_data.paddingH,
        tiling_data.paddingW,
        tiling_data.poolKernelSize,
        tiling_data.poolStride,
        tiling_data.poolPadding);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ops/amax.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/max_pool3d.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/sub.h>
#include "pytorch_npu_helper.hpp"

#include <vector>

namespace {
int64_t ComputeTransposedOutputDim(int64_t input, int64_t kernel, int64_t stride, int64_t padding, int64_t outputPadding)
{
    return (input - 1) * stride - 2 * padding + kernel + outputPadding;
}

int64_t ComputePoolOutputDim(int64_t input, int64_t kernel, int64_t stride, int64_t padding)
{
    const int64_t numerator = input + padding * 2 - kernel;
    if (numerator < 0) {
        return 0;
    }
    return numerator / stride + 1;
}
}

at::Tensor conv_transpose3d_max_pool_softmax_subtract_swish_max_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &conv_bias,
    const at::Tensor &subtract_bias,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_d,
    int64_t padding_h,
    int64_t padding_w,
    int64_t output_padding_d,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t pool_kernel_size,
    int64_t pool_stride,
    int64_t pool_padding)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D tensor");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be 1D");
    TORCH_CHECK(subtract_bias.dim() == 1, "subtract_bias must be 1D");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");
    TORCH_CHECK(weight.size(1) == conv_bias.size(0), "conv_bias size mismatch");
    TORCH_CHECK(weight.size(1) == subtract_bias.size(0), "subtract_bias size mismatch");

    const std::vector<int64_t> stride = {stride_d, stride_h, stride_w};
    const std::vector<int64_t> padding = {padding_d, padding_h, padding_w};
    const std::vector<int64_t> dilation = {1, 1, 1};
    const std::vector<int64_t> outputPadding = {output_padding_d, output_padding_h, output_padding_w};
    const std::vector<int64_t> poolKernel = {pool_kernel_size, pool_kernel_size, pool_kernel_size};
    const std::vector<int64_t> poolStrideVec = {pool_stride, pool_stride, pool_stride};
    const std::vector<int64_t> poolPaddingVec = {pool_padding, pool_padding, pool_padding};
    const c10::optional<at::Tensor> convBiasOpt = conv_bias;

    at::Tensor conv = at::convolution(
        x,
        weight,
        convBiasOpt,
        stride,
        padding,
        dilation,
        true,
        outputPadding,
        1);
    at::Tensor pooled = at::max_pool3d(
        conv,
        poolKernel,
        poolStrideVec,
        poolPaddingVec,
        dilation,
        false);
    at::Tensor normalized = at::softmax(pooled, 1, c10::nullopt);
    at::Tensor shifted = at::sub(normalized, at::reshape(subtract_bias, {1, subtract_bias.numel(), 1, 1, 1}));
    at::Tensor swish = shifted * at::sigmoid(shifted);
    std::vector<int64_t> reduceDim = {1};
    return at::amax(swish, reduceDim, false);
}

/* EXEC_NPU_CMD(aclnnConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom, x, weight, conv_bias, subtract_bias, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, output_padding_d, output_padding_h, output_padding_w, pool_kernel_size, pool_stride, pool_padding, result); */

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transpose3d_max_pool_softmax_subtract_swish_max_custom",
        &conv_transpose3d_max_pool_softmax_subtract_swish_max_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose3d_max_pool_softmax_subtract_swish_max_custom",
        &conv_transpose3d_max_pool_softmax_subtract_swish_max_custom_impl_npu,
        "ConvTranspose3d + MaxPool3d + Softmax + Subtract + Swish + Max custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


def _normalize_3d(value):
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError("expected an int or length-3 tuple")
    return tuple(value)


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        pool_stride,
        pool_padding,
    ):
        super(ModelNew, self).__init__()
        kernel_size = _normalize_3d(kernel_size)
        stride = _normalize_3d(stride)
        padding = _normalize_3d(padding)
        output_padding = _normalize_3d(output_padding)

        self.conv_transpose = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,
        )
        self.subtract = torch.nn.Parameter(torch.randn(out_channels))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def forward(self, x):
        return custom_ops_lib.conv_transpose3d_max_pool_softmax_subtract_swish_max_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.subtract,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.output_padding[0],
            self.output_padding[1],
            self.output_padding[2],
            self.pool_kernel_size,
            self.pool_stride,
            self.pool_padding,
        )
'''
