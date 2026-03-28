project_json_src='''
[
    {
        "op": "ConvTranspose3dSoftmaxSigmoidCustom",
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
                "name": "bias",
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
                "name": "dilation_d",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "dilation_h",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "dilation_w",
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
                "name": "groups",
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
BEGIN_TILING_DATA_DEF(ConvTranspose3dSoftmaxSigmoidCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelDepth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, strideD);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, paddingD);
    TILING_DATA_FIELD_DEF(uint32_t, paddingH);
    TILING_DATA_FIELD_DEF(uint32_t, paddingW);
    TILING_DATA_FIELD_DEF(uint32_t, dilationD);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
    TILING_DATA_FIELD_DEF(uint32_t, outputPaddingD);
    TILING_DATA_FIELD_DEF(uint32_t, outputPaddingH);
    TILING_DATA_FIELD_DEF(uint32_t, outputPaddingW);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose3dSoftmaxSigmoidCustom,
    ConvTranspose3dSoftmaxSigmoidCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose3d_softmax_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeTransposedOutputDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t outputPadding)
{
    if (stride <= 0 || dilation <= 0) {
        return 0;
    }
    const int64_t effectiveKernel = dilation * (kernel - 1) + 1;
    const int64_t output = (input - 1) * stride - 2 * padding + effectiveKernel + outputPadding;
    return output > 0 ? static_cast<uint32_t>(output) : 0;
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
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *strideDPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *strideHPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *strideWPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *paddingDPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *dilationDPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t *dilationHPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t *dilationWPtr = attrs->GetAttrPointer<int64_t>(8);
    const int64_t *outputPaddingDPtr = attrs->GetAttrPointer<int64_t>(9);
    const int64_t *outputPaddingHPtr = attrs->GetAttrPointer<int64_t>(10);
    const int64_t *outputPaddingWPtr = attrs->GetAttrPointer<int64_t>(11);
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(12);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        dilationDPtr == nullptr || dilationHPtr == nullptr || dilationWPtr == nullptr ||
        outputPaddingDPtr == nullptr || outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr ||
        groupsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputDepth = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(4));
    const uint32_t weightInputChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t groupOutChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelDepth = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(4));
    const uint32_t groups = static_cast<uint32_t>(*groupsPtr);

    if (groups == 0 || inChannels != weightInputChannels || inChannels % groups != 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t outChannels = groupOutChannels * groups;
    const uint32_t outputDepth = ComputeTransposedOutputDim(
        xShape.GetDim(2), wShape.GetDim(2), *strideDPtr, *paddingDPtr, *dilationDPtr, *outputPaddingDPtr);
    const uint32_t outputHeight = ComputeTransposedOutputDim(
        xShape.GetDim(3), wShape.GetDim(3), *strideHPtr, *paddingHPtr, *dilationHPtr, *outputPaddingHPtr);
    const uint32_t outputWidth = ComputeTransposedOutputDim(
        xShape.GetDim(4), wShape.GetDim(4), *strideWPtr, *paddingWPtr, *dilationWPtr, *outputPaddingWPtr);

    if (bShape.GetDim(0) != static_cast<int64_t>(outChannels) ||
        outputDepth == 0 || outputHeight == 0 || outputWidth == 0) {
        return ge::GRAPH_FAILED;
    }

    ConvTranspose3dSoftmaxSigmoidCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelDepth(kernelDepth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputDepth(outputDepth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_strideD(static_cast<uint32_t>(*strideDPtr));
    tiling.set_strideH(static_cast<uint32_t>(*strideHPtr));
    tiling.set_strideW(static_cast<uint32_t>(*strideWPtr));
    tiling.set_paddingD(static_cast<uint32_t>(*paddingDPtr));
    tiling.set_paddingH(static_cast<uint32_t>(*paddingHPtr));
    tiling.set_paddingW(static_cast<uint32_t>(*paddingWPtr));
    tiling.set_dilationD(static_cast<uint32_t>(*dilationDPtr));
    tiling.set_dilationH(static_cast<uint32_t>(*dilationHPtr));
    tiling.set_dilationW(static_cast<uint32_t>(*dilationWPtr));
    tiling.set_outputPaddingD(static_cast<uint32_t>(*outputPaddingDPtr));
    tiling.set_outputPaddingH(static_cast<uint32_t>(*outputPaddingHPtr));
    tiling.set_outputPaddingW(static_cast<uint32_t>(*outputPaddingWPtr));
    tiling.set_groups(groups);

    context->SetBlockDim(batchSize == 0 ? 1 : batchSize);
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
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || weightShape == nullptr || biasShape == nullptr || attrs == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }

    const int64_t *strideDPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *strideHPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *strideWPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *paddingDPtr = attrs->GetAttrPointer<int64_t>(3);
    const int64_t *paddingHPtr = attrs->GetAttrPointer<int64_t>(4);
    const int64_t *paddingWPtr = attrs->GetAttrPointer<int64_t>(5);
    const int64_t *dilationDPtr = attrs->GetAttrPointer<int64_t>(6);
    const int64_t *dilationHPtr = attrs->GetAttrPointer<int64_t>(7);
    const int64_t *dilationWPtr = attrs->GetAttrPointer<int64_t>(8);
    const int64_t *outputPaddingDPtr = attrs->GetAttrPointer<int64_t>(9);
    const int64_t *outputPaddingHPtr = attrs->GetAttrPointer<int64_t>(10);
    const int64_t *outputPaddingWPtr = attrs->GetAttrPointer<int64_t>(11);
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(12);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        dilationDPtr == nullptr || dilationHPtr == nullptr || dilationWPtr == nullptr ||
        outputPaddingDPtr == nullptr || outputPaddingHPtr == nullptr || outputPaddingWPtr == nullptr ||
        groupsPtr == nullptr || *groupsPtr <= 0) {
        return GRAPH_FAILED;
    }

    const int64_t outChannels = weightShape->GetDim(1) * (*groupsPtr);
    const int64_t outDepth = ComputeTransposedOutputDim(
        inputShape->GetDim(2), weightShape->GetDim(2), *strideDPtr, *paddingDPtr, *dilationDPtr, *outputPaddingDPtr);
    const int64_t outHeight = ComputeTransposedOutputDim(
        inputShape->GetDim(3), weightShape->GetDim(3), *strideHPtr, *paddingHPtr, *dilationHPtr, *outputPaddingHPtr);
    const int64_t outWidth = ComputeTransposedOutputDim(
        inputShape->GetDim(4), weightShape->GetDim(4), *strideWPtr, *paddingWPtr, *dilationWPtr, *outputPaddingWPtr);
    if (biasShape->GetDim(0) != outChannels || outDepth <= 0 || outHeight <= 0 || outWidth <= 0) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, outChannels);
    outputShape->SetDim(2, outDepth);
    outputShape->SetDim(3, outHeight);
    outputShape->SetDim(4, outWidth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose3dSoftmaxSigmoidCustom : public OpDef {
public:
    explicit ConvTranspose3dSoftmaxSigmoidCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("stride_d").AttrType(REQUIRED).Int();
        this->Attr("stride_h").AttrType(REQUIRED).Int();
        this->Attr("stride_w").AttrType(REQUIRED).Int();
        this->Attr("padding_d").AttrType(REQUIRED).Int();
        this->Attr("padding_h").AttrType(REQUIRED).Int();
        this->Attr("padding_w").AttrType(REQUIRED).Int();
        this->Attr("dilation_d").AttrType(REQUIRED).Int();
        this->Attr("dilation_h").AttrType(REQUIRED).Int();
        this->Attr("dilation_w").AttrType(REQUIRED).Int();
        this->Attr("output_padding_d").AttrType(REQUIRED).Int();
        this->Attr("output_padding_h").AttrType(REQUIRED).Int();
        this->Attr("output_padding_w").AttrType(REQUIRED).Int();
        this->Attr("groups").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dSoftmaxSigmoidCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTranspose3dSoftmaxSigmoid {
public:
    __aicore__ inline KernelConvTranspose3dSoftmaxSigmoid() {}

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
        uint32_t outputDepth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t strideD,
        uint32_t strideH,
        uint32_t strideW,
        uint32_t paddingD,
        uint32_t paddingH,
        uint32_t paddingW,
        uint32_t dilationD,
        uint32_t dilationH,
        uint32_t dilationW,
        uint32_t groups)
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
        this->outputDepth = outputDepth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->strideD = strideD;
        this->strideH = strideH;
        this->strideW = strideW;
        this->paddingD = paddingD;
        this->paddingH = paddingH;
        this->paddingW = paddingW;
        this->dilationD = dilationD;
        this->dilationH = dilationH;
        this->dilationW = dilationW;
        this->groups = groups;
        this->blockIdx = GetBlockIdx();
        this->inputPlaneStride = inputHeight * inputWidth;
        this->inputChannelStride = inputDepth * this->inputPlaneStride;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->outputPlaneStride = outputHeight * outputWidth;
        this->outputChannelStride = outputDepth * this->outputPlaneStride;
        this->outputBatchStride = outChannels * this->outputChannelStride;
        this->weightOutputChannelsPerGroup = outChannels / groups;
        this->inputChannelsPerGroup = inChannels / groups;
        this->weightOutputStride = kernelDepth * kernelHeight * kernelWidth;
        this->weightInputStride = this->weightOutputChannelsPerGroup * this->weightOutputStride;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inChannels * this->weightInputStride);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->outputBatchStride;
        const float epsilon = 1.0e-12f;

        for (uint32_t outD = 0; outD < this->outputDepth; ++outD) {
            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    float channelMax = -3.4028235e38f;
                    for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                        const float value = ComputeConvTransposeValue(xBatchBase, outChannel, outD, outH, outW);
                        if (value > channelMax) {
                            channelMax = value;
                        }
                    }

                    float expSum = 0.0f;
                    for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                        expSum += FastExp(ComputeConvTransposeValue(xBatchBase, outChannel, outD, outH, outW) - channelMax);
                    }
                    expSum = expSum < epsilon ? epsilon : expSum;

                    for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                        const float value = ComputeConvTransposeValue(xBatchBase, outChannel, outD, outH, outW);
                        const float softmaxValue = FastExp(value - channelMax) / expSum;
                        const float sigmoidValue = Sigmoid(softmaxValue);
                        const uint32_t outOffset =
                            yBatchBase +
                            outChannel * this->outputChannelStride +
                            outD * this->outputPlaneStride +
                            outH * this->outputWidth +
                            outW;
                        yGm.SetValue(outOffset, sigmoidValue);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline float ComputeConvTransposeValue(
        uint32_t xBatchBase,
        uint32_t outChannel,
        uint32_t outD,
        uint32_t outH,
        uint32_t outW) const
    {
        const uint32_t groupIdx = outChannel / this->weightOutputChannelsPerGroup;
        const uint32_t groupOutChannel = outChannel % this->weightOutputChannelsPerGroup;
        const uint32_t inChannelStart = groupIdx * this->inputChannelsPerGroup;
        const uint32_t inChannelEnd = inChannelStart + this->inputChannelsPerGroup;
        float sum = biasGm.GetValue(outChannel);

        for (uint32_t inChannel = inChannelStart; inChannel < inChannelEnd; ++inChannel) {
            const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
            const uint32_t wChannelBase =
                inChannel * this->weightInputStride + groupOutChannel * this->weightOutputStride;
            for (uint32_t kernelD = 0; kernelD < this->kernelDepth; ++kernelD) {
                const int32_t numerD =
                    static_cast<int32_t>(outD) + static_cast<int32_t>(this->paddingD) -
                    static_cast<int32_t>(kernelD) * static_cast<int32_t>(this->dilationD);
                if (numerD < 0 || numerD % static_cast<int32_t>(this->strideD) != 0) {
                    continue;
                }
                const int32_t inD = numerD / static_cast<int32_t>(this->strideD);
                if (inD < 0 || inD >= static_cast<int32_t>(this->inputDepth)) {
                    continue;
                }
                for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                    const int32_t numerH =
                        static_cast<int32_t>(outH) + static_cast<int32_t>(this->paddingH) -
                        static_cast<int32_t>(kernelH) * static_cast<int32_t>(this->dilationH);
                    if (numerH < 0 || numerH % static_cast<int32_t>(this->strideH) != 0) {
                        continue;
                    }
                    const int32_t inH = numerH / static_cast<int32_t>(this->strideH);
                    if (inH < 0 || inH >= static_cast<int32_t>(this->inputHeight)) {
                        continue;
                    }
                    for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                        const int32_t numerW =
                            static_cast<int32_t>(outW) + static_cast<int32_t>(this->paddingW) -
                            static_cast<int32_t>(kernelW) * static_cast<int32_t>(this->dilationW);
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
        const float kLn2 = 0.69314718056f;
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
    GlobalTensor<float> biasGm;
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
    uint32_t outputDepth;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t strideD;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingD;
    uint32_t paddingH;
    uint32_t paddingW;
    uint32_t dilationD;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t groups;
    uint32_t blockIdx;
    uint32_t inputPlaneStride;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t outputPlaneStride;
    uint32_t outputChannelStride;
    uint32_t outputBatchStride;
    uint32_t weightInputStride;
    uint32_t weightOutputStride;
    uint32_t weightOutputChannelsPerGroup;
    uint32_t inputChannelsPerGroup;
};

extern "C" __global__ __aicore__ void conv_transpose3d_softmax_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dSoftmaxSigmoid op;
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
        tiling_data.outputDepth,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.strideD,
        tiling_data.strideH,
        tiling_data.strideW,
        tiling_data.paddingD,
        tiling_data.paddingH,
        tiling_data.paddingW,
        tiling_data.dilationD,
        tiling_data.dilationH,
        tiling_data.dilationW,
        tiling_data.groups);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_transpose3d_softmax_sigmoid_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_d,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_d,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t output_padding_d,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t groups)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");
    TORCH_CHECK(x.size(1) % groups == 0, "input channels must be divisible by groups");
    TORCH_CHECK(bias.size(0) == weight.size(1) * groups, "bias shape mismatch");

    const int64_t outChannels = weight.size(1) * groups;
    const int64_t outD =
        (x.size(2) - 1) * stride_d - 2 * padding_d + dilation_d * (weight.size(2) - 1) + output_padding_d + 1;
    const int64_t outH =
        (x.size(3) - 1) * stride_h - 2 * padding_h + dilation_h * (weight.size(3) - 1) + output_padding_h + 1;
    const int64_t outW =
        (x.size(4) - 1) * stride_w - 2 * padding_w + dilation_w * (weight.size(4) - 1) + output_padding_w + 1;
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "invalid transposed-convolution output shape");

    at::Tensor conv = at::empty({x.size(0), outChannels, outD, outH, outW}, x.options());
    auto convBias = c10::optional<at::Tensor>(bias);
    std::vector<int64_t> strideVec = {stride_d, stride_h, stride_w};
    std::vector<int64_t> paddingVec = {padding_d, padding_h, padding_w};
    std::vector<int64_t> dilationVec = {dilation_d, dilation_h, dilation_w};
    std::vector<int64_t> outputPaddingVec = {output_padding_d, output_padding_h, output_padding_w};
    at::IntArrayRef strideArray(strideVec);
    at::IntArrayRef paddingArray(paddingVec);
    at::IntArrayRef dilationArray(dilationVec);
    at::IntArrayRef outputPaddingArray(outputPaddingVec);
    const bool transposed = true;
    const int8_t cubeMathType = 0;

    EXEC_NPU_CMD(
        aclnnConvolution,
        x,
        weight,
        convBias,
        strideArray,
        paddingArray,
        dilationArray,
        transposed,
        outputPaddingArray,
        groups,
        conv,
        cubeMathType);

    at::Tensor probs = at::softmax(conv, 1, c10::nullopt);
    return at::sigmoid(probs);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transpose3d_softmax_sigmoid_custom", &conv_transpose3d_softmax_sigmoid_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose3d_softmax_sigmoid_custom",
        &conv_transpose3d_softmax_sigmoid_custom_impl_npu,
        "conv_transpose3d_softmax_sigmoid_custom");
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
    return tuple(int(v) for v in value)


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
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
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = tuple(self.conv_transpose.dilation)
        self.groups = self.conv_transpose.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_transpose3d_softmax_sigmoid_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self.output_padding[0],
            self.output_padding[1],
            self.output_padding[2],
            self.groups,
        )
'''
