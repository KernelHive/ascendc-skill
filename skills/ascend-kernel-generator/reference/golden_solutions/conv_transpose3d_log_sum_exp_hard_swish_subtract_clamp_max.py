project_json_src='''
[
    {
        "op": "ConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustom",
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
                "name": "conv_bias",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "subtract_bias",
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
BEGIN_TILING_DATA_DEF(ConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustomTilingData)
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
    ConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustom,
    ConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max_custom_tiling.h"
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
    const gert::StorageShape *convBiasShape = context->GetInputShape(2);
    const gert::StorageShape *subtractBiasShape = context->GetInputShape(3);
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr || subtractBiasShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    const auto cbShape = convBiasShape->GetStorageShape();
    const auto sbShape = subtractBiasShape->GetStorageShape();
    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
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
    const uint32_t outChannels = groupOutChannels * groups;

    if (groups == 0 || inChannels != weightInputChannels || inChannels % groups != 0) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint32_t>(cbShape.GetShapeSize()) != outChannels ||
        static_cast<uint32_t>(sbShape.GetShapeSize()) != outChannels) {
        return ge::GRAPH_FAILED;
    }

    ConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelDepth(kernelDepth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputDepth(
        ComputeTransposedOutputDim(inputDepth, kernelDepth, *strideDPtr, *paddingDPtr, *dilationDPtr, *outputPaddingDPtr));
    tiling.set_outputHeight(
        ComputeTransposedOutputDim(inputHeight, kernelHeight, *strideHPtr, *paddingHPtr, *dilationHPtr, *outputPaddingHPtr));
    tiling.set_outputWidth(
        ComputeTransposedOutputDim(inputWidth, kernelWidth, *strideWPtr, *paddingWPtr, *dilationWPtr, *outputPaddingWPtr));
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
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
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
    if (convBiasShape->GetShapeSize() != outChannels || subtractBiasShape->GetShapeSize() != outChannels) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, 1);
    outputShape->SetDim(
        2,
        ComputeTransposedOutputDim(
            inputShape->GetDim(2),
            weightShape->GetDim(2),
            *strideDPtr,
            *paddingDPtr,
            *dilationDPtr,
            *outputPaddingDPtr));
    outputShape->SetDim(
        3,
        ComputeTransposedOutputDim(
            inputShape->GetDim(3),
            weightShape->GetDim(3),
            *strideHPtr,
            *paddingHPtr,
            *dilationHPtr,
            *outputPaddingHPtr));
    outputShape->SetDim(
        4,
        ComputeTransposedOutputDim(
            inputShape->GetDim(4),
            weightShape->GetDim(4),
            *strideWPtr,
            *paddingWPtr,
            *dilationWPtr,
            *outputPaddingWPtr));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustom : public OpDef {
public:
    explicit ConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustom(const char *name)
        : OpDef(name)
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

OP_ADD(ConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTranspose3dLogSumExpHardSwishSubtractClampMax {
public:
    static constexpr float LN2 = 0.69314718056f;

    __aicore__ inline KernelConvTranspose3dLogSumExpHardSwishSubtractClampMax() {}

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
        this->outputBatchStride = this->outputChannelStride;
        this->weightOutputChannelsPerGroup = outChannels / groups;
        this->inputChannelsPerGroup = inChannels / groups;
        this->weightOutputStride = kernelDepth * kernelHeight * kernelWidth;
        this->weightInputStride = this->weightOutputChannelsPerGroup * this->weightOutputStride;

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

        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->outputBatchStride;

        for (uint32_t outD = 0; outD < this->outputDepth; ++outD) {
            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    float maxValue = -3.40282347e+38f;
                    for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                        const float convValue = ComputeConvTransposeValue(xBatchBase, outChannel, outD, outH, outW);
                        if (convValue > maxValue) {
                            maxValue = convValue;
                        }
                    }

                    float sumExp = 0.0f;
                    for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                        const float convValue = ComputeConvTransposeValue(xBatchBase, outChannel, outD, outH, outW);
                        sumExp += FastExp(convValue - maxValue);
                    }
                    float fusedValue = maxValue + FastLogPositive(sumExp);
                    fusedValue = fusedValue * Sigmoid(fusedValue + 3.0f) * (1.0f / 6.0f);

                    float reduced = -3.40282347e+38f;
                    for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                        float candidate = fusedValue - subtractBiasGm.GetValue(outChannel);
                        if (candidate < -1.0f) {
                            candidate = -1.0f;
                        } else if (candidate > 1.0f) {
                            candidate = 1.0f;
                        }
                        if (candidate > reduced) {
                            reduced = candidate;
                        }
                    }

                    yGm.SetValue(
                        yBatchBase + outD * this->outputPlaneStride + outH * this->outputWidth + outW,
                        reduced);
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
        float sum = convBiasGm.GetValue(outChannel);
        const uint32_t groupIdx = outChannel / this->weightOutputChannelsPerGroup;
        const uint32_t groupOutChannel = outChannel % this->weightOutputChannelsPerGroup;
        const uint32_t inChannelStart = groupIdx * this->inputChannelsPerGroup;
        const uint32_t inChannelEnd = inChannelStart + this->inputChannelsPerGroup;

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
        if (x < -20.0f) {
            return 0.0f;
        }
        int32_t k = 0;
        while (x > 0.5f * LN2) {
            x -= LN2;
            ++k;
        }
        while (x < -0.5f * LN2) {
            x += LN2;
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

    __aicore__ inline float FastLogPositive(float x) const
    {
        if (x <= 0.0f) {
            return -3.40282347e+38f;
        }
        int32_t k = 0;
        while (x > 2.0f) {
            x *= 0.5f;
            ++k;
        }
        while (x < 1.0f) {
            x *= 2.0f;
            --k;
        }
        const float y = (x - 1.0f) / (x + 1.0f);
        const float y2 = y * y;
        const float y3 = y2 * y;
        const float y5 = y3 * y2;
        const float y7 = y5 * y2;
        const float series = 2.0f * (y + y3 / 3.0f + y5 / 5.0f + y7 / 7.0f);
        return static_cast<float>(k) * LN2 + series;
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

extern "C" __global__ __aicore__ void conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max_custom(
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
    KernelConvTranspose3dLogSumExpHardSwishSubtractClampMax op;
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
#include <ATen/Functions.h>
#include <ATen/ops/amax.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/logsumexp.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/sub.h>
#include "pytorch_npu_helper.hpp"
#include <vector>

/* EXEC_NPU_CMD(aclnnConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustom, x, weight, conv_bias, subtract_bias, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, dilation_d, dilation_h, dilation_w, output_padding_d, output_padding_h, output_padding_w, groups, result); */

at::Tensor conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max_custom_impl_npu(
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
    TORCH_CHECK(conv_bias.numel() == weight.size(1) * groups, "conv_bias size mismatch");
    TORCH_CHECK(subtract_bias.numel() == weight.size(1) * groups, "subtract_bias size mismatch");
    TORCH_CHECK(groups > 0, "groups must be positive");

    const std::vector<int64_t> stride = {stride_d, stride_h, stride_w};
    const std::vector<int64_t> padding = {padding_d, padding_h, padding_w};
    const std::vector<int64_t> dilation = {dilation_d, dilation_h, dilation_w};
    const std::vector<int64_t> output_padding = {output_padding_d, output_padding_h, output_padding_w};
    const c10::optional<at::Tensor> convBiasOpt = conv_bias.reshape({conv_bias.numel()});

    at::Tensor conv = at::convolution(
        x,
        weight,
        convBiasOpt,
        stride,
        padding,
        dilation,
        true,
        output_padding,
        groups);

    std::vector<int64_t> reduceChannel = {1};
    at::Tensor lse = at::logsumexp(conv, reduceChannel, true);
    at::Tensor hardSwishLike = at::mul(lse, at::sigmoid(lse + 3.0)) / 6.0;
    at::Tensor subtractBiasView = at::reshape(subtract_bias, {1, subtract_bias.numel(), 1, 1, 1});
    at::Tensor shifted = at::sub(hardSwishLike, subtractBiasView);
    at::Tensor clamped = at::clamp(shifted, -1.0, 1.0);
    return at::amax(clamped, reduceChannel, true);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max_custom",
        &conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max_custom",
        &conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max_custom_impl_npu,
        "conv_transpose3d + logsumexp + sigmoid-shifted hard-swish-like + subtract + clamp + max");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))
        self.stride = (stride, stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (1, 1, 1)
        self.output_padding = (0, 0, 0)
        self.groups = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bias,
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
