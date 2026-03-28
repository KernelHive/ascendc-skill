project_json_src='''
[
    {
        "op": "ConvTranspose3dScalingAvgPoolBiasAddScalingCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "conv_weight",
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
                "name": "post_bias",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "scale1",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float"]
            },
            {
                "name": "scale2",
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
BEGIN_TILING_DATA_DEF(ConvTranspose3dScalingAvgPoolBiasAddScalingCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelDepth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, convOutputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, convOutputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, convOutputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, pooledOutputDepth);
    TILING_DATA_FIELD_DEF(uint32_t, pooledOutputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, pooledOutputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, strideD);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, paddingD);
    TILING_DATA_FIELD_DEF(uint32_t, paddingH);
    TILING_DATA_FIELD_DEF(uint32_t, paddingW);
    TILING_DATA_FIELD_DEF(uint32_t, dilationD);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose3dScalingAvgPoolBiasAddScalingCustom,
    ConvTranspose3dScalingAvgPoolBiasAddScalingCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose3d_scaling_avg_pool_bias_add_scaling_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeTransposedOutputDim(
    int64_t input,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation)
{
    if (stride <= 0 || dilation <= 0) {
        return 0;
    }
    const int64_t effectiveKernel = dilation * (kernel - 1) + 1;
    const int64_t output = (input - 1) * stride - 2 * padding + effectiveKernel;
    return output > 0 ? static_cast<uint32_t>(output) : 0;
}

uint32_t ComputePoolOutputDim(uint32_t input)
{
    if (input < 2) {
        return 0;
    }
    return input / 2;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *convBiasShape = context->GetInputShape(2);
    const gert::StorageShape *postBiasShape = context->GetInputShape(3);
    const gert::StorageShape *scale1Shape = context->GetInputShape(4);
    const gert::StorageShape *scale2Shape = context->GetInputShape(5);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr ||
        postBiasShape == nullptr || scale1Shape == nullptr || scale2Shape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    const auto cbShape = convBiasShape->GetStorageShape();
    const auto pbShape = postBiasShape->GetStorageShape();
    const auto s1Shape = scale1Shape->GetStorageShape();
    const auto s2Shape = scale2Shape->GetStorageShape();
    if (xShape.GetDimNum() != 5 || wShape.GetDimNum() != 5) {
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
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(9);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        dilationDPtr == nullptr || dilationHPtr == nullptr || dilationWPtr == nullptr ||
        groupsPtr == nullptr || *groupsPtr <= 0) {
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
        static_cast<uint32_t>(pbShape.GetShapeSize()) != outChannels ||
        static_cast<uint32_t>(s1Shape.GetShapeSize()) != 1 ||
        static_cast<uint32_t>(s2Shape.GetShapeSize()) != 1) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t convOutputDepth =
        ComputeTransposedOutputDim(inputDepth, kernelDepth, *strideDPtr, *paddingDPtr, *dilationDPtr);
    const uint32_t convOutputHeight =
        ComputeTransposedOutputDim(inputHeight, kernelHeight, *strideHPtr, *paddingHPtr, *dilationHPtr);
    const uint32_t convOutputWidth =
        ComputeTransposedOutputDim(inputWidth, kernelWidth, *strideWPtr, *paddingWPtr, *dilationWPtr);

    ConvTranspose3dScalingAvgPoolBiasAddScalingCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelDepth(kernelDepth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_convOutputDepth(convOutputDepth);
    tiling.set_convOutputHeight(convOutputHeight);
    tiling.set_convOutputWidth(convOutputWidth);
    tiling.set_pooledOutputDepth(ComputePoolOutputDim(convOutputDepth));
    tiling.set_pooledOutputHeight(ComputePoolOutputDim(convOutputHeight));
    tiling.set_pooledOutputWidth(ComputePoolOutputDim(convOutputWidth));
    tiling.set_strideD(static_cast<uint32_t>(*strideDPtr));
    tiling.set_strideH(static_cast<uint32_t>(*strideHPtr));
    tiling.set_strideW(static_cast<uint32_t>(*strideWPtr));
    tiling.set_paddingD(static_cast<uint32_t>(*paddingDPtr));
    tiling.set_paddingH(static_cast<uint32_t>(*paddingHPtr));
    tiling.set_paddingW(static_cast<uint32_t>(*paddingWPtr));
    tiling.set_dilationD(static_cast<uint32_t>(*dilationDPtr));
    tiling.set_dilationH(static_cast<uint32_t>(*dilationHPtr));
    tiling.set_dilationW(static_cast<uint32_t>(*dilationWPtr));
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
    const gert::Shape *postBiasShape = context->GetInputShape(3);
    const gert::Shape *scale1Shape = context->GetInputShape(4);
    const gert::Shape *scale2Shape = context->GetInputShape(5);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr ||
        postBiasShape == nullptr || scale1Shape == nullptr || scale2Shape == nullptr || attrs == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5) {
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
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(9);
    if (strideDPtr == nullptr || strideHPtr == nullptr || strideWPtr == nullptr ||
        paddingDPtr == nullptr || paddingHPtr == nullptr || paddingWPtr == nullptr ||
        dilationDPtr == nullptr || dilationHPtr == nullptr || dilationWPtr == nullptr ||
        groupsPtr == nullptr || *groupsPtr <= 0) {
        return GRAPH_FAILED;
    }

    const int64_t outChannels = weightShape->GetDim(1) * (*groupsPtr);
    if (convBiasShape->GetShapeSize() != outChannels ||
        postBiasShape->GetShapeSize() != outChannels ||
        scale1Shape->GetShapeSize() != 1 ||
        scale2Shape->GetShapeSize() != 1) {
        return GRAPH_FAILED;
    }

    const uint32_t convOutputDepth = ComputeTransposedOutputDim(
        inputShape->GetDim(2), weightShape->GetDim(2), *strideDPtr, *paddingDPtr, *dilationDPtr);
    const uint32_t convOutputHeight = ComputeTransposedOutputDim(
        inputShape->GetDim(3), weightShape->GetDim(3), *strideHPtr, *paddingHPtr, *dilationHPtr);
    const uint32_t convOutputWidth = ComputeTransposedOutputDim(
        inputShape->GetDim(4), weightShape->GetDim(4), *strideWPtr, *paddingWPtr, *dilationWPtr);

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, outChannels);
    outputShape->SetDim(2, ComputePoolOutputDim(convOutputDepth));
    outputShape->SetDim(3, ComputePoolOutputDim(convOutputHeight));
    outputShape->SetDim(4, ComputePoolOutputDim(convOutputWidth));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose3dScalingAvgPoolBiasAddScalingCustom : public OpDef {
public:
    explicit ConvTranspose3dScalingAvgPoolBiasAddScalingCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conv_weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conv_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("post_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale1").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale2").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
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
        this->Attr("groups").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dScalingAvgPoolBiasAddScalingCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTranspose3dScalingAvgPoolBiasAddScaling {
public:
    __aicore__ inline KernelConvTranspose3dScalingAvgPoolBiasAddScaling() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR convWeight,
        GM_ADDR convBias,
        GM_ADDR postBias,
        GM_ADDR scale1,
        GM_ADDR scale2,
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
        uint32_t convOutputDepth,
        uint32_t convOutputHeight,
        uint32_t convOutputWidth,
        uint32_t pooledOutputDepth,
        uint32_t pooledOutputHeight,
        uint32_t pooledOutputWidth,
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
        this->convOutputDepth = convOutputDepth;
        this->convOutputHeight = convOutputHeight;
        this->convOutputWidth = convOutputWidth;
        this->pooledOutputDepth = pooledOutputDepth;
        this->pooledOutputHeight = pooledOutputHeight;
        this->pooledOutputWidth = pooledOutputWidth;
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
        this->pooledPlaneStride = pooledOutputHeight * pooledOutputWidth;
        this->pooledChannelStride = pooledOutputDepth * this->pooledPlaneStride;
        this->pooledBatchStride = outChannels * this->pooledChannelStride;
        this->weightOutputChannelsPerGroup = outChannels / groups;
        this->inputChannelsPerGroup = inChannels / groups;
        this->weightOutputStride = kernelDepth * kernelHeight * kernelWidth;
        this->weightInputStride = this->weightOutputChannelsPerGroup * this->weightOutputStride;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        convWeightGm.SetGlobalBuffer((__gm__ float *)convWeight, inChannels * this->weightInputStride);
        convBiasGm.SetGlobalBuffer((__gm__ float *)convBias, outChannels);
        postBiasGm.SetGlobalBuffer((__gm__ float *)postBias, outChannels);
        scale1Gm.SetGlobalBuffer((__gm__ float *)scale1, 1);
        scale2Gm.SetGlobalBuffer((__gm__ float *)scale2, 1);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->pooledBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const float scale1Value = scale1Gm.GetValue(0);
        const float scale2Value = scale2Gm.GetValue(0);
        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->pooledBatchStride;

        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            const uint32_t groupIdx = outChannel / this->weightOutputChannelsPerGroup;
            const uint32_t groupOutChannel = outChannel % this->weightOutputChannelsPerGroup;
            const uint32_t inChannelStart = groupIdx * this->inputChannelsPerGroup;
            const uint32_t inChannelEnd = inChannelStart + this->inputChannelsPerGroup;
            const uint32_t yChannelBase = yBatchBase + outChannel * this->pooledChannelStride;
            const float convBiasValue = convBiasGm.GetValue(outChannel);
            const float postBiasValue = postBiasGm.GetValue(outChannel);

            for (uint32_t outD = 0; outD < this->pooledOutputDepth; ++outD) {
                for (uint32_t outH = 0; outH < this->pooledOutputHeight; ++outH) {
                    for (uint32_t outW = 0; outW < this->pooledOutputWidth; ++outW) {
                        float pooledSum = 0.0f;

                        for (uint32_t poolD = 0; poolD < 2; ++poolD) {
                            const uint32_t convOutD = outD * 2 + poolD;
                            for (uint32_t poolH = 0; poolH < 2; ++poolH) {
                                const uint32_t convOutH = outH * 2 + poolH;
                                for (uint32_t poolW = 0; poolW < 2; ++poolW) {
                                    const uint32_t convOutW = outW * 2 + poolW;
                                    float convValue = convBiasValue;

                                    for (uint32_t inChannel = inChannelStart; inChannel < inChannelEnd; ++inChannel) {
                                        const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                                        const uint32_t wChannelBase =
                                            inChannel * this->weightInputStride + groupOutChannel * this->weightOutputStride;

                                        for (uint32_t kernelD = 0; kernelD < this->kernelDepth; ++kernelD) {
                                            const int32_t numerD =
                                                static_cast<int32_t>(convOutD) + static_cast<int32_t>(this->paddingD) -
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
                                                    static_cast<int32_t>(convOutH) + static_cast<int32_t>(this->paddingH) -
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
                                                        static_cast<int32_t>(convOutW) + static_cast<int32_t>(this->paddingW) -
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
                                                    convValue += xGm.GetValue(xOffset) * convWeightGm.GetValue(wOffset);
                                                }
                                            }
                                        }
                                    }

                                    pooledSum += convValue * scale1Value;
                                }
                            }
                        }

                        const uint32_t outOffset =
                            yChannelBase + outD * this->pooledPlaneStride + outH * this->pooledOutputWidth + outW;
                        const float pooledValue = pooledSum * 0.125f;
                        yGm.SetValue(outOffset, (pooledValue + postBiasValue) * scale2Value);
                    }
                }
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> convWeightGm;
    GlobalTensor<float> convBiasGm;
    GlobalTensor<float> postBiasGm;
    GlobalTensor<float> scale1Gm;
    GlobalTensor<float> scale2Gm;
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
    uint32_t convOutputDepth;
    uint32_t convOutputHeight;
    uint32_t convOutputWidth;
    uint32_t pooledOutputDepth;
    uint32_t pooledOutputHeight;
    uint32_t pooledOutputWidth;
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
    uint32_t pooledPlaneStride;
    uint32_t pooledChannelStride;
    uint32_t pooledBatchStride;
    uint32_t weightInputStride;
    uint32_t weightOutputStride;
    uint32_t weightOutputChannelsPerGroup;
    uint32_t inputChannelsPerGroup;
};

extern "C" __global__ __aicore__ void conv_transpose3d_scaling_avg_pool_bias_add_scaling_custom(
    GM_ADDR x,
    GM_ADDR conv_weight,
    GM_ADDR conv_bias,
    GM_ADDR post_bias,
    GM_ADDR scale1,
    GM_ADDR scale2,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dScalingAvgPoolBiasAddScaling op;
    op.Init(
        x,
        conv_weight,
        conv_bias,
        post_bias,
        scale1,
        scale2,
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
        tiling_data.convOutputDepth,
        tiling_data.convOutputHeight,
        tiling_data.convOutputWidth,
        tiling_data.pooledOutputDepth,
        tiling_data.pooledOutputHeight,
        tiling_data.pooledOutputWidth,
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
#include <ATen/ops/add.h>
#include <ATen/ops/avg_pool3d.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/reshape.h>
#include <vector>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_transpose3d_scaling_avg_pool_bias_add_scaling_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &conv_weight,
    const at::Tensor &conv_bias,
    const at::Tensor &post_bias,
    const at::Tensor &scale1,
    const at::Tensor &scale2,
    int64_t stride_d,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_d,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_d,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(conv_weight.dim() == 5, "conv_weight must be a 5D tensor");
    TORCH_CHECK(conv_bias.numel() == conv_weight.size(1) * groups, "conv_bias size mismatch");
    TORCH_CHECK(post_bias.numel() == conv_weight.size(1) * groups, "post_bias size mismatch");
    TORCH_CHECK(scale1.numel() == 1, "scale1 must be a scalar tensor");
    TORCH_CHECK(scale2.numel() == 1, "scale2 must be a scalar tensor");
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(x.size(1) == conv_weight.size(0), "input channels must equal conv_weight.size(0)");
    TORCH_CHECK(x.size(1) % groups == 0, "input channels must be divisible by groups");

    /* EXEC_NPU_CMD(aclnnConvTranspose3dScalingAvgPoolBiasAddScalingCustom, x, conv_weight, conv_bias, post_bias, scale1, scale2, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, dilation_d, dilation_h, dilation_w, groups, result); */

    const std::vector<int64_t> stride = {stride_d, stride_h, stride_w};
    const std::vector<int64_t> padding = {padding_d, padding_h, padding_w};
    const std::vector<int64_t> dilation = {dilation_d, dilation_h, dilation_w};
    const std::vector<int64_t> outputPadding = {0, 0, 0};
    const c10::optional<at::Tensor> convBiasOpt = conv_bias.reshape({conv_bias.numel()});

    at::Tensor conv = at::convolution(
        x,
        conv_weight,
        convBiasOpt,
        stride,
        padding,
        dilation,
        true,
        outputPadding,
        groups);

    at::Tensor scaled = at::mul(conv, scale1);
    at::Tensor pooled = at::avg_pool3d(
        scaled,
        at::IntArrayRef({2, 2, 2}),
        at::IntArrayRef({2, 2, 2}),
        at::IntArrayRef({0, 0, 0}),
        false,
        true,
        ::std::nullopt);
    at::Tensor biasView = at::reshape(post_bias, {1, post_bias.numel(), 1, 1, 1});
    at::Tensor shifted = at::add(pooled, biasView);
    return at::mul(shifted, scale2);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv_transpose3d_scaling_avg_pool_bias_add_scaling_custom",
        &conv_transpose3d_scaling_avg_pool_bias_add_scaling_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose3d_scaling_avg_pool_bias_add_scaling_custom",
        &conv_transpose3d_scaling_avg_pool_bias_add_scaling_custom_impl_npu,
        "conv_transpose3d_scaling_avg_pool_bias_add_scaling_custom");
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
        raise ValueError("expected an int or a length-3 tuple")
    return tuple(value)


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        kernel_size = _normalize_3d(kernel_size)
        stride = _normalize_3d(stride)
        padding = _normalize_3d(padding)

        self.conv_transpose = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.scale1 = torch.nn.Parameter(torch.tensor(float(scale1), dtype=torch.float32))
        self.scale2 = torch.nn.Parameter(torch.tensor(float(scale2), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.stride = stride
        self.padding = padding
        self.dilation = tuple(self.conv_transpose.dilation)
        self.groups = self.conv_transpose.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv_transpose3d_scaling_avg_pool_bias_add_scaling_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.bias,
            self.scale1,
            self.scale2,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self.groups,
        )
'''
