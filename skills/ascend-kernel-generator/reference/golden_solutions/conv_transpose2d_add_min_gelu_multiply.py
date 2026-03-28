project_json_src='''
[
    {
        "op": "ConvTranspose2dAddMinGeluMultiplyCustom",
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
                "name": "add_value",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "multiply_value",
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
                "name": "output_padding",
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
BEGIN_TILING_DATA_DEF(ConvTranspose2dAddMinGeluMultiplyCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, stride);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(uint32_t, outputPadding);
    TILING_DATA_FIELD_DEF(uint32_t, groups);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    ConvTranspose2dAddMinGeluMultiplyCustom,
    ConvTranspose2dAddMinGeluMultiplyCustomTilingData)
}
"""

host_operator_src="""
#include "conv_transpose2d_add_min_gelu_multiply_custom_tiling.h"
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
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *convBiasShape = context->GetInputShape(2);
    const gert::StorageShape *addValueShape = context->GetInputShape(3);
    const gert::StorageShape *multiplyValueShape = context->GetInputShape(4);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr ||
        addValueShape == nullptr || multiplyValueShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    const auto cbShape = convBiasShape->GetStorageShape();
    const auto addShape = addValueShape->GetStorageShape();
    const auto mulShape = multiplyValueShape->GetStorageShape();
    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 4 || cbShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    if (addShape.GetShapeSize() != 1 || mulShape.GetShapeSize() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *outputPaddingPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(3);
    if (stridePtr == nullptr || paddingPtr == nullptr || outputPaddingPtr == nullptr || groupsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t inputHeight = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t inputWidth = static_cast<uint32_t>(xShape.GetDim(3));
    const uint32_t weightInputChannels = static_cast<uint32_t>(wShape.GetDim(0));
    const uint32_t groupOutChannels = static_cast<uint32_t>(wShape.GetDim(1));
    const uint32_t kernelHeight = static_cast<uint32_t>(wShape.GetDim(2));
    const uint32_t kernelWidth = static_cast<uint32_t>(wShape.GetDim(3));
    const uint32_t stride = static_cast<uint32_t>(*stridePtr);
    const uint32_t padding = static_cast<uint32_t>(*paddingPtr);
    const uint32_t outputPadding = static_cast<uint32_t>(*outputPaddingPtr);
    const uint32_t groups = static_cast<uint32_t>(*groupsPtr);

    if (groups == 0 || stride == 0 || outputPadding >= stride) {
        return ge::GRAPH_FAILED;
    }
    if (inChannels != weightInputChannels || inChannels % groups != 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t outChannels = groupOutChannels * groups;
    if (cbShape.GetDim(0) != static_cast<int64_t>(outChannels)) {
        return ge::GRAPH_FAILED;
    }

    ConvTranspose2dAddMinGeluMultiplyCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputHeight(
        ComputeTransposedOutputDim(inputHeight, kernelHeight, stride, padding, outputPadding));
    tiling.set_outputWidth(
        ComputeTransposedOutputDim(inputWidth, kernelWidth, stride, padding, outputPadding));
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_outputPadding(outputPadding);
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
    const gert::Shape *convBiasShape = context->GetInputShape(2);
    const gert::Shape *addValueShape = context->GetInputShape(3);
    const gert::Shape *multiplyValueShape = context->GetInputShape(4);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr ||
        addValueShape == nullptr || multiplyValueShape == nullptr || attrs == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4 || convBiasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (addValueShape->GetShapeSize() != 1 || multiplyValueShape->GetShapeSize() != 1) {
        return GRAPH_FAILED;
    }

    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *outputPaddingPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *groupsPtr = attrs->GetAttrPointer<int64_t>(3);
    if (stridePtr == nullptr || paddingPtr == nullptr || outputPaddingPtr == nullptr || groupsPtr == nullptr) {
        return GRAPH_FAILED;
    }
    if (*stridePtr <= 0 || *paddingPtr < 0 || *outputPaddingPtr < 0 || *groupsPtr <= 0) {
        return GRAPH_FAILED;
    }
    if (*outputPaddingPtr >= *stridePtr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDim(1) != weightShape->GetDim(0) || inputShape->GetDim(1) % *groupsPtr != 0) {
        return GRAPH_FAILED;
    }

    const int64_t outChannels = weightShape->GetDim(1) * (*groupsPtr);
    if (convBiasShape->GetDim(0) != outChannels) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, outChannels);
    outputShape->SetDim(
        2,
        ComputeTransposedOutputDim(
            inputShape->GetDim(2),
            weightShape->GetDim(2),
            *stridePtr,
            *paddingPtr,
            *outputPaddingPtr));
    outputShape->SetDim(
        3,
        ComputeTransposedOutputDim(
            inputShape->GetDim(3),
            weightShape->GetDim(3),
            *stridePtr,
            *paddingPtr,
            *outputPaddingPtr));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ConvTranspose2dAddMinGeluMultiplyCustom : public OpDef {
public:
    explicit ConvTranspose2dAddMinGeluMultiplyCustom(const char *name) : OpDef(name)
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
        this->Input("add_value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("multiply_value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->Attr("padding").AttrType(REQUIRED).Int();
        this->Attr("output_padding").AttrType(REQUIRED).Int();
        this->Attr("groups").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose2dAddMinGeluMultiplyCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConvTranspose2dAddMinGeluMultiply {
public:
    __aicore__ inline KernelConvTranspose2dAddMinGeluMultiply() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR convBias,
        GM_ADDR addValue,
        GM_ADDR multiplyValue,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t inputHeight,
        uint32_t inputWidth,
        uint32_t kernelHeight,
        uint32_t kernelWidth,
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t stride,
        uint32_t padding,
        uint32_t groups)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->stride = stride;
        this->padding = padding;
        this->groups = groups;
        this->blockIdx = GetBlockIdx();
        this->inputChannelStride = inputHeight * inputWidth;
        this->outputChannelStride = outputHeight * outputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->outputBatchStride = outChannels * this->outputChannelStride;
        this->weightOutputChannelsPerGroup = outChannels / groups;
        this->inputChannelsPerGroup = inChannels / groups;
        this->weightInputStride = this->weightOutputChannelsPerGroup * kernelHeight * kernelWidth;
        this->weightOutputStride = kernelHeight * kernelWidth;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inChannels * this->weightInputStride);
        convBiasGm.SetGlobalBuffer((__gm__ float *)convBias, outChannels);
        addValueGm.SetGlobalBuffer((__gm__ float *)addValue, 1);
        multiplyValueGm.SetGlobalBuffer((__gm__ float *)multiplyValue, 1);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline float Gelu(float value) const
    {
        if (value <= -4.0f) {
            return kGeluTable[0];
        }
        if (value >= 4.0f) {
            return value;
        }

        constexpr float kStart = -4.0f;
        constexpr float kInvStep = 3.2e+01f;
        const float scaled = (value - kStart) * kInvStep;
        int32_t idx = static_cast<int32_t>(scaled);
        if (idx < 0) {
            idx = 0;
        } else if (idx > 255) {
            idx = 255;
        }
        const float frac = scaled - static_cast<float>(idx);
        const float left = kGeluTable[idx];
        const float right = kGeluTable[idx + 1];
        return left + (right - left) * frac;
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const float addScalar = addValueGm.GetValue(0);
        const float multiplyScalar = multiplyValueGm.GetValue(0);
        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->outputBatchStride;

        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            const uint32_t groupIdx = outChannel / this->weightOutputChannelsPerGroup;
            const uint32_t groupOutChannel = outChannel % this->weightOutputChannelsPerGroup;
            const uint32_t inChannelStart = groupIdx * this->inputChannelsPerGroup;
            const uint32_t inChannelEnd = inChannelStart + this->inputChannelsPerGroup;
            const uint32_t yChannelBase = yBatchBase + outChannel * this->outputChannelStride;

            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    float sum = convBiasGm.GetValue(outChannel);
                    for (uint32_t inChannel = inChannelStart; inChannel < inChannelEnd; ++inChannel) {
                        const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                        const uint32_t wChannelBase =
                            inChannel * this->weightInputStride + groupOutChannel * this->weightOutputStride;
                        for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                            const int32_t numerH =
                                static_cast<int32_t>(outH) + static_cast<int32_t>(this->padding) -
                                static_cast<int32_t>(kernelH);
                            if (numerH < 0 || numerH % static_cast<int32_t>(this->stride) != 0) {
                                continue;
                            }
                            const int32_t inH = numerH / static_cast<int32_t>(this->stride);
                            if (inH < 0 || inH >= static_cast<int32_t>(this->inputHeight)) {
                                continue;
                            }

                            for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                const int32_t numerW =
                                    static_cast<int32_t>(outW) + static_cast<int32_t>(this->padding) -
                                    static_cast<int32_t>(kernelW);
                                if (numerW < 0 || numerW % static_cast<int32_t>(this->stride) != 0) {
                                    continue;
                                }
                                const int32_t inW = numerW / static_cast<int32_t>(this->stride);
                                if (inW < 0 || inW >= static_cast<int32_t>(this->inputWidth)) {
                                    continue;
                                }

                                const uint32_t xOffset =
                                    xChannelBase +
                                    static_cast<uint32_t>(inH) * this->inputWidth +
                                    static_cast<uint32_t>(inW);
                                const uint32_t wOffset = wChannelBase + kernelH * this->kernelWidth + kernelW;
                                sum += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                            }
                        }
                    }

                    sum += addScalar;
                    if (sum > 0.0f) {
                        sum = 0.0f;
                    }
                    const float activated = Gelu(sum);
                    yGm.SetValue(yChannelBase + outH * this->outputWidth + outW, activated * multiplyScalar);
                }
            }
        }
    }

private:
    static constexpr float kGeluTable[257] = {
        -6.457034100e-05f, -7.503348203e-05f, -8.703943845e-05f, -1.007909042e-04f, -1.165134850e-04f, -1.344577775e-04f, -1.549015730e-04f, -1.781521857e-04f,
        -2.045489032e-04f, -2.344655588e-04f, -2.683132195e-04f, -3.065429871e-04f, -3.496489028e-04f, -3.981709500e-04f, -4.526981436e-04f, -5.138716934e-04f,
        -5.823882302e-04f, -6.590030765e-04f, -7.445335449e-04f, -8.398622451e-04f, -9.459403757e-04f, -1.063790978e-03f, -1.194512126e-03f, -1.339280020e-03f,
        -1.499351955e-03f, -1.676069138e-03f, -1.870859305e-03f, -2.085239109e-03f, -2.320816242e-03f, -2.579291241e-03f, -2.862458948e-03f, -3.172209563e-03f,
        -3.510529263e-03f, -3.879500318e-03f, -4.281300684e-03f, -4.718202992e-03f, -5.192572915e-03f, -5.706866837e-03f, -6.263628794e-03f, -6.865486625e-03f,
        -7.515147296e-03f, -8.215391335e-03f, -8.969066349e-03f, -9.779079568e-03f, -1.064838937e-02f, -1.157999579e-02f, -1.257692985e-02f, -1.364224192e-02f,
        -1.477898877e-02f, -1.599021958e-02f, -1.727896069e-02f, -1.864819914e-02f, -2.010086507e-02f, -2.163981284e-02f, -2.326780099e-02f, -2.498747097e-02f,
        -2.680132474e-02f, -2.871170126e-02f, -3.072075180e-02f, -3.283041427e-02f, -3.504238658e-02f, -3.735809904e-02f, -3.977868597e-02f, -4.230495651e-02f,
        -4.493736489e-02f, -4.767598010e-02f, -5.052045521e-02f, -5.346999644e-02f, -5.652333209e-02f, -5.967868158e-02f, -6.293372474e-02f, -6.628557143e-02f,
        -6.973073188e-02f, -7.326508778e-02f, -7.688386446e-02f, -8.058160426e-02f, -8.435214144e-02f, -8.818857876e-02f, -9.208326602e-02f, -9.602778081e-02f,
        -1.000129116e-01f, -1.040286438e-01f, -1.080641480e-01f, -1.121077723e-01f, -1.161470372e-01f, -1.201686343e-01f, -1.241584290e-01f, -1.281014666e-01f,
        -1.319819829e-01f, -1.357834189e-01f, -1.394884400e-01f, -1.430789592e-01f, -1.465361651e-01f, -1.498405550e-01f, -1.529719712e-01f, -1.559096434e-01f,
        -1.586322346e-01f, -1.611178913e-01f, -1.633442992e-01f, -1.652887411e-01f, -1.669281608e-01f, -1.682392290e-01f, -1.691984139e-01f, -1.697820538e-01f,
        -1.699664337e-01f, -1.697278635e-01f, -1.690427588e-01f, -1.678877229e-01f, -1.662396309e-01f, -1.640757137e-01f, -1.613736428e-01f, -1.581116151e-01f,
        -1.542684371e-01f, -1.498236075e-01f, -1.447573993e-01f, -1.390509384e-01f, -1.326862812e-01f, -1.256464882e-01f, -1.179156946e-01f, -1.094791765e-01f,
        -1.003234139e-01f, -9.043614765e-02f, -7.980643264e-02f, -6.842468460e-02f, -5.628272182e-02f, -4.337380081e-02f, -2.969264568e-02f, -1.523547133e-02f,
        0.000000000e+00f, 1.601452867e-02f, 3.280735432e-02f, 5.037619919e-02f, 6.871727818e-02f, 8.782531540e-02f, 1.076935674e-01f, 1.283138524e-01f,
        1.496765861e-01f, 1.717708235e-01f, 1.945843054e-01f, 2.181035118e-01f, 2.423137188e-01f, 2.671990616e-01f, 2.927426007e-01f, 3.189263925e-01f,
        3.457315629e-01f, 3.731383849e-01f, 4.011263572e-01f, 4.296742863e-01f, 4.587603691e-01f, 4.883622771e-01f, 5.184572412e-01f, 5.490221365e-01f,
        5.800335663e-01f, 6.114679462e-01f, 6.433015861e-01f, 6.755107710e-01f, 7.080718392e-01f, 7.409612589e-01f, 7.741557008e-01f, 8.076321087e-01f,
        8.413677654e-01f, 8.753403566e-01f, 9.095280288e-01f, 9.439094450e-01f, 9.784638349e-01f, 1.013171041e+00f, 1.048011560e+00f, 1.082966581e+00f,
        1.118018017e+00f, 1.153148533e+00f, 1.188341571e+00f, 1.223581366e+00f, 1.258852963e+00f, 1.294142228e+00f, 1.329435852e+00f, 1.364721356e+00f,
        1.399987088e+00f, 1.435222219e+00f, 1.470416734e+00f, 1.505561421e+00f, 1.540647859e+00f, 1.575668396e+00f, 1.610616136e+00f, 1.645484912e+00f,
        1.680269268e+00f, 1.714964429e+00f, 1.749566275e+00f, 1.784071318e+00f, 1.818476668e+00f, 1.852780004e+00f, 1.886979545e+00f, 1.921074020e+00f,
        1.955062635e+00f, 1.988945043e+00f, 2.022721314e+00f, 2.056391901e+00f, 2.089957613e+00f, 2.123419586e+00f, 2.156779248e+00f, 2.190038299e+00f,
        2.223198675e+00f, 2.256262529e+00f, 2.289232199e+00f, 2.322110187e+00f, 2.354899135e+00f, 2.387601801e+00f, 2.420221039e+00f, 2.452759780e+00f,
        2.485221011e+00f, 2.517607758e+00f, 2.549923070e+00f, 2.582170004e+00f, 2.614351611e+00f, 2.646470920e+00f, 2.678530934e+00f, 2.710534609e+00f,
        2.742484853e+00f, 2.774384513e+00f, 2.806236371e+00f, 2.838043133e+00f, 2.869807427e+00f, 2.901531797e+00f, 2.933218699e+00f, 2.964870500e+00f,
        2.996489471e+00f, 3.028077790e+00f, 3.059637541e+00f, 3.091170709e+00f, 3.122679184e+00f, 3.154164761e+00f, 3.185629141e+00f, 3.217073931e+00f,
        3.248500648e+00f, 3.279910720e+00f, 3.311305488e+00f, 3.342686209e+00f, 3.374054060e+00f, 3.405410138e+00f, 3.436755466e+00f, 3.468090997e+00f,
        3.499417612e+00f, 3.530736128e+00f, 3.562047302e+00f, 3.593351829e+00f, 3.624650351e+00f, 3.655943457e+00f, 3.687231687e+00f, 3.718515534e+00f,
        3.749795451e+00f, 3.781071848e+00f, 3.812345098e+00f, 3.843615542e+00f, 3.874883487e+00f, 3.906149209e+00f, 3.937412961e+00f, 3.968674967e+00f,
        3.999935430e+00f,
    };

    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> convBiasGm;
    GlobalTensor<float> addValueGm;
    GlobalTensor<float> multiplyValueGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelHeight = 0;
    uint32_t kernelWidth = 0;
    uint32_t outputHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t stride = 0;
    uint32_t padding = 0;
    uint32_t groups = 0;
    uint32_t blockIdx = 0;
    uint32_t inputChannelStride = 0;
    uint32_t outputChannelStride = 0;
    uint32_t inputBatchStride = 0;
    uint32_t outputBatchStride = 0;
    uint32_t weightInputStride = 0;
    uint32_t weightOutputStride = 0;
    uint32_t weightOutputChannelsPerGroup = 0;
    uint32_t inputChannelsPerGroup = 0;
};

extern "C" __global__ __aicore__ void conv_transpose2d_add_min_gelu_multiply_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR conv_bias,
    GM_ADDR add_value,
    GM_ADDR multiply_value,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose2dAddMinGeluMultiply op;
    op.Init(
        x,
        weight,
        conv_bias,
        add_value,
        multiply_value,
        y,
        tiling_data.batchSize,
        tiling_data.inChannels,
        tiling_data.outChannels,
        tiling_data.inputHeight,
        tiling_data.inputWidth,
        tiling_data.kernelHeight,
        tiling_data.kernelWidth,
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.stride,
        tiling_data.padding,
        tiling_data.groups);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv_transpose2d_add_min_gelu_multiply_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &convBias,
    const at::Tensor &addValue,
    const at::Tensor &multiplyValue,
    int64_t stride,
    int64_t padding,
    int64_t outputPadding,
    int64_t groups)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(convBias.dim() == 1, "convBias must be a 1D tensor");
    TORCH_CHECK(addValue.numel() == 1, "addValue must contain exactly one element");
    TORCH_CHECK(multiplyValue.numel() == 1, "multiplyValue must contain exactly one element");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(outputPadding >= 0, "output padding must be non-negative");
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(outputPadding < stride, "output padding must be smaller than stride");
    TORCH_CHECK(x.size(1) == weight.size(0), "input channels must match weight.size(0)");
    TORCH_CHECK(x.size(1) % groups == 0, "input channels must be divisible by groups");

    const int64_t outChannels = weight.size(1) * groups;
    TORCH_CHECK(convBias.size(0) == outChannels, "convBias size must match output channels");

    const int64_t outH = (x.size(2) - 1) * stride - 2 * padding + weight.size(2) + outputPadding;
    const int64_t outW = (x.size(3) - 1) * stride - 2 * padding + weight.size(3) + outputPadding;
    TORCH_CHECK(outH > 0 && outW > 0, "invalid output shape");

    at::Tensor result = at::empty({x.size(0), outChannels, outH, outW}, x.options());
    EXEC_NPU_CMD(
        aclnnConvTranspose2dAddMinGeluMultiplyCustom,
        x,
        weight,
        convBias,
        addValue,
        multiplyValue,
        stride,
        padding,
        outputPadding,
        groups,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv_transpose2d_add_min_gelu_multiply_custom", &conv_transpose2d_add_min_gelu_multiply_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv_transpose2d_add_min_gelu_multiply_custom",
        &conv_transpose2d_add_min_gelu_multiply_impl_npu,
        "conv_transpose2d_add_min_gelu_multiply_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        add_value,
        multiply_value,
    ):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.conv_transpose = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )
        self.add_value = torch.nn.Parameter(torch.tensor([add_value], dtype=torch.float32), requires_grad=False)
        self.multiply_value = torch.nn.Parameter(torch.tensor([multiply_value], dtype=torch.float32), requires_grad=False)
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.conv_transpose.weight.zero_()
            self.conv_transpose.bias.zero_()
            for in_channel in range(self.conv_transpose.weight.size(0)):
                for out_channel in range(self.conv_transpose.weight.size(1)):
                    self.conv_transpose.weight[in_channel, out_channel, 1, 1] = (
                        float(in_channel + out_channel + 1) / 16.0
                    )
            for out_channel in range(self.conv_transpose.bias.numel()):
                self.conv_transpose.bias[out_channel] = -0.25 + out_channel * 0.125

    def forward(self, x):
        return custom_ops_lib.conv_transpose2d_add_min_gelu_multiply_custom(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.add_value,
            self.multiply_value,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )
'''
