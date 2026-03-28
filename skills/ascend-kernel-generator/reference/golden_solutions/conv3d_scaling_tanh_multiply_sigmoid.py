project_json_src='''
[
    {
        "op": "Conv3dScalingTanhMultiplySigmoidCustom",
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
                "name": "scaling_factor",
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
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dScalingTanhMultiplySigmoidCustomTilingData)
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
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Conv3dScalingTanhMultiplySigmoidCustom,
    Conv3dScalingTanhMultiplySigmoidCustomTilingData)
}
"""

host_operator_src="""
#include "conv3d_scaling_tanh_multiply_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
bool IsChannelBroadcastShapeValid(const gert::Shape* shape, int64_t outChannels)
{
    return shape != nullptr &&
           shape->GetDimNum() == 4 &&
           shape->GetDim(0) == outChannels &&
           shape->GetDim(1) == 1 &&
           shape->GetDim(2) == 1 &&
           shape->GetDim(3) == 1;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xStorage = context->GetInputShape(0);
    const gert::StorageShape* weightStorage = context->GetInputShape(1);
    const gert::StorageShape* convBiasStorage = context->GetInputShape(2);
    const gert::StorageShape* scalingStorage = context->GetInputShape(3);
    const gert::StorageShape* biasStorage = context->GetInputShape(4);
    if (xStorage == nullptr || weightStorage == nullptr || convBiasStorage == nullptr ||
        scalingStorage == nullptr || biasStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& xShape = xStorage->GetStorageShape();
    const gert::Shape& weightShape = weightStorage->GetStorageShape();
    const gert::Shape& convBiasShape = convBiasStorage->GetStorageShape();
    const gert::Shape& scalingShape = scalingStorage->GetStorageShape();
    const gert::Shape& biasShape = biasStorage->GetStorageShape();
    if (xShape.GetDimNum() != 5 || weightShape.GetDimNum() != 5 || convBiasShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    if (xShape.GetDim(1) != weightShape.GetDim(1) || convBiasShape.GetDim(0) != weightShape.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }
    if (!IsChannelBroadcastShapeValid(&scalingShape, weightShape.GetDim(0)) ||
        !IsChannelBroadcastShapeValid(&biasShape, weightShape.GetDim(0))) {
        return ge::GRAPH_FAILED;
    }

    const int64_t outD = xShape.GetDim(2) - weightShape.GetDim(2) + 1;
    const int64_t outH = xShape.GetDim(3) - weightShape.GetDim(3) + 1;
    const int64_t outW = xShape.GetDim(4) - weightShape.GetDim(4) + 1;
    if (outD <= 0 || outH <= 0 || outW <= 0) {
        return ge::GRAPH_FAILED;
    }

    Conv3dScalingTanhMultiplySigmoidCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(xShape.GetDim(0)));
    tiling.set_inChannels(static_cast<uint32_t>(xShape.GetDim(1)));
    tiling.set_outChannels(static_cast<uint32_t>(weightShape.GetDim(0)));
    tiling.set_inputDepth(static_cast<uint32_t>(xShape.GetDim(2)));
    tiling.set_inputHeight(static_cast<uint32_t>(xShape.GetDim(3)));
    tiling.set_inputWidth(static_cast<uint32_t>(xShape.GetDim(4)));
    tiling.set_kernelDepth(static_cast<uint32_t>(weightShape.GetDim(2)));
    tiling.set_kernelHeight(static_cast<uint32_t>(weightShape.GetDim(3)));
    tiling.set_kernelWidth(static_cast<uint32_t>(weightShape.GetDim(4)));
    tiling.set_outputDepth(static_cast<uint32_t>(outD));
    tiling.set_outputHeight(static_cast<uint32_t>(outH));
    tiling.set_outputWidth(static_cast<uint32_t>(outW));

    context->SetBlockDim(xShape.GetDim(0) <= 0 ? 1 : static_cast<uint32_t>(xShape.GetDim(0)));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape* convBiasShape = context->GetInputShape(2);
    const gert::Shape* scalingShape = context->GetInputShape(3);
    const gert::Shape* biasShape = context->GetInputShape(4);
    if (xShape == nullptr || weightShape == nullptr || convBiasShape == nullptr ||
        scalingShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5 || convBiasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(1) || convBiasShape->GetDim(0) != weightShape->GetDim(0)) {
        return GRAPH_FAILED;
    }
    if (!IsChannelBroadcastShapeValid(scalingShape, weightShape->GetDim(0)) ||
        !IsChannelBroadcastShapeValid(biasShape, weightShape->GetDim(0))) {
        return GRAPH_FAILED;
    }

    const int64_t outD = xShape->GetDim(2) - weightShape->GetDim(2) + 1;
    const int64_t outH = xShape->GetDim(3) - weightShape->GetDim(3) + 1;
    const int64_t outW = xShape->GetDim(4) - weightShape->GetDim(4) + 1;
    if (outD <= 0 || outH <= 0 || outW <= 0) {
        return GRAPH_FAILED;
    }

    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(5);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, weightShape->GetDim(0));
    yShape->SetDim(2, outD);
    yShape->SetDim(3, outH);
    yShape->SetDim(4, outW);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv3dScalingTanhMultiplySigmoidCustom : public OpDef {
public:
    explicit Conv3dScalingTanhMultiplySigmoidCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conv_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scaling_factor").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dScalingTanhMultiplySigmoidCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConv3dScalingTanhMultiplySigmoid {
public:
    __aicore__ inline KernelConv3dScalingTanhMultiplySigmoid() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR convBias,
        GM_ADDR scalingFactor,
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
        uint32_t outputWidth)
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
        this->blockIdx = GetBlockIdx();

        this->inputPlaneStride = inputHeight * inputWidth;
        this->inputChannelStride = inputDepth * this->inputPlaneStride;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->outputPlaneStride = outputHeight * outputWidth;
        this->outputChannelStride = outputDepth * this->outputPlaneStride;
        this->outputBatchStride = outChannels * this->outputChannelStride;
        this->weightPlaneStride = kernelHeight * kernelWidth;
        this->weightInStride = kernelDepth * this->weightPlaneStride;
        this->weightOutStride = inChannels * this->weightInStride;

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float*)weight, outChannels * this->weightOutStride);
        convBiasGm.SetGlobalBuffer((__gm__ float*)convBias, outChannels);
        scalingFactorGm.SetGlobalBuffer((__gm__ float*)scalingFactor, outChannels);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }

        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->outputBatchStride;

        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            const uint32_t weightBase = outChannel * this->weightOutStride;
            const uint32_t yChannelBase = yBatchBase + outChannel * this->outputChannelStride;
            const float convBiasValue = convBiasGm.GetValue(outChannel);
            const float scalingValue = scalingFactorGm.GetValue(outChannel);
            const float biasValue = biasGm.GetValue(outChannel);
            for (uint32_t outD = 0; outD < this->outputDepth; ++outD) {
                for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                    for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                        float acc = convBiasValue;
                        for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                            const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                            const uint32_t wChannelBase = weightBase + inChannel * this->weightInStride;
                            for (uint32_t kernelD = 0; kernelD < this->kernelDepth; ++kernelD) {
                                const uint32_t inD = outD + kernelD;
                                const uint32_t xDepthBase = xChannelBase + inD * this->inputPlaneStride;
                                const uint32_t wDepthBase = wChannelBase + kernelD * this->weightPlaneStride;
                                for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                                    const uint32_t inH = outH + kernelH;
                                    const uint32_t xRowBase = xDepthBase + inH * this->inputWidth;
                                    const uint32_t wRowBase = wDepthBase + kernelH * this->kernelWidth;
                                    for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                        const uint32_t inW = outW + kernelW;
                                        acc += xGm.GetValue(xRowBase + inW) * weightGm.GetValue(wRowBase + kernelW);
                                    }
                                }
                            }
                        }

                        float value = acc * scalingValue;
                        value = TanhApprox(value);
                        value = value * biasValue;
                        value = SigmoidApprox(value);
                        yGm.SetValue(
                            yChannelBase + outD * this->outputPlaneStride + outH * this->outputWidth + outW,
                            value);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline float Interpolate(
        float value,
        float rangeStart,
        float rangeEnd,
        const float* table) const
    {
        if (value <= rangeStart) {
            return table[0];
        }
        if (value >= rangeEnd) {
            return table[256];
        }

        const float scaled = (value - rangeStart) * (256.0f / (rangeEnd - rangeStart));
        int32_t idx = static_cast<int32_t>(scaled);
        if (idx < 0) {
            idx = 0;
        } else if (idx > 255) {
            idx = 255;
        }
        const float frac = scaled - static_cast<float>(idx);
        const float left = table[idx];
        const float right = table[idx + 1];
        return left + (right - left) * frac;
    }

    __aicore__ inline float TanhApprox(float value) const
    {
        return Interpolate(value, -8.0f, 8.0f, kTanhTable);
    }

    __aicore__ inline float SigmoidApprox(float value) const
    {
        return Interpolate(value, -12.0f, 12.0f, kSigmoidTable);
    }

    static constexpr float kSigmoidTable[257] = {
        6.144174602e-06f, 6.748051590e-06f, 7.411279872e-06f, 8.139692625e-06f, 8.939696305e-06f, 9.818326984e-06f, 1.078331222e-05f, 1.184313903e-05f,
        1.300712847e-05f, 1.428551765e-05f, 1.568954973e-05f, 1.723157275e-05f, 1.892514825e-05f, 2.078517044e-05f, 2.282799719e-05f, 2.507159385e-05f,
        2.753569111e-05f, 3.024195853e-05f, 3.321419495e-05f, 3.647853768e-05f, 4.006369223e-05f, 4.400118458e-05f, 4.832563819e-05f, 5.307507823e-05f,
        5.829126566e-05f, 6.402006410e-05f, 7.031184267e-05f, 7.722191834e-05f, 8.481104172e-05f, 9.314593043e-05f, 1.022998547e-04f, 1.123532806e-04f,
        1.233945760e-04f, 1.355207856e-04f, 1.488384826e-04f, 1.634647025e-04f, 1.795279693e-04f, 1.971694220e-04f, 2.165440499e-04f, 2.378220506e-04f,
        2.611903191e-04f, 2.868540824e-04f, 3.150386941e-04f, 3.459916032e-04f, 3.799845148e-04f, 4.173157607e-04f, 4.583129006e-04f, 5.033355755e-04f,
        5.527786369e-04f, 6.070755804e-04f, 6.667023092e-04f, 7.321812617e-04f, 8.040859356e-04f, 8.830458478e-04f, 9.697519678e-04f, 1.064962672e-03f,
        1.169510265e-03f, 1.284308120e-03f, 1.410358497e-03f, 1.548761094e-03f, 1.700722411e-03f, 1.867565978e-03f, 2.050743540e-03f, 2.251847280e-03f,
        2.472623157e-03f, 2.714985469e-03f, 2.981032730e-03f, 3.273064971e-03f, 3.593602581e-03f, 3.945406807e-03f, 4.331502021e-03f, 4.755199910e-03f,
        5.220125694e-03f, 5.730246515e-03f, 6.289902137e-03f, 6.903838071e-03f, 7.577241268e-03f, 8.315778477e-03f, 9.125637389e-03f, 1.001357062e-02f,
        1.098694263e-02f, 1.205377952e-02f, 1.322282175e-02f, 1.450357969e-02f, 1.590639171e-02f, 1.744248474e-02f, 1.912403675e-02f, 2.096424082e-02f,
        2.297736991e-02f, 2.517884170e-02f, 2.758528223e-02f, 3.021458707e-02f, 3.308597839e-02f, 3.622005586e-02f, 3.963883910e-02f, 4.336579876e-02f,
        4.742587318e-02f, 5.184546665e-02f, 5.665242531e-02f, 6.187598572e-02f, 6.754669114e-02f, 7.369626971e-02f, 8.035746882e-02f, 8.756383952e-02f,
        9.534946490e-02f, 1.037486269e-01f, 1.127954063e-01f, 1.225232125e-01f, 1.329642402e-01f, 1.441488530e-01f, 1.561048974e-01f, 1.688569521e-01f,
        1.824255238e-01f, 1.968262036e-01f, 2.120688044e-01f, 2.281565022e-01f, 2.450850131e-01f, 2.628418374e-01f, 2.814056074e-01f, 3.007455789e-01f,
        3.208213008e-01f, 3.415824994e-01f, 3.629692055e-01f, 3.849121445e-01f, 4.073334000e-01f, 4.301473486e-01f, 4.532618480e-01f, 4.765796511e-01f,
        5.000000000e-01f, 5.234203489e-01f, 5.467381520e-01f, 5.698526514e-01f, 5.926666000e-01f, 6.150878555e-01f, 6.370307945e-01f, 6.584175006e-01f,
        6.791786992e-01f, 6.992544211e-01f, 7.185943926e-01f, 7.371581626e-01f, 7.549149869e-01f, 7.718434978e-01f, 7.879311956e-01f, 8.031737964e-01f,
        8.175744762e-01f, 8.311430479e-01f, 8.438951026e-01f, 8.558511470e-01f, 8.670357598e-01f, 8.774767875e-01f, 8.872045937e-01f, 8.962513731e-01f,
        9.046505351e-01f, 9.124361605e-01f, 9.196425312e-01f, 9.263037303e-01f, 9.324533089e-01f, 9.381240143e-01f, 9.433475747e-01f, 9.481545334e-01f,
        9.525741268e-01f, 9.566342012e-01f, 9.603611609e-01f, 9.637799441e-01f, 9.669140216e-01f, 9.697854129e-01f, 9.724147178e-01f, 9.748211583e-01f,
        9.770226301e-01f, 9.790357592e-01f, 9.808759632e-01f, 9.825575153e-01f, 9.840936083e-01f, 9.854964203e-01f, 9.867771782e-01f, 9.879462205e-01f,
        9.890130574e-01f, 9.899864294e-01f, 9.908743626e-01f, 9.916842215e-01f, 9.924227587e-01f, 9.930961619e-01f, 9.937100979e-01f, 9.942697535e-01f,
        9.947798743e-01f, 9.952448001e-01f, 9.956684980e-01f, 9.960545932e-01f, 9.964063974e-01f, 9.967269350e-01f, 9.970189673e-01f, 9.972850145e-01f,
        9.975273768e-01f, 9.977481527e-01f, 9.979492565e-01f, 9.981324340e-01f, 9.982992776e-01f, 9.984512389e-01f, 9.985896415e-01f, 9.987156919e-01f,
        9.988304897e-01f, 9.989350373e-01f, 9.990302480e-01f, 9.991169542e-01f, 9.991959141e-01f, 9.992678187e-01f, 9.993332977e-01f, 9.993929244e-01f,
        9.994472214e-01f, 9.994966644e-01f, 9.995416871e-01f, 9.995826842e-01f, 9.996200155e-01f, 9.996540084e-01f, 9.996849613e-01f, 9.997131459e-01f,
        9.997388097e-01f, 9.997621779e-01f, 9.997834560e-01f, 9.998028306e-01f, 9.998204720e-01f, 9.998365353e-01f, 9.998511615e-01f, 9.998644792e-01f,
        9.998766054e-01f, 9.998876467e-01f, 9.998977001e-01f, 9.999068541e-01f, 9.999151890e-01f, 9.999227781e-01f, 9.999296882e-01f, 9.999359799e-01f,
        9.999417087e-01f, 9.999469249e-01f, 9.999516744e-01f, 9.999559988e-01f, 9.999599363e-01f, 9.999635215e-01f, 9.999667858e-01f, 9.999697580e-01f,
        9.999724643e-01f, 9.999749284e-01f, 9.999771720e-01f, 9.999792148e-01f, 9.999810749e-01f, 9.999827684e-01f, 9.999843105e-01f, 9.999857145e-01f,
        9.999869929e-01f, 9.999881569e-01f, 9.999892167e-01f, 9.999901817e-01f, 9.999910603e-01f, 9.999918603e-01f, 9.999925887e-01f, 9.999932519e-01f,
        9.999938558e-01f,
    };

    static constexpr float kTanhTable[257] = {
        -9.999997749e-01f, -9.999997450e-01f, -9.999997110e-01f, -9.999996725e-01f, -9.999996289e-01f, -9.999995795e-01f, -9.999995235e-01f, -9.999994601e-01f,
        -9.999993882e-01f, -9.999993067e-01f, -9.999992144e-01f, -9.999991098e-01f, -9.999989913e-01f, -9.999988570e-01f, -9.999987048e-01f, -9.999985324e-01f,
        -9.999983369e-01f, -9.999981155e-01f, -9.999978646e-01f, -9.999975803e-01f, -9.999972581e-01f, -9.999968930e-01f, -9.999964793e-01f, -9.999960105e-01f,
        -9.999954794e-01f, -9.999948774e-01f, -9.999941954e-01f, -9.999934225e-01f, -9.999925467e-01f, -9.999915543e-01f, -9.999904298e-01f, -9.999891556e-01f,
        -9.999877117e-01f, -9.999860755e-01f, -9.999842215e-01f, -9.999821206e-01f, -9.999797400e-01f, -9.999770425e-01f, -9.999739857e-01f, -9.999705220e-01f,
        -9.999665972e-01f, -9.999621497e-01f, -9.999571101e-01f, -9.999513995e-01f, -9.999449286e-01f, -9.999375962e-01f, -9.999292875e-01f, -9.999198726e-01f,
        -9.999092043e-01f, -9.998971156e-01f, -9.998834175e-01f, -9.998678957e-01f, -9.998503075e-01f, -9.998303779e-01f, -9.998077952e-01f, -9.997822062e-01f,
        -9.997532108e-01f, -9.997203558e-01f, -9.996831276e-01f, -9.996409441e-01f, -9.995931460e-01f, -9.995389866e-01f, -9.994776194e-01f, -9.994080858e-01f,
        -9.993292997e-01f, -9.992400310e-01f, -9.991388858e-01f, -9.990242858e-01f, -9.988944427e-01f, -9.987473317e-01f, -9.985806592e-01f, -9.983918281e-01f,
        -9.981778976e-01f, -9.979355379e-01f, -9.976609795e-01f, -9.973499552e-01f, -9.969976355e-01f, -9.965985552e-01f, -9.961465307e-01f, -9.956345671e-01f,
        -9.950547537e-01f, -9.943981461e-01f, -9.936546343e-01f, -9.928127948e-01f, -9.918597246e-01f, -9.907808556e-01f, -9.895597486e-01f, -9.881778623e-01f,
        -9.866142982e-01f, -9.848455175e-01f, -9.828450292e-01f, -9.805830470e-01f, -9.780261147e-01f, -9.751366983e-01f, -9.718727459e-01f, -9.681872166e-01f,
        -9.640275801e-01f, -9.593352933e-01f, -9.540452602e-01f, -9.480852856e-01f, -9.413755385e-01f, -9.338280432e-01f, -9.253462253e-01f, -9.158245442e-01f,
        -9.051482536e-01f, -8.931933404e-01f, -8.798266997e-01f, -8.649066177e-01f, -8.482836400e-01f, -8.298019100e-01f, -8.093010702e-01f, -7.866188121e-01f,
        -7.615941560e-01f, -7.340715196e-01f, -7.039056039e-01f, -6.709670742e-01f, -6.351489524e-01f, -5.963735555e-01f, -5.545997223e-01f, -5.098299737e-01f,
        -4.621171573e-01f, -4.115700557e-01f, -3.583573984e-01f, -3.027097293e-01f, -2.449186624e-01f, -1.853331999e-01f, -1.243530018e-01f, -6.241874675e-02f,
        0.000000000e+00f, 6.241874675e-02f, 1.243530018e-01f, 1.853331999e-01f, 2.449186624e-01f, 3.027097293e-01f, 3.583573984e-01f, 4.115700557e-01f,
        4.621171573e-01f, 5.098299737e-01f, 5.545997223e-01f, 5.963735555e-01f, 6.351489524e-01f, 6.709670742e-01f, 7.039056039e-01f, 7.340715196e-01f,
        7.615941560e-01f, 7.866188121e-01f, 8.093010702e-01f, 8.298019100e-01f, 8.482836400e-01f, 8.649066177e-01f, 8.798266997e-01f, 8.931933404e-01f,
        9.051482536e-01f, 9.158245442e-01f, 9.253462253e-01f, 9.338280432e-01f, 9.413755385e-01f, 9.480852856e-01f, 9.540452602e-01f, 9.593352933e-01f,
        9.640275801e-01f, 9.681872166e-01f, 9.718727459e-01f, 9.751366983e-01f, 9.780261147e-01f, 9.805830470e-01f, 9.828450292e-01f, 9.848455175e-01f,
        9.866142982e-01f, 9.881778623e-01f, 9.895597486e-01f, 9.907808556e-01f, 9.918597246e-01f, 9.928127948e-01f, 9.936546343e-01f, 9.943981461e-01f,
        9.950547537e-01f, 9.956345671e-01f, 9.961465307e-01f, 9.965985552e-01f, 9.969976355e-01f, 9.973499552e-01f, 9.976609795e-01f, 9.979355379e-01f,
        9.981778976e-01f, 9.983918281e-01f, 9.985806592e-01f, 9.987473317e-01f, 9.988944427e-01f, 9.990242858e-01f, 9.991388858e-01f, 9.992400310e-01f,
        9.993292997e-01f, 9.994080858e-01f, 9.994776194e-01f, 9.995389866e-01f, 9.995931460e-01f, 9.996409441e-01f, 9.996831276e-01f, 9.997203558e-01f,
        9.997532108e-01f, 9.997822062e-01f, 9.998077952e-01f, 9.998303779e-01f, 9.998503075e-01f, 9.998678957e-01f, 9.998834175e-01f, 9.998971156e-01f,
        9.999092043e-01f, 9.999198726e-01f, 9.999292875e-01f, 9.999375962e-01f, 9.999449286e-01f, 9.999513995e-01f, 9.999571101e-01f, 9.999621497e-01f,
        9.999665972e-01f, 9.999705220e-01f, 9.999739857e-01f, 9.999770425e-01f, 9.999797400e-01f, 9.999821206e-01f, 9.999842215e-01f, 9.999860755e-01f,
        9.999877117e-01f, 9.999891556e-01f, 9.999904298e-01f, 9.999915543e-01f, 9.999925467e-01f, 9.999934225e-01f, 9.999941954e-01f, 9.999948774e-01f,
        9.999954794e-01f, 9.999960105e-01f, 9.999964793e-01f, 9.999968930e-01f, 9.999972581e-01f, 9.999975803e-01f, 9.999978646e-01f, 9.999981155e-01f,
        9.999983369e-01f, 9.999985324e-01f, 9.999987048e-01f, 9.999988570e-01f, 9.999989913e-01f, 9.999991098e-01f, 9.999992144e-01f, 9.999993067e-01f,
        9.999993882e-01f, 9.999994601e-01f, 9.999995235e-01f, 9.999995795e-01f, 9.999996289e-01f, 9.999996725e-01f, 9.999997110e-01f, 9.999997450e-01f,
        9.999997749e-01f,
    };

    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> convBiasGm;
    GlobalTensor<float> scalingFactorGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t batchSize = 0;
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    uint32_t inputDepth = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelDepth = 0;
    uint32_t kernelHeight = 0;
    uint32_t kernelWidth = 0;
    uint32_t outputDepth = 0;
    uint32_t outputHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t blockIdx = 0;
    uint32_t inputPlaneStride = 0;
    uint32_t inputChannelStride = 0;
    uint32_t inputBatchStride = 0;
    uint32_t outputPlaneStride = 0;
    uint32_t outputChannelStride = 0;
    uint32_t outputBatchStride = 0;
    uint32_t weightPlaneStride = 0;
    uint32_t weightInStride = 0;
    uint32_t weightOutStride = 0;
};

extern "C" __global__ __aicore__ void conv3d_scaling_tanh_multiply_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR convBias,
    GM_ADDR scalingFactor,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dScalingTanhMultiplySigmoid op;
    op.Init(
        x,
        weight,
        convBias,
        scalingFactor,
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
        tiling_data.outputWidth);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv3d_scaling_tanh_multiply_sigmoid_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& convBias,
    const at::Tensor& scalingFactor,
    const at::Tensor& bias)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D OIDHW tensor");
    TORCH_CHECK(convBias.dim() == 1, "conv_bias must be a 1D tensor");
    TORCH_CHECK(scalingFactor.dim() == 4, "scaling_factor must be a 4D tensor");
    TORCH_CHECK(bias.dim() == 4, "bias must be a 4D tensor");
    TORCH_CHECK(x.size(1) == weight.size(1), "x channels must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == convBias.size(0), "conv_bias length must match out_channels");
    TORCH_CHECK(scalingFactor.size(0) == weight.size(0), "scaling_factor channel dim must match out_channels");
    TORCH_CHECK(bias.size(0) == weight.size(0), "bias channel dim must match out_channels");
    TORCH_CHECK(
        scalingFactor.size(1) == 1 && scalingFactor.size(2) == 1 && scalingFactor.size(3) == 1,
        "scaling_factor must have shape [out_channels, 1, 1, 1]");
    TORCH_CHECK(
        bias.size(1) == 1 && bias.size(2) == 1 && bias.size(3) == 1,
        "bias must have shape [out_channels, 1, 1, 1]");

    const int64_t outD = x.size(2) - weight.size(2) + 1;
    const int64_t outH = x.size(3) - weight.size(3) + 1;
    const int64_t outW = x.size(4) - weight.size(4) + 1;
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "invalid conv3d output shape");

    at::Tensor result = at::empty({x.size(0), weight.size(0), outD, outH, outW}, x.options());
    EXEC_NPU_CMD(aclnnConv3dScalingTanhMultiplySigmoidCustom, x, weight, convBias, scalingFactor, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv3d_scaling_tanh_multiply_sigmoid_custom",
        &conv3d_scaling_tanh_multiply_sigmoid_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv3d_scaling_tanh_multiply_sigmoid_custom",
        &conv3d_scaling_tanh_multiply_sigmoid_impl_npu,
        "conv3d_scaling_tanh_multiply_sigmoid_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = torch.nn.Parameter(torch.randn(bias_shape))
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv3d_scaling_tanh_multiply_sigmoid_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.scaling_factor,
            self.bias,
        )
'''
