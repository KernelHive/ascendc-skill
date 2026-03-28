project_json_src='''
[
    {
        "op": "Conv2dTanhScalingBiasAddMaxCustom",
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
                "name": "add_bias",
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
                "name": "scaling_factor",
                "param_type": "required",
                "type": "float"
            },
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
BEGIN_TILING_DATA_DEF(Conv2dTanhScalingBiasAddMaxCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, convOutputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, convOutputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, poolKernelSize);
    TILING_DATA_FIELD_DEF(float, scalingFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Conv2dTanhScalingBiasAddMaxCustom,
    Conv2dTanhScalingBiasAddMaxCustomTilingData)
}
"""

host_operator_src="""
#include "conv2d_tanh_scaling_bias_add_max_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeConvOutputDim(int64_t input, int64_t kernel)
{
    if (input < kernel || kernel <= 0) {
        return 0;
    }
    return static_cast<uint32_t>(input - kernel + 1);
}

uint32_t ComputePoolOutputDim(int64_t input, int64_t kernel)
{
    if (input < kernel || kernel <= 0) {
        return 0;
    }
    return static_cast<uint32_t>((input - kernel) / kernel + 1);
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::StorageShape *weightShape = context->GetInputShape(1);
    const gert::StorageShape *convBiasShape = context->GetInputShape(2);
    const gert::StorageShape *addBiasShape = context->GetInputShape(3);
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr || addBiasShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = inputShape->GetStorageShape();
    const auto wShape = weightShape->GetStorageShape();
    const auto convBShape = convBiasShape->GetStorageShape();
    const auto addBShape = addBiasShape->GetStorageShape();
    if (xShape.GetDimNum() != 4 || wShape.GetDimNum() != 4 || convBShape.GetDimNum() != 1 || addBShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const float *scalingFactorPtr = attrs->GetAttrPointer<float>(0);
    const int64_t *poolKernelSizePtr = attrs->GetAttrPointer<int64_t>(1);
    if (scalingFactorPtr == nullptr || poolKernelSizePtr == nullptr) {
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
    const uint32_t convBiasSize = static_cast<uint32_t>(convBShape.GetDim(0));
    const uint32_t addBiasChannels = static_cast<uint32_t>(addBShape.GetDim(0));
    const uint32_t addBiasHeight = static_cast<uint32_t>(addBShape.GetDim(1));
    const uint32_t addBiasWidth = static_cast<uint32_t>(addBShape.GetDim(2));
    const uint32_t poolKernelSize = static_cast<uint32_t>(*poolKernelSizePtr);

    if (inChannels != weightInChannels || convBiasSize != outChannels || addBiasChannels != outChannels ||
        addBiasHeight != 1 || addBiasWidth != 1 || poolKernelSize == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t convOutputHeight = ComputeConvOutputDim(inputHeight, kernelHeight);
    const uint32_t convOutputWidth = ComputeConvOutputDim(inputWidth, kernelWidth);
    const uint32_t outputHeight = ComputePoolOutputDim(convOutputHeight, poolKernelSize);
    const uint32_t outputWidth = ComputePoolOutputDim(convOutputWidth, poolKernelSize);

    Conv2dTanhScalingBiasAddMaxCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_convOutputHeight(convOutputHeight);
    tiling.set_convOutputWidth(convOutputWidth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_poolKernelSize(poolKernelSize);
    tiling.set_scalingFactor(*scalingFactorPtr);

    const uint32_t blockDim = batchSize == 0 ? 1 : (batchSize > 8 ? 8 : batchSize);
    context->SetBlockDim(blockDim);
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
    const gert::Shape *addBiasShape = context->GetInputShape(3);
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr || addBiasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4 || convBiasShape->GetDimNum() != 1 ||
        addBiasShape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }
    const int64_t *poolKernelSizePtr = attrs->GetAttrPointer<int64_t>(1);
    if (poolKernelSizePtr == nullptr || *poolKernelSizePtr <= 0) {
        return GRAPH_FAILED;
    }

    if (inputShape->GetDim(1) != weightShape->GetDim(1) || weightShape->GetDim(0) != convBiasShape->GetDim(0) ||
        weightShape->GetDim(0) != addBiasShape->GetDim(0) || addBiasShape->GetDim(1) != 1 ||
        addBiasShape->GetDim(2) != 1) {
        return GRAPH_FAILED;
    }

    const int64_t convOutputHeight = ComputeConvOutputDim(inputShape->GetDim(2), weightShape->GetDim(2));
    const int64_t convOutputWidth = ComputeConvOutputDim(inputShape->GetDim(3), weightShape->GetDim(3));
    const int64_t outputHeight = ComputePoolOutputDim(convOutputHeight, *poolKernelSizePtr);
    const int64_t outputWidth = ComputePoolOutputDim(convOutputWidth, *poolKernelSizePtr);

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
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
class Conv2dTanhScalingBiasAddMaxCustom : public OpDef {
public:
    explicit Conv2dTanhScalingBiasAddMaxCustom(const char *name) : OpDef(name)
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
        this->Input("add_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scaling_factor").AttrType(REQUIRED).Float();
        this->Attr("pool_kernel_size").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dTanhScalingBiasAddMaxCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConv2dTanhScalingBiasAddMax {
public:
    __aicore__ inline KernelConv2dTanhScalingBiasAddMax() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR convBias,
        GM_ADDR addBias,
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
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t poolKernelSize,
        float scalingFactor)
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
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->poolKernelSize = poolKernelSize;
        this->scalingFactor = scalingFactor;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum();
        this->inputChannelStride = inputHeight * inputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->weightOutStride = inChannels * kernelHeight * kernelWidth;
        this->weightInStride = kernelHeight * kernelWidth;
        this->outputChannelStride = outputHeight * outputWidth;
        this->outputBatchStride = outChannels * this->outputChannelStride;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * this->weightOutStride);
        convBiasGm.SetGlobalBuffer((__gm__ float *)convBias, outChannels);
        addBiasGm.SetGlobalBuffer((__gm__ float *)addBias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->outputHeight == 0 || this->outputWidth == 0) {
            return;
        }

        for (uint32_t batchIdx = this->blockIdx; batchIdx < this->batchSize; batchIdx += this->blockNum) {
            const uint32_t inputBatchBase = batchIdx * this->inputBatchStride;
            const uint32_t outputBatchBase = batchIdx * this->outputBatchStride;
            for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                const uint32_t weightBase = outChannel * this->weightOutStride;
                const float convBiasValue = convBiasGm.GetValue(outChannel);
                const float addBiasValue = addBiasGm.GetValue(outChannel);
                const uint32_t outputChannelBase = outputBatchBase + outChannel * this->outputChannelStride;
                for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                    const uint32_t poolStartH = outH * this->poolKernelSize;
                    for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                        const uint32_t poolStartW = outW * this->poolKernelSize;
                        float maxValue = 0.0f;
                        bool hasValue = false;
                        for (uint32_t poolH = 0; poolH < this->poolKernelSize; ++poolH) {
                            const uint32_t convH = poolStartH + poolH;
                            for (uint32_t poolW = 0; poolW < this->poolKernelSize; ++poolW) {
                                const uint32_t convW = poolStartW + poolW;
                                float convValue = convBiasValue;
                                for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                                    const uint32_t inputChannelBase =
                                        inputBatchBase + inChannel * this->inputChannelStride;
                                    const uint32_t weightChannelBase =
                                        weightBase + inChannel * this->weightInStride;
                                    for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                                        const uint32_t inH = convH + kernelH;
                                        const uint32_t inputRowBase =
                                            inputChannelBase + inH * this->inputWidth;
                                        const uint32_t weightRowBase =
                                            weightChannelBase + kernelH * this->kernelWidth;
                                        for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                            convValue += xGm.GetValue(inputRowBase + convW + kernelW) *
                                                weightGm.GetValue(weightRowBase + kernelW);
                                        }
                                    }
                                }

                                float fusedValue = ApplyTanh(convValue);
                                fusedValue = fusedValue * this->scalingFactor + addBiasValue;
                                if (!hasValue || fusedValue > maxValue) {
                                    maxValue = fusedValue;
                                    hasValue = true;
                                }
                            }
                        }
                        yGm.SetValue(outputChannelBase + outH * this->outputWidth + outW, maxValue);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline float ApplyTanh(float x) const
    {
        if (x <= -8.0f) {
            return -9.999997749e-01f;
        }
        if (x >= 8.0f) {
            return 9.999997749e-01f;
        }

        constexpr float kStart = -8.0f;
        constexpr float kInvStep = 3.2e+01f;
        const float scaled = (x - kStart) * kInvStep;
        int32_t idx = static_cast<int32_t>(scaled);
        if (idx < 0) {
            idx = 0;
        } else if (idx > 511) {
            idx = 511;
        }
        const float frac = scaled - static_cast<float>(idx);
        const float left = kTanhTable[idx];
        const float right = kTanhTable[idx + 1];
        return left + (right - left) * frac;
    }

private:
    static constexpr float kTanhTable[513] = {
        -9.999997749e-01f, -9.999997604e-01f, -9.999997450e-01f, -9.999997285e-01f, -9.999997110e-01f, -9.999996924e-01f, -9.999996725e-01f, -9.999996514e-01f,
        -9.999996289e-01f, -9.999996050e-01f, -9.999995795e-01f, -9.999995524e-01f, -9.999995235e-01f, -9.999994928e-01f, -9.999994601e-01f, -9.999994253e-01f,
        -9.999993882e-01f, -9.999993487e-01f, -9.999993067e-01f, -9.999992620e-01f, -9.999992144e-01f, -9.999991638e-01f, -9.999991098e-01f, -9.999990524e-01f,
        -9.999989913e-01f, -9.999989262e-01f, -9.999988570e-01f, -9.999987833e-01f, -9.999987048e-01f, -9.999986213e-01f, -9.999985324e-01f, -9.999984377e-01f,
        -9.999983369e-01f, -9.999982297e-01f, -9.999981155e-01f, -9.999979940e-01f, -9.999978646e-01f, -9.999977269e-01f, -9.999975803e-01f, -9.999974242e-01f,
        -9.999972581e-01f, -9.999970812e-01f, -9.999968930e-01f, -9.999966926e-01f, -9.999964793e-01f, -9.999962522e-01f, -9.999960105e-01f, -9.999957532e-01f,
        -9.999954794e-01f, -9.999951878e-01f, -9.999948774e-01f, -9.999945471e-01f, -9.999941954e-01f, -9.999938210e-01f, -9.999934225e-01f, -9.999929983e-01f,
        -9.999925467e-01f, -9.999920660e-01f, -9.999915543e-01f, -9.999910096e-01f, -9.999904298e-01f, -9.999898126e-01f, -9.999891556e-01f, -9.999884562e-01f,
        -9.999877117e-01f, -9.999869191e-01f, -9.999860755e-01f, -9.999851774e-01f, -9.999842215e-01f, -9.999832039e-01f, -9.999821206e-01f, -9.999809675e-01f,
        -9.999797400e-01f, -9.999784334e-01f, -9.999770425e-01f, -9.999755618e-01f, -9.999739857e-01f, -9.999723080e-01f, -9.999705220e-01f, -9.999686209e-01f,
        -9.999665972e-01f, -9.999644429e-01f, -9.999621497e-01f, -9.999597086e-01f, -9.999571101e-01f, -9.999543440e-01f, -9.999513995e-01f, -9.999482651e-01f,
        -9.999449286e-01f, -9.999413769e-01f, -9.999375962e-01f, -9.999335716e-01f, -9.999292875e-01f, -9.999247271e-01f, -9.999198726e-01f, -9.999147051e-01f,
        -9.999092043e-01f, -9.999033487e-01f, -9.998971156e-01f, -9.998904805e-01f, -9.998834175e-01f, -9.998758990e-01f, -9.998678957e-01f, -9.998593763e-01f,
        -9.998503075e-01f, -9.998406540e-01f, -9.998303779e-01f, -9.998194392e-01f, -9.998077952e-01f, -9.997954003e-01f, -9.997822062e-01f, -9.997681613e-01f,
        -9.997532108e-01f, -9.997372964e-01f, -9.997203558e-01f, -9.997023230e-01f, -9.996831276e-01f, -9.996626945e-01f, -9.996409441e-01f, -9.996177914e-01f,
        -9.995931460e-01f, -9.995669119e-01f, -9.995389866e-01f, -9.995092610e-01f, -9.994776194e-01f, -9.994439381e-01f, -9.994080858e-01f, -9.993699226e-01f,
        -9.993292997e-01f, -9.992860587e-01f, -9.992400310e-01f, -9.991910370e-01f, -9.991388858e-01f, -9.990833742e-01f, -9.990242858e-01f, -9.989613903e-01f,
        -9.988944427e-01f, -9.988231824e-01f, -9.987473317e-01f, -9.986665954e-01f, -9.985806592e-01f, -9.984891887e-01f, -9.983918281e-01f, -9.982881987e-01f,
        -9.981778976e-01f, -9.980604961e-01f, -9.979355379e-01f, -9.978025379e-01f, -9.976609795e-01f, -9.975103134e-01f, -9.973499552e-01f, -9.971792830e-01f,
        -9.969976355e-01f, -9.968043090e-01f, -9.965985552e-01f, -9.963795779e-01f, -9.961465307e-01f, -9.958985129e-01f, -9.956345671e-01f, -9.953536750e-01f,
        -9.950547537e-01f, -9.947366521e-01f, -9.943981461e-01f, -9.940379345e-01f, -9.936546343e-01f, -9.932467752e-01f, -9.928127948e-01f, -9.923510327e-01f,
        -9.918597246e-01f, -9.913369960e-01f, -9.907808556e-01f, -9.901891886e-01f, -9.895597486e-01f, -9.888901506e-01f, -9.881778623e-01f, -9.874201957e-01f,
        -9.866142982e-01f, -9.857571425e-01f, -9.848455175e-01f, -9.838760169e-01f, -9.828450292e-01f, -9.817487252e-01f, -9.805830470e-01f, -9.793436950e-01f,
        -9.780261147e-01f, -9.766254840e-01f, -9.751366983e-01f, -9.735543565e-01f, -9.718727459e-01f, -9.700858268e-01f, -9.681872166e-01f, -9.661701735e-01f,
        -9.640275801e-01f, -9.617519265e-01f, -9.593352933e-01f, -9.567693345e-01f, -9.540452602e-01f, -9.511538199e-01f, -9.480852856e-01f, -9.448294355e-01f,
        -9.413755385e-01f, -9.377123389e-01f, -9.338280432e-01f, -9.297103072e-01f, -9.253462253e-01f, -9.207223218e-01f, -9.158245442e-01f, -9.106382595e-01f,
        -9.051482536e-01f, -8.993387348e-01f, -8.931933404e-01f, -8.866951494e-01f, -8.798266997e-01f, -8.725700115e-01f, -8.649066177e-01f, -8.568176011e-01f,
        -8.482836400e-01f, -8.392850624e-01f, -8.298019100e-01f, -8.198140121e-01f, -8.093010702e-01f, -7.982427545e-01f, -7.866188121e-01f, -7.744091874e-01f,
        -7.615941560e-01f, -7.481544703e-01f, -7.340715196e-01f, -7.193275010e-01f, -7.039056039e-01f, -6.877902051e-01f, -6.709670742e-01f, -6.534235881e-01f,
        -6.351489524e-01f, -6.161344271e-01f, -5.963735555e-01f, -5.758623913e-01f, -5.545997223e-01f, -5.325872862e-01f, -5.098299737e-01f, -4.863360172e-01f,
        -4.621171573e-01f, -4.371887851e-01f, -4.115700557e-01f, -3.852839663e-01f, -3.583573984e-01f, -3.308211175e-01f, -3.027097293e-01f, -2.740615890e-01f,
        -2.449186624e-01f, -2.153263397e-01f, -1.853331999e-01f, -1.549907304e-01f, -1.243530018e-01f, -9.347630397e-02f, -6.241874675e-02f, -3.123983145e-02f,
        0.000000000e+00f, 3.123983145e-02f, 6.241874675e-02f, 9.347630397e-02f, 1.243530018e-01f, 1.549907304e-01f, 1.853331999e-01f, 2.153263397e-01f,
        2.449186624e-01f, 2.740615890e-01f, 3.027097293e-01f, 3.308211175e-01f, 3.583573984e-01f, 3.852839663e-01f, 4.115700557e-01f, 4.371887851e-01f,
        4.621171573e-01f, 4.863360172e-01f, 5.098299737e-01f, 5.325872862e-01f, 5.545997223e-01f, 5.758623913e-01f, 5.963735555e-01f, 6.161344271e-01f,
        6.351489524e-01f, 6.534235881e-01f, 6.709670742e-01f, 6.877902051e-01f, 7.039056039e-01f, 7.193275010e-01f, 7.340715196e-01f, 7.481544703e-01f,
        7.615941560e-01f, 7.744091874e-01f, 7.866188121e-01f, 7.982427545e-01f, 8.093010702e-01f, 8.198140121e-01f, 8.298019100e-01f, 8.392850624e-01f,
        8.482836400e-01f, 8.568176011e-01f, 8.649066177e-01f, 8.725700115e-01f, 8.798266997e-01f, 8.866951494e-01f, 8.931933404e-01f, 8.993387348e-01f,
        9.051482536e-01f, 9.106382595e-01f, 9.158245442e-01f, 9.207223218e-01f, 9.253462253e-01f, 9.297103072e-01f, 9.338280432e-01f, 9.377123389e-01f,
        9.413755385e-01f, 9.448294355e-01f, 9.480852856e-01f, 9.511538199e-01f, 9.540452602e-01f, 9.567693345e-01f, 9.593352933e-01f, 9.617519265e-01f,
        9.640275801e-01f, 9.661701735e-01f, 9.681872166e-01f, 9.700858268e-01f, 9.718727459e-01f, 9.735543565e-01f, 9.751366983e-01f, 9.766254840e-01f,
        9.780261147e-01f, 9.793436950e-01f, 9.805830470e-01f, 9.817487252e-01f, 9.828450292e-01f, 9.838760169e-01f, 9.848455175e-01f, 9.857571425e-01f,
        9.866142982e-01f, 9.874201957e-01f, 9.881778623e-01f, 9.888901506e-01f, 9.895597486e-01f, 9.901891886e-01f, 9.907808556e-01f, 9.913369960e-01f,
        9.918597246e-01f, 9.923510327e-01f, 9.928127948e-01f, 9.932467752e-01f, 9.936546343e-01f, 9.940379345e-01f, 9.943981461e-01f, 9.947366521e-01f,
        9.950547537e-01f, 9.953536750e-01f, 9.956345671e-01f, 9.958985129e-01f, 9.961465307e-01f, 9.963795779e-01f, 9.965985552e-01f, 9.968043090e-01f,
        9.969976355e-01f, 9.971792830e-01f, 9.973499552e-01f, 9.975103134e-01f, 9.976609795e-01f, 9.978025379e-01f, 9.979355379e-01f, 9.980604961e-01f,
        9.981778976e-01f, 9.982881987e-01f, 9.983918281e-01f, 9.984891887e-01f, 9.985806592e-01f, 9.986665954e-01f, 9.987473317e-01f, 9.988231824e-01f,
        9.988944427e-01f, 9.989613903e-01f, 9.990242858e-01f, 9.990833742e-01f, 9.991388858e-01f, 9.991910370e-01f, 9.992400310e-01f, 9.992860587e-01f,
        9.993292997e-01f, 9.993699226e-01f, 9.994080858e-01f, 9.994439381e-01f, 9.994776194e-01f, 9.995092610e-01f, 9.995389866e-01f, 9.995669119e-01f,
        9.995931460e-01f, 9.996177914e-01f, 9.996409441e-01f, 9.996626945e-01f, 9.996831276e-01f, 9.997023230e-01f, 9.997203558e-01f, 9.997372964e-01f,
        9.997532108e-01f, 9.997681613e-01f, 9.997822062e-01f, 9.997954003e-01f, 9.998077952e-01f, 9.998194392e-01f, 9.998303779e-01f, 9.998406540e-01f,
        9.998503075e-01f, 9.998593763e-01f, 9.998678957e-01f, 9.998758990e-01f, 9.998834175e-01f, 9.998904805e-01f, 9.998971156e-01f, 9.999033487e-01f,
        9.999092043e-01f, 9.999147051e-01f, 9.999198726e-01f, 9.999247271e-01f, 9.999292875e-01f, 9.999335716e-01f, 9.999375962e-01f, 9.999413769e-01f,
        9.999449286e-01f, 9.999482651e-01f, 9.999513995e-01f, 9.999543440e-01f, 9.999571101e-01f, 9.999597086e-01f, 9.999621497e-01f, 9.999644429e-01f,
        9.999665972e-01f, 9.999686209e-01f, 9.999705220e-01f, 9.999723080e-01f, 9.999739857e-01f, 9.999755618e-01f, 9.999770425e-01f, 9.999784334e-01f,
        9.999797400e-01f, 9.999809675e-01f, 9.999821206e-01f, 9.999832039e-01f, 9.999842215e-01f, 9.999851774e-01f, 9.999860755e-01f, 9.999869191e-01f,
        9.999877117e-01f, 9.999884562e-01f, 9.999891556e-01f, 9.999898126e-01f, 9.999904298e-01f, 9.999910096e-01f, 9.999915543e-01f, 9.999920660e-01f,
        9.999925467e-01f, 9.999929983e-01f, 9.999934225e-01f, 9.999938210e-01f, 9.999941954e-01f, 9.999945471e-01f, 9.999948774e-01f, 9.999951878e-01f,
        9.999954794e-01f, 9.999957532e-01f, 9.999960105e-01f, 9.999962522e-01f, 9.999964793e-01f, 9.999966926e-01f, 9.999968930e-01f, 9.999970812e-01f,
        9.999972581e-01f, 9.999974242e-01f, 9.999975803e-01f, 9.999977269e-01f, 9.999978646e-01f, 9.999979940e-01f, 9.999981155e-01f, 9.999982297e-01f,
        9.999983369e-01f, 9.999984377e-01f, 9.999985324e-01f, 9.999986213e-01f, 9.999987048e-01f, 9.999987833e-01f, 9.999988570e-01f, 9.999989262e-01f,
        9.999989913e-01f, 9.999990524e-01f, 9.999991098e-01f, 9.999991638e-01f, 9.999992144e-01f, 9.999992620e-01f, 9.999993067e-01f, 9.999993487e-01f,
        9.999993882e-01f, 9.999994253e-01f, 9.999994601e-01f, 9.999994928e-01f, 9.999995235e-01f, 9.999995524e-01f, 9.999995795e-01f, 9.999996050e-01f,
        9.999996289e-01f, 9.999996514e-01f, 9.999996725e-01f, 9.999996924e-01f, 9.999997110e-01f, 9.999997285e-01f, 9.999997450e-01f, 9.999997604e-01f,
        9.999997749e-01f,
    };

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> convBiasGm;
    GlobalTensor<float> addBiasGm;
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
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t poolKernelSize;
    uint32_t blockIdx;
    uint32_t blockNum;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t weightOutStride;
    uint32_t weightInStride;
    uint32_t outputChannelStride;
    uint32_t outputBatchStride;
    float scalingFactor;
};

extern "C" __global__ __aicore__ void conv2d_tanh_scaling_bias_add_max_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR convBias,
    GM_ADDR addBias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dTanhScalingBiasAddMax op;
    op.Init(
        x,
        weight,
        convBias,
        addBias,
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
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.poolKernelSize,
        tiling_data.scalingFactor);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv2d_tanh_scaling_bias_add_max_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &conv_bias,
    const at::Tensor &add_bias,
    double scaling_factor,
    int64_t pool_kernel_size)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D OIHW tensor");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be a 1D tensor");
    TORCH_CHECK(add_bias.dim() == 3, "add_bias must be a 3D tensor shaped [C, 1, 1]");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == conv_bias.size(0), "conv_bias length must match output channels");
    TORCH_CHECK(weight.size(0) == add_bias.size(0), "add_bias channels must match output channels");
    TORCH_CHECK(add_bias.size(1) == 1 && add_bias.size(2) == 1, "add_bias shape must be [C, 1, 1]");
    TORCH_CHECK(pool_kernel_size > 0, "pool_kernel_size must be positive");

    const int64_t convOutputHeight = x.size(2) - weight.size(2) + 1;
    const int64_t convOutputWidth = x.size(3) - weight.size(3) + 1;
    TORCH_CHECK(convOutputHeight >= 0 && convOutputWidth >= 0, "invalid convolution output shape");
    const int64_t outputHeight =
        convOutputHeight < pool_kernel_size ? 0 : (convOutputHeight - pool_kernel_size) / pool_kernel_size + 1;
    const int64_t outputWidth =
        convOutputWidth < pool_kernel_size ? 0 : (convOutputWidth - pool_kernel_size) / pool_kernel_size + 1;

    at::Tensor result = at::empty({x.size(0), weight.size(0), outputHeight, outputWidth}, x.options());
    EXEC_NPU_CMD(
        aclnnConv2dTanhScalingBiasAddMaxCustom,
        x,
        weight,
        conv_bias,
        add_bias,
        scaling_factor,
        pool_kernel_size,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv2d_tanh_scaling_bias_add_max_custom",
        &conv2d_tanh_scaling_bias_add_max_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv2d_tanh_scaling_bias_add_max_custom",
        &conv2d_tanh_scaling_bias_add_max_custom_impl_npu,
        "conv2d_tanh_scaling_bias_add_max_custom");
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
        scaling_factor: float,
        bias_shape,
        pool_kernel_size: int,
    ) -> None:
        super().__init__()
        self.scaling_factor = float(scaling_factor)
        self.pool_kernel_size = int(pool_kernel_size)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = torch.nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv2d_tanh_scaling_bias_add_max_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.bias,
            self.scaling_factor,
            self.pool_kernel_size,
        )
'''
