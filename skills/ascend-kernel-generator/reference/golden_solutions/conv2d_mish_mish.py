project_json_src='''
[
    {
        "op": "Conv2dMishMishCustom",
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
BEGIN_TILING_DATA_DEF(Conv2dMishMishCustomTilingData)
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
    TILING_DATA_FIELD_DEF(uint32_t, dilation);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dMishMishCustom, Conv2dMishMishCustomTilingData)
}
"""

host_operator_src="""
#include "conv2d_mish_mish_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeOutputDim(int64_t input, int64_t kernelSize, int64_t stride, int64_t padding, int64_t dilation)
{
    if (stride <= 0 || dilation <= 0) {
        return 0;
    }
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
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *dilationPtr = attrs->GetAttrPointer<int64_t>(2);
    if (stridePtr == nullptr || paddingPtr == nullptr || dilationPtr == nullptr) {
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

    if (inChannels != weightInChannels || biasLength != outChannels) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t stride = static_cast<uint32_t>(*stridePtr);
    const uint32_t padding = static_cast<uint32_t>(*paddingPtr);
    const uint32_t dilation = static_cast<uint32_t>(*dilationPtr);
    const uint32_t outputHeight = ComputeOutputDim(inputHeight, kernelHeight, stride, padding, dilation);
    const uint32_t outputWidth = ComputeOutputDim(inputWidth, kernelWidth, stride, padding, dilation);

    Conv2dMishMishCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_dilation(dilation);

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
    if (inputShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDim(1) != weightShape->GetDim(1) || biasShape->GetDim(0) != weightShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }
    const int64_t *stridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *paddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *dilationPtr = attrs->GetAttrPointer<int64_t>(2);
    if (stridePtr == nullptr || paddingPtr == nullptr || dilationPtr == nullptr) {
        return GRAPH_FAILED;
    }

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, weightShape->GetDim(0));
    outputShape->SetDim(
        2,
        ComputeOutputDim(inputShape->GetDim(2), weightShape->GetDim(2), *stridePtr, *paddingPtr, *dilationPtr));
    outputShape->SetDim(
        3,
        ComputeOutputDim(inputShape->GetDim(3), weightShape->GetDim(3), *stridePtr, *paddingPtr, *dilationPtr));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv2dMishMishCustom : public OpDef {
public:
    explicit Conv2dMishMishCustom(const char *name) : OpDef(name)
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
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->Attr("padding").AttrType(REQUIRED).Int();
        this->Attr("dilation").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dMishMishCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConv2dMishMish {
public:
    __aicore__ inline KernelConv2dMishMish() {}

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
        uint32_t outputHeight,
        uint32_t outputWidth,
        uint32_t stride,
        uint32_t padding,
        uint32_t dilation)
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
        this->dilation = dilation;
        this->blockIdx = GetBlockIdx();
        this->inputChannelStride = inputHeight * inputWidth;
        this->outputChannelStride = outputHeight * outputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->outputBatchStride = outChannels * this->outputChannelStride;
        this->weightOutStride = inChannels * kernelHeight * kernelWidth;
        this->weightInStride = kernelHeight * kernelWidth;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * this->weightOutStride);
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
        const int32_t inputHeight = static_cast<int32_t>(this->inputHeight);
        const int32_t inputWidth = static_cast<int32_t>(this->inputWidth);

        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            const uint32_t weightBase = outChannel * this->weightOutStride;
            const uint32_t yChannelBase = yBatchBase + outChannel * this->outputChannelStride;
            const float biasValue = biasGm.GetValue(outChannel);
            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                const int32_t startH =
                    static_cast<int32_t>(outH) * static_cast<int32_t>(this->stride) -
                    static_cast<int32_t>(this->padding);
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    const int32_t startW =
                        static_cast<int32_t>(outW) * static_cast<int32_t>(this->stride) -
                        static_cast<int32_t>(this->padding);
                    float sum = biasValue;
                    for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                        const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                        const uint32_t wChannelBase = weightBase + inChannel * this->weightInStride;
                        for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                            const int32_t inH =
                                startH + static_cast<int32_t>(kernelH) * static_cast<int32_t>(this->dilation);
                            if (inH < 0 || inH >= inputHeight) {
                                continue;
                            }
                            for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                const int32_t inW =
                                    startW + static_cast<int32_t>(kernelW) * static_cast<int32_t>(this->dilation);
                                if (inW < 0 || inW >= inputWidth) {
                                    continue;
                                }

                                const uint32_t xOffset =
                                    xChannelBase +
                                    static_cast<uint32_t>(inH) * this->inputWidth +
                                    static_cast<uint32_t>(inW);
                                const uint32_t wOffset =
                                    wChannelBase + kernelH * this->kernelWidth + kernelW;
                                sum += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                            }
                        }
                    }
                    const float mish1 = ApplyMish(sum);
                    const float mish2 = ApplyMish(mish1);
                    yGm.SetValue(yChannelBase + outH * this->outputWidth + outW, mish2);
                }
            }
        }
    }

private:
    __aicore__ inline float ApplyMish(float x) const
    {
        if (x <= -8.0f) {
            return -2.683250883e-03f;
        }
        if (x >= 8.0f) {
            return x;
        }
        constexpr float kStart = -8.0f;
        constexpr float kInvStep = 1.6e+01f;
        const float scaled = (x - kStart) * kInvStep;
        int32_t idx = static_cast<int32_t>(scaled);
        if (idx < 0) {
            idx = 0;
        } else if (idx > 255) {
            idx = 255;
        }
        const float frac = scaled - static_cast<float>(idx);
        const float left = kMishTable[idx];
        const float right = kMishTable[idx + 1];
        return left + (right - left) * frac;
    }

private:
    static constexpr float kMishTable[257] = {
        -2.683250883e-03f, -2.833960146e-03f, -2.992946582e-03f, -3.160650785e-03f, -3.337535725e-03f, -3.524087806e-03f, -3.720817954e-03f, -3.928262756e-03f,
        -4.146985643e-03f, -4.377578112e-03f, -4.620661002e-03f, -4.876885810e-03f, -5.146936059e-03f, -5.431528714e-03f, -5.731415640e-03f, -6.047385119e-03f,
        -6.380263410e-03f, -6.730916353e-03f, -7.100251035e-03f, -7.489217491e-03f, -7.898810457e-03f, -8.330071170e-03f, -8.784089214e-03f, -9.262004398e-03f,
        -9.765008692e-03f, -1.029434818e-02f, -1.085132507e-02f, -1.143729969e-02f, -1.205369258e-02f, -1.270198654e-02f, -1.338372871e-02f, -1.410053266e-02f,
        -1.485408048e-02f, -1.564612485e-02f, -1.647849109e-02f, -1.735307918e-02f, -1.827186571e-02f, -1.923690578e-02f, -2.025033485e-02f, -2.131437041e-02f,
        -2.243131362e-02f, -2.360355070e-02f, -2.483355427e-02f, -2.612388440e-02f, -2.747718941e-02f, -2.889620649e-02f, -3.038376195e-02f, -3.194277109e-02f,
        -3.357623773e-02f, -3.528725327e-02f, -3.707899518e-02f, -3.895472503e-02f, -4.091778571e-02f, -4.297159813e-02f, -4.511965692e-02f, -4.736552537e-02f,
        -4.971282931e-02f, -5.216524991e-02f, -5.472651528e-02f, -5.740039066e-02f, -6.019066719e-02f, -6.310114903e-02f, -6.613563866e-02f, -6.929792024e-02f,
        -7.259174079e-02f, -7.602078904e-02f, -7.958867164e-02f, -8.329888665e-02f, -8.715479388e-02f, -9.115958198e-02f, -9.531623181e-02f, -9.962747606e-02f,
        -1.040957545e-01f, -1.087231647e-01f, -1.135114078e-01f, -1.184617292e-01f, -1.235748535e-01f, -1.288509127e-01f, -1.342893693e-01f, -1.398889315e-01f,
        -1.456474613e-01f, -1.515618749e-01f, -1.576280355e-01f, -1.638406370e-01f, -1.701930794e-01f, -1.766773345e-01f, -1.832838034e-01f, -1.900011628e-01f,
        -1.968162039e-01f, -2.037136597e-01f, -2.106760247e-01f, -2.176833648e-01f, -2.247131192e-01f, -2.317398955e-01f, -2.387352578e-01f, -2.456675111e-01f,
        -2.525014827e-01f, -2.591983037e-01f, -2.657151937e-01f, -2.720052518e-01f, -2.780172592e-01f, -2.836954965e-01f, -2.889795847e-01f, -2.938043527e-01f,
        -2.980997422e-01f, -3.017907556e-01f, -3.047974579e-01f, -3.070350411e-01f, -3.084139611e-01f, -3.088401592e-01f, -3.082153764e-01f, -3.064375730e-01f,
        -3.034014614e-01f, -2.989991594e-01f, -2.931209730e-01f, -2.856563081e-01f, -2.764947145e-01f, -2.655270558e-01f, -2.526467968e-01f, -2.377513952e-01f,
        -2.207437747e-01f, -2.015338554e-01f, -1.800401088e-01f, -1.561910984e-01f, -1.299269637e-01f, -1.012008033e-01f, -6.997990718e-02f, -3.624679280e-02f,
        0.000000000e+00f, 3.874539399e-02f, 7.995757085e-02f, 1.235881000e-01f, 1.695724097e-01f, 2.178305969e-01f, 2.682685597e-01f, 3.207794223e-01f,
        3.752452113e-01f, 4.315387328e-01f, 4.895255911e-01f, 5.490662876e-01f, 6.100183352e-01f, 6.722383283e-01f, 7.355839110e-01f, 7.999155951e-01f,
        8.650983883e-01f, 9.310032025e-01f, 9.975080244e-01f, 1.064498838e+00f, 1.131870304e+00f, 1.199526199e+00f, 1.267379638e+00f, 1.335353098e+00f,
        1.403378266e+00f, 1.471395744e+00f, 1.539354633e+00f, 1.607212023e+00f, 1.674932424e+00f, 1.742487148e+00f, 1.809853669e+00f, 1.877014980e+00f,
        1.943958960e+00f, 2.010677759e+00f, 2.077167223e+00f, 2.143426343e+00f, 2.209456758e+00f, 2.275262292e+00f, 2.340848544e+00f, 2.406222520e+00f,
        2.471392305e+00f, 2.536366779e+00f, 2.601155376e+00f, 2.665767871e+00f, 2.730214204e+00f, 2.794504335e+00f, 2.858648119e+00f, 2.922655216e+00f,
        2.986535005e+00f, 3.050296534e+00f, 3.113948471e+00f, 3.177499074e+00f, 3.240956173e+00f, 3.304327158e+00f, 3.367618977e+00f, 3.430838137e+00f,
        3.493990715e+00f, 3.557082369e+00f, 3.620118352e+00f, 3.683103528e+00f, 3.746042393e+00f, 3.808939091e+00f, 3.871797438e+00f, 3.934620939e+00f,
        3.997412807e+00f, 4.060175986e+00f, 4.122913168e+00f, 4.185626811e+00f, 4.248319155e+00f, 4.310992243e+00f, 4.373647932e+00f, 4.436287912e+00f,
        4.498913715e+00f, 4.561526733e+00f, 4.624128228e+00f, 4.686719341e+00f, 4.749301105e+00f, 4.811874457e+00f, 4.874440240e+00f, 4.936999217e+00f,
        4.999552078e+00f, 5.062099442e+00f, 5.124641870e+00f, 5.187179867e+00f, 5.249713886e+00f, 5.312244337e+00f, 5.374771587e+00f, 5.437295968e+00f,
        5.499817777e+00f, 5.562337281e+00f, 5.624854720e+00f, 5.687370309e+00f, 5.749884243e+00f, 5.812396695e+00f, 5.874907820e+00f, 5.937417758e+00f,
        5.999926634e+00f, 6.062434561e+00f, 6.124941638e+00f, 6.187447957e+00f, 6.249953596e+00f, 6.312458630e+00f, 6.374963121e+00f, 6.437467129e+00f,
        6.499970704e+00f, 6.562473893e+00f, 6.624976737e+00f, 6.687479274e+00f, 6.749981535e+00f, 6.812483552e+00f, 6.874985349e+00f, 6.937486952e+00f,
        6.999988380e+00f, 7.062489653e+00f, 7.124990787e+00f, 7.187491797e+00f, 7.249992697e+00f, 7.312493499e+00f, 7.374994214e+00f, 7.437494850e+00f,
        7.499995417e+00f, 7.562495921e+00f, 7.624996370e+00f, 7.687496770e+00f, 7.749997127e+00f, 7.812497444e+00f, 7.874997726e+00f, 7.937497977e+00f,
        7.999998201e+00f,
    };

private:
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
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
    uint32_t blockIdx;
    uint32_t inputChannelStride;
    uint32_t outputChannelStride;
    uint32_t inputBatchStride;
    uint32_t outputBatchStride;
    uint32_t weightOutStride;
    uint32_t weightInStride;
};

extern "C" __global__ __aicore__ void conv2d_mish_mish_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dMishMish op;
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
        tiling_data.outputHeight,
        tiling_data.outputWidth,
        tiling_data.stride,
        tiling_data.padding,
        tiling_data.dilation);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv2d_mish_mish_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D OIHW tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == bias.size(0), "bias length must match output channels");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation > 0, "dilation must be positive");

    const int64_t effectiveKernelH = dilation * (weight.size(2) - 1) + 1;
    const int64_t effectiveKernelW = dilation * (weight.size(3) - 1) + 1;
    const int64_t outputHeight = (x.size(2) + padding * 2 - effectiveKernelH) / stride + 1;
    const int64_t outputWidth = (x.size(3) + padding * 2 - effectiveKernelW) / stride + 1;
    TORCH_CHECK(outputHeight >= 0 && outputWidth >= 0, "invalid convolution output shape");

    at::Tensor result = at::empty({x.size(0), weight.size(0), outputHeight, outputWidth}, x.options());
    EXEC_NPU_CMD(aclnnConv2dMishMishCustom, x, weight, bias, stride, padding, dilation, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_mish_mish_custom", &conv2d_mish_mish_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_mish_mish_custom", &conv2d_mish_mish_impl_npu, "conv2d_mish_mish_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv2d_mish_mish_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.stride,
            self.padding,
            self.dilation,
        )
'''
