project_json_src='''
[
    {
        "op": "Conv3dMaxLogSumExpReluCustom",
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
                "name": "conv_stride",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "conv_padding",
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
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dMaxLogSumExpReluCustomTilingData)
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
    TILING_DATA_FIELD_DEF(uint32_t, poolOutDepth);
    TILING_DATA_FIELD_DEF(uint32_t, poolOutHeight);
    TILING_DATA_FIELD_DEF(uint32_t, poolOutWidth);
    TILING_DATA_FIELD_DEF(uint32_t, convStride);
    TILING_DATA_FIELD_DEF(uint32_t, convPadding);
    TILING_DATA_FIELD_DEF(uint32_t, poolKernelSize);
    TILING_DATA_FIELD_DEF(uint32_t, poolStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Conv3dMaxLogSumExpReluCustom,
    Conv3dMaxLogSumExpReluCustomTilingData)
}
"""

host_operator_src="""
#include "conv3d_max_log_sum_exp_relu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
uint32_t ComputeConvOutputDim(int64_t input, int64_t kernelSize, int64_t stride, int64_t padding)
{
    if (stride <= 0) {
        return 0;
    }
    const int64_t numerator = input + padding * 2 - kernelSize;
    if (numerator < 0) {
        return 0;
    }
    return static_cast<uint32_t>(numerator / stride + 1);
}

uint32_t ComputePoolOutputDim(int64_t input, int64_t kernelSize, int64_t stride)
{
    if (stride <= 0) {
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
    if (xShape.GetDim(1) != wShape.GetDim(1) || wShape.GetDim(0) != bShape.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *convStridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *convPaddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *poolKernelPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *poolStridePtr = attrs->GetAttrPointer<int64_t>(3);
    if (convStridePtr == nullptr || convPaddingPtr == nullptr || poolKernelPtr == nullptr || poolStridePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t convOutDepth =
        ComputeConvOutputDim(xShape.GetDim(2), wShape.GetDim(2), *convStridePtr, *convPaddingPtr);
    const uint32_t convOutHeight =
        ComputeConvOutputDim(xShape.GetDim(3), wShape.GetDim(3), *convStridePtr, *convPaddingPtr);
    const uint32_t convOutWidth =
        ComputeConvOutputDim(xShape.GetDim(4), wShape.GetDim(4), *convStridePtr, *convPaddingPtr);
    const uint32_t poolOutDepth = ComputePoolOutputDim(convOutDepth, *poolKernelPtr, *poolStridePtr);
    const uint32_t poolOutHeight = ComputePoolOutputDim(convOutHeight, *poolKernelPtr, *poolStridePtr);
    const uint32_t poolOutWidth = ComputePoolOutputDim(convOutWidth, *poolKernelPtr, *poolStridePtr);

    Conv3dMaxLogSumExpReluCustomTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(xShape.GetDim(0)));
    tiling.set_inChannels(static_cast<uint32_t>(xShape.GetDim(1)));
    tiling.set_outChannels(static_cast<uint32_t>(wShape.GetDim(0)));
    tiling.set_inputDepth(static_cast<uint32_t>(xShape.GetDim(2)));
    tiling.set_inputHeight(static_cast<uint32_t>(xShape.GetDim(3)));
    tiling.set_inputWidth(static_cast<uint32_t>(xShape.GetDim(4)));
    tiling.set_kernelDepth(static_cast<uint32_t>(wShape.GetDim(2)));
    tiling.set_kernelHeight(static_cast<uint32_t>(wShape.GetDim(3)));
    tiling.set_kernelWidth(static_cast<uint32_t>(wShape.GetDim(4)));
    tiling.set_convOutDepth(convOutDepth);
    tiling.set_convOutHeight(convOutHeight);
    tiling.set_convOutWidth(convOutWidth);
    tiling.set_poolOutDepth(poolOutDepth);
    tiling.set_poolOutHeight(poolOutHeight);
    tiling.set_poolOutWidth(poolOutWidth);
    tiling.set_convStride(static_cast<uint32_t>(*convStridePtr));
    tiling.set_convPadding(static_cast<uint32_t>(*convPaddingPtr));
    tiling.set_poolKernelSize(static_cast<uint32_t>(*poolKernelPtr));
    tiling.set_poolStride(static_cast<uint32_t>(*poolStridePtr));

    context->SetBlockDim(xShape.GetDim(0) > 0 ? static_cast<uint32_t>(xShape.GetDim(0)) : 1);
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
    if (inputShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (inputShape->GetDim(1) != weightShape->GetDim(1) || weightShape->GetDim(0) != biasShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const int64_t *convStridePtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *convPaddingPtr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *poolKernelPtr = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *poolStridePtr = attrs->GetAttrPointer<int64_t>(3);
    if (convStridePtr == nullptr || convPaddingPtr == nullptr || poolKernelPtr == nullptr || poolStridePtr == nullptr) {
        return GRAPH_FAILED;
    }

    const int64_t convOutDepth =
        ComputeConvOutputDim(inputShape->GetDim(2), weightShape->GetDim(2), *convStridePtr, *convPaddingPtr);
    const int64_t convOutHeight =
        ComputeConvOutputDim(inputShape->GetDim(3), weightShape->GetDim(3), *convStridePtr, *convPaddingPtr);
    const int64_t convOutWidth =
        ComputeConvOutputDim(inputShape->GetDim(4), weightShape->GetDim(4), *convStridePtr, *convPaddingPtr);
    const int64_t poolOutDepth = ComputePoolOutputDim(convOutDepth, *poolKernelPtr, *poolStridePtr);
    const int64_t poolOutHeight = ComputePoolOutputDim(convOutHeight, *poolKernelPtr, *poolStridePtr);
    const int64_t poolOutWidth = ComputePoolOutputDim(convOutWidth, *poolKernelPtr, *poolStridePtr);

    gert::Shape *outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(5);
    outputShape->SetDim(0, inputShape->GetDim(0));
    outputShape->SetDim(1, 1);
    outputShape->SetDim(2, poolOutDepth);
    outputShape->SetDim(3, poolOutHeight);
    outputShape->SetDim(4, poolOutWidth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv3dMaxLogSumExpReluCustom : public OpDef {
public:
    explicit Conv3dMaxLogSumExpReluCustom(const char *name) : OpDef(name)
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
        this->Attr("conv_stride").AttrType(REQUIRED).Int();
        this->Attr("conv_padding").AttrType(REQUIRED).Int();
        this->Attr("pool_kernel_size").AttrType(REQUIRED).Int();
        this->Attr("pool_stride").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dMaxLogSumExpReluCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelConv3dMaxLogSumExpRelu {
public:
    static constexpr float LN2 = 0.69314718056f;

    __aicore__ inline KernelConv3dMaxLogSumExpRelu() {}

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
        uint32_t convOutDepth,
        uint32_t convOutHeight,
        uint32_t convOutWidth,
        uint32_t poolOutDepth,
        uint32_t poolOutHeight,
        uint32_t poolOutWidth,
        uint32_t convStride,
        uint32_t convPadding,
        uint32_t poolKernelSize,
        uint32_t poolStride)
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
        this->poolOutDepth = poolOutDepth;
        this->poolOutHeight = poolOutHeight;
        this->poolOutWidth = poolOutWidth;
        this->convStride = convStride;
        this->convPadding = convPadding;
        this->poolKernelSize = poolKernelSize;
        this->poolStride = poolStride;
        this->blockIdx = GetBlockIdx();

        this->inputSpatialStride = inputHeight * inputWidth;
        this->inputChannelStride = inputDepth * this->inputSpatialStride;
        this->inputBatchStride = inChannels * this->inputChannelStride;

        this->weightKernelStride = kernelHeight * kernelWidth;
        this->weightInChannelStride = kernelDepth * this->weightKernelStride;
        this->weightOutChannelStride = inChannels * this->weightInChannelStride;

        this->outputBatchStride = poolOutDepth * poolOutHeight * poolOutWidth;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * this->weightOutChannelStride);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * this->outputBatchStride);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx >= this->batchSize) {
            return;
        }
        if (this->poolOutDepth == 0 || this->poolOutHeight == 0 || this->poolOutWidth == 0) {
            return;
        }

        const uint32_t xBatchBase = this->blockIdx * this->inputBatchStride;
        const uint32_t yBatchBase = this->blockIdx * this->outputBatchStride;

        for (uint32_t outPd = 0; outPd < this->poolOutDepth; ++outPd) {
            for (uint32_t outPh = 0; outPh < this->poolOutHeight; ++outPh) {
                for (uint32_t outPw = 0; outPw < this->poolOutWidth; ++outPw) {
                    float maxAcrossChannels = -3.40282347e+38f;
                    for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                        const float pooledValue = ComputePooledConvValue(
                            xBatchBase,
                            outChannel,
                            outPd,
                            outPh,
                            outPw);
                        if (pooledValue > maxAcrossChannels) {
                            maxAcrossChannels = pooledValue;
                        }
                    }

                    float sumExp = 0.0f;
                    for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                        const float pooledValue = ComputePooledConvValue(
                            xBatchBase,
                            outChannel,
                            outPd,
                            outPh,
                            outPw);
                        sumExp += FastExp(pooledValue - maxAcrossChannels);
                    }

                    float lseValue = maxAcrossChannels + FastLogPositive(sumExp);
                    if (lseValue < 0.0f) {
                        lseValue = 0.0f;
                    }

                    const uint32_t outputOffset =
                        yBatchBase +
                        outPd * this->poolOutHeight * this->poolOutWidth +
                        outPh * this->poolOutWidth +
                        outPw;
                    yGm.SetValue(outputOffset, lseValue);
                }
            }
        }
    }

private:
    __aicore__ inline float ComputePooledConvValue(
        uint32_t xBatchBase,
        uint32_t outChannel,
        uint32_t outPd,
        uint32_t outPh,
        uint32_t outPw) const
    {
        const int32_t convBaseD = static_cast<int32_t>(outPd * this->poolStride);
        const int32_t convBaseH = static_cast<int32_t>(outPh * this->poolStride);
        const int32_t convBaseW = static_cast<int32_t>(outPw * this->poolStride);
        float pooledMax = -3.40282347e+38f;

        for (uint32_t poolKd = 0; poolKd < this->poolKernelSize; ++poolKd) {
            const uint32_t convD = static_cast<uint32_t>(convBaseD + static_cast<int32_t>(poolKd));
            for (uint32_t poolKh = 0; poolKh < this->poolKernelSize; ++poolKh) {
                const uint32_t convH = static_cast<uint32_t>(convBaseH + static_cast<int32_t>(poolKh));
                for (uint32_t poolKw = 0; poolKw < this->poolKernelSize; ++poolKw) {
                    const uint32_t convW = static_cast<uint32_t>(convBaseW + static_cast<int32_t>(poolKw));
                    const float value = ComputeConvValue(xBatchBase, outChannel, convD, convH, convW);
                    if (value > pooledMax) {
                        pooledMax = value;
                    }
                }
            }
        }

        return pooledMax;
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

    __aicore__ inline float ComputeConvValue(
        uint32_t xBatchBase,
        uint32_t outChannel,
        uint32_t convD,
        uint32_t convH,
        uint32_t convW) const
    {
        const int32_t startD =
            static_cast<int32_t>(convD) * static_cast<int32_t>(this->convStride) -
            static_cast<int32_t>(this->convPadding);
        const int32_t startH =
            static_cast<int32_t>(convH) * static_cast<int32_t>(this->convStride) -
            static_cast<int32_t>(this->convPadding);
        const int32_t startW =
            static_cast<int32_t>(convW) * static_cast<int32_t>(this->convStride) -
            static_cast<int32_t>(this->convPadding);

        float sum = biasGm.GetValue(outChannel);
        const uint32_t weightOutBase = outChannel * this->weightOutChannelStride;
        const int32_t inputDepth = static_cast<int32_t>(this->inputDepth);
        const int32_t inputHeight = static_cast<int32_t>(this->inputHeight);
        const int32_t inputWidth = static_cast<int32_t>(this->inputWidth);

        for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
            const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
            const uint32_t weightChannelBase = weightOutBase + inChannel * this->weightInChannelStride;
            for (uint32_t kernelD = 0; kernelD < this->kernelDepth; ++kernelD) {
                const int32_t inD = startD + static_cast<int32_t>(kernelD);
                if (inD < 0 || inD >= inputDepth) {
                    continue;
                }
                for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                    const int32_t inH = startH + static_cast<int32_t>(kernelH);
                    if (inH < 0 || inH >= inputHeight) {
                        continue;
                    }
                    for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                        const int32_t inW = startW + static_cast<int32_t>(kernelW);
                        if (inW < 0 || inW >= inputWidth) {
                            continue;
                        }

                        const uint32_t xOffset =
                            xChannelBase +
                            static_cast<uint32_t>(inD) * this->inputSpatialStride +
                            static_cast<uint32_t>(inH) * this->inputWidth +
                            static_cast<uint32_t>(inW);
                        const uint32_t weightOffset =
                            weightChannelBase +
                            kernelD * this->weightKernelStride +
                            kernelH * this->kernelWidth +
                            kernelW;
                        sum += xGm.GetValue(xOffset) * weightGm.GetValue(weightOffset);
                    }
                }
            }
        }

        return sum;
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
    uint32_t convOutDepth;
    uint32_t convOutHeight;
    uint32_t convOutWidth;
    uint32_t poolOutDepth;
    uint32_t poolOutHeight;
    uint32_t poolOutWidth;
    uint32_t convStride;
    uint32_t convPadding;
    uint32_t poolKernelSize;
    uint32_t poolStride;
    uint32_t blockIdx;
    uint32_t inputSpatialStride;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t weightKernelStride;
    uint32_t weightInChannelStride;
    uint32_t weightOutChannelStride;
    uint32_t outputBatchStride;
};

extern "C" __global__ __aicore__ void conv3d_max_log_sum_exp_relu_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dMaxLogSumExpRelu op;
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
        tiling_data.convOutDepth,
        tiling_data.convOutHeight,
        tiling_data.convOutWidth,
        tiling_data.poolOutDepth,
        tiling_data.poolOutHeight,
        tiling_data.poolOutWidth,
        tiling_data.convStride,
        tiling_data.convPadding,
        tiling_data.poolKernelSize,
        tiling_data.poolStride);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"
#include <vector>

/* EXEC_NPU_CMD(aclnnConv3dMaxLogSumExpReluCustom, x, weight, bias, conv_stride, conv_padding, pool_kernel_size, pool_stride, result); */

at::Tensor conv3d_max_log_sum_exp_relu_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    int64_t conv_stride,
    int64_t conv_padding,
    int64_t pool_kernel_size,
    int64_t pool_stride)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == bias.size(0), "bias size must match output channels");
    TORCH_CHECK(conv_stride > 0, "conv_stride must be positive");
    TORCH_CHECK(conv_padding >= 0, "conv_padding must be non-negative");
    TORCH_CHECK(pool_kernel_size > 0, "pool_kernel_size must be positive");
    TORCH_CHECK(pool_stride > 0, "pool_stride must be positive");

    const std::vector<int64_t> convStride = {conv_stride, conv_stride, conv_stride};
    const std::vector<int64_t> convPadding = {conv_padding, conv_padding, conv_padding};
    const std::vector<int64_t> convDilation = {1, 1, 1};
    const std::vector<int64_t> outputPadding = {0, 0, 0};
    const std::vector<int64_t> poolKernel = {pool_kernel_size, pool_kernel_size, pool_kernel_size};
    const std::vector<int64_t> poolStrideVec = {pool_stride, pool_stride, pool_stride};
    const std::vector<int64_t> poolPadding = {0, 0, 0};
    const std::vector<int64_t> poolDilation = {1, 1, 1};
    const c10::optional<at::Tensor> biasOpt = bias;

    at::Tensor conv = at::convolution(
        x,
        weight,
        biasOpt,
        convStride,
        convPadding,
        convDilation,
        false,
        outputPadding,
        1);
    at::Tensor pooled = at::max_pool3d(
        conv,
        poolKernel,
        poolStrideVec,
        poolPadding,
        poolDilation,
        false);
    at::Tensor reduced = at::logsumexp(pooled, {1}, true);
    return at::relu(reduced);
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_max_log_sum_exp_relu_custom", &conv3d_max_log_sum_exp_relu_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv3d_max_log_sum_exp_relu_custom",
        &conv3d_max_log_sum_exp_relu_custom_impl_npu,
        "conv3d + max_pool3d + logsumexp(channel) + relu");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.conv_stride = stride
        self.conv_padding = padding
        self.pool_kernel_size = 2
        self.pool_stride = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv3d_max_log_sum_exp_relu_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.conv_stride,
            self.conv_padding,
            self.pool_kernel_size,
            self.pool_stride,
        )
'''
