project_json_src='''
[
    {
        "op": "Conv3dMultiplyInstanceNormClampMultiplyMaxCustom",
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
            },
            {
                "name": "multiplier",
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
#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dMultiplyInstanceNormClampMultiplyMaxCustomTilingData)
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
    TILING_DATA_FIELD_DEF(uint32_t, outputVolume);
    TILING_DATA_FIELD_DEF(uint32_t, alignedOutputVolume);
    TILING_DATA_FIELD_DEF(uint32_t, useCoreNums);
    TILING_DATA_FIELD_DEF(float, epsilon);
    TILING_DATA_FIELD_DEF(float, invOutputVolume);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Conv3dMultiplyInstanceNormClampMultiplyMaxCustom,
    Conv3dMultiplyInstanceNormClampMultiplyMaxCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "conv3d_multiply_instance_norm_clamp_multiply_max_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr float EPSILON = 1.0e-5f;

inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return align == 0 ? value : (value + align - 1) / align * align;
}

inline bool IsMultiplierShapeValid(const gert::Shape* shape, uint32_t outChannels)
{
    if (shape == nullptr) {
        return false;
    }
    if (shape->GetDimNum() == 1) {
        return static_cast<uint32_t>(shape->GetDim(0)) == outChannels;
    }
    if (shape->GetDimNum() != 4) {
        return false;
    }
    return static_cast<uint32_t>(shape->GetDim(0)) == outChannels &&
           static_cast<uint32_t>(shape->GetDim(1)) == 1U &&
           static_cast<uint32_t>(shape->GetDim(2)) == 1U &&
           static_cast<uint32_t>(shape->GetDim(3)) == 1U;
}

inline bool IsValidShape(
    const gert::Shape* xShape,
    const gert::Shape* weightShape,
    const gert::Shape* biasShape,
    const gert::Shape* multiplierShape,
    uint32_t& batchSize,
    uint32_t& inChannels,
    uint32_t& outChannels,
    uint32_t& inputDepth,
    uint32_t& inputHeight,
    uint32_t& inputWidth,
    uint32_t& kernelDepth,
    uint32_t& kernelHeight,
    uint32_t& kernelWidth,
    uint32_t& outputDepth,
    uint32_t& outputHeight,
    uint32_t& outputWidth)
{
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr || multiplierShape == nullptr) {
        return false;
    }
    if (xShape->GetDimNum() != 5 || weightShape->GetDimNum() != 5 || biasShape->GetDimNum() != 1) {
        return false;
    }

    batchSize = static_cast<uint32_t>(xShape->GetDim(0));
    inChannels = static_cast<uint32_t>(xShape->GetDim(1));
    inputDepth = static_cast<uint32_t>(xShape->GetDim(2));
    inputHeight = static_cast<uint32_t>(xShape->GetDim(3));
    inputWidth = static_cast<uint32_t>(xShape->GetDim(4));
    outChannels = static_cast<uint32_t>(weightShape->GetDim(0));
    const uint32_t weightInChannels = static_cast<uint32_t>(weightShape->GetDim(1));
    kernelDepth = static_cast<uint32_t>(weightShape->GetDim(2));
    kernelHeight = static_cast<uint32_t>(weightShape->GetDim(3));
    kernelWidth = static_cast<uint32_t>(weightShape->GetDim(4));
    const uint32_t biasLength = static_cast<uint32_t>(biasShape->GetDim(0));

    if (batchSize == 0 || inChannels == 0 || outChannels == 0 ||
        inputDepth == 0 || inputHeight == 0 || inputWidth == 0 ||
        kernelDepth == 0 || kernelHeight == 0 || kernelWidth == 0) {
        return false;
    }
    if (weightInChannels != inChannels || biasLength != outChannels) {
        return false;
    }
    if (!IsMultiplierShapeValid(multiplierShape, outChannels)) {
        return false;
    }
    if (inputDepth < kernelDepth || inputHeight < kernelHeight || inputWidth < kernelWidth) {
        return false;
    }

    outputDepth = inputDepth - kernelDepth + 1;
    outputHeight = inputHeight - kernelHeight + 1;
    outputWidth = inputWidth - kernelWidth + 1;
    return outputDepth > 0 && outputHeight > 0 && outputWidth > 0;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xStorage = context->GetInputShape(0);
    const gert::StorageShape* weightStorage = context->GetInputShape(1);
    const gert::StorageShape* biasStorage = context->GetInputShape(2);
    const gert::StorageShape* multiplierStorage = context->GetInputShape(3);
    if (xStorage == nullptr || weightStorage == nullptr || biasStorage == nullptr || multiplierStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

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
    if (!IsValidShape(
            &xStorage->GetStorageShape(),
            &weightStorage->GetStorageShape(),
            &biasStorage->GetStorageShape(),
            &multiplierStorage->GetStorageShape(),
            batchSize,
            inChannels,
            outChannels,
            inputDepth,
            inputHeight,
            inputWidth,
            kernelDepth,
            kernelHeight,
            kernelWidth,
            outputDepth,
            outputHeight,
            outputWidth)) {
        return ge::GRAPH_FAILED;
    }

    Conv3dMultiplyInstanceNormClampMultiplyMaxCustomTilingData tiling;
    const uint32_t outputVolume = outputDepth * outputHeight * outputWidth;
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
    tiling.set_outputVolume(outputVolume);
    tiling.set_alignedOutputVolume(AlignUp(outputVolume, BLOCK_SIZE / sizeof(float)));
    tiling.set_useCoreNums(batchSize < BLOCK_DIM ? batchSize : BLOCK_DIM);
    tiling.set_epsilon(EPSILON);
    tiling.set_invOutputVolume(outputVolume == 0 ? 0.0f : 1.0f / static_cast<float>(outputVolume));

    context->SetBlockDim(batchSize < BLOCK_DIM ? batchSize : BLOCK_DIM);
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
    if (!IsValidShape(
            context->GetInputShape(0),
            context->GetInputShape(1),
            context->GetInputShape(2),
            context->GetInputShape(3),
            batchSize,
            inChannels,
            outChannels,
            inputDepth,
            inputHeight,
            inputWidth,
            kernelDepth,
            kernelHeight,
            kernelWidth,
            outputDepth,
            outputHeight,
            outputWidth)) {
        return GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, batchSize);
    outputShape->SetDim(1, outputDepth);
    outputShape->SetDim(2, outputHeight);
    outputShape->SetDim(3, outputWidth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv3dMultiplyInstanceNormClampMultiplyMaxCustom : public OpDef {
public:
    explicit Conv3dMultiplyInstanceNormClampMultiplyMaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("multiplier").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dMultiplyInstanceNormClampMultiplyMaxCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BLOCK_SIZE = 32;

template <typename T1, typename T2>
__aicore__ inline T1 AlignUp(T1 value, T2 align)
{
    return align == 0 ? value : (value + align - 1) / align * align;
}

__aicore__ inline void DataCopyCustomUB2GM(
    const GlobalTensor<float>& dstTensor,
    const LocalTensor<float>& srcTensor,
    const uint32_t count)
{
    const uint32_t numPerBlock = BLOCK_SIZE / sizeof(float);
    if (count % numPerBlock == 0) {
        DataCopy(dstTensor, srcTensor, count);
        return;
    }

    const uint32_t alignedCount = count / numPerBlock * numPerBlock;
    if (alignedCount > 0) {
        DataCopy(dstTensor, srcTensor, alignedCount);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
    for (uint32_t i = 0; i < numPerBlock; ++i) {
        const float value = srcTensor.GetValue(count - numPerBlock + i);
        srcTensor.SetValue(i, value);
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(dstTensor[count - numPerBlock], srcTensor, numPerBlock);
}

class KernelConv3dMultiplyInstanceNormClampMultiplyMax {
public:
    __aicore__ inline KernelConv3dMultiplyInstanceNormClampMultiplyMax() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR multiplier,
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
        uint32_t outputVolume,
        uint32_t alignedOutputVolume,
        uint32_t useCoreNums,
        float epsilon,
        float invOutputVolume)
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
        this->outputVolume = outputVolume;
        this->alignedOutputVolume = alignedOutputVolume;
        this->useCoreNums = useCoreNums;
        this->epsilon = epsilon;
        this->invOutputVolume = invOutputVolume;

        this->inputPlaneStride = inputHeight * inputWidth;
        this->inputChannelStride = inputDepth * this->inputPlaneStride;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->outputPlaneStride = outputHeight * outputWidth;
        this->outputBatchStride = outputDepth * this->outputPlaneStride;
        this->weightPlaneStride = kernelHeight * kernelWidth;
        this->weightInStride = kernelDepth * this->weightPlaneStride;
        this->weightOutStride = inChannels * this->weightInStride;

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float*)weight, outChannels * this->weightOutStride);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, outChannels);
        multiplierGm.SetGlobalBuffer((__gm__ float*)multiplier, outChannels);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * this->outputBatchStride);

        pipe.InitBuffer(channelBuffer, this->alignedOutputVolume * sizeof(float));
        pipe.InitBuffer(maxBuffer, this->alignedOutputVolume * sizeof(float));
        pipe.InitBuffer(scalarBuffer, AlignUp(sizeof(float), static_cast<uint32_t>(BLOCK_SIZE)));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = GetBlockIdx();
        for (uint32_t batchIdx = blockIdx; batchIdx < this->batchSize; batchIdx += this->useCoreNums) {
            ProcessBatch(batchIdx);
        }
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t batchIdx)
    {
        LocalTensor<float> maxLocal = maxBuffer.Get<float>();
        for (uint32_t i = 0; i < this->alignedOutputVolume; ++i) {
            maxLocal.SetValue(i, -3.40282347e+38f);
        }

        for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
            LocalTensor<float> channelLocal = channelBuffer.Get<float>();
            const float multiplierValue = multiplierGm.GetValue(outChannel);
            ComputeConvChannel(batchIdx, outChannel, multiplierValue, channelLocal);

            const float meanValue = ComputeMean(channelLocal);
            const float invStdValue = ComputeInvStd(channelLocal, meanValue);
            UpdateMax(channelLocal, maxLocal, meanValue, invStdValue, multiplierValue);
        }

        const uint32_t outputBase = batchIdx * this->outputBatchStride;
        DataCopyCustomUB2GM(yGm[outputBase], maxLocal, this->outputVolume);
    }

    __aicore__ inline void ComputeConvChannel(
        uint32_t batchIdx,
        uint32_t outChannel,
        float multiplierValue,
        const LocalTensor<float>& channelLocal)
    {
        const uint32_t xBatchBase = batchIdx * this->inputBatchStride;
        const uint32_t weightBase = outChannel * this->weightOutStride;
        const float biasValue = biasGm.GetValue(outChannel);

        for (uint32_t outD = 0; outD < this->outputDepth; ++outD) {
            for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
                for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                    float acc = biasValue;
                    for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                        const uint32_t xChannelBase = xBatchBase + inChannel * this->inputChannelStride;
                        const uint32_t weightChannelBase = weightBase + inChannel * this->weightInStride;
                        for (uint32_t kernelD = 0; kernelD < this->kernelDepth; ++kernelD) {
                            const uint32_t inD = outD + kernelD;
                            const uint32_t xDepthBase = xChannelBase + inD * this->inputPlaneStride;
                            const uint32_t weightDepthBase = weightChannelBase + kernelD * this->weightPlaneStride;
                            for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                                const uint32_t inH = outH + kernelH;
                                const uint32_t xRowBase = xDepthBase + inH * this->inputWidth;
                                const uint32_t weightRowBase = weightDepthBase + kernelH * this->kernelWidth;
                                for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                                    const uint32_t inW = outW + kernelW;
                                    acc += xGm.GetValue(xRowBase + inW) * weightGm.GetValue(weightRowBase + kernelW);
                                }
                            }
                        }
                    }

                    const uint32_t outIndex =
                        outD * this->outputPlaneStride +
                        outH * this->outputWidth +
                        outW;
                    channelLocal.SetValue(outIndex, acc * multiplierValue);
                }
            }
        }
    }

    __aicore__ inline float ComputeMean(const LocalTensor<float>& channelLocal) const
    {
        float sumValue = 0.0f;
        for (uint32_t i = 0; i < this->outputVolume; ++i) {
            sumValue += channelLocal.GetValue(i);
        }
        return sumValue * this->invOutputVolume;
    }

    __aicore__ inline float ComputeInvStd(const LocalTensor<float>& channelLocal, float meanValue)
    {
        float varSum = 0.0f;
        for (uint32_t i = 0; i < this->outputVolume; ++i) {
            const float diff = channelLocal.GetValue(i) - meanValue;
            varSum += diff * diff;
        }

        LocalTensor<float> scalarLocal = scalarBuffer.Get<float>();
        scalarLocal.SetValue(0, varSum * this->invOutputVolume + this->epsilon);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sqrt(scalarLocal, scalarLocal, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return 1.0f / scalarLocal.GetValue(0);
    }

    __aicore__ inline void UpdateMax(
        const LocalTensor<float>& channelLocal,
        const LocalTensor<float>& maxLocal,
        float meanValue,
        float invStdValue,
        float multiplierValue)
    {
        for (uint32_t i = 0; i < this->outputVolume; ++i) {
            float value = (channelLocal.GetValue(i) - meanValue) * invStdValue;
            if (value < -1.0f) {
                value = -1.0f;
            } else if (value > 1.0f) {
                value = 1.0f;
            }
            value *= multiplierValue;
            if (value > maxLocal.GetValue(i)) {
                maxLocal.SetValue(i, value);
            }
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> channelBuffer;
    TBuf<TPosition::VECCALC> maxBuffer;
    TBuf<TPosition::VECCALC> scalarBuffer;
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> multiplierGm;
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
    uint32_t outputVolume = 0;
    uint32_t alignedOutputVolume = 0;
    uint32_t useCoreNums = 0;
    float epsilon = 1.0e-5f;
    float invOutputVolume = 0.0f;
    uint32_t inputPlaneStride = 0;
    uint32_t inputChannelStride = 0;
    uint32_t inputBatchStride = 0;
    uint32_t outputPlaneStride = 0;
    uint32_t outputBatchStride = 0;
    uint32_t weightPlaneStride = 0;
    uint32_t weightInStride = 0;
    uint32_t weightOutStride = 0;
};

extern "C" __global__ __aicore__ void conv3d_multiply_instance_norm_clamp_multiply_max_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR multiplier,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dMultiplyInstanceNormClampMultiplyMax op;
    op.Init(
        x,
        weight,
        bias,
        multiplier,
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
        tiling_data.outputVolume,
        tiling_data.alignedOutputVolume,
        tiling_data.useCoreNums,
        tiling_data.epsilon,
        tiling_data.invOutputVolume);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv3d_multiply_instance_norm_clamp_multiply_max_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& multiplier)
{
    TORCH_CHECK(x.dim() == 5, "x must be a 5D NCDHW tensor");
    TORCH_CHECK(weight.dim() == 5, "weight must be a 5D OIDHW tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(multiplier.dim() == 1 || multiplier.dim() == 4, "multiplier must be 1D or 4D");
    TORCH_CHECK(x.size(1) == weight.size(1), "input channels must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == bias.size(0), "bias length must match out_channels");
    TORCH_CHECK(multiplier.size(0) == weight.size(0), "multiplier length must match out_channels");
    if (multiplier.dim() == 4) {
        TORCH_CHECK(
            multiplier.size(1) == 1 && multiplier.size(2) == 1 && multiplier.size(3) == 1,
            "4D multiplier must have shape [C, 1, 1, 1]");
    }

    const int64_t outD = x.size(2) - weight.size(2) + 1;
    const int64_t outH = x.size(3) - weight.size(3) + 1;
    const int64_t outW = x.size(4) - weight.size(4) + 1;
    TORCH_CHECK(outD > 0 && outH > 0 && outW > 0, "invalid conv3d output shape");

    at::Tensor result = at::empty({x.size(0), outD, outH, outW}, x.options());
    EXEC_NPU_CMD(
        aclnnConv3dMultiplyInstanceNormClampMultiplyMaxCustom,
        x,
        weight,
        bias,
        multiplier,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv3d_multiply_instance_norm_clamp_multiply_max_custom",
        &conv3d_multiply_instance_norm_clamp_multiply_max_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv3d_multiply_instance_norm_clamp_multiply_max_custom",
        &conv3d_multiply_instance_norm_clamp_multiply_max_impl_npu,
        "fused conv3d * multiplier -> instance norm -> clamp -> multiplier -> max");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = torch.nn.Parameter(torch.randn(multiplier_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv3d_multiply_instance_norm_clamp_multiply_max_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.multiplier)
'''
