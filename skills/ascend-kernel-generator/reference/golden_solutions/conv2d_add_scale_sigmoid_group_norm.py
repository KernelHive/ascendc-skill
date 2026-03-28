project_json_src='''
[
    {
        "op": "Conv2dAddScaleSigmoidGroupNormCustom",
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
            },
            {
                "name": "scale",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "gamma",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "beta",
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
                "name": "num_groups",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "epsilon",
                "param_type": "required",
                "type": "float"
            }
        ]
    }
]
'''

host_tiling_src="""
#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dAddScaleSigmoidGroupNormCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, outputHw);
    TILING_DATA_FIELD_DEF(uint32_t, alignedOutputHw);
    TILING_DATA_FIELD_DEF(uint32_t, numGroups);
    TILING_DATA_FIELD_DEF(uint32_t, channelsPerGroup);
    TILING_DATA_FIELD_DEF(uint32_t, groupElemCount);
    TILING_DATA_FIELD_DEF(uint32_t, alignedGroupBufferCount);
    TILING_DATA_FIELD_DEF(float, epsilon);
    TILING_DATA_FIELD_DEF(float, invGroupElemCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Conv2dAddScaleSigmoidGroupNormCustom,
    Conv2dAddScaleSigmoidGroupNormCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "conv2d_add_scale_sigmoid_group_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t BLOCK_SIZE = 32;

inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return align == 0 ? value : (value + align - 1) / align * align;
}

inline bool IsOneDimVectorWithLength(const gert::Shape* shape, uint32_t expectedLength)
{
    return shape != nullptr &&
        shape->GetDimNum() == 1 &&
        static_cast<uint32_t>(shape->GetDim(0)) == expectedLength;
}

inline bool IsValidShape(
    const gert::Shape* inputShape,
    const gert::Shape* weightShape,
    const gert::Shape* convBiasShape,
    const gert::Shape* addBiasShape,
    const gert::Shape* scaleShape,
    const gert::Shape* gammaShape,
    const gert::Shape* betaShape,
    uint32_t numGroups,
    uint32_t& batchSize,
    uint32_t& inChannels,
    uint32_t& outChannels,
    uint32_t& inputHeight,
    uint32_t& inputWidth,
    uint32_t& kernelHeight,
    uint32_t& kernelWidth,
    uint32_t& outputHeight,
    uint32_t& outputWidth,
    uint32_t& channelsPerGroup)
{
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr ||
        addBiasShape == nullptr || scaleShape == nullptr || gammaShape == nullptr || betaShape == nullptr) {
        return false;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4) {
        return false;
    }

    batchSize = static_cast<uint32_t>(inputShape->GetDim(0));
    inChannels = static_cast<uint32_t>(inputShape->GetDim(1));
    inputHeight = static_cast<uint32_t>(inputShape->GetDim(2));
    inputWidth = static_cast<uint32_t>(inputShape->GetDim(3));
    outChannels = static_cast<uint32_t>(weightShape->GetDim(0));
    const uint32_t weightInChannels = static_cast<uint32_t>(weightShape->GetDim(1));
    kernelHeight = static_cast<uint32_t>(weightShape->GetDim(2));
    kernelWidth = static_cast<uint32_t>(weightShape->GetDim(3));

    if (batchSize == 0 || inChannels == 0 || outChannels == 0 || inputHeight == 0 || inputWidth == 0 ||
        kernelHeight == 0 || kernelWidth == 0 || numGroups == 0) {
        return false;
    }
    if (weightInChannels != inChannels || outChannels % numGroups != 0) {
        return false;
    }
    if (inputHeight < kernelHeight || inputWidth < kernelWidth) {
        return false;
    }
    if (!IsOneDimVectorWithLength(convBiasShape, outChannels) ||
        !IsOneDimVectorWithLength(addBiasShape, outChannels) ||
        !IsOneDimVectorWithLength(scaleShape, outChannels) ||
        !IsOneDimVectorWithLength(gammaShape, outChannels) ||
        !IsOneDimVectorWithLength(betaShape, outChannels)) {
        return false;
    }

    outputHeight = inputHeight - kernelHeight + 1;
    outputWidth = inputWidth - kernelWidth + 1;
    channelsPerGroup = outChannels / numGroups;
    return outputHeight > 0 && outputWidth > 0 && channelsPerGroup > 0;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    const gert::StorageShape* weightStorageShape = context->GetInputShape(1);
    const gert::StorageShape* convBiasStorageShape = context->GetInputShape(2);
    const gert::StorageShape* addBiasStorageShape = context->GetInputShape(3);
    const gert::StorageShape* scaleStorageShape = context->GetInputShape(4);
    const gert::StorageShape* gammaStorageShape = context->GetInputShape(5);
    const gert::StorageShape* betaStorageShape = context->GetInputShape(6);
    if (inputStorageShape == nullptr || weightStorageShape == nullptr || convBiasStorageShape == nullptr ||
        addBiasStorageShape == nullptr || scaleStorageShape == nullptr || gammaStorageShape == nullptr ||
        betaStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t* numGroupsPtr = attrs->GetAttrPointer<int64_t>(0);
    const float* epsilonPtr = attrs->GetAttrPointer<float>(1);
    if (numGroupsPtr == nullptr || epsilonPtr == nullptr || *numGroupsPtr <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t batchSize = 0;
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelHeight = 0;
    uint32_t kernelWidth = 0;
    uint32_t outputHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t channelsPerGroup = 0;
    if (!IsValidShape(
            &inputStorageShape->GetStorageShape(),
            &weightStorageShape->GetStorageShape(),
            &convBiasStorageShape->GetStorageShape(),
            &addBiasStorageShape->GetStorageShape(),
            &scaleStorageShape->GetStorageShape(),
            &gammaStorageShape->GetStorageShape(),
            &betaStorageShape->GetStorageShape(),
            static_cast<uint32_t>(*numGroupsPtr),
            batchSize,
            inChannels,
            outChannels,
            inputHeight,
            inputWidth,
            kernelHeight,
            kernelWidth,
            outputHeight,
            outputWidth,
            channelsPerGroup)) {
        return ge::GRAPH_FAILED;
    }

    Conv2dAddScaleSigmoidGroupNormCustomTilingData tiling;
    const uint32_t outputHw = outputHeight * outputWidth;
    const uint32_t alignedOutputHw = AlignUp(outputHw, BLOCK_SIZE / sizeof(float));
    const uint32_t groupElemCount = channelsPerGroup * outputHw;
    const uint32_t alignedGroupBufferCount = channelsPerGroup * alignedOutputHw;

    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_outputHw(outputHw);
    tiling.set_alignedOutputHw(alignedOutputHw);
    tiling.set_numGroups(static_cast<uint32_t>(*numGroupsPtr));
    tiling.set_channelsPerGroup(channelsPerGroup);
    tiling.set_groupElemCount(groupElemCount);
    tiling.set_alignedGroupBufferCount(alignedGroupBufferCount);
    tiling.set_epsilon(*epsilonPtr);
    tiling.set_invGroupElemCount(groupElemCount == 0 ? 0.0f : 1.0f / static_cast<float>(groupElemCount));

    context->SetBlockDim(batchSize < BLOCK_DIM ? batchSize : BLOCK_DIM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape* convBiasShape = context->GetInputShape(2);
    const gert::Shape* addBiasShape = context->GetInputShape(3);
    const gert::Shape* scaleShape = context->GetInputShape(4);
    const gert::Shape* gammaShape = context->GetInputShape(5);
    const gert::Shape* betaShape = context->GetInputShape(6);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t* numGroupsPtr = attrs->GetAttrPointer<int64_t>(0);
    if (numGroupsPtr == nullptr || *numGroupsPtr <= 0) {
        return GRAPH_FAILED;
    }

    uint32_t batchSize = 0;
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelHeight = 0;
    uint32_t kernelWidth = 0;
    uint32_t outputHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t channelsPerGroup = 0;
    if (!IsValidShape(
            inputShape,
            weightShape,
            convBiasShape,
            addBiasShape,
            scaleShape,
            gammaShape,
            betaShape,
            static_cast<uint32_t>(*numGroupsPtr),
            batchSize,
            inChannels,
            outChannels,
            inputHeight,
            inputWidth,
            kernelHeight,
            kernelWidth,
            outputHeight,
            outputWidth,
            channelsPerGroup)) {
        return GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, batchSize);
    outputShape->SetDim(1, outChannels);
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
class Conv2dAddScaleSigmoidGroupNormCustom : public OpDef {
public:
    explicit Conv2dAddScaleSigmoidGroupNormCustom(const char* name) : OpDef(name)
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
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("num_groups").AttrType(REQUIRED).Int();
        this->Attr("epsilon").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dAddScaleSigmoidGroupNormCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;

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

class KernelConv2dAddScaleSigmoidGroupNormCustom {
public:
    __aicore__ inline KernelConv2dAddScaleSigmoidGroupNormCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR convBias,
        GM_ADDR addBias,
        GM_ADDR scale,
        GM_ADDR gamma,
        GM_ADDR beta,
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
        uint32_t outputHw,
        uint32_t alignedOutputHw,
        uint32_t numGroups,
        uint32_t channelsPerGroup,
        uint32_t groupElemCount,
        uint32_t alignedGroupBufferCount,
        float epsilon,
        float invGroupElemCount)
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
        this->outputHw = outputHw;
        this->alignedOutputHw = alignedOutputHw;
        this->numGroups = numGroups;
        this->channelsPerGroup = channelsPerGroup;
        this->groupElemCount = groupElemCount;
        this->alignedGroupBufferCount = alignedGroupBufferCount;
        this->epsilon = epsilon;
        this->invGroupElemCount = invGroupElemCount;
        this->blockDim = GetBlockNum();

        this->inputChannelStride = inputHeight * inputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->weightOutStride = inChannels * kernelHeight * kernelWidth;
        this->weightInStride = kernelHeight * kernelWidth;
        this->outputChannelStride = outputHw;
        this->outputBatchStride = outChannels * outputHw;

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float*)weight, outChannels * this->weightOutStride);
        convBiasGm.SetGlobalBuffer((__gm__ float*)convBias, outChannels);
        addBiasGm.SetGlobalBuffer((__gm__ float*)addBias, outChannels);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale, outChannels);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, outChannels);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, outChannels);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * this->outputBatchStride);

        pipe.InitBuffer(calcBuf, this->alignedGroupBufferCount * 2 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->groupElemCount == 0) {
            return;
        }

        for (uint32_t batchIdx = GetBlockIdx(); batchIdx < this->batchSize; batchIdx += this->blockDim) {
            for (uint32_t groupIdx = 0; groupIdx < this->numGroups; ++groupIdx) {
                LocalTensor<float> groupLocal = calcBuf.Get<float>();
                LocalTensor<float> tmp1 = calcBuf.Get<float>()[this->alignedGroupBufferCount];

                ComputeGroup(batchIdx, groupIdx, groupLocal, tmp1);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const float meanValue = ComputeMean(groupLocal);
                const float invStdValue = ComputeInvStd(groupLocal, tmp1, meanValue);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                NormalizeAndStore(batchIdx, groupIdx, groupLocal, tmp1, meanValue, invStdValue);
            }
        }
    }

private:
    __aicore__ inline void ComputeGroup(
        uint32_t batchIdx,
        uint32_t groupIdx,
        LocalTensor<float>& groupLocal,
        LocalTensor<float>& tmpLocal)
    {
        const uint32_t channelStart = groupIdx * this->channelsPerGroup;
        for (uint32_t channelOffset = 0; channelOffset < this->channelsPerGroup; ++channelOffset) {
            const uint32_t outChannel = channelStart + channelOffset;
            LocalTensor<float> channelLocal = groupLocal[channelOffset * this->alignedOutputHw];
            LocalTensor<float> channelTmp = tmpLocal[channelOffset * this->alignedOutputHw];
            ComputeConvChannel(batchIdx, outChannel, channelLocal);
            ApplyAddScaleSigmoid(channelLocal, channelTmp, outChannel);
        }
    }

    __aicore__ inline void ComputeConvChannel(
        uint32_t batchIdx,
        uint32_t outChannel,
        LocalTensor<float>& channelLocal)
    {
        const uint32_t inputBatchBase = batchIdx * this->inputBatchStride;
        const uint32_t weightBase = outChannel * this->weightOutStride;
        const float convBiasValue = convBiasGm.GetValue(outChannel);

        uint32_t outIndex = 0;
        for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
            for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                float sum = convBiasValue;
                for (uint32_t inChannel = 0; inChannel < this->inChannels; ++inChannel) {
                    const uint32_t inputChannelBase = inputBatchBase + inChannel * this->inputChannelStride;
                    const uint32_t weightChannelBase = weightBase + inChannel * this->weightInStride;
                    for (uint32_t kernelH = 0; kernelH < this->kernelHeight; ++kernelH) {
                        const uint32_t inputH = outH + kernelH;
                        const uint32_t inputRowBase = inputChannelBase + inputH * this->inputWidth;
                        const uint32_t weightRowBase = weightChannelBase + kernelH * this->kernelWidth;
                        for (uint32_t kernelW = 0; kernelW < this->kernelWidth; ++kernelW) {
                            const uint32_t xOffset = inputRowBase + outW + kernelW;
                            const uint32_t wOffset = weightRowBase + kernelW;
                            sum += xGm.GetValue(xOffset) * weightGm.GetValue(wOffset);
                        }
                    }
                }
                channelLocal.SetValue(outIndex, sum);
                ++outIndex;
            }
        }
    }

    __aicore__ inline void ApplyAddScaleSigmoid(
        LocalTensor<float>& channelLocal,
        LocalTensor<float>& channelTmp,
        uint32_t outChannel)
    {
        const float addBiasValue = addBiasGm.GetValue(outChannel);
        const float scaleValue = scaleGm.GetValue(outChannel);
        Adds(channelLocal, channelLocal, addBiasValue, this->outputHw);
        PipeBarrier<PIPE_V>();
        Muls(channelLocal, channelLocal, scaleValue, this->outputHw);
        PipeBarrier<PIPE_V>();
        Sigmoid(channelTmp, channelLocal, this->outputHw);
        PipeBarrier<PIPE_V>();
        Adds(channelLocal, channelTmp, 0.0f, this->outputHw);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float ComputeMean(LocalTensor<float>& groupLocal)
    {
        float sumValue = 0.0f;
        for (uint32_t channelOffset = 0; channelOffset < this->channelsPerGroup; ++channelOffset) {
            LocalTensor<float> channelLocal = groupLocal[channelOffset * this->alignedOutputHw];
            for (uint32_t i = 0; i < this->outputHw; ++i) {
                sumValue += channelLocal.GetValue(i);
            }
        }
        return sumValue * this->invGroupElemCount;
    }

    __aicore__ inline float ComputeInvStd(
        LocalTensor<float>& groupLocal,
        LocalTensor<float>& tmpLocal,
        float meanValue)
    {
        float squareSumValue = 0.0f;
        for (uint32_t channelOffset = 0; channelOffset < this->channelsPerGroup; ++channelOffset) {
            LocalTensor<float> channelLocal = groupLocal[channelOffset * this->alignedOutputHw];
            for (uint32_t i = 0; i < this->outputHw; ++i) {
                const float centered = channelLocal.GetValue(i) - meanValue;
                squareSumValue += centered * centered;
            }
        }
        const float varianceValue = squareSumValue * this->invGroupElemCount + this->epsilon;
        LocalTensor<float> tmpScalar = tmpLocal;
        tmpScalar.SetValue(0, varianceValue);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sqrt(tmpScalar, tmpScalar, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return 1.0f / tmpScalar.GetValue(0);
    }

    __aicore__ inline void NormalizeAndStore(
        uint32_t batchIdx,
        uint32_t groupIdx,
        LocalTensor<float>& groupLocal,
        LocalTensor<float>& tmpLocal,
        float meanValue,
        float invStdValue)
    {
        const uint32_t channelStart = groupIdx * this->channelsPerGroup;
        for (uint32_t channelOffset = 0; channelOffset < this->channelsPerGroup; ++channelOffset) {
            const uint32_t outChannel = channelStart + channelOffset;
            LocalTensor<float> channelLocal = groupLocal[channelOffset * this->alignedOutputHw];
            LocalTensor<float> channelTmp = tmpLocal[channelOffset * this->alignedOutputHw];
            const float gammaValue = gammaGm.GetValue(outChannel);
            const float betaValue = betaGm.GetValue(outChannel);

            Adds(channelTmp, channelLocal, -meanValue, this->outputHw);
            PipeBarrier<PIPE_V>();
            Muls(channelTmp, channelTmp, invStdValue, this->outputHw);
            PipeBarrier<PIPE_V>();
            Muls(channelTmp, channelTmp, gammaValue, this->outputHw);
            PipeBarrier<PIPE_V>();
            Adds(channelTmp, channelTmp, betaValue, this->outputHw);
            PipeBarrier<PIPE_V>();

            const uint32_t outputBase =
                batchIdx * this->outputBatchStride + outChannel * this->outputChannelStride;
            DataCopyCustomUB2GM(yGm[outputBase], channelTmp, this->outputHw);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> calcBuf;
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> convBiasGm;
    GlobalTensor<float> addBiasGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> gammaGm;
    GlobalTensor<float> betaGm;
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
    uint32_t outputHw;
    uint32_t alignedOutputHw;
    uint32_t numGroups;
    uint32_t channelsPerGroup;
    uint32_t groupElemCount;
    uint32_t alignedGroupBufferCount;
    uint32_t blockDim;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t weightOutStride;
    uint32_t weightInStride;
    uint32_t outputChannelStride;
    uint32_t outputBatchStride;
    float epsilon;
    float invGroupElemCount;
};

extern "C" __global__ __aicore__ void conv2d_add_scale_sigmoid_group_norm_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR convBias,
    GM_ADDR addBias,
    GM_ADDR scale,
    GM_ADDR gamma,
    GM_ADDR beta,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dAddScaleSigmoidGroupNormCustom op;
    op.Init(
        x,
        weight,
        convBias,
        addBias,
        scale,
        gamma,
        beta,
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
        tiling_data.outputHw,
        tiling_data.alignedOutputHw,
        tiling_data.numGroups,
        tiling_data.channelsPerGroup,
        tiling_data.groupElemCount,
        tiling_data.alignedGroupBufferCount,
        tiling_data.epsilon,
        tiling_data.invGroupElemCount);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv2d_add_scale_sigmoid_group_norm_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_bias,
    const at::Tensor& add_bias,
    const at::Tensor& scale,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    int64_t num_groups,
    double epsilon)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be a 1D tensor");
    TORCH_CHECK(add_bias.dim() == 1, "add_bias must be a 1D tensor");
    TORCH_CHECK(scale.dim() == 1, "scale must be a 1D tensor");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be a 1D tensor");
    TORCH_CHECK(beta.dim() == 1, "beta must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(1), "x.size(1) must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == conv_bias.size(0), "conv_bias size must match weight.size(0)");
    TORCH_CHECK(weight.size(0) == add_bias.size(0), "add_bias size must match weight.size(0)");
    TORCH_CHECK(weight.size(0) == scale.size(0), "scale size must match weight.size(0)");
    TORCH_CHECK(weight.size(0) == gamma.size(0), "gamma size must match weight.size(0)");
    TORCH_CHECK(weight.size(0) == beta.size(0), "beta size must match weight.size(0)");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(weight.size(0) % num_groups == 0, "out_channels must be divisible by num_groups");
    TORCH_CHECK(x.size(2) >= weight.size(2) && x.size(3) >= weight.size(3), "kernel must fit inside input");

    const int64_t outputHeight = x.size(2) - weight.size(2) + 1;
    const int64_t outputWidth = x.size(3) - weight.size(3) + 1;
    at::Tensor result = at::empty({x.size(0), weight.size(0), outputHeight, outputWidth}, x.options());
    EXEC_NPU_CMD(
        aclnnConv2dAddScaleSigmoidGroupNormCustom,
        x,
        weight,
        conv_bias,
        add_bias,
        scale,
        gamma,
        beta,
        num_groups,
        epsilon,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv2d_add_scale_sigmoid_group_norm_custom",
        &conv2d_add_scale_sigmoid_group_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv2d_add_scale_sigmoid_group_norm_custom",
        &conv2d_add_scale_sigmoid_group_norm_custom_impl_npu,
        "conv2d + add + scale + sigmoid + group_norm");
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
        num_groups: int,
        bias_shape,
        scale_shape,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        self.epsilon = float(epsilon)
        self.conv2d = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.add_bias = torch.nn.Parameter(torch.randn(bias_shape))
        self.scale = torch.nn.Parameter(torch.randn(scale_shape))
        self.group_norm = torch.nn.GroupNorm(self.num_groups, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv2d_add_scale_sigmoid_group_norm_custom(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            self.add_bias.reshape(-1),
            self.scale.reshape(-1),
            self.group_norm.weight,
            self.group_norm.bias,
            self.num_groups,
            self.epsilon,
        )
'''
