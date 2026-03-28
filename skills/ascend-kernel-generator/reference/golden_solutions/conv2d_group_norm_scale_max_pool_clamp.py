project_json_src='''
[
    {
        "op": "Conv2dGroupNormScaleMaxPoolClampCustom",
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
            },
            {
                "name": "pool_kernel_size",
                "param_type": "required",
                "type": "int"
            },
            {
                "name": "clamp_min",
                "param_type": "required",
                "type": "float"
            },
            {
                "name": "clamp_max",
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
BEGIN_TILING_DATA_DEF(Conv2dGroupNormScaleMaxPoolClampCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, outChannels);
    TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
    TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
    TILING_DATA_FIELD_DEF(uint32_t, convOutputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, convOutputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, convOutputHw);
    TILING_DATA_FIELD_DEF(uint32_t, alignedConvOutputHw);
    TILING_DATA_FIELD_DEF(uint32_t, pooledOutputHeight);
    TILING_DATA_FIELD_DEF(uint32_t, pooledOutputWidth);
    TILING_DATA_FIELD_DEF(uint32_t, pooledOutputHw);
    TILING_DATA_FIELD_DEF(uint32_t, numGroups);
    TILING_DATA_FIELD_DEF(uint32_t, channelsPerGroup);
    TILING_DATA_FIELD_DEF(uint32_t, groupElemCount);
    TILING_DATA_FIELD_DEF(uint32_t, alignedGroupBufferCount);
    TILING_DATA_FIELD_DEF(uint32_t, poolKernelSize);
    TILING_DATA_FIELD_DEF(float, epsilon);
    TILING_DATA_FIELD_DEF(float, invGroupElemCount);
    TILING_DATA_FIELD_DEF(float, clampMin);
    TILING_DATA_FIELD_DEF(float, clampMax);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Conv2dGroupNormScaleMaxPoolClampCustom,
    Conv2dGroupNormScaleMaxPoolClampCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "conv2d_group_norm_scale_max_pool_clamp_custom_tiling.h"
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
    const gert::Shape* scaleShape,
    const gert::Shape* gammaShape,
    const gert::Shape* betaShape,
    uint32_t numGroups,
    uint32_t poolKernelSize,
    uint32_t& batchSize,
    uint32_t& inChannels,
    uint32_t& outChannels,
    uint32_t& inputHeight,
    uint32_t& inputWidth,
    uint32_t& kernelHeight,
    uint32_t& kernelWidth,
    uint32_t& convOutputHeight,
    uint32_t& convOutputWidth,
    uint32_t& pooledOutputHeight,
    uint32_t& pooledOutputWidth,
    uint32_t& channelsPerGroup)
{
    if (inputShape == nullptr || weightShape == nullptr || convBiasShape == nullptr ||
        scaleShape == nullptr || gammaShape == nullptr || betaShape == nullptr) {
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
        kernelHeight == 0 || kernelWidth == 0 || numGroups == 0 || poolKernelSize == 0) {
        return false;
    }
    if (weightInChannels != inChannels || outChannels % numGroups != 0) {
        return false;
    }
    if (inputHeight < kernelHeight || inputWidth < kernelWidth) {
        return false;
    }
    if (!IsOneDimVectorWithLength(convBiasShape, outChannels) ||
        !IsOneDimVectorWithLength(scaleShape, outChannels) ||
        !IsOneDimVectorWithLength(gammaShape, outChannels) ||
        !IsOneDimVectorWithLength(betaShape, outChannels)) {
        return false;
    }

    convOutputHeight = inputHeight - kernelHeight + 1;
    convOutputWidth = inputWidth - kernelWidth + 1;
    if (convOutputHeight < poolKernelSize || convOutputWidth < poolKernelSize) {
        return false;
    }
    pooledOutputHeight = (convOutputHeight - poolKernelSize) / poolKernelSize + 1;
    pooledOutputWidth = (convOutputWidth - poolKernelSize) / poolKernelSize + 1;
    channelsPerGroup = outChannels / numGroups;
    return pooledOutputHeight > 0 && pooledOutputWidth > 0 && channelsPerGroup > 0;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    const gert::StorageShape* weightStorageShape = context->GetInputShape(1);
    const gert::StorageShape* convBiasStorageShape = context->GetInputShape(2);
    const gert::StorageShape* scaleStorageShape = context->GetInputShape(3);
    const gert::StorageShape* gammaStorageShape = context->GetInputShape(4);
    const gert::StorageShape* betaStorageShape = context->GetInputShape(5);
    if (inputStorageShape == nullptr || weightStorageShape == nullptr || convBiasStorageShape == nullptr ||
        scaleStorageShape == nullptr || gammaStorageShape == nullptr || betaStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t* numGroupsPtr = attrs->GetAttrPointer<int64_t>(0);
    const float* epsilonPtr = attrs->GetAttrPointer<float>(1);
    const int64_t* poolKernelSizePtr = attrs->GetAttrPointer<int64_t>(2);
    const float* clampMinPtr = attrs->GetAttrPointer<float>(3);
    const float* clampMaxPtr = attrs->GetAttrPointer<float>(4);
    if (numGroupsPtr == nullptr || epsilonPtr == nullptr || poolKernelSizePtr == nullptr ||
        clampMinPtr == nullptr || clampMaxPtr == nullptr || *numGroupsPtr <= 0 || *poolKernelSizePtr <= 0 ||
        *clampMinPtr > *clampMaxPtr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t batchSize = 0;
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelHeight = 0;
    uint32_t kernelWidth = 0;
    uint32_t convOutputHeight = 0;
    uint32_t convOutputWidth = 0;
    uint32_t pooledOutputHeight = 0;
    uint32_t pooledOutputWidth = 0;
    uint32_t channelsPerGroup = 0;
    if (!IsValidShape(
            &inputStorageShape->GetStorageShape(),
            &weightStorageShape->GetStorageShape(),
            &convBiasStorageShape->GetStorageShape(),
            &scaleStorageShape->GetStorageShape(),
            &gammaStorageShape->GetStorageShape(),
            &betaStorageShape->GetStorageShape(),
            static_cast<uint32_t>(*numGroupsPtr),
            static_cast<uint32_t>(*poolKernelSizePtr),
            batchSize,
            inChannels,
            outChannels,
            inputHeight,
            inputWidth,
            kernelHeight,
            kernelWidth,
            convOutputHeight,
            convOutputWidth,
            pooledOutputHeight,
            pooledOutputWidth,
            channelsPerGroup)) {
        return ge::GRAPH_FAILED;
    }

    Conv2dGroupNormScaleMaxPoolClampCustomTilingData tiling;
    const uint32_t convOutputHw = convOutputHeight * convOutputWidth;
    const uint32_t alignedConvOutputHw = AlignUp(convOutputHw, BLOCK_SIZE / sizeof(float));
    const uint32_t groupElemCount = channelsPerGroup * convOutputHw;
    const uint32_t alignedGroupBufferCount = channelsPerGroup * alignedConvOutputHw;

    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_convOutputHeight(convOutputHeight);
    tiling.set_convOutputWidth(convOutputWidth);
    tiling.set_convOutputHw(convOutputHw);
    tiling.set_alignedConvOutputHw(alignedConvOutputHw);
    tiling.set_pooledOutputHeight(pooledOutputHeight);
    tiling.set_pooledOutputWidth(pooledOutputWidth);
    tiling.set_pooledOutputHw(pooledOutputHeight * pooledOutputWidth);
    tiling.set_numGroups(static_cast<uint32_t>(*numGroupsPtr));
    tiling.set_channelsPerGroup(channelsPerGroup);
    tiling.set_groupElemCount(groupElemCount);
    tiling.set_alignedGroupBufferCount(alignedGroupBufferCount);
    tiling.set_poolKernelSize(static_cast<uint32_t>(*poolKernelSizePtr));
    tiling.set_epsilon(*epsilonPtr);
    tiling.set_invGroupElemCount(groupElemCount == 0 ? 0.0f : 1.0f / static_cast<float>(groupElemCount));
    tiling.set_clampMin(*clampMinPtr);
    tiling.set_clampMax(*clampMaxPtr);

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
    const gert::Shape* scaleShape = context->GetInputShape(3);
    const gert::Shape* gammaShape = context->GetInputShape(4);
    const gert::Shape* betaShape = context->GetInputShape(5);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t* numGroupsPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* poolKernelSizePtr = attrs->GetAttrPointer<int64_t>(2);
    if (numGroupsPtr == nullptr || poolKernelSizePtr == nullptr || *numGroupsPtr <= 0 || *poolKernelSizePtr <= 0) {
        return GRAPH_FAILED;
    }

    uint32_t batchSize = 0;
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelHeight = 0;
    uint32_t kernelWidth = 0;
    uint32_t convOutputHeight = 0;
    uint32_t convOutputWidth = 0;
    uint32_t pooledOutputHeight = 0;
    uint32_t pooledOutputWidth = 0;
    uint32_t channelsPerGroup = 0;
    if (!IsValidShape(
            inputShape,
            weightShape,
            convBiasShape,
            scaleShape,
            gammaShape,
            betaShape,
            static_cast<uint32_t>(*numGroupsPtr),
            static_cast<uint32_t>(*poolKernelSizePtr),
            batchSize,
            inChannels,
            outChannels,
            inputHeight,
            inputWidth,
            kernelHeight,
            kernelWidth,
            convOutputHeight,
            convOutputWidth,
            pooledOutputHeight,
            pooledOutputWidth,
            channelsPerGroup)) {
        return GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDimNum(4);
    outputShape->SetDim(0, batchSize);
    outputShape->SetDim(1, outChannels);
    outputShape->SetDim(2, pooledOutputHeight);
    outputShape->SetDim(3, pooledOutputWidth);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Conv2dGroupNormScaleMaxPoolClampCustom : public OpDef {
public:
    explicit Conv2dGroupNormScaleMaxPoolClampCustom(const char* name) : OpDef(name)
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
        this->Attr("pool_kernel_size").AttrType(REQUIRED).Int();
        this->Attr("clamp_min").AttrType(REQUIRED).Float();
        this->Attr("clamp_max").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dGroupNormScaleMaxPoolClampCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;

class KernelConv2dGroupNormScaleMaxPoolClampCustom {
public:
    __aicore__ inline KernelConv2dGroupNormScaleMaxPoolClampCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR convBias,
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
        uint32_t convOutputHeight,
        uint32_t convOutputWidth,
        uint32_t convOutputHw,
        uint32_t alignedConvOutputHw,
        uint32_t pooledOutputHeight,
        uint32_t pooledOutputWidth,
        uint32_t pooledOutputHw,
        uint32_t numGroups,
        uint32_t channelsPerGroup,
        uint32_t groupElemCount,
        uint32_t alignedGroupBufferCount,
        uint32_t poolKernelSize,
        float epsilon,
        float invGroupElemCount,
        float clampMin,
        float clampMax)
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
        this->convOutputHw = convOutputHw;
        this->alignedConvOutputHw = alignedConvOutputHw;
        this->pooledOutputHeight = pooledOutputHeight;
        this->pooledOutputWidth = pooledOutputWidth;
        this->pooledOutputHw = pooledOutputHw;
        this->numGroups = numGroups;
        this->channelsPerGroup = channelsPerGroup;
        this->groupElemCount = groupElemCount;
        this->alignedGroupBufferCount = alignedGroupBufferCount;
        this->poolKernelSize = poolKernelSize;
        this->epsilon = epsilon;
        this->invGroupElemCount = invGroupElemCount;
        this->clampMin = clampMin;
        this->clampMax = clampMax;
        this->blockDim = GetBlockNum();

        this->inputChannelStride = inputHeight * inputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->weightOutStride = inChannels * kernelHeight * kernelWidth;
        this->weightInStride = kernelHeight * kernelWidth;
        this->convOutputChannelStride = convOutputHw;
        this->convOutputBatchStride = outChannels * convOutputHw;
        this->pooledOutputChannelStride = pooledOutputHw;
        this->pooledOutputBatchStride = outChannels * pooledOutputHw;

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float*)weight, outChannels * this->weightOutStride);
        convBiasGm.SetGlobalBuffer((__gm__ float*)convBias, outChannels);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale, outChannels);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, outChannels);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, outChannels);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * this->pooledOutputBatchStride);

        pipe.InitBuffer(calcBuf, this->alignedGroupBufferCount * 2 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->groupElemCount == 0) {
            return;
        }

        for (uint32_t batchIdx = GetBlockIdx(); batchIdx < this->batchSize; batchIdx += this->blockDim) {
            for (uint32_t groupIdx = 0; groupIdx < this->numGroups; ++groupIdx) {
                LocalTensor<float> convGroupLocal = calcBuf.Get<float>();
                LocalTensor<float> normGroupLocal = calcBuf.Get<float>()[this->alignedGroupBufferCount];

                ComputeGroup(batchIdx, groupIdx, convGroupLocal);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const float meanValue = ComputeMean(convGroupLocal);
                const float invStdValue = ComputeInvStd(convGroupLocal, normGroupLocal, meanValue);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                NormalizeScaleAndStore(batchIdx, groupIdx, convGroupLocal, normGroupLocal, meanValue, invStdValue);
            }
        }
    }

private:
    __aicore__ inline void ComputeGroup(
        uint32_t batchIdx,
        uint32_t groupIdx,
        LocalTensor<float>& convGroupLocal)
    {
        const uint32_t channelStart = groupIdx * this->channelsPerGroup;
        for (uint32_t channelOffset = 0; channelOffset < this->channelsPerGroup; ++channelOffset) {
            const uint32_t outChannel = channelStart + channelOffset;
            LocalTensor<float> channelLocal = convGroupLocal[channelOffset * this->alignedConvOutputHw];
            ComputeConvChannel(batchIdx, outChannel, channelLocal);
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
        for (uint32_t outH = 0; outH < this->convOutputHeight; ++outH) {
            for (uint32_t outW = 0; outW < this->convOutputWidth; ++outW) {
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

    __aicore__ inline float ComputeMean(LocalTensor<float>& groupLocal)
    {
        float sumValue = 0.0f;
        for (uint32_t channelOffset = 0; channelOffset < this->channelsPerGroup; ++channelOffset) {
            LocalTensor<float> channelLocal = groupLocal[channelOffset * this->alignedConvOutputHw];
            for (uint32_t i = 0; i < this->convOutputHw; ++i) {
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
            LocalTensor<float> channelLocal = groupLocal[channelOffset * this->alignedConvOutputHw];
            for (uint32_t i = 0; i < this->convOutputHw; ++i) {
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

    __aicore__ inline void NormalizeScaleAndStore(
        uint32_t batchIdx,
        uint32_t groupIdx,
        LocalTensor<float>& convGroupLocal,
        LocalTensor<float>& normGroupLocal,
        float meanValue,
        float invStdValue)
    {
        const uint32_t channelStart = groupIdx * this->channelsPerGroup;
        for (uint32_t channelOffset = 0; channelOffset < this->channelsPerGroup; ++channelOffset) {
            const uint32_t outChannel = channelStart + channelOffset;
            LocalTensor<float> convChannelLocal = convGroupLocal[channelOffset * this->alignedConvOutputHw];
            LocalTensor<float> normChannelLocal = normGroupLocal[channelOffset * this->alignedConvOutputHw];
            const float gammaValue = gammaGm.GetValue(outChannel);
            const float betaValue = betaGm.GetValue(outChannel);
            const float scaleValue = scaleGm.GetValue(outChannel);

            Adds(normChannelLocal, convChannelLocal, -meanValue, this->convOutputHw);
            PipeBarrier<PIPE_V>();
            Muls(normChannelLocal, normChannelLocal, invStdValue, this->convOutputHw);
            PipeBarrier<PIPE_V>();
            Muls(normChannelLocal, normChannelLocal, gammaValue, this->convOutputHw);
            PipeBarrier<PIPE_V>();
            Adds(normChannelLocal, normChannelLocal, betaValue, this->convOutputHw);
            PipeBarrier<PIPE_V>();
            Muls(normChannelLocal, normChannelLocal, scaleValue, this->convOutputHw);
            PipeBarrier<PIPE_V>();

            MaxPoolAndClampStore(batchIdx, outChannel, normChannelLocal);
        }
    }

    __aicore__ inline void MaxPoolAndClampStore(
        uint32_t batchIdx,
        uint32_t outChannel,
        LocalTensor<float>& normChannelLocal)
    {
        const uint32_t outputBase =
            batchIdx * this->pooledOutputBatchStride + outChannel * this->pooledOutputChannelStride;
        for (uint32_t outH = 0; outH < this->pooledOutputHeight; ++outH) {
            const uint32_t startH = outH * this->poolKernelSize;
            for (uint32_t outW = 0; outW < this->pooledOutputWidth; ++outW) {
                const uint32_t startW = outW * this->poolKernelSize;
                float maxValue = normChannelLocal.GetValue(startH * this->convOutputWidth + startW);
                for (uint32_t kernelH = 0; kernelH < this->poolKernelSize; ++kernelH) {
                    const uint32_t inH = startH + kernelH;
                    const uint32_t rowBase = inH * this->convOutputWidth;
                    for (uint32_t kernelW = 0; kernelW < this->poolKernelSize; ++kernelW) {
                        const uint32_t inW = startW + kernelW;
                        const float value = normChannelLocal.GetValue(rowBase + inW);
                        if (value > maxValue) {
                            maxValue = value;
                        }
                    }
                }
                if (maxValue < this->clampMin) {
                    maxValue = this->clampMin;
                } else if (maxValue > this->clampMax) {
                    maxValue = this->clampMax;
                }
                yGm.SetValue(outputBase + outH * this->pooledOutputWidth + outW, maxValue);
            }
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> calcBuf;
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> convBiasGm;
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
    uint32_t convOutputHeight;
    uint32_t convOutputWidth;
    uint32_t convOutputHw;
    uint32_t alignedConvOutputHw;
    uint32_t pooledOutputHeight;
    uint32_t pooledOutputWidth;
    uint32_t pooledOutputHw;
    uint32_t numGroups;
    uint32_t channelsPerGroup;
    uint32_t groupElemCount;
    uint32_t alignedGroupBufferCount;
    uint32_t poolKernelSize;
    uint32_t blockDim;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t weightOutStride;
    uint32_t weightInStride;
    uint32_t convOutputChannelStride;
    uint32_t convOutputBatchStride;
    uint32_t pooledOutputChannelStride;
    uint32_t pooledOutputBatchStride;
    float epsilon;
    float invGroupElemCount;
    float clampMin;
    float clampMax;
};

extern "C" __global__ __aicore__ void conv2d_group_norm_scale_max_pool_clamp_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR convBias,
    GM_ADDR scale,
    GM_ADDR gamma,
    GM_ADDR beta,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dGroupNormScaleMaxPoolClampCustom op;
    op.Init(
        x,
        weight,
        convBias,
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
        tiling_data.convOutputHeight,
        tiling_data.convOutputWidth,
        tiling_data.convOutputHw,
        tiling_data.alignedConvOutputHw,
        tiling_data.pooledOutputHeight,
        tiling_data.pooledOutputWidth,
        tiling_data.pooledOutputHw,
        tiling_data.numGroups,
        tiling_data.channelsPerGroup,
        tiling_data.groupElemCount,
        tiling_data.alignedGroupBufferCount,
        tiling_data.poolKernelSize,
        tiling_data.epsilon,
        tiling_data.invGroupElemCount,
        tiling_data.clampMin,
        tiling_data.clampMax);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv2d_group_norm_scale_max_pool_clamp_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_bias,
    const at::Tensor& scale,
    const at::Tensor& gamma,
    const at::Tensor& beta,
    int64_t num_groups,
    double epsilon,
    int64_t pool_kernel_size,
    double clamp_min,
    double clamp_max)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be a 1D tensor");
    TORCH_CHECK(scale.dim() == 1, "scale must be a 1D tensor");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be a 1D tensor");
    TORCH_CHECK(beta.dim() == 1, "beta must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(1), "x.size(1) must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == conv_bias.size(0), "conv_bias size must match weight.size(0)");
    TORCH_CHECK(weight.size(0) == scale.size(0), "scale size must match weight.size(0)");
    TORCH_CHECK(weight.size(0) == gamma.size(0), "gamma size must match weight.size(0)");
    TORCH_CHECK(weight.size(0) == beta.size(0), "beta size must match weight.size(0)");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(weight.size(0) % num_groups == 0, "out_channels must be divisible by num_groups");
    TORCH_CHECK(pool_kernel_size > 0, "pool_kernel_size must be positive");
    TORCH_CHECK(clamp_min <= clamp_max, "clamp_min must not exceed clamp_max");
    TORCH_CHECK(x.size(2) >= weight.size(2) && x.size(3) >= weight.size(3), "kernel must fit inside input");

    const int64_t convOutputHeight = x.size(2) - weight.size(2) + 1;
    const int64_t convOutputWidth = x.size(3) - weight.size(3) + 1;
    TORCH_CHECK(
        convOutputHeight >= pool_kernel_size && convOutputWidth >= pool_kernel_size,
        "pool kernel must fit inside conv output");
    const int64_t pooledOutputHeight = (convOutputHeight - pool_kernel_size) / pool_kernel_size + 1;
    const int64_t pooledOutputWidth = (convOutputWidth - pool_kernel_size) / pool_kernel_size + 1;

    at::Tensor result = at::empty({x.size(0), weight.size(0), pooledOutputHeight, pooledOutputWidth}, x.options());
    EXEC_NPU_CMD(
        aclnnConv2dGroupNormScaleMaxPoolClampCustom,
        x,
        weight,
        conv_bias,
        scale,
        gamma,
        beta,
        num_groups,
        epsilon,
        pool_kernel_size,
        clamp_min,
        clamp_max,
        result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl(
        "conv2d_group_norm_scale_max_pool_clamp_custom",
        &conv2d_group_norm_scale_max_pool_clamp_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv2d_group_norm_scale_max_pool_clamp_custom",
        &conv2d_group_norm_scale_max_pool_clamp_custom_impl_npu,
        "conv2d + group_norm + scale + max_pool + clamp");
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
        scale_shape,
        maxpool_kernel_size: int,
        clamp_min: float,
        clamp_max: float,
    ) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
        self.epsilon = float(1e-5)
        self.maxpool_kernel_size = int(maxpool_kernel_size)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )
        self.group_norm = torch.nn.GroupNorm(self.num_groups, out_channels)
        self.scale = torch.nn.Parameter(torch.ones(scale_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv2d_group_norm_scale_max_pool_clamp_custom(
            x,
            self.conv.weight,
            self.conv.bias,
            self.scale.reshape(-1),
            self.group_norm.weight,
            self.group_norm.bias,
            self.num_groups,
            self.epsilon,
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max,
        )
'''
