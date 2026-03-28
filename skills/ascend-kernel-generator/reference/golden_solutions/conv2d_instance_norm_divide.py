project_json_src='''
[
    {
        "op": "Conv2dInstanceNormDivideCustom",
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
                "name": "divide_by",
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
BEGIN_TILING_DATA_DEF(Conv2dInstanceNormDivideCustomTilingData)
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
    TILING_DATA_FIELD_DEF(float, epsilon);
    TILING_DATA_FIELD_DEF(float, divideBy);
    TILING_DATA_FIELD_DEF(float, invOutputHw);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    Conv2dInstanceNormDivideCustom,
    Conv2dInstanceNormDivideCustomTilingData)
}
"""

host_operator_src="""
#include <cstdint>
#include "conv2d_instance_norm_divide_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr float EPSILON = 1e-5f;

inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return align == 0 ? value : (value + align - 1) / align * align;
}

inline bool IsValidShape(
    const gert::Shape* inputShape,
    const gert::Shape* weightShape,
    const gert::Shape* biasShape,
    uint32_t& batchSize,
    uint32_t& inChannels,
    uint32_t& outChannels,
    uint32_t& inputHeight,
    uint32_t& inputWidth,
    uint32_t& kernelHeight,
    uint32_t& kernelWidth,
    uint32_t& outputHeight,
    uint32_t& outputWidth)
{
    if (inputShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return false;
    }
    if (inputShape->GetDimNum() != 4 || weightShape->GetDimNum() != 4 || biasShape->GetDimNum() != 1) {
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
    const uint32_t biasLength = static_cast<uint32_t>(biasShape->GetDim(0));

    if (batchSize == 0 || inChannels == 0 || outChannels == 0 || inputHeight == 0 || inputWidth == 0 ||
        kernelHeight == 0 || kernelWidth == 0) {
        return false;
    }
    if (weightInChannels != inChannels || biasLength != outChannels) {
        return false;
    }
    if (inputHeight < kernelHeight || inputWidth < kernelWidth) {
        return false;
    }

    outputHeight = inputHeight - kernelHeight + 1;
    outputWidth = inputWidth - kernelWidth + 1;
    return outputHeight > 0 && outputWidth > 0;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputStorageShape = context->GetInputShape(0);
    const gert::StorageShape* weightStorageShape = context->GetInputShape(1);
    const gert::StorageShape* biasStorageShape = context->GetInputShape(2);
    if (inputStorageShape == nullptr || weightStorageShape == nullptr || biasStorageShape == nullptr) {
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
    if (!IsValidShape(
            &inputStorageShape->GetStorageShape(),
            &weightStorageShape->GetStorageShape(),
            &biasStorageShape->GetStorageShape(),
            batchSize,
            inChannels,
            outChannels,
            inputHeight,
            inputWidth,
            kernelHeight,
            kernelWidth,
            outputHeight,
            outputWidth)) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const float* divideByPtr = attrs->GetAttrPointer<float>(0);
    if (divideByPtr == nullptr || *divideByPtr == 0.0f) {
        return ge::GRAPH_FAILED;
    }

    Conv2dInstanceNormDivideCustomTilingData tiling;
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_outputHw(outputHeight * outputWidth);
    tiling.set_alignedOutputHw(AlignUp(outputHeight * outputWidth, BLOCK_SIZE / sizeof(float)));
    tiling.set_epsilon(EPSILON);
    tiling.set_divideBy(*divideByPtr);
    tiling.set_invOutputHw(outputHeight * outputWidth == 0 ? 0.0f : 1.0f / static_cast<float>(outputHeight * outputWidth));

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
    const gert::Shape* biasShape = context->GetInputShape(2);

    uint32_t batchSize = 0;
    uint32_t inChannels = 0;
    uint32_t outChannels = 0;
    uint32_t inputHeight = 0;
    uint32_t inputWidth = 0;
    uint32_t kernelHeight = 0;
    uint32_t kernelWidth = 0;
    uint32_t outputHeight = 0;
    uint32_t outputWidth = 0;
    if (!IsValidShape(
            inputShape,
            weightShape,
            biasShape,
            batchSize,
            inChannels,
            outChannels,
            inputHeight,
            inputWidth,
            kernelHeight,
            kernelWidth,
            outputHeight,
            outputWidth)) {
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
class Conv2dInstanceNormDivideCustom : public OpDef {
public:
    explicit Conv2dInstanceNormDivideCustom(const char* name) : OpDef(name)
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
        this->Attr("divide_by").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dInstanceNormDivideCustom);
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

class KernelConv2dInstanceNormDivideCustom {
public:
    __aicore__ inline KernelConv2dInstanceNormDivideCustom() {}

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
        uint32_t outputHw,
        uint32_t alignedOutputHw,
        float epsilon,
        float divideBy,
        float invOutputHw)
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
        this->epsilon = epsilon;
        this->invOutputHw = invOutputHw;
        this->normScale = divideBy == 0.0f ? 0.0f : 1.0f / divideBy;
        this->blockDim = GetBlockNum();

        this->inputChannelStride = inputHeight * inputWidth;
        this->inputBatchStride = inChannels * this->inputChannelStride;
        this->weightOutStride = inChannels * kernelHeight * kernelWidth;
        this->weightInStride = kernelHeight * kernelWidth;
        this->outputChannelStride = outputHw;
        this->outputBatchStride = outChannels * outputHw;

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * this->inputBatchStride);
        weightGm.SetGlobalBuffer((__gm__ float*)weight, outChannels * this->weightOutStride);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * this->outputBatchStride);

        pipe.InitBuffer(calcBuf, this->alignedOutputHw * 4 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->outputHw == 0) {
            return;
        }

        for (uint32_t batchIdx = GetBlockIdx(); batchIdx < this->batchSize; batchIdx += this->blockDim) {
            for (uint32_t outChannel = 0; outChannel < this->outChannels; ++outChannel) {
                LocalTensor<float> convLocal = calcBuf.Get<float>();
                LocalTensor<float> tmp1 = calcBuf.Get<float>()[this->alignedOutputHw];
                LocalTensor<float> tmp2 = calcBuf.Get<float>()[this->alignedOutputHw * 2];
                LocalTensor<float> tmp3 = calcBuf.Get<float>()[this->alignedOutputHw * 3];

                ComputeConv(batchIdx, outChannel, convLocal);
                const float meanValue = ComputeMean(convLocal, tmp1, tmp2);
                const float invStdValue = ComputeInvStd(convLocal, tmp1, tmp2, tmp3, meanValue);
                NormalizeAndStore(batchIdx, outChannel, convLocal, tmp1, meanValue, invStdValue);
            }
        }
    }

private:
    __aicore__ inline void ComputeConv(
        uint32_t batchIdx,
        uint32_t outChannel,
        LocalTensor<float>& convLocal)
    {
        const uint32_t inputBatchBase = batchIdx * this->inputBatchStride;
        const uint32_t weightBase = outChannel * this->weightOutStride;
        const float biasValue = biasGm.GetValue(outChannel);

        uint32_t outIndex = 0;
        for (uint32_t outH = 0; outH < this->outputHeight; ++outH) {
            for (uint32_t outW = 0; outW < this->outputWidth; ++outW) {
                float sum = biasValue;
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
                convLocal.SetValue(outIndex, sum);
                ++outIndex;
            }
        }
    }

    __aicore__ inline float ComputeMean(
        LocalTensor<float>& convLocal,
        LocalTensor<float>& tmp1,
        LocalTensor<float>& tmp2)
    {
        Adds(tmp1, convLocal, 0.0f, this->outputHw);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(tmp1, tmp1, tmp2, this->outputHw);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float sumValue = tmp1.GetValue(0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        return sumValue * this->invOutputHw;
    }

    __aicore__ inline float ComputeInvStd(
        LocalTensor<float>& convLocal,
        LocalTensor<float>& tmp1,
        LocalTensor<float>& tmp2,
        LocalTensor<float>& tmp3,
        float meanValue)
    {
        Adds(tmp1, convLocal, -meanValue, this->outputHw);
        PipeBarrier<PIPE_V>();
        Mul(tmp2, tmp1, tmp1, this->outputHw);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(tmp2, tmp2, tmp3, this->outputHw);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float varianceValue = tmp2.GetValue(0) * this->invOutputHw + this->epsilon;
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);

        tmp1.SetValue(0, varianceValue);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sqrt(tmp1, tmp1, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return 1.0f / tmp1.GetValue(0);
    }

    __aicore__ inline void NormalizeAndStore(
        uint32_t batchIdx,
        uint32_t outChannel,
        LocalTensor<float>& convLocal,
        LocalTensor<float>& tmp1,
        float meanValue,
        float invStdValue)
    {
        Adds(tmp1, convLocal, -meanValue, this->outputHw);
        PipeBarrier<PIPE_V>();
        Muls(tmp1, tmp1, invStdValue * this->normScale, this->outputHw);
        PipeBarrier<PIPE_V>();

        const uint32_t outputBase =
            batchIdx * this->outputBatchStride + outChannel * this->outputChannelStride;
        DataCopyCustomUB2GM(yGm[outputBase], tmp1, this->outputHw);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> calcBuf;
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
    uint32_t outputHw;
    uint32_t alignedOutputHw;
    uint32_t blockDim;
    uint32_t inputChannelStride;
    uint32_t inputBatchStride;
    uint32_t weightOutStride;
    uint32_t weightInStride;
    uint32_t outputChannelStride;
    uint32_t outputBatchStride;
    float epsilon;
    float invOutputHw;
    float normScale;
};

extern "C" __global__ __aicore__ void conv2d_instance_norm_divide_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dInstanceNormDivideCustom op;
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
        tiling_data.outputHw,
        tiling_data.alignedOutputHw,
        tiling_data.epsilon,
        tiling_data.divideBy,
        tiling_data.invOutputHw);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor conv2d_instance_norm_divide_custom_impl_npu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double divide_by)
{
    TORCH_CHECK(x.dim() == 4, "x must be a 4D NCHW tensor");
    TORCH_CHECK(weight.dim() == 4, "weight must be a 4D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(1), "x.size(1) must match weight.size(1)");
    TORCH_CHECK(weight.size(0) == bias.size(0), "bias size must match weight.size(0)");
    TORCH_CHECK(x.size(2) >= weight.size(2) && x.size(3) >= weight.size(3), "kernel must fit inside input");
    TORCH_CHECK(divide_by != 0.0, "divide_by must be non-zero");

    const int64_t outputHeight = x.size(2) - weight.size(2) + 1;
    const int64_t outputWidth = x.size(3) - weight.size(3) + 1;
    at::Tensor result = at::empty({x.size(0), weight.size(0), outputHeight, outputWidth}, x.options());
    EXEC_NPU_CMD(aclnnConv2dInstanceNormDivideCustom, x, weight, bias, divide_by, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_instance_norm_divide_custom", &conv2d_instance_norm_divide_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv2d_instance_norm_divide_custom",
        &conv2d_instance_norm_divide_custom_impl_npu,
        "conv2d + instance_norm + divide");
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
        divide_by: float,
    ) -> None:
        super().__init__()
        self.divide_by = float(divide_by)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.conv2d_instance_norm_divide_custom(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            self.divide_by,
        )
'''
