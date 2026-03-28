project_json_src='''
[
    {
        "op": "InstanceNormCustom",
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
BEGIN_TILING_DATA_DEF(InstanceNormCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalNc);
TILING_DATA_FIELD_DEF(uint32_t, useCoreNums);
TILING_DATA_FIELD_DEF(uint32_t, ncPerCore);
TILING_DATA_FIELD_DEF(uint32_t, ncPerCoreTail);
TILING_DATA_FIELD_DEF(uint32_t, reduceNums);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, invReduceNums);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InstanceNormCustom, InstanceNormCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "instance_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t TILE_LENGTH = 2048;
constexpr float EPSILON = 1e-5f;

template <typename T1, typename T2>
inline T1 CeilDiv(T1 a, T2 b)
{
    return b == 0 ? a : (a + b - 1) / b;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t n = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t c = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t h = static_cast<uint32_t>(shape.GetDim(2));
    const uint32_t w = static_cast<uint32_t>(shape.GetDim(3));
    const uint32_t totalNc = n * c;
    const uint32_t reduceNums = h * w;
    const uint32_t useCoreNums = totalNc < BLOCK_DIM ? totalNc : BLOCK_DIM;
    const uint32_t ncPerCore = CeilDiv(totalNc, useCoreNums == 0 ? 1U : useCoreNums);
    const uint32_t ncPerCoreTail = totalNc == 0 ? 0 : totalNc - ncPerCore * (useCoreNums - 1);
    const uint32_t tileLength = reduceNums < TILE_LENGTH ? reduceNums : TILE_LENGTH;
    const uint32_t tileNum = CeilDiv(reduceNums, tileLength == 0 ? 1U : tileLength);
    const uint32_t lastTileLength = reduceNums == 0 ? 0 : reduceNums - (tileNum - 1) * tileLength;

    InstanceNormCustomTilingData tiling;
    context->SetBlockDim(useCoreNums == 0 ? 1 : useCoreNums);
    tiling.set_totalNc(totalNc);
    tiling.set_useCoreNums(useCoreNums == 0 ? 1 : useCoreNums);
    tiling.set_ncPerCore(ncPerCore);
    tiling.set_ncPerCoreTail(ncPerCoreTail);
    tiling.set_reduceNums(reduceNums);
    tiling.set_tileLength(tileLength);
    tiling.set_tileNum(tileNum);
    tiling.set_lastTileLength(lastTileLength);
    tiling.set_epsilon(EPSILON);
    tiling.set_invReduceNums(reduceNums == 0 ? 0.0f : 1.0f / static_cast<float>(reduceNums));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class InstanceNormCustom : public OpDef {
public:
    explicit InstanceNormCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(InstanceNormCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BLOCK_SIZE = 32;

template <typename T1, typename T2>
__aicore__ inline T1 AlignUp(T1 value, T2 align)
{
    return align == 0 ? value : (value + align - 1) / align * align;
}

__aicore__ inline void DataCopyCustomGM2UB(
    const LocalTensor<float>& dstTensor,
    const GlobalTensor<float>& srcTensor,
    const uint32_t count)
{
    const uint32_t numPerBlock = BLOCK_SIZE / sizeof(float);
    if (count % numPerBlock == 0) {
        DataCopy(dstTensor, srcTensor, count);
    } else {
        DataCopy(dstTensor, srcTensor, AlignUp(count, numPerBlock));
    }
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
        float value = srcTensor.GetValue(count - numPerBlock + i);
        srcTensor.SetValue(i, value);
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(dstTensor[count - numPerBlock], srcTensor, numPerBlock);
}

class KernelInstanceNorm {
public:
    __aicore__ inline KernelInstanceNorm() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t totalNc,
        uint32_t useCoreNums,
        uint32_t ncPerCore,
        uint32_t ncPerCoreTail,
        uint32_t reduceNums,
        uint32_t tileLength,
        uint32_t tileNum,
        uint32_t lastTileLength,
        float epsilon,
        float invReduceNums)
    {
        this->reduceNums = reduceNums;
        this->tileLength = tileLength;
        this->tileNum = tileNum;
        this->lastTileLength = lastTileLength;
        this->epsilon = epsilon;
        this->invReduceNums = invReduceNums;

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t sliceCount = blockIdx + 1 == useCoreNums ? ncPerCoreTail : ncPerCore;
        const uint32_t sliceOffset = blockIdx * ncPerCore;
        this->localSliceCount = sliceCount;

        xGm.SetGlobalBuffer((__gm__ float*)x + sliceOffset * reduceNums, sliceCount * reduceNums);
        yGm.SetGlobalBuffer((__gm__ float*)y + sliceOffset * reduceNums, sliceCount * reduceNums);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(calcBuf, this->tileLength * 3 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->reduceNums == 0 || this->localSliceCount == 0) {
            return;
        }

        for (uint32_t sliceIdx = 0; sliceIdx < this->localSliceCount; ++sliceIdx) {
            const uint32_t sliceOffset = sliceIdx * this->reduceNums;
            float meanValue = ComputeMean(sliceOffset);
            float invStdValue = ComputeInvStd(sliceOffset, meanValue);
            NormalizeSlice(sliceOffset, meanValue, invStdValue);
        }
    }

private:
    __aicore__ inline uint32_t GetCurrentLength(uint32_t tileIdx) const
    {
        return tileIdx + 1 == this->tileNum ? this->lastTileLength : this->tileLength;
    }

    __aicore__ inline void CopyIn(uint32_t gmOffset, uint32_t currentLength)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopyCustomGM2UB(xLocal, xGm[gmOffset], currentLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline float ComputeMean(uint32_t sliceOffset)
    {
        float sumValue = 0.0f;
        for (uint32_t tileIdx = 0; tileIdx < this->tileNum; ++tileIdx) {
            const uint32_t currentLength = GetCurrentLength(tileIdx);
            CopyIn(sliceOffset + tileIdx * this->tileLength, currentLength);
            LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> sumTensor = calcBuf.Get<float>();
            LocalTensor<float> reduceTmp = calcBuf.Get<float>()[this->tileLength];

            Adds(sumTensor, xLocal, 0.0f, currentLength);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(sumTensor, sumTensor, reduceTmp, currentLength);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            sumValue += sumTensor.GetValue(0);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);

            inQueueX.FreeTensor(xLocal);
        }
        return sumValue * this->invReduceNums;
    }

    __aicore__ inline float ComputeInvStd(uint32_t sliceOffset, float meanValue)
    {
        float varSum = 0.0f;
        for (uint32_t tileIdx = 0; tileIdx < this->tileNum; ++tileIdx) {
            const uint32_t currentLength = GetCurrentLength(tileIdx);
            CopyIn(sliceOffset + tileIdx * this->tileLength, currentLength);
            LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> centered = calcBuf.Get<float>();
            LocalTensor<float> square = calcBuf.Get<float>()[this->tileLength];
            LocalTensor<float> reduceTmp = calcBuf.Get<float>()[this->tileLength * 2];

            Adds(centered, xLocal, -meanValue, currentLength);
            PipeBarrier<PIPE_V>();
            Mul(square, centered, centered, currentLength);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(square, square, reduceTmp, currentLength);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            varSum += square.GetValue(0);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);

            inQueueX.FreeTensor(xLocal);
        }

        LocalTensor<float> scalarTensor = calcBuf.Get<float>();
        scalarTensor.SetValue(0, varSum * this->invReduceNums + this->epsilon);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sqrt(scalarTensor, scalarTensor, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return 1.0f / scalarTensor.GetValue(0);
    }

    __aicore__ inline void NormalizeSlice(uint32_t sliceOffset, float meanValue, float invStdValue)
    {
        for (uint32_t tileIdx = 0; tileIdx < this->tileNum; ++tileIdx) {
            const uint32_t currentLength = GetCurrentLength(tileIdx);
            const uint32_t gmOffset = sliceOffset + tileIdx * this->tileLength;
            CopyIn(gmOffset, currentLength);

            LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            Adds(yLocal, xLocal, -meanValue, currentLength);
            PipeBarrier<PIPE_V>();
            Muls(yLocal, yLocal, invStdValue, currentLength);
            PipeBarrier<PIPE_V>();
            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xLocal);
            CopyOut(gmOffset, currentLength);
        }
    }

    __aicore__ inline void CopyOut(uint32_t gmOffset, uint32_t currentLength)
    {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopyCustomUB2GM(yGm[gmOffset], yLocal, currentLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<TPosition::VECCALC> calcBuf;
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    uint32_t localSliceCount;
    uint32_t reduceNums;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t lastTileLength;
    float epsilon;
    float invReduceNums;
};

extern "C" __global__ __aicore__ void instance_norm_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelInstanceNorm op;
    op.Init(
        x,
        y,
        tiling_data.totalNc,
        tiling_data.useCoreNums,
        tiling_data.ncPerCore,
        tiling_data.ncPerCoreTail,
        tiling_data.reduceNums,
        tiling_data.tileLength,
        tiling_data.tileNum,
        tiling_data.lastTileLength,
        tiling_data.epsilon,
        tiling_data.invReduceNums);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor instance_norm_custom_impl_npu(const at::Tensor& x)
{
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnInstanceNormCustom, x, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("instance_norm_custom", &instance_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("instance_norm_custom", &instance_norm_custom_impl_npu, "instance norm custom op");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.instance_norm_custom(x)
'''
