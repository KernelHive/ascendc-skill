project_json_src='''
[
    {
        "op": "LogSoftmaxCustom",
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
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogSoftmaxCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rowCount);
TILING_DATA_FIELD_DEF(uint32_t, colCount);
TILING_DATA_FIELD_DEF(uint32_t, tmpSize);
TILING_DATA_FIELD_DEF_STRUCT(LogSoftMaxTiling, logSoftmaxTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LogSoftmaxCustom, LogSoftmaxCustomTilingData)
}
"""

host_operator_src="""
#include "log_softmax_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    LogSoftmaxCustomTilingData tiling;
    const auto inputShape = context->GetInputTensor(0)->GetStorageShape();
    const size_t dimNum = inputShape.GetDimNum();
    if (dimNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t rowCount = 1;
    for (size_t i = 0; i + 1 < dimNum; ++i) {
        rowCount *= static_cast<uint32_t>(inputShape.GetDim(i));
    }
    const uint32_t colCount = static_cast<uint32_t>(inputShape.GetDim(dimNum - 1));
    if (colCount == 0) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_rowCount(rowCount);
    tiling.set_colCount(colCount);

    ge::Shape srcShape({1, static_cast<int64_t>(colCount)});
    const uint32_t localWorkSpaceSize = AscendC::GetLogSoftMaxMinTmpSize(srcShape, sizeof(float), false);
    tiling.set_tmpSize(localWorkSpaceSize);
    AscendC::LogSoftMaxTilingFunc(srcShape, sizeof(float), localWorkSpaceSize, tiling.logSoftmaxTilingData);

    context->SetBlockDim(BLOCK_DIM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = localWorkSpaceSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class LogSoftmaxCustom : public OpDef {
public:
    explicit LogSoftmaxCustom(const char *name) : OpDef(name)
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

OP_ADD(LogSoftmaxCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

class KernelLogSoftmax {
public:
    static constexpr float LN_10 = 2.302585093f;
    __aicore__ inline KernelLogSoftmax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t rowCount, uint32_t colCount,
        uint32_t tmpSize, const LogSoftMaxTiling &logSoftmaxTiling)
    {
        this->rowCount = rowCount;
        this->colCount = colCount;
        this->tmpSize = tmpSize;
        this->logSoftmaxTiling = logSoftmaxTiling;

        xGm.SetGlobalBuffer((__gm__ float *)x, rowCount * colCount);
        yGm.SetGlobalBuffer((__gm__ float *)y, rowCount * colCount);
        pipe.InitBuffer(inQueueX, 1, colCount * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, colCount * sizeof(float));
        pipe.InitBuffer(sumBuffer, 8 * sizeof(float));
        pipe.InitBuffer(maxBuffer, 8 * sizeof(float));
        pipe.InitBuffer(tmpBuffer, tmpSize);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t rowIdx = 0; rowIdx < rowCount; ++rowIdx) {
            CopyIn(rowIdx);
            Compute();
            CopyOut(rowIdx);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[rowIdx * colCount], colCount);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> sumLocal = sumBuffer.Get<float>();
        AscendC::LocalTensor<float> maxLocal = maxBuffer.Get<float>();
        AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuffer.Get<uint8_t>();
        AscendC::SoftMaxShapeInfo srcShape = {1, colCount, 1, colCount};
        AscendC::LogSoftMax<float>(yLocal, sumLocal, maxLocal, xLocal, tmpLocal, logSoftmaxTiling, srcShape);
        AscendC::Muls(yLocal, yLocal, LN_10, colCount);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[rowIdx * colCount], yLocal, colCount);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuffer;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    LogSoftMaxTiling logSoftmaxTiling;
    uint32_t rowCount;
    uint32_t colCount;
    uint32_t tmpSize;
};

extern "C" __global__ __aicore__ void log_softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelLogSoftmax op;
    op.Init(x, y, tiling_data.rowCount, tiling_data.colCount, tiling_data.tmpSize, tiling_data.logSoftmaxTilingData);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor log_softmax_impl_npu(const at::Tensor &x) {
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnLogSoftmaxCustom, x, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("log_softmax_custom", &log_softmax_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("log_softmax_custom", &log_softmax_impl_npu, "stable log_softmax on the last dimension");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.log_softmax_custom(x)
'''
