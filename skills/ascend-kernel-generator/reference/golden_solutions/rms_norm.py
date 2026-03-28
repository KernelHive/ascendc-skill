project_json_src='''
[
    {
        "op": "RmsNormCustom",
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
                "name": "gamma",
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
BEGIN_TILING_DATA_DEF(RmsNormCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rowCount);
TILING_DATA_FIELD_DEF(uint32_t, colCount);
TILING_DATA_FIELD_DEF(uint32_t, useCoreNums);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCoreTail);
TILING_DATA_FIELD_DEF(uint32_t, tileRows);
TILING_DATA_FIELD_DEF(uint32_t, tailTileRows);
TILING_DATA_FIELD_DEF(uint32_t, mainTmpBufSize);
TILING_DATA_FIELD_DEF(uint32_t, tailTmpBufSize);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, invColCount);
TILING_DATA_FIELD_DEF_STRUCT(RmsNormTiling, mainRmsNormTilingData);
TILING_DATA_FIELD_DEF_STRUCT(RmsNormTiling, tailRmsNormTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNormCustom, RmsNormCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "rms_norm_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t TARGET_TILE_ELEMS = 4096;
constexpr float EPSILON = 1e-5f;

template <typename T1, typename T2>
inline T1 CeilDiv(T1 a, T2 b)
{
    return b == 0 ? a : (a + b - 1) / b;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto &shape = inputShape->GetStorageShape();
    if (shape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t rowCount = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t colCount = static_cast<uint32_t>(shape.GetDim(1));
    if (rowCount == 0 || colCount == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t useCoreNums = rowCount < BLOCK_DIM ? rowCount : BLOCK_DIM;
    const uint32_t rowsPerCore = CeilDiv(rowCount, useCoreNums == 0 ? 1U : useCoreNums);
    const uint32_t rowsPerCoreTail = rowCount - rowsPerCore * (useCoreNums - 1);
    uint32_t tileRows = TARGET_TILE_ELEMS / colCount;
    tileRows = tileRows == 0 ? 1 : tileRows;
    tileRows = tileRows > rowsPerCore ? rowsPerCore : tileRows;
    tileRows = tileRows == 0 ? 1 : tileRows;
    const uint32_t tailTileRows = rowsPerCoreTail == 0 ? tileRows : (rowsPerCoreTail < tileRows ? rowsPerCoreTail : tileRows);

    ge::Shape mainShape({1, static_cast<int64_t>(tileRows), static_cast<int64_t>(colCount)});
    ge::Shape tailShape({1, static_cast<int64_t>(tailTileRows), static_cast<int64_t>(colCount)});

    uint32_t mainMaxValue = 0;
    uint32_t mainMinValue = 0;
    uint32_t tailMaxValue = 0;
    uint32_t tailMinValue = 0;
    if (!AscendC::GetRmsNormMaxMinTmpSize(mainShape, sizeof(float), mainMaxValue, mainMinValue, false)) {
        return ge::GRAPH_FAILED;
    }
    if (!AscendC::GetRmsNormMaxMinTmpSize(tailShape, sizeof(float), tailMaxValue, tailMinValue, false)) {
        return ge::GRAPH_FAILED;
    }

    RmsNormCustomTilingData tiling;
    tiling.set_rowCount(rowCount);
    tiling.set_colCount(colCount);
    tiling.set_useCoreNums(useCoreNums == 0 ? 1 : useCoreNums);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_rowsPerCoreTail(rowsPerCoreTail);
    tiling.set_tileRows(tileRows);
    tiling.set_tailTileRows(tailTileRows);
    tiling.set_mainTmpBufSize(mainMinValue);
    tiling.set_tailTmpBufSize(tailMinValue);
    tiling.set_epsilon(EPSILON);
    tiling.set_invColCount(1.0f / static_cast<float>(colCount));

    if (!AscendC::GetRmsNormTilingInfo(mainShape, mainShape, mainMinValue, sizeof(float), tiling.mainRmsNormTilingData, false)) {
        return ge::GRAPH_FAILED;
    }
    if (!AscendC::GetRmsNormTilingInfo(tailShape, tailShape, tailMinValue, sizeof(float), tiling.tailRmsNormTilingData, false)) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(useCoreNums == 0 ? 1 : useCoreNums);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
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
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class RmsNormCustom : public OpDef {
public:
    explicit RmsNormCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
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

OP_ADD(RmsNormCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelRmsNorm {
public:
    __aicore__ inline KernelRmsNorm() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR gamma,
        GM_ADDR y,
        uint32_t rowCount,
        uint32_t colCount,
        uint32_t useCoreNums,
        uint32_t rowsPerCore,
        uint32_t rowsPerCoreTail,
        uint32_t tileRows,
        uint32_t tailTileRows,
        uint32_t mainTmpBufSize,
        uint32_t tailTmpBufSize,
        float epsilon,
        float invColCount,
        const RmsNormTiling &mainTiling,
        const RmsNormTiling &tailTiling)
    {
        this->colCount = colCount;
        this->tileRows = tileRows;
        this->tailTileRows = tailTileRows;
        this->epsilon = epsilon;
        this->invColCount = invColCount;
        (void)rowCount;
        (void)mainTmpBufSize;
        (void)tailTmpBufSize;
        (void)mainTiling;
        (void)tailTiling;

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t localRows = blockIdx + 1 == useCoreNums ? rowsPerCoreTail : rowsPerCore;
        const uint32_t rowOffset = blockIdx * rowsPerCore;
        this->localRows = localRows;

        xGm.SetGlobalBuffer((__gm__ float *)x + rowOffset * colCount, localRows * colCount);
        yGm.SetGlobalBuffer((__gm__ float *)y + rowOffset * colCount, localRows * colCount);
        gammaGm.SetGlobalBuffer((__gm__ float *)gamma, colCount);

        const uint32_t maxTmpBufSize = mainTmpBufSize > tailTmpBufSize ? mainTmpBufSize : tailTmpBufSize;

        (void)maxTmpBufSize;
        (void)BUFFER_NUM;
        pipe.InitBuffer(gammaBuf, colCount * sizeof(float));
        pipe.InitBuffer(scalarBuf, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->localRows == 0) {
            return;
        }

        LocalTensor<float> gammaLocal = gammaBuf.Get<float>();
        DataCopy(gammaLocal, gammaGm, this->colCount);
        LocalTensor<float> scalarLocal = scalarBuf.Get<float>();

        for (uint32_t row = 0; row < this->localRows; ++row) {
            const uint32_t base = row * this->colCount;
            float sumSq = 0.0f;
            for (uint32_t col = 0; col < this->colCount; ++col) {
                const float value = xGm.GetValue(base + col);
                sumSq += value * value;
            }

            scalarLocal.SetValue(0, sumSq * this->invColCount + this->epsilon);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Sqrt(scalarLocal, scalarLocal, 1);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const float invRms = 1.0f / scalarLocal.GetValue(0);

            for (uint32_t col = 0; col < this->colCount; ++col) {
                const float value = xGm.GetValue(base + col);
                const float gamma = gammaLocal.GetValue(col);
                yGm.SetValue(base + col, value * invRms * gamma);
            }
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> gammaBuf;
    TBuf<TPosition::VECCALC> scalarBuf;
    GlobalTensor<float> xGm;
    GlobalTensor<float> gammaGm;
    GlobalTensor<float> yGm;
    uint32_t localRows;
    uint32_t colCount;
    uint32_t tileRows;
    uint32_t tailTileRows;
    float epsilon;
    float invColCount;
};

extern "C" __global__ __aicore__ void rms_norm_custom(
    GM_ADDR x,
    GM_ADDR gamma,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelRmsNorm op;
    op.Init(
        x,
        gamma,
        y,
        tiling_data.rowCount,
        tiling_data.colCount,
        tiling_data.useCoreNums,
        tiling_data.rowsPerCore,
        tiling_data.rowsPerCoreTail,
        tiling_data.tileRows,
        tiling_data.tailTileRows,
        tiling_data.mainTmpBufSize,
        tiling_data.tailTmpBufSize,
        tiling_data.epsilon,
        tiling_data.invColCount,
        tiling_data.mainRmsNormTilingData,
        tiling_data.tailRmsNormTilingData);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor rms_norm_custom_impl_npu(const at::Tensor &x)
{
    TORCH_CHECK(x.dim() == 4, "rms_norm_custom expects a 4D NCHW tensor");
    at::Tensor xContiguous = x.contiguous();
    const auto n = xContiguous.size(0);
    const auto c = xContiguous.size(1);
    const auto h = xContiguous.size(2);
    const auto w = xContiguous.size(3);

    at::Tensor xNHWC = xContiguous.permute({0, 2, 3, 1}).contiguous();
    at::Tensor xRows = xNHWC.reshape({n * h * w, c});
    at::Tensor gamma = at::ones({c}, xRows.options());
    at::Tensor yRows = at::empty_like(xRows);
    EXEC_NPU_CMD(aclnnRmsNormCustom, xRows, gamma, yRows);
    at::Tensor yNHWC = yRows.reshape({n, h, w, c});
    return yNHWC.permute({0, 3, 1, 2}).contiguous();
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("rms_norm_custom", &rms_norm_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_custom", &rms_norm_custom_impl_npu, "rms norm along channel dimension");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.rms_norm_custom(x)
'''
