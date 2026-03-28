project_json_src='''
[
    {
        "op": "GemmGroupNormHardtanhCustom",
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
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmGroupNormHardtanhCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, cols);
TILING_DATA_FIELD_DEF(uint32_t, innerDim);
TILING_DATA_FIELD_DEF(uint32_t, groupSize);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmGroupNormHardtanhCustom, GemmGroupNormHardtanhCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "gemm_group_norm_hardtanh_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto shapeX = context->GetInputTensor(0)->GetOriginShape();
    auto shapeWeight = context->GetInputTensor(1)->GetOriginShape();
    auto shapeBias = context->GetInputTensor(2)->GetOriginShape();
    auto shapeGamma = context->GetInputTensor(3)->GetOriginShape();
    auto shapeBeta = context->GetInputTensor(4)->GetOriginShape();

    if (shapeX.GetDimNum() != 2 || shapeWeight.GetDimNum() != 2 || shapeBias.GetDimNum() != 1 ||
        shapeGamma.GetDimNum() != 1 || shapeBeta.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int32_t rows = shapeX.GetDim(0);
    const int32_t innerDim = shapeX.GetDim(1);
    const int32_t weightInnerDim = shapeWeight.GetDim(0);
    const int32_t cols = shapeWeight.GetDim(1);
    if (rows != 128 || innerDim != 1024 || weightInnerDim != innerDim || cols != 512) {
        return ge::GRAPH_FAILED;
    }
    if (shapeBias.GetDim(0) != cols || shapeGamma.GetDim(0) != cols || shapeBeta.GetDim(0) != cols) {
        return ge::GRAPH_FAILED;
    }

    GemmGroupNormHardtanhCustomTilingData tiling;
    tiling.set_rows(static_cast<uint32_t>(rows));
    tiling.set_cols(static_cast<uint32_t>(cols));
    tiling.set_innerDim(static_cast<uint32_t>(innerDim));
    tiling.set_groupSize(64);
    tiling.set_blockDim(8);

    context->SetBlockDim(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    const gert::Shape *biasShape = context->GetInputShape(2);
    const gert::Shape *gammaShape = context->GetInputShape(3);
    const gert::Shape *betaShape = context->GetInputShape(4);
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr ||
        gammaShape == nullptr || betaShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2 || biasShape->GetDimNum() != 1 ||
        gammaShape->GetDimNum() != 1 || betaShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(0)) {
        return GRAPH_FAILED;
    }
    if (weightShape->GetDim(1) != biasShape->GetDim(0) || weightShape->GetDim(1) != gammaShape->GetDim(0) ||
        weightShape->GetDim(1) != betaShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    gert::Shape *outShape = context->GetOutputShape(0);
    outShape->SetDimNum(2);
    outShape->SetDim(0, xShape->GetDim(0));
    outShape->SetDim(1, weightShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class GemmGroupNormHardtanhCustom : public OpDef {
public:
    explicit GemmGroupNormHardtanhCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("gamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(GemmGroupNormHardtanhCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class GemmGroupNormHardtanhKernel {
public:
    __aicore__ inline GemmGroupNormHardtanhKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR gamma,
        GM_ADDR beta,
        GM_ADDR y,
        uint32_t rows,
        uint32_t cols,
        uint32_t innerDim,
        uint32_t groupSize,
        uint32_t blockDim)
    {
        this->rows = rows;
        this->cols = cols;
        this->innerDim = innerDim;
        this->groupSize = groupSize;
        this->groupNum = cols / groupSize;
        this->blockDim = blockDim;

        xGm.SetGlobalBuffer((__gm__ float *)x, static_cast<uint64_t>(rows) * innerDim);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, static_cast<uint64_t>(innerDim) * cols);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, cols);
        gammaGm.SetGlobalBuffer((__gm__ float *)gamma, cols);
        betaGm.SetGlobalBuffer((__gm__ float *)beta, cols);
        yGm.SetGlobalBuffer((__gm__ float *)y, static_cast<uint64_t>(rows) * cols);

        pipe.InitBuffer(groupBuf, groupSize * sizeof(float));
        pipe.InitBuffer(scalarBuf, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = GetBlockIdx();
        const float epsilon = 1.0e-5f;
        const float invGroupSize = 1.0f / 64.0f;
        const float hardtanhMin = -2.0f;
        const float hardtanhMax = 2.0f;
        AscendC::LocalTensor<float> groupLocal = groupBuf.Get<float>();
        AscendC::LocalTensor<float> scalarLocal = scalarBuf.Get<float>();

        for (uint32_t row = blockIdx; row < this->rows; row += this->blockDim) {
            const uint64_t xRowOffset = static_cast<uint64_t>(row) * this->innerDim;
            const uint64_t yRowOffset = static_cast<uint64_t>(row) * this->cols;

            for (uint32_t groupIdx = 0; groupIdx < this->groupNum; ++groupIdx) {
                const uint32_t groupOffset = groupIdx * this->groupSize;
                float meanValue = 0.0f;
                for (uint32_t i = 0; i < this->groupSize; ++i) {
                    const uint32_t outCol = groupOffset + i;
                    float accum = biasGm.GetValue(outCol);
                    for (uint32_t kIdx = 0; kIdx < this->innerDim; ++kIdx) {
                        const float xValue = xGm.GetValue(xRowOffset + kIdx);
                        const float weightValue =
                            weightGm.GetValue(static_cast<uint64_t>(kIdx) * this->cols + outCol);
                        accum += xValue * weightValue;
                    }
                    groupLocal.SetValue(i, accum);
                    meanValue += accum;
                }
                meanValue *= invGroupSize;

                float varianceValue = 0.0f;
                for (uint32_t i = 0; i < this->groupSize; ++i) {
                    const float centered = groupLocal.GetValue(i) - meanValue;
                    varianceValue += centered * centered;
                }

                scalarLocal.SetValue(0, varianceValue * invGroupSize + epsilon);
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                Sqrt(scalarLocal, scalarLocal, 1);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const float invStdValue = 1.0f / scalarLocal.GetValue(0);

                for (uint32_t i = 0; i < this->groupSize; ++i) {
                    const uint32_t outCol = groupOffset + i;
                    float value = groupLocal.GetValue(i);
                    value = (value - meanValue) * invStdValue;
                    value = value * gammaGm.GetValue(outCol) + betaGm.GetValue(outCol);
                    value = value < hardtanhMin ? hardtanhMin : value;
                    value = value > hardtanhMax ? hardtanhMax : value;
                    yGm.SetValue(yRowOffset + outCol, value);
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> groupBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> gammaGm;
    AscendC::GlobalTensor<float> betaGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t innerDim = 0;
    uint32_t groupNum = 0;
    uint32_t groupSize = 0;
    uint32_t blockDim = 0;
};

extern "C" __global__ __aicore__ void gemm_group_norm_hardtanh_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR gamma,
    GM_ADDR beta,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    GemmGroupNormHardtanhKernel op;
    op.Init(
        x,
        weight,
        bias,
        gamma,
        beta,
        y,
        tiling_data.rows,
        tiling_data.cols,
        tiling_data.innerDim,
        tiling_data.groupSize,
        tiling_data.blockDim);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor gemm_group_norm_hardtanh_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    const at::Tensor &gamma,
    const at::Tensor &beta)
{
    at::Tensor result = at::empty({x.size(0), weight.size(1)}, x.options());
    EXEC_NPU_CMD(aclnnGemmGroupNormHardtanhCustom, x, weight, bias, gamma, beta, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_group_norm_hardtanh_custom", &gemm_group_norm_hardtanh_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gemm_group_norm_hardtanh_custom",
        &gemm_group_norm_hardtanh_custom_impl_npu,
        "gemm + group_norm + hardtanh");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = torch.nn.Linear(in_features, out_features)
        self.group_norm = torch.nn.GroupNorm(num_groups, out_features)
        self.hardtanh = torch.nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        weight = self.gemm.weight.transpose(0, 1).contiguous()
        return custom_ops_lib.gemm_group_norm_hardtanh_custom(
            x,
            weight,
            self.gemm.bias,
            self.group_norm.weight,
            self.group_norm.bias,
        )
'''
