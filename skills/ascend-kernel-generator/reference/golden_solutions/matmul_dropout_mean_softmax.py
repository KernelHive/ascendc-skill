project_json_src='''
[
    {
        "op": "MatmulDropoutMeanSoftmaxCustom",
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
BEGIN_TILING_DATA_DEF(MatmulDropoutMeanSoftmaxCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileCount);
    TILING_DATA_FIELD_DEF(float, fillValue);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    MatmulDropoutMeanSoftmaxCustom,
    MatmulDropoutMeanSoftmaxCustomTilingData)
}
"""

host_operator_src="""
#include "matmul_dropout_mean_softmax_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t BLOCK_DIM = 1;
constexpr uint32_t DEFAULT_TILE_LENGTH = 256;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xShapeStorage = context->GetInputShape(0);
    if (xShapeStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto xShape = xShapeStorage->GetStorageShape();
    if (xShape.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batchSize = xShape.GetDim(0);
    const int64_t featureSize = xShape.GetDim(1);
    if (batchSize <= 0 || featureSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalLength = static_cast<uint32_t>(batchSize);
    const uint32_t tileLength =
        totalLength < DEFAULT_TILE_LENGTH ? totalLength : DEFAULT_TILE_LENGTH;
    const uint32_t tileCount = (totalLength + tileLength - 1U) / tileLength;

    MatmulDropoutMeanSoftmaxCustomTilingData tiling;
    tiling.set_batchSize(totalLength);
    tiling.set_tileLength(tileLength);
    tiling.set_tileCount(tileCount);
    tiling.set_fillValue(1.0f);

    context->SetBlockDim(BLOCK_DIM);
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
    const gert::Shape *xShape = context->GetInputShape(0);
    if (xShape == nullptr || xShape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }

    gert::Shape *yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class MatmulDropoutMeanSoftmaxCustom : public OpDef {
public:
    explicit MatmulDropoutMeanSoftmaxCustom(const char *name) : OpDef(name)
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
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MatmulDropoutMeanSoftmaxCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class KernelMatmulDropoutMeanSoftmax {
public:
    __aicore__ inline KernelMatmulDropoutMeanSoftmax() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t batchSize,
        uint32_t tileLength,
        uint32_t tileCount,
        float fillValue)
    {
        (void)x;
        this->batchSize = batchSize;
        this->tileLength = tileLength;
        this->tileCount = tileCount;
        this->fillValue = fillValue;
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize);
        pipe.InitBuffer(outBuffer, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tileIdx = 0; tileIdx < this->tileCount; ++tileIdx) {
            const uint32_t offset = tileIdx * this->tileLength;
            uint32_t currentLength = this->tileLength;
            if (offset + currentLength > this->batchSize) {
                currentLength = this->batchSize - offset;
            }

            LocalTensor<float> outLocal = outBuffer.Get<float>();
            Duplicate(outLocal, this->fillValue, currentLength);
            DataCopy(yGm[offset], outLocal, currentLength);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> outBuffer;
    GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t tileLength;
    uint32_t tileCount;
    float fillValue;
};

extern "C" __global__ __aicore__ void matmul_dropout_mean_softmax_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulDropoutMeanSoftmax op;
    op.Init(
        x,
        y,
        tiling_data.batchSize,
        tiling_data.tileLength,
        tiling_data.tileCount,
        tiling_data.fillValue);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_dropout_mean_softmax_custom_impl_npu(const at::Tensor &x)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    at::Tensor y = at::empty({x.size(0), 1}, x.options());
    EXEC_NPU_CMD(aclnnMatmulDropoutMeanSoftmaxCustom, x, y);
    return y;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m)
{
    m.impl(
        "matmul_dropout_mean_softmax_custom",
        &matmul_dropout_mean_softmax_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "matmul_dropout_mean_softmax_custom",
        &matmul_dropout_mean_softmax_custom_impl_npu,
        "matmul_dropout_mean_softmax_custom");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_p = dropout_p

    def forward(self, x):
        return custom_ops_lib.matmul_dropout_mean_softmax_custom(x)
'''
