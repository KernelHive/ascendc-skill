project_json_src='''
[
    {
        "op": "MatmulWithDiagonalMatricesCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "diag",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "float"
                ]
            },
            {
                "name": "matrix",
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
                "name": "out",
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
BEGIN_TILING_DATA_DEF(MatmulWithDiagonalMatricesCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulWithDiagonalMatricesCustom, MatmulWithDiagonalMatricesCustomTilingData)
}
"""

host_operator_src="""
#include "matmul_with_diagonal_matrices_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 2048;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulWithDiagonalMatricesCustomTilingData tiling;
    const uint32_t totalLength = context->GetInputShape(1)->GetOriginShape().GetShapeSize();

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* matrixShape = context->GetInputShape(1);
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = *matrixShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(1));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MatmulWithDiagonalMatricesCustom : public OpDef {
public:
    explicit MatmulWithDiagonalMatricesCustom(const char* name) : OpDef(name)
    {
        this->Input("diag")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("matrix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MatmulWithDiagonalMatricesCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmulWithDiagonalMatrices {
public:
    __aicore__ inline KernelMatmulWithDiagonalMatrices() {}

    __aicore__ inline void Init(
        GM_ADDR diag,
        GM_ADDR matrix,
        GM_ADDR out,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        diagGm.SetGlobalBuffer((__gm__ DTYPE_DIAG*)diag + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        matrixGm.SetGlobalBuffer((__gm__ DTYPE_MATRIX*)matrix + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)out + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueDiag, BUFFER_NUM, this->tileLength * sizeof(DTYPE_DIAG));
        pipe.InitBuffer(inQueueMatrix, BUFFER_NUM, this->tileLength * sizeof(DTYPE_MATRIX));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(DTYPE_OUT));
    }

    __aicore__ inline void Process()
    {
        const int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_DIAG> diagLocal = inQueueDiag.AllocTensor<DTYPE_DIAG>();
        AscendC::LocalTensor<DTYPE_MATRIX> matrixLocal = inQueueMatrix.AllocTensor<DTYPE_MATRIX>();
        AscendC::DataCopy(diagLocal, diagGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(matrixLocal, matrixGm[progress * this->tileLength], this->tileLength);
        inQueueDiag.EnQue(diagLocal);
        inQueueMatrix.EnQue(matrixLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DTYPE_DIAG> diagLocal = inQueueDiag.DeQue<DTYPE_DIAG>();
        AscendC::LocalTensor<DTYPE_MATRIX> matrixLocal = inQueueMatrix.DeQue<DTYPE_MATRIX>();
        AscendC::LocalTensor<DTYPE_OUT> outLocal = outQueue.AllocTensor<DTYPE_OUT>();
        AscendC::Mul(outLocal, diagLocal, matrixLocal, this->tileLength);
        outQueue.EnQue<DTYPE_OUT>(outLocal);
        inQueueDiag.FreeTensor(diagLocal);
        inQueueMatrix.FreeTensor(matrixLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_OUT> outLocal = outQueue.DeQue<DTYPE_OUT>();
        AscendC::DataCopy(outGm[progress * this->tileLength], outLocal, this->tileLength);
        outQueue.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueDiag;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueMatrix;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<DTYPE_DIAG> diagGm;
    AscendC::GlobalTensor<DTYPE_MATRIX> matrixGm;
    AscendC::GlobalTensor<DTYPE_OUT> outGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void matmul_with_diagonal_matrices_custom(
    GM_ADDR diag,
    GM_ADDR matrix,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulWithDiagonalMatrices op;
    op.Init(diag, matrix, out, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor matmul_with_diagonal_matrices_impl_npu(const at::Tensor& diag, const at::Tensor& matrix)
{
    at::Tensor diagExpanded = diag.reshape({diag.size(0), 1}).expand({diag.size(0), matrix.size(1)}).contiguous();
    at::Tensor matrixContiguous = matrix.contiguous();
    at::Tensor result = at::empty_like(matrixContiguous);
    EXEC_NPU_CMD(aclnnMatmulWithDiagonalMatricesCustom, diagExpanded, matrixContiguous, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_with_diagonal_matrices_custom", &matmul_with_diagonal_matrices_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "matmul_with_diagonal_matrices_custom",
        &matmul_with_diagonal_matrices_impl_npu,
        "matrix multiply with diagonal matrix");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return custom_ops_lib.matmul_with_diagonal_matrices_custom(a, b)
'''
