project_json_src='''
[
    {
        "op": "MatmulDivideGeluCustom",
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
                "name": "divisor",
                "param_type": "required",
                "type": "float"
            }
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulDivideGeluCustomTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(float, reciprocalDivisor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulDivideGeluCustom, MatmulDivideGeluCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "matmul_divide_gelu_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace optiling {
constexpr uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto shapeX = context->GetInputTensor(0)->GetOriginShape();
    auto shapeWeight = context->GetInputTensor(1)->GetOriginShape();
    auto shapeBias = context->GetInputTensor(2)->GetOriginShape();

    if (shapeX.GetDimNum() != 2 || shapeWeight.GetDimNum() != 2 || shapeBias.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int32_t m = shapeX.GetDim(0);
    const int32_t k = shapeX.GetDim(1);
    const int32_t weightK = shapeWeight.GetDim(0);
    const int32_t n = shapeWeight.GetDim(1);
    if (k != weightK || shapeBias.GetDim(0) != n) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *divisorPtr = attrs == nullptr ? nullptr : attrs->GetAttrPointer<float>(0);
    if (divisorPtr == nullptr || *divisorPtr == 0.0f) {
        return ge::GRAPH_FAILED;
    }

    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(m, n, k);
    cubeTiling.SetOrgShape(m, n, k);
    cubeTiling.SetFixSplit(128, 128, -1);
    cubeTiling.SetBias(true);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    MatmulDivideGeluCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_totalLength(static_cast<uint32_t>(m * n));
    tiling.set_tileNum(TILE_NUM);
    tiling.set_reciprocalDivisor(1.0f / *divisorPtr);

    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        context->SetBlockDim(2);
        context->SetTilingKey(2);
    } else {
        context->SetBlockDim(1);
        context->SetTilingKey(1);
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    const gert::Shape *biasShape = context->GetInputShape(2);
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(0) || weightShape->GetDim(1) != biasShape->GetDim(0)) {
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
class MatmulDivideGeluCustom : public OpDef {
public:
    explicit MatmulDivideGeluCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("divisor").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MatmulDivideGeluCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

constexpr int32_t BUFFER_NUM = 2;

template <typename aType, typename bType, typename cType, typename biasType>
class MatmulDivideGeluKernel {
public:
    __aicore__ inline MatmulDivideGeluKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t totalLength,
        uint32_t tileNum,
        float reciprocalDivisor,
        const TCubeTiling &tiling)
    {
        this->tiling = tiling;
        this->tileNum = tileNum;
        this->totalLength = totalLength;
        this->reciprocalDivisor = reciprocalDivisor;

        xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(x), tiling.M * tiling.Ka);
        weightGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(weight), tiling.Kb * tiling.N);
        yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(y), tiling.M * tiling.N);
        biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);

        int32_t offsetA = 0;
        int32_t offsetB = 0;
        int32_t offsetY = 0;
        int32_t offsetBias = 0;
        CalcOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetY, offsetBias);
        xGlobal = xGlobal[offsetA];
        weightGlobal = weightGlobal[offsetB];
        yGlobal = yGlobal[offsetY];
        biasGlobal = biasGlobal[offsetBias];

        const uint32_t mBlocks = CeilDiv(tiling.M, tiling.singleCoreM);
        const uint32_t mCoreIdx = GetBlockIdx() % mBlocks;
        const uint32_t nCoreIdx = GetBlockIdx() / mBlocks;
        this->validM = MinU32(static_cast<uint32_t>(tiling.singleCoreM), static_cast<uint32_t>(tiling.M - mCoreIdx * tiling.singleCoreM));
        this->validN = MinU32(static_cast<uint32_t>(tiling.singleCoreN), static_cast<uint32_t>(tiling.N - nCoreIdx * tiling.singleCoreN));
        this->tileLength = CeilDiv(this->validN, this->tileNum * BUFFER_NUM);

        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(cType));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(cType));
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(cType));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(cType));
    }

    template <bool setTmpSpace = false>
    __aicore__ inline void Process()
    {
        if constexpr (setTmpSpace) {
            AscendC::TBuf<> tmpMMFormatUb;
            AscendC::LocalTensor<uint8_t> mmformatUb;
            pipe.InitBuffer(tmpMMFormatUb, AscendC::TOTAL_VEC_LOCAL_SIZE);
            mmformatUb = tmpMMFormatUb.Get<uint8_t>(AscendC::TOTAL_VEC_LOCAL_SIZE);
            matmulObj.SetLocalWorkspace(mmformatUb);
        }

        matmulObj.SetTensorA(xGlobal);
        matmulObj.SetTensorB(weightGlobal);
        matmulObj.SetBias(biasGlobal);
        matmulObj.IterateAll(yGlobal);
        matmulObj.End();

        for (uint32_t row = 0; row < this->validM; ++row) {
            const uint32_t loopCount = this->tileNum * BUFFER_NUM;
            for (uint32_t i = 0; i < loopCount; ++i) {
                const uint32_t colOffset = i * this->tileLength;
                if (colOffset >= this->validN) {
                    break;
                }
                const uint32_t currentLength = MinU32(this->tileLength, this->validN - colOffset);
                const uint32_t gmOffset = row * tiling.N + colOffset;
                CopyIn(gmOffset, currentLength);
                Compute(currentLength);
                CopyOut(gmOffset, currentLength);
            }
        }
    }

    Matmul<
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>,
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    AscendC::TPipe pipe;

private:
    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t lhs, uint32_t rhs)
    {
        return (lhs + rhs - 1) / rhs;
    }

    __aicore__ inline void CalcOffset(
        int32_t blockIdx,
        const TCubeTiling &tiling,
        int32_t &offsetA,
        int32_t &offsetB,
        int32_t &offsetY,
        int32_t &offsetBias)
    {
        const int32_t mBlocks = CeilDiv(static_cast<uint32_t>(tiling.M), static_cast<uint32_t>(tiling.singleCoreM));
        const int32_t mCoreIdx = blockIdx % mBlocks;
        const int32_t nCoreIdx = blockIdx / mBlocks;

        offsetA = mCoreIdx * tiling.Ka * tiling.singleCoreM;
        offsetB = nCoreIdx * tiling.singleCoreN;
        offsetY = mCoreIdx * tiling.N * tiling.singleCoreM + nCoreIdx * tiling.singleCoreN;
        offsetBias = nCoreIdx * tiling.singleCoreN;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t currentLength)
    {
        AscendC::LocalTensor<cType> yLocal = inQueueY.AllocTensor<cType>();
        AscendC::DataCopy(yLocal, yGlobal[offset], currentLength);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t currentLength)
    {
        AscendC::LocalTensor<cType> yLocal = inQueueY.DeQue<cType>();
        AscendC::LocalTensor<cType> outLocal = outQueueY.AllocTensor<cType>();
        AscendC::LocalTensor<cType> tmpLocal1 = tmpBuffer1.Get<cType>();
        AscendC::LocalTensor<cType> tmpLocal2 = tmpBuffer2.Get<cType>();

        AscendC::Muls(tmpLocal1, yLocal, this->reciprocalDivisor, currentLength);
        AscendC::Mul(tmpLocal2, tmpLocal1, tmpLocal1, currentLength);
        AscendC::Mul(tmpLocal2, tmpLocal2, tmpLocal1, currentLength);
        AscendC::Muls(tmpLocal2, tmpLocal2, 0.0455399241f, currentLength);
        AscendC::Add(tmpLocal2, tmpLocal2, tmpLocal1, currentLength);
        AscendC::Muls(tmpLocal2, tmpLocal2, -1.595769122f, currentLength);
        AscendC::Exp(tmpLocal2, tmpLocal2, currentLength);
        AscendC::Adds(tmpLocal2, tmpLocal2, 1.0f, currentLength);
        AscendC::Div(outLocal, tmpLocal1, tmpLocal2, currentLength);

        outQueueY.EnQue<cType>(outLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t currentLength)
    {
        AscendC::LocalTensor<cType> outLocal = outQueueY.DeQue<cType>();
        AscendC::DataCopy(yGlobal[offset], outLocal, currentLength);
        outQueueY.FreeTensor(outLocal);
    }

private:
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer2;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<aType> xGlobal;
    AscendC::GlobalTensor<bType> weightGlobal;
    AscendC::GlobalTensor<cType> yGlobal;
    AscendC::GlobalTensor<biasType> biasGlobal;
    TCubeTiling tiling;
    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t validM;
    uint32_t validN;
    float reciprocalDivisor;
};

extern "C" __global__ __aicore__ void matmul_divide_gelu_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulDivideGeluKernel<float, float, float, float> op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulObj, &tilingData.cubeTilingData);
    op.Init(
        x,
        weight,
        bias,
        y,
        tilingData.totalLength,
        tilingData.tileNum,
        tilingData.reciprocalDivisor,
        tilingData.cubeTilingData);
    if (TILING_KEY_IS(1)) {
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        op.Process<true>();
    }
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor matmul_divide_gelu_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    double divisor)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(0), "x.size(1) must match weight.size(0)");
    TORCH_CHECK(weight.size(1) == bias.size(0), "bias size must match weight.size(1)");
    TORCH_CHECK(divisor != 0.0, "divisor must be non-zero");

    at::Tensor result = at::empty({x.size(0), weight.size(1)}, x.options());
    EXEC_NPU_CMD(aclnnMatmulDivideGeluCustom, x, weight, bias, divisor, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_divide_gelu_custom", &matmul_divide_gelu_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_divide_gelu_custom", &matmul_divide_gelu_custom_impl_npu, "matmul + divide + gelu");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.divisor = float(divisor)

    def forward(self, x):
        weight = self.linear.weight.transpose(0, 1).contiguous()
        return custom_ops_lib.matmul_divide_gelu_custom(
            x,
            weight,
            self.linear.bias,
            self.divisor,
        )
'''
