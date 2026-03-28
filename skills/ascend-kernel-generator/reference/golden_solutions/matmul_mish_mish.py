project_json_src='''
[
    {
        "op": "MatmulMishMishCustom",
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
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulMishMishCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, mDim);
TILING_DATA_FIELD_DEF(uint32_t, nDim);
TILING_DATA_FIELD_DEF(uint32_t, kDim);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulMishMishCustom, MatmulMishMishCustomTilingData)
} // namespace optiling
"""

host_operator_src="""
#include "matmul_mish_mish_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto shapeX = context->GetInputTensor(0)->GetOriginShape();
    auto shapeWeight = context->GetInputTensor(1)->GetOriginShape();
    auto shapeBias = context->GetInputTensor(2)->GetOriginShape();

    if (shapeX.GetDimNum() != 2 || shapeWeight.GetDimNum() != 2 || shapeBias.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int32_t m = static_cast<int32_t>(shapeX.GetDim(0));
    const int32_t k = static_cast<int32_t>(shapeX.GetDim(1));
    const int32_t weightK = static_cast<int32_t>(shapeWeight.GetDim(0));
    const int32_t n = static_cast<int32_t>(shapeWeight.GetDim(1));
    if (m <= 0 || k <= 0 || n <= 0 || weightK != k || shapeBias.GetDim(0) != n) {
        return ge::GRAPH_FAILED;
    }

    MatmulMishMishCustomTilingData tiling;
    tiling.set_mDim(static_cast<uint32_t>(m));
    tiling.set_nDim(static_cast<uint32_t>(n));
    tiling.set_kDim(static_cast<uint32_t>(k));
    tiling.set_blockDim(1);

    context->SetBlockDim(1);
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
    if (xShape == nullptr || weightShape == nullptr || biasShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDimNum() != 2 || weightShape->GetDimNum() != 2 || biasShape->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    if (xShape->GetDim(1) != weightShape->GetDim(0) || weightShape->GetDim(1) != biasShape->GetDim(0)) {
        return GRAPH_FAILED;
    }

    gert::Shape *yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, weightShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class MatmulMishMishCustom : public OpDef {
public:
    explicit MatmulMishMishCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(MatmulMishMishCustom);
} // namespace ops
"""

kernel_src="""
#include "kernel_operator.h"

using namespace AscendC;

class MatmulMishMishKernel {
public:
    __aicore__ inline MatmulMishMishKernel() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR weight,
        GM_ADDR bias,
        GM_ADDR y,
        uint32_t mDim,
        uint32_t nDim,
        uint32_t kDim,
        uint32_t blockDim)
    {
        this->mDim = mDim;
        this->nDim = nDim;
        this->kDim = kDim;
        this->blockDim = blockDim;
        xGm.SetGlobalBuffer((__gm__ float *)x, static_cast<uint64_t>(mDim) * kDim);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, static_cast<uint64_t>(kDim) * nDim);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, nDim);
        yGm.SetGlobalBuffer((__gm__ float *)y, static_cast<uint64_t>(mDim) * nDim);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = GetBlockIdx();
        for (uint32_t row = coreIdx; row < this->mDim; row += this->blockDim) {
            const uint64_t xOffset = static_cast<uint64_t>(row) * this->kDim;
            const uint64_t yOffset = static_cast<uint64_t>(row) * this->nDim;
            for (uint32_t col = 0; col < this->nDim; ++col) {
                float acc = biasGm.GetValue(col);
                for (uint32_t kk = 0; kk < this->kDim; ++kk) {
                    const float xValue = xGm.GetValue(xOffset + kk);
                    const float wValue = weightGm.GetValue(static_cast<uint64_t>(kk) * this->nDim + col);
                    acc += xValue * wValue;
                }
                const float mish1 = ApplyMish(acc);
                const float mish2 = ApplyMish(mish1);
                yGm.SetValue(yOffset + col, mish2);
            }
        }
    }

private:
    __aicore__ inline float ApplyMish(float x) const
    {
        if (x <= -8.0f) {
            return -2.683250883e-03f;
        }
        if (x >= 8.0f) {
            return x;
        }
        constexpr float kStart = -8.0f;
        constexpr float kInvStep = 1.6e+01f;
        const float scaled = (x - kStart) * kInvStep;
        int32_t idx = static_cast<int32_t>(scaled);
        if (idx < 0) {
            idx = 0;
        } else if (idx > 255) {
            idx = 255;
        }
        const float frac = scaled - static_cast<float>(idx);
        const float left = kMishTable[idx];
        const float right = kMishTable[idx + 1];
        return left + (right - left) * frac;
    }

private:
    static constexpr float kMishTable[257] = {
        -2.683250883e-03f, -2.833960146e-03f, -2.992946582e-03f, -3.160650785e-03f, -3.337535725e-03f, -3.524087806e-03f, -3.720817954e-03f, -3.928262756e-03f,
        -4.146985643e-03f, -4.377578112e-03f, -4.620661002e-03f, -4.876885810e-03f, -5.146936059e-03f, -5.431528714e-03f, -5.731415640e-03f, -6.047385119e-03f,
        -6.380263410e-03f, -6.730916353e-03f, -7.100251035e-03f, -7.489217491e-03f, -7.898810457e-03f, -8.330071170e-03f, -8.784089214e-03f, -9.262004398e-03f,
        -9.765008692e-03f, -1.029434818e-02f, -1.085132507e-02f, -1.143729969e-02f, -1.205369258e-02f, -1.270198654e-02f, -1.338372871e-02f, -1.410053266e-02f,
        -1.485408048e-02f, -1.564612485e-02f, -1.647849109e-02f, -1.735307918e-02f, -1.827186571e-02f, -1.923690578e-02f, -2.025033485e-02f, -2.131437041e-02f,
        -2.243131362e-02f, -2.360355070e-02f, -2.483355427e-02f, -2.612388440e-02f, -2.747718941e-02f, -2.889620649e-02f, -3.038376195e-02f, -3.194277109e-02f,
        -3.357623773e-02f, -3.528725327e-02f, -3.707899518e-02f, -3.895472503e-02f, -4.091778571e-02f, -4.297159813e-02f, -4.511965692e-02f, -4.736552537e-02f,
        -4.971282931e-02f, -5.216524991e-02f, -5.472651528e-02f, -5.740039066e-02f, -6.019066719e-02f, -6.310114903e-02f, -6.613563866e-02f, -6.929792024e-02f,
        -7.259174079e-02f, -7.602078904e-02f, -7.958867164e-02f, -8.329888665e-02f, -8.715479388e-02f, -9.115958198e-02f, -9.531623181e-02f, -9.962747606e-02f,
        -1.040957545e-01f, -1.087231647e-01f, -1.135114078e-01f, -1.184617292e-01f, -1.235748535e-01f, -1.288509127e-01f, -1.342893693e-01f, -1.398889315e-01f,
        -1.456474613e-01f, -1.515618749e-01f, -1.576280355e-01f, -1.638406370e-01f, -1.701930794e-01f, -1.766773345e-01f, -1.832838034e-01f, -1.900011628e-01f,
        -1.968162039e-01f, -2.037136597e-01f, -2.106760247e-01f, -2.176833648e-01f, -2.247131192e-01f, -2.317398955e-01f, -2.387352578e-01f, -2.456675111e-01f,
        -2.525014827e-01f, -2.591983037e-01f, -2.657151937e-01f, -2.720052518e-01f, -2.780172592e-01f, -2.836954965e-01f, -2.889795847e-01f, -2.938043527e-01f,
        -2.980997422e-01f, -3.017907556e-01f, -3.047974579e-01f, -3.070350411e-01f, -3.084139611e-01f, -3.088401592e-01f, -3.082153764e-01f, -3.064375730e-01f,
        -3.034014614e-01f, -2.989991594e-01f, -2.931209730e-01f, -2.856563081e-01f, -2.764947145e-01f, -2.655270558e-01f, -2.526467968e-01f, -2.377513952e-01f,
        -2.207437747e-01f, -2.015338554e-01f, -1.800401088e-01f, -1.561910984e-01f, -1.299269637e-01f, -1.012008033e-01f, -6.997990718e-02f, -3.624679280e-02f,
        0.000000000e+00f, 3.874539399e-02f, 7.995757085e-02f, 1.235881000e-01f, 1.695724097e-01f, 2.178305969e-01f, 2.682685597e-01f, 3.207794223e-01f,
        3.752452113e-01f, 4.315387328e-01f, 4.895255911e-01f, 5.490662876e-01f, 6.100183352e-01f, 6.722383283e-01f, 7.355839110e-01f, 7.999155951e-01f,
        8.650983883e-01f, 9.310032025e-01f, 9.975080244e-01f, 1.064498838e+00f, 1.131870304e+00f, 1.199526199e+00f, 1.267379638e+00f, 1.335353098e+00f,
        1.403378266e+00f, 1.471395744e+00f, 1.539354633e+00f, 1.607212023e+00f, 1.674932424e+00f, 1.742487148e+00f, 1.809853669e+00f, 1.877014980e+00f,
        1.943958960e+00f, 2.010677759e+00f, 2.077167223e+00f, 2.143426343e+00f, 2.209456758e+00f, 2.275262292e+00f, 2.340848544e+00f, 2.406222520e+00f,
        2.471392305e+00f, 2.536366779e+00f, 2.601155376e+00f, 2.665767871e+00f, 2.730214204e+00f, 2.794504335e+00f, 2.858648119e+00f, 2.922655216e+00f,
        2.986535005e+00f, 3.050296534e+00f, 3.113948471e+00f, 3.177499074e+00f, 3.240956173e+00f, 3.304327158e+00f, 3.367618977e+00f, 3.430838137e+00f,
        3.493990719e+00f, 3.557082395e+00f, 3.620118439e+00f, 3.683103748e+00f, 3.746042860e+00f, 3.808939973e+00f, 3.871799000e+00f, 3.934623589e+00f,
        3.997417150e+00f, 4.060182860e+00f, 4.122923685e+00f, 4.185642389e+00f, 4.248341555e+00f, 4.311023599e+00f, 4.373690785e+00f, 4.436345239e+00f,
        4.498988949e+00f, 4.561623780e+00f, 4.624251479e+00f, 4.686873684e+00f, 4.749491940e+00f, 4.812107697e+00f, 4.874722321e+00f, 4.937337099e+00f,
        4.999953240e+00f, 5.062571877e+00f, 5.125194074e+00f, 5.187820829e+00f, 5.250453077e+00f, 5.313091694e+00f, 5.375737501e+00f, 5.438391264e+00f,
        5.501053696e+00f, 5.563725461e+00f, 5.626407174e+00f, 5.689099400e+00f, 5.751802664e+00f, 5.814517454e+00f, 5.877244220e+00f, 5.939983383e+00f,
        6.002735335e+00f, 6.065500441e+00f, 6.128279040e+00f, 6.191071446e+00f, 6.253877949e+00f, 6.316698816e+00f, 6.379534291e+00f, 6.442384602e+00f,
        6.505249958e+00f, 6.568130554e+00f, 6.631026571e+00f, 6.693938176e+00f, 6.756865528e+00f, 6.819808770e+00f, 6.882768036e+00f, 6.945743449e+00f,
        7.008735120e+00f, 7.071743153e+00f, 7.134767638e+00f, 7.197808661e+00f, 7.260866297e+00f, 7.323940619e+00f, 7.387031687e+00f, 7.450139557e+00f,
        7.513264273e+00f
    };

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    uint32_t mDim;
    uint32_t nDim;
    uint32_t kDim;
    uint32_t blockDim;
};

extern "C" __global__ __aicore__ void matmul_mish_mish_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulMishMishKernel op;
    op.Init(x, weight, bias, y, tilingData.mDim, tilingData.nDim, tilingData.kDim, tilingData.blockDim);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor matmul_mish_mish_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");
    TORCH_CHECK(x.size(1) == weight.size(0), "x.size(1) must equal weight.size(0)");
    TORCH_CHECK(weight.size(1) == bias.size(0), "weight.size(1) must equal bias.size(0)");

    at::Tensor result = at::empty({x.size(0), weight.size(1)}, x.options());
    EXEC_NPU_CMD(aclnnMatmulMishMishCustom, x, weight, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("matmul_mish_mish_custom", &matmul_mish_mish_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_mish_mish_custom", &matmul_mish_mish_custom_impl_npu, "matmul + mish + mish");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        weight = self.linear.weight.transpose(0, 1).contiguous()
        return custom_ops_lib.matmul_mish_mish_custom(x, weight, self.linear.bias)
'''
