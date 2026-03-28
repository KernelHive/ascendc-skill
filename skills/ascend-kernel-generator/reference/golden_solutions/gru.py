project_json_src = '''
[
    {
        "op": "GruCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float16"]
            },
            {
                "name": "w_ih",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float16"]
            },
            {
                "name": "w_hh",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float16"]
            },
            {
                "name": "b_ih",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float16"]
            },
            {
                "name": "b_hh",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float16"]
            },
            {
                "name": "h0",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float16"]
            }
        ],
        "output_desc": [
            {
                "name": "y",
                "param_type": "required",
                "format": ["ND"],
                "type": ["float16"]
            }
        ]
    }
]
'''

host_tiling_src = """
#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GruCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, seqLen);
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inputSize);
    TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GruCustom, GruCustomTilingData)
}
"""

host_operator_src = """
#include <cstdint>
#include "gru_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xShape = context->GetInputShape(0);
    const gert::StorageShape *wIhShape = context->GetInputShape(1);
    const gert::StorageShape *wHhShape = context->GetInputShape(2);
    const gert::StorageShape *bIhShape = context->GetInputShape(3);
    const gert::StorageShape *bHhShape = context->GetInputShape(4);
    const gert::StorageShape *h0Shape = context->GetInputShape(5);
    if (xShape == nullptr || wIhShape == nullptr || wHhShape == nullptr || bIhShape == nullptr ||
        bHhShape == nullptr || h0Shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &xStorage = xShape->GetStorageShape();
    const auto &wIhStorage = wIhShape->GetStorageShape();
    const auto &wHhStorage = wHhShape->GetStorageShape();
    const auto &bIhStorage = bIhShape->GetStorageShape();
    const auto &bHhStorage = bHhShape->GetStorageShape();
    const auto &h0Storage = h0Shape->GetStorageShape();
    if (xStorage.GetDimNum() != 3 || wIhStorage.GetDimNum() != 2 || wHhStorage.GetDimNum() != 2 ||
        bIhStorage.GetDimNum() != 1 || bHhStorage.GetDimNum() != 1 || h0Storage.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t seqLen = static_cast<uint32_t>(xStorage.GetDim(0));
    const uint32_t batchSize = static_cast<uint32_t>(xStorage.GetDim(1));
    const uint32_t inputSize = static_cast<uint32_t>(xStorage.GetDim(2));
    const uint32_t hiddenSize = static_cast<uint32_t>(h0Storage.GetDim(2));

    if (wIhStorage.GetDim(0) != static_cast<int64_t>(3 * hiddenSize) ||
        wIhStorage.GetDim(1) != static_cast<int64_t>(inputSize) ||
        wHhStorage.GetDim(0) != static_cast<int64_t>(3 * hiddenSize) ||
        wHhStorage.GetDim(1) != static_cast<int64_t>(hiddenSize) ||
        bIhStorage.GetDim(0) != static_cast<int64_t>(3 * hiddenSize) ||
        bHhStorage.GetDim(0) != static_cast<int64_t>(3 * hiddenSize) ||
        h0Storage.GetDim(0) != 1 ||
        h0Storage.GetDim(1) != static_cast<int64_t>(batchSize)) {
        return ge::GRAPH_FAILED;
    }

    GruCustomTilingData tiling;
    tiling.set_seqLen(seqLen);
    tiling.set_batchSize(batchSize);
    tiling.set_inputSize(inputSize);
    tiling.set_hiddenSize(hiddenSize);

    context->SetBlockDim(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    const gert::Shape *h0Shape = context->GetInputShape(5);
    if (xShape == nullptr || h0Shape == nullptr || xShape->GetDimNum() != 3 || h0Shape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }

    gert::Shape *yShape = context->GetOutputShape(0);
    yShape->SetDimNum(3);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, xShape->GetDim(1));
    yShape->SetDim(2, h0Shape->GetDim(2));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class GruCustom : public OpDef {
public:
    explicit GruCustom(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("w_ih").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("w_hh").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("b_ih").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("b_hh").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("h0").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(GruCustom);
}
"""

kernel_src = """
#include <cstdint>
#include "kernel_operator.h"

class KernelGru {
public:
    __aicore__ inline KernelGru() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR wIh,
        GM_ADDR wHh,
        GM_ADDR bIh,
        GM_ADDR bHh,
        GM_ADDR h0,
        GM_ADDR y,
        uint32_t seqLen,
        uint32_t batchSize,
        uint32_t inputSize,
        uint32_t hiddenSize)
    {
        this->seqLen = seqLen;
        this->batchSize = batchSize;
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, static_cast<uint64_t>(seqLen) * batchSize * inputSize);
        wIhGm.SetGlobalBuffer((__gm__ DTYPE_W_IH *)wIh, static_cast<uint64_t>(3 * hiddenSize) * inputSize);
        wHhGm.SetGlobalBuffer((__gm__ DTYPE_W_HH *)wHh, static_cast<uint64_t>(3 * hiddenSize) * hiddenSize);
        bIhGm.SetGlobalBuffer((__gm__ DTYPE_B_IH *)bIh, 3 * hiddenSize);
        bHhGm.SetGlobalBuffer((__gm__ DTYPE_B_HH *)bHh, 3 * hiddenSize);
        h0Gm.SetGlobalBuffer((__gm__ DTYPE_H0 *)h0, static_cast<uint64_t>(batchSize) * hiddenSize);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, static_cast<uint64_t>(seqLen) * batchSize * hiddenSize);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t t = 0; t < seqLen; ++t) {
            for (uint32_t b = 0; b < batchSize; ++b) {
                const uint64_t xBase = (static_cast<uint64_t>(t) * batchSize + b) * inputSize;
                const uint64_t yBase = (static_cast<uint64_t>(t) * batchSize + b) * hiddenSize;
                const uint64_t prevBase = (t == 0)
                    ? static_cast<uint64_t>(b) * hiddenSize
                    : (static_cast<uint64_t>(t - 1) * batchSize + b) * hiddenSize;

                for (uint32_t h = 0; h < hiddenSize; ++h) {
                    const float prevHidden = LoadPrevHidden(t, prevBase, h);
                    const float resetGate = ApproxSigmoid(
                        DotInput(xBase, h) +
                        DotHidden(t, prevBase, h) +
                        static_cast<float>(bIhGm.GetValue(h)) +
                        static_cast<float>(bHhGm.GetValue(h)));
                    const float updateGate = ApproxSigmoid(
                        DotInput(xBase, hiddenSize + h) +
                        DotHidden(t, prevBase, hiddenSize + h) +
                        static_cast<float>(bIhGm.GetValue(hiddenSize + h)) +
                        static_cast<float>(bHhGm.GetValue(hiddenSize + h)));
                    const float candidate = ApproxTanh(
                        DotInput(xBase, 2 * hiddenSize + h) +
                        static_cast<float>(bIhGm.GetValue(2 * hiddenSize + h)) +
                        resetGate * (
                            DotHidden(t, prevBase, 2 * hiddenSize + h) +
                            static_cast<float>(bHhGm.GetValue(2 * hiddenSize + h))));
                    const float hiddenValue = (1.0f - updateGate) * candidate + updateGate * prevHidden;
                    yGm.SetValue(yBase + h, static_cast<DTYPE_Y>(hiddenValue));
                }
            }
        }
    }

private:
    __aicore__ inline float LoadPrevHidden(uint32_t t, uint64_t prevBase, uint32_t h) const
    {
        if (t == 0) {
            return static_cast<float>(h0Gm.GetValue(prevBase + h));
        }
        return static_cast<float>(yGm.GetValue(prevBase + h));
    }

    __aicore__ inline float DotInput(uint64_t xBase, uint32_t gateIndex) const
    {
        const uint64_t weightBase = static_cast<uint64_t>(gateIndex) * inputSize;
        float acc = 0.0f;
        for (uint32_t i = 0; i < inputSize; ++i) {
            acc += static_cast<float>(xGm.GetValue(xBase + i)) * static_cast<float>(wIhGm.GetValue(weightBase + i));
        }
        return acc;
    }

    __aicore__ inline float DotHidden(uint32_t t, uint64_t prevBase, uint32_t gateIndex) const
    {
        const uint64_t weightBase = static_cast<uint64_t>(gateIndex) * hiddenSize;
        float acc = 0.0f;
        for (uint32_t i = 0; i < hiddenSize; ++i) {
            const float hidden = LoadPrevHidden(t, prevBase, i);
            acc += hidden * static_cast<float>(wHhGm.GetValue(weightBase + i));
        }
        return acc;
    }

    __aicore__ inline float ApproxSigmoid(float x) const
    {
        return 0.5f * (ApproxTanh(0.5f * x) + 1.0f);
    }

    __aicore__ inline float ApproxTanh(float x) const
    {
        if (x <= -8.0f) {
            return -9.999997749e-01f;
        }
        if (x >= 8.0f) {
            return 9.999997749e-01f;
        }

        constexpr float kStart = -8.0f;
        constexpr float kInvStep = 3.2e+01f;
        const float scaled = (x - kStart) * kInvStep;
        int32_t idx = static_cast<int32_t>(scaled);
        if (idx < 0) {
            idx = 0;
        } else if (idx > 511) {
            idx = 511;
        }
        const float frac = scaled - static_cast<float>(idx);
        const float left = kTanhTable[idx];
        const float right = kTanhTable[idx + 1];
        return left + (right - left) * frac;
    }

private:
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_W_IH> wIhGm;
    AscendC::GlobalTensor<DTYPE_W_HH> wHhGm;
    AscendC::GlobalTensor<DTYPE_B_IH> bIhGm;
    AscendC::GlobalTensor<DTYPE_B_HH> bHhGm;
    AscendC::GlobalTensor<DTYPE_H0> h0Gm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t seqLen = 0;
    uint32_t batchSize = 0;
    uint32_t inputSize = 0;
    uint32_t hiddenSize = 0;

    static constexpr float kTanhTable[513] = {
        -9.999997749e-01f, -9.999997604e-01f, -9.999997450e-01f, -9.999997285e-01f, -9.999997110e-01f, -9.999996924e-01f, -9.999996725e-01f, -9.999996514e-01f,
        -9.999996289e-01f, -9.999996050e-01f, -9.999995795e-01f, -9.999995524e-01f, -9.999995235e-01f, -9.999994928e-01f, -9.999994601e-01f, -9.999994253e-01f,
        -9.999993882e-01f, -9.999993487e-01f, -9.999993067e-01f, -9.999992620e-01f, -9.999992144e-01f, -9.999991638e-01f, -9.999991098e-01f, -9.999990524e-01f,
        -9.999989913e-01f, -9.999989262e-01f, -9.999988570e-01f, -9.999987833e-01f, -9.999987048e-01f, -9.999986213e-01f, -9.999985324e-01f, -9.999984377e-01f,
        -9.999983369e-01f, -9.999982297e-01f, -9.999981155e-01f, -9.999979940e-01f, -9.999978646e-01f, -9.999977269e-01f, -9.999975803e-01f, -9.999974242e-01f,
        -9.999972581e-01f, -9.999970812e-01f, -9.999968930e-01f, -9.999966926e-01f, -9.999964793e-01f, -9.999962522e-01f, -9.999960105e-01f, -9.999957532e-01f,
        -9.999954794e-01f, -9.999951878e-01f, -9.999948774e-01f, -9.999945471e-01f, -9.999941954e-01f, -9.999938210e-01f, -9.999934225e-01f, -9.999929983e-01f,
        -9.999925467e-01f, -9.999920660e-01f, -9.999915543e-01f, -9.999910096e-01f, -9.999904298e-01f, -9.999898126e-01f, -9.999891556e-01f, -9.999884562e-01f,
        -9.999877117e-01f, -9.999869191e-01f, -9.999860755e-01f, -9.999851774e-01f, -9.999842215e-01f, -9.999832039e-01f, -9.999821206e-01f, -9.999809675e-01f,
        -9.999797400e-01f, -9.999784334e-01f, -9.999770425e-01f, -9.999755618e-01f, -9.999739857e-01f, -9.999723080e-01f, -9.999705220e-01f, -9.999686209e-01f,
        -9.999665972e-01f, -9.999644429e-01f, -9.999621497e-01f, -9.999597086e-01f, -9.999571101e-01f, -9.999543440e-01f, -9.999513995e-01f, -9.999482651e-01f,
        -9.999449286e-01f, -9.999413769e-01f, -9.999375962e-01f, -9.999335716e-01f, -9.999292875e-01f, -9.999247271e-01f, -9.999198726e-01f, -9.999147051e-01f,
        -9.999092043e-01f, -9.999033487e-01f, -9.998971156e-01f, -9.998904805e-01f, -9.998834175e-01f, -9.998758990e-01f, -9.998678957e-01f, -9.998593763e-01f,
        -9.998503075e-01f, -9.998406540e-01f, -9.998303779e-01f, -9.998194392e-01f, -9.998077952e-01f, -9.997954003e-01f, -9.997822062e-01f, -9.997681613e-01f,
        -9.997532108e-01f, -9.997372964e-01f, -9.997203558e-01f, -9.997023230e-01f, -9.996831276e-01f, -9.996626945e-01f, -9.996409441e-01f, -9.996177914e-01f,
        -9.995931460e-01f, -9.995669119e-01f, -9.995389866e-01f, -9.995092610e-01f, -9.994776194e-01f, -9.994439381e-01f, -9.994080858e-01f, -9.993699226e-01f,
        -9.993292997e-01f, -9.992860587e-01f, -9.992400310e-01f, -9.991910370e-01f, -9.991388858e-01f, -9.990833742e-01f, -9.990242858e-01f, -9.989613903e-01f,
        -9.988944427e-01f, -9.988231824e-01f, -9.987473317e-01f, -9.986665954e-01f, -9.985806592e-01f, -9.984891887e-01f, -9.983918281e-01f, -9.982881987e-01f,
        -9.981778976e-01f, -9.980604961e-01f, -9.979355379e-01f, -9.978025379e-01f, -9.976609795e-01f, -9.975103134e-01f, -9.973499552e-01f, -9.971792830e-01f,
        -9.969976355e-01f, -9.968043090e-01f, -9.965985552e-01f, -9.963795779e-01f, -9.961465307e-01f, -9.958985129e-01f, -9.956345671e-01f, -9.953536750e-01f,
        -9.950547537e-01f, -9.947366521e-01f, -9.943981461e-01f, -9.940379345e-01f, -9.936546343e-01f, -9.932467752e-01f, -9.928127948e-01f, -9.923510327e-01f,
        -9.918597246e-01f, -9.913369960e-01f, -9.907808556e-01f, -9.901891886e-01f, -9.895597486e-01f, -9.888901506e-01f, -9.881778623e-01f, -9.874201957e-01f,
        -9.866142982e-01f, -9.857571425e-01f, -9.848455175e-01f, -9.838760169e-01f, -9.828450292e-01f, -9.817487252e-01f, -9.805830470e-01f, -9.793436950e-01f,
        -9.780261147e-01f, -9.766254840e-01f, -9.751366983e-01f, -9.735543565e-01f, -9.718727459e-01f, -9.700858268e-01f, -9.681872166e-01f, -9.661701735e-01f,
        -9.640275801e-01f, -9.617519265e-01f, -9.593352933e-01f, -9.567693345e-01f, -9.540452602e-01f, -9.511538199e-01f, -9.480852856e-01f, -9.448294355e-01f,
        -9.413755385e-01f, -9.377123389e-01f, -9.338280432e-01f, -9.297103072e-01f, -9.253462253e-01f, -9.207223218e-01f, -9.158245442e-01f, -9.106382595e-01f,
        -9.051482536e-01f, -8.993387348e-01f, -8.931933404e-01f, -8.866951494e-01f, -8.798266997e-01f, -8.725700115e-01f, -8.649066177e-01f, -8.568176011e-01f,
        -8.482836400e-01f, -8.392850624e-01f, -8.298019100e-01f, -8.198140121e-01f, -8.093010702e-01f, -7.982427545e-01f, -7.866188121e-01f, -7.744091874e-01f,
        -7.615941560e-01f, -7.481544703e-01f, -7.340715196e-01f, -7.193275010e-01f, -7.039056039e-01f, -6.877902051e-01f, -6.709670742e-01f, -6.534235881e-01f,
        -6.351489524e-01f, -6.161344271e-01f, -5.963735555e-01f, -5.758623913e-01f, -5.545997223e-01f, -5.325872862e-01f, -5.098299737e-01f, -4.863360172e-01f,
        -4.621171573e-01f, -4.371887851e-01f, -4.115700557e-01f, -3.852839663e-01f, -3.583573984e-01f, -3.308211175e-01f, -3.027097293e-01f, -2.740615890e-01f,
        -2.449186624e-01f, -2.153263397e-01f, -1.853331999e-01f, -1.549907304e-01f, -1.243530018e-01f, -9.347630397e-02f, -6.241874675e-02f, -3.123983145e-02f,
        0.000000000e+00f, 3.123983145e-02f, 6.241874675e-02f, 9.347630397e-02f, 1.243530018e-01f, 1.549907304e-01f, 1.853331999e-01f, 2.153263397e-01f,
        2.449186624e-01f, 2.740615890e-01f, 3.027097293e-01f, 3.308211175e-01f, 3.583573984e-01f, 3.852839663e-01f, 4.115700557e-01f, 4.371887851e-01f,
        4.621171573e-01f, 4.863360172e-01f, 5.098299737e-01f, 5.325872862e-01f, 5.545997223e-01f, 5.758623913e-01f, 5.963735555e-01f, 6.161344271e-01f,
        6.351489524e-01f, 6.534235881e-01f, 6.709670742e-01f, 6.877902051e-01f, 7.039056039e-01f, 7.193275010e-01f, 7.340715196e-01f, 7.481544703e-01f,
        7.615941560e-01f, 7.744091874e-01f, 7.866188121e-01f, 7.982427545e-01f, 8.093010702e-01f, 8.198140121e-01f, 8.298019100e-01f, 8.392850624e-01f,
        8.482836400e-01f, 8.568176011e-01f, 8.649066177e-01f, 8.725700115e-01f, 8.798266997e-01f, 8.866951494e-01f, 8.931933404e-01f, 8.993387348e-01f,
        9.051482536e-01f, 9.106382595e-01f, 9.158245442e-01f, 9.207223218e-01f, 9.253462253e-01f, 9.297103072e-01f, 9.338280432e-01f, 9.377123389e-01f,
        9.413755385e-01f, 9.448294355e-01f, 9.480852856e-01f, 9.511538199e-01f, 9.540452602e-01f, 9.567693345e-01f, 9.593352933e-01f, 9.617519265e-01f,
        9.640275801e-01f, 9.661701735e-01f, 9.681872166e-01f, 9.700858268e-01f, 9.718727459e-01f, 9.735543565e-01f, 9.751366983e-01f, 9.766254840e-01f,
        9.780261147e-01f, 9.793436950e-01f, 9.805830470e-01f, 9.817487252e-01f, 9.828450292e-01f, 9.838760169e-01f, 9.848455175e-01f, 9.857571425e-01f,
        9.866142982e-01f, 9.874201957e-01f, 9.881778623e-01f, 9.888901506e-01f, 9.895597486e-01f, 9.901891886e-01f, 9.907808556e-01f, 9.913369960e-01f,
        9.918597246e-01f, 9.923510327e-01f, 9.928127948e-01f, 9.932467752e-01f, 9.936546343e-01f, 9.940379345e-01f, 9.943981461e-01f, 9.947366521e-01f,
        9.950547537e-01f, 9.953536750e-01f, 9.956345671e-01f, 9.958985129e-01f, 9.961465307e-01f, 9.963795779e-01f, 9.965985552e-01f, 9.968043090e-01f,
        9.969976355e-01f, 9.971792830e-01f, 9.973499552e-01f, 9.975103134e-01f, 9.976609795e-01f, 9.978025379e-01f, 9.979355379e-01f, 9.980604961e-01f,
        9.981778976e-01f, 9.982881987e-01f, 9.983918281e-01f, 9.984891887e-01f, 9.985806592e-01f, 9.986665954e-01f, 9.987473317e-01f, 9.988231824e-01f,
        9.988944427e-01f, 9.989613903e-01f, 9.990242858e-01f, 9.990833742e-01f, 9.991388858e-01f, 9.991910370e-01f, 9.992400310e-01f, 9.992860587e-01f,
        9.993292997e-01f, 9.993699226e-01f, 9.994080858e-01f, 9.994439381e-01f, 9.994776194e-01f, 9.995092610e-01f, 9.995389866e-01f, 9.995669119e-01f,
        9.995931460e-01f, 9.996177914e-01f, 9.996409441e-01f, 9.996626945e-01f, 9.996831276e-01f, 9.997023230e-01f, 9.997203558e-01f, 9.997372964e-01f,
        9.997532108e-01f, 9.997681613e-01f, 9.997822062e-01f, 9.997954003e-01f, 9.998077952e-01f, 9.998194392e-01f, 9.998303779e-01f, 9.998406540e-01f,
        9.998503075e-01f, 9.998593763e-01f, 9.998678957e-01f, 9.998758990e-01f, 9.998834175e-01f, 9.998904805e-01f, 9.998971156e-01f, 9.999033487e-01f,
        9.999092043e-01f, 9.999147051e-01f, 9.999198726e-01f, 9.999247271e-01f, 9.999292875e-01f, 9.999335716e-01f, 9.999375962e-01f, 9.999413769e-01f,
        9.999449286e-01f, 9.999482651e-01f, 9.999513995e-01f, 9.999543440e-01f, 9.999571101e-01f, 9.999597086e-01f, 9.999621497e-01f, 9.999644429e-01f,
        9.999665972e-01f, 9.999686209e-01f, 9.999705220e-01f, 9.999723080e-01f, 9.999739857e-01f, 9.999755618e-01f, 9.999770425e-01f, 9.999784334e-01f,
        9.999797400e-01f, 9.999809675e-01f, 9.999821206e-01f, 9.999832039e-01f, 9.999842215e-01f, 9.999851774e-01f, 9.999860755e-01f, 9.999869191e-01f,
        9.999877117e-01f, 9.999884562e-01f, 9.999891556e-01f, 9.999898126e-01f, 9.999904298e-01f, 9.999910096e-01f, 9.999915543e-01f, 9.999920660e-01f,
        9.999925467e-01f, 9.999929983e-01f, 9.999934225e-01f, 9.999938210e-01f, 9.999941954e-01f, 9.999945471e-01f, 9.999948774e-01f, 9.999951878e-01f,
        9.999954794e-01f, 9.999957532e-01f, 9.999960105e-01f, 9.999962522e-01f, 9.999964793e-01f, 9.999966926e-01f, 9.999968930e-01f, 9.999970812e-01f,
        9.999972581e-01f, 9.999974242e-01f, 9.999975803e-01f, 9.999977269e-01f, 9.999978646e-01f, 9.999979940e-01f, 9.999981155e-01f, 9.999982297e-01f,
        9.999983369e-01f, 9.999984377e-01f, 9.999985324e-01f, 9.999986213e-01f, 9.999987048e-01f, 9.999987833e-01f, 9.999988570e-01f, 9.999989262e-01f,
        9.999989913e-01f, 9.999990524e-01f, 9.999991098e-01f, 9.999991638e-01f, 9.999992144e-01f, 9.999992620e-01f, 9.999993067e-01f, 9.999993487e-01f,
        9.999993882e-01f, 9.999994253e-01f, 9.999994601e-01f, 9.999994928e-01f, 9.999995235e-01f, 9.999995524e-01f, 9.999995795e-01f, 9.999996050e-01f,
        9.999996289e-01f, 9.999996514e-01f, 9.999996725e-01f, 9.999996924e-01f, 9.999997110e-01f, 9.999997285e-01f, 9.999997450e-01f, 9.999997604e-01f,
        9.999997749e-01f
    };
};

extern "C" __global__ __aicore__ void gru_custom(
    GM_ADDR x,
    GM_ADDR w_ih,
    GM_ADDR w_hh,
    GM_ADDR b_ih,
    GM_ADDR b_hh,
    GM_ADDR h0,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelGru op;
    op.Init(x, w_ih, w_hh, b_ih, b_hh, h0, y, tiling_data.seqLen, tiling_data.batchSize, tiling_data.inputSize, tiling_data.hiddenSize);
    op.Process();
}
"""

python_bind_src = """
#include <torch/library.h>
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

at::Tensor gru_custom_impl_npu(
    const at::Tensor &x,
    const at::Tensor &w_ih,
    const at::Tensor &w_hh,
    const at::Tensor &b_ih,
    const at::Tensor &b_hh,
    const at::Tensor &h0)
{
    TORCH_CHECK(x.dim() == 3, "x must be [seq, batch, input]");
    TORCH_CHECK(w_ih.dim() == 2 && w_hh.dim() == 2, "weights must be rank-2");
    TORCH_CHECK(b_ih.dim() == 1 && b_hh.dim() == 1, "biases must be rank-1");
    TORCH_CHECK(h0.dim() == 3, "h0 must be [1, batch, hidden]");
    TORCH_CHECK(h0.size(0) == 1, "only single-layer GRU is supported");
    TORCH_CHECK(x.size(2) == w_ih.size(1), "input size mismatch");
    TORCH_CHECK(h0.size(2) == w_hh.size(1), "hidden size mismatch");
    TORCH_CHECK(w_ih.size(0) == 3 * h0.size(2), "w_ih shape mismatch");
    TORCH_CHECK(w_hh.size(0) == 3 * h0.size(2), "w_hh shape mismatch");
    TORCH_CHECK(b_ih.size(0) == 3 * h0.size(2), "b_ih shape mismatch");
    TORCH_CHECK(b_hh.size(0) == 3 * h0.size(2), "b_hh shape mismatch");
    TORCH_CHECK(x.size(1) == h0.size(1), "batch size mismatch");

    at::Tensor result = at::empty({x.size(0), x.size(1), h0.size(2)}, x.options());
    EXEC_NPU_CMD(aclnnGruCustom, x, w_ih, w_hh, b_ih, b_hh, h0, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gru_custom", &gru_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gru_custom", &gru_custom_impl_npu, "single-layer gru");
}
"""

model_src = '''
import torch
import torch_npu
import custom_ops_lib


class ModelNew(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.w_ih = torch.nn.Parameter(torch.randn(3 * hidden_size, input_size, dtype=torch.float16))
        self.w_hh = torch.nn.Parameter(torch.randn(3 * hidden_size, hidden_size, dtype=torch.float16))
        self.b_ih = torch.nn.Parameter(torch.randn(3 * hidden_size, dtype=torch.float16))
        self.b_hh = torch.nn.Parameter(torch.randn(3 * hidden_size, dtype=torch.float16))
        self.register_buffer("h0", torch.randn(1, 2, hidden_size, dtype=torch.float16))

    def forward(self, x):
        return custom_ops_lib.gru_custom(x, self.w_ih, self.w_hh, self.b_ih, self.b_hh, self.h0)
'''
