
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelConvTranspose2dAddMinGeluMultiplyCustom {
public:
    __aicore__ inline KernelConvTranspose2dAddMinGeluMultiplyCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t N, uint32_t C, uint32_t HW,
                               uint32_t /*total*/,
                               float addv, float mulv)
    {
        this->N = N;
        this->C = C;
        this->HW = HW;
        this->addv = addv;
        this->mulv = mulv;

        xBase = reinterpret_cast<__gm__ float*>(x);
        yBase = reinterpret_cast<__gm__ float*>(y);

        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());

        // Split by N (batch). Each core handles a contiguous range of batch items.
        const uint32_t perCoreN = (this->N + blockNum - 1u) / blockNum;
        const uint32_t startN = blockIdx * perCoreN;

        uint32_t countN = 0;
        if (startN < this->N) {
            const uint32_t remain = this->N - startN;
            countN = (remain < perCoreN) ? remain : perCoreN;
        }
        this->nStart = startN;
        this->nCount = countN;

        // UB scalar buffer used for Rsqrt/Exp on single element.
        pipe.InitBuffer(qSc, BUFFER_NUM, 8 * sizeof(float));
        sc = qSc.AllocTensor<float>();
    }

    __aicore__ inline void Process()
    {
        if (nCount == 0) { qSc.FreeTensor(sc); return; }
        if (N == 0 || C == 0 || HW == 0) { qSc.FreeTensor(sc); return; }

        const uint64_t CHW = static_cast<uint64_t>(C) * static_cast<uint64_t>(HW);

        for (uint32_t ni = 0; ni < nCount; ++ni) {
            const uint32_t n = nStart + ni;
            const uint64_t base = static_cast<uint64_t>(n) * CHW;
            ComputeOneBatch(base, CHW);
        }

        qSc.FreeTensor(sc);
    }

private:
    __aicore__ inline float InvSqrtFast(float v)
    {
        sc.SetValue(0, v);
        AscendC::Rsqrt<float>(sc, sc, 1);
        return sc.GetValue(0);
    }

    __aicore__ inline float ExpFast(float v)
    {
        sc.SetValue(0, v);
        AscendC::Exp<float>(sc, sc, 1);
        return sc.GetValue(0);
    }

    __aicore__ inline float TanhApprox(float x)
    {
        // tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
        const float e2x = ExpFast(2.0f * x);
        return (e2x - 1.0f) / (e2x + 1.0f);
    }

    __aicore__ inline float GeluApprox(float x)
    {
        // 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        const float k0 = 0.7978845608028654f; // sqrt(2/pi)
        const float k1 = 0.044715f;
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float t = k0 * (x + k1 * x3);
        const float th = TanhApprox(t);
        return 0.5f * x * (1.0f + th);
    }

    __aicore__ inline void ComputeOneBatch(uint64_t base, uint64_t len)
    {
        // Elementwise epilogue:
        // v = x + addv
        // v = min(v, 0.0)
        // v = gelu(v)
        // v = v * mulv
        uint64_t i = 0;
        const uint64_t len4 = len & ~3ULL;

        for (; i < len4; i += 4) {
            float v0 = xBase[base + i + 0] + addv;
            float v1 = xBase[base + i + 1] + addv;
            float v2 = xBase[base + i + 2] + addv;
            float v3 = xBase[base + i + 3] + addv;

            if (v0 > 0.0f) v0 = 0.0f;
            if (v1 > 0.0f) v1 = 0.0f;
            if (v2 > 0.0f) v2 = 0.0f;
            if (v3 > 0.0f) v3 = 0.0f;

            v0 = GeluApprox(v0) * mulv;
            v1 = GeluApprox(v1) * mulv;
            v2 = GeluApprox(v2) * mulv;
            v3 = GeluApprox(v3) * mulv;

            yBase[base + i + 0] = v0;
            yBase[base + i + 1] = v1;
            yBase[base + i + 2] = v2;
            yBase[base + i + 3] = v3;
        }
        for (; i < len; ++i) {
            float v = xBase[base + i] + addv;
            if (v > 0.0f) v = 0.0f;
            yBase[base + i] = GeluApprox(v) * mulv;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qSc;
    AscendC::LocalTensor<float> sc;

    __gm__ float* xBase = nullptr;
    __gm__ float* yBase = nullptr;

    uint32_t N = 0, C = 0, HW = 0;
    uint32_t nStart = 0, nCount = 0;
    float addv = 0.5f;
    float mulv = 2.0f;
};

extern "C" __global__ __aicore__ void conv_transpose2d_add_min_gelu_multiply_custom(GM_ADDR x, GM_ADDR y,
                                                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose2dAddMinGeluMultiplyCustom op;
    op.Init(x, y, t.N, t.C, t.HW, t.total, t.addv, t.mulv);
    op.Process();
}
