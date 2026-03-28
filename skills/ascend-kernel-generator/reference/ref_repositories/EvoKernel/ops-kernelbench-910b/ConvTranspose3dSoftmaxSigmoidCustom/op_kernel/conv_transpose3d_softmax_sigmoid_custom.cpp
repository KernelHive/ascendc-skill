
#include "kernel_operator.h"

// Fused specialized operator:
//   y0 = conv_transpose3d(x, weight, bias, stride=2, pad=1, outpad=1, dil=1, groups=1)
//   y1 = softmax(y0, dim=1)   // over Cout=64
//   y  = sigmoid(y1)
//
// Specialization contract (must match python_bind checks):
//   x:      [N=16, Cin=32, Din=16, Hin=32, Win=32] float32 contiguous NCDHW
//   weight: [Cin=32, Cout=64, Kd=3, Kh=3, Kw=3] float32 (PyTorch ConvTranspose3d layout)
//   bias:   [Cout=64] float32 contiguous
//   y:      [N=16, Cout=64, Dout=32, Hout=64, Wout=64] float32 contiguous NCDHW
//
// Notes:
// - Avoid ReduceMax/ReduceSum signature/workspace issues by using explicit scalar reductions for Cout=64.
// - Use vector Exp + sigmoid ops in UB for math throughput.
// - Parallelize over spatial rows (n,od,oh,ow). Each core computes all 64 channels for its assigned rows.

class KernelConvTranspose3dSoftmaxSigmoidCustom {
public:
    __aicore__ inline KernelConvTranspose3dSoftmaxSigmoidCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t blockDim, uint32_t totalRows)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        bGm.SetGlobalBuffer((__gm__ float*)b);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        blockDim_ = (blockDim == 0) ? 1 : blockDim;
        totalRows_ = (totalRows == 0) ? 1 : totalRows;

        pipe.InitBuffer(accQ_, 1, COUT * sizeof(float));
        pipe.InitBuffer(tmpQ_, 1, COUT * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t bid = (int64_t)AscendC::GetBlockIdx();
        int64_t bdim = (int64_t)blockDim_;
        if (bdim <= 0) bdim = 1;

        const int64_t TOTAL = (int64_t)totalRows_; // == N*COUT*DOUT*HOUT*WOUT / COUT? no, here totalRows == output elems
        // We set totalRows to output element count; we want per-(n,od,oh,ow) rows, so divide by COUT.
        // Output shape is fixed; derive rows count deterministically.
        constexpr int64_t ROWS = (int64_t)N * DOUT * HOUT * WOUT;

        const int64_t rowsPerBlock = (ROWS + bdim - 1) / bdim;
        int64_t rowStart = bid * rowsPerBlock;
        int64_t rowEnd = rowStart + rowsPerBlock;
        if (rowStart > ROWS) rowStart = ROWS;
        if (rowEnd > ROWS) rowEnd = ROWS;

        AscendC::LocalTensor<float> accUb = accQ_.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpUb = tmpQ_.AllocTensor<float>();

        for (int64_t linear = rowStart; linear < rowEnd; ++linear) {
            int64_t t = linear;
            const int32_t ow = (int32_t)(t % WOUT); t /= WOUT;
            const int32_t oh = (int32_t)(t % HOUT); t /= HOUT;
            const int32_t od = (int32_t)(t % DOUT); t /= DOUT;
            const int32_t n  = (int32_t)t;

            // accUb = bias
            AscendC::DataCopy(accUb, bGm, COUT);

            const int64_t xBaseN = (int64_t)n * (int64_t)CIN * X_STRIDE_C;

            // Gather-form deconv mapping:
            // for each (kd,kh,kw): num = out + pad - k*dil; must be divisible by stride; in = num/stride
            for (int32_t ci = 0; ci < CIN; ++ci) {
                const int64_t xBaseCi = xBaseN + (int64_t)ci * X_STRIDE_C;
                const int64_t wBaseCi = (int64_t)ci * W_STRIDE_CI;

                for (int32_t kd = 0; kd < KD; ++kd) {
                    const int32_t numD = od + PAD - kd * DIL;
                    if (numD < 0) continue;
                    if ((numD % STR) != 0) continue;
                    const int32_t id = numD / STR;
                    if ((uint32_t)id >= (uint32_t)DIN) continue;

                    const int64_t xBaseD = xBaseCi + (int64_t)id * X_STRIDE_D;
                    const int64_t wBaseKd = wBaseCi + (int64_t)kd * W_STRIDE_KD;

                    for (int32_t kh = 0; kh < KH; ++kh) {
                        const int32_t numH = oh + PAD - kh * DIL;
                        if (numH < 0) continue;
                        if ((numH % STR) != 0) continue;
                        const int32_t ih = numH / STR;
                        if ((uint32_t)ih >= (uint32_t)HIN) continue;

                        const int64_t xBaseH = xBaseD + (int64_t)ih * X_STRIDE_H;
                        const int64_t wBaseKh = wBaseKd + (int64_t)kh * W_STRIDE_KH;

                        for (int32_t kw = 0; kw < KW; ++kw) {
                            const int32_t numW = ow + PAD - kw * DIL;
                            if (numW < 0) continue;
                            if ((numW % STR) != 0) continue;
                            const int32_t iw = numW / STR;
                            if ((uint32_t)iw >= (uint32_t)WIN) continue;

                            const float xv = xGm.GetValue((uint64_t)(xBaseH + (int64_t)iw));
                            if (xv == 0.0f) continue;

                            // For fixed Cout=64, accumulate scalar per channel (no extra co2 loop beyond 64).
                            const int64_t wKOff = wBaseKh + (int64_t)kw;
#pragma unroll
                            for (int32_t co = 0; co < COUT; ++co) {
                                const int64_t wIdx = wKOff + (int64_t)co * W_STRIDE_CO;
                                const float wv = wGm.GetValue((uint64_t)wIdx);
                                accUb.SetValue((uint32_t)co, accUb.GetValue((uint32_t)co) + xv * wv);
                            }
                        }
                    }
                }
            }

            // softmax over COUT (scalar reductions to avoid Reduce* API signature issues)
            float rowMax = accUb.GetValue(0);
#pragma unroll
            for (int32_t co = 1; co < COUT; ++co) {
                float v = accUb.GetValue((uint32_t)co);
                if (v > rowMax) rowMax = v;
            }

            // accUb = exp(accUb - rowMax)
#pragma unroll
            for (int32_t co = 0; co < COUT; ++co) {
                tmpUb.SetValue((uint32_t)co, accUb.GetValue((uint32_t)co) - rowMax);
            }
            AscendC::Exp(tmpUb, tmpUb, COUT);

            float rowSum = 0.0f;
#pragma unroll
            for (int32_t co = 0; co < COUT; ++co) {
                rowSum += tmpUb.GetValue((uint32_t)co);
            }
            const float invSum = 1.0f / (rowSum + 1e-20f);
            AscendC::Muls(tmpUb, tmpUb, invSum, COUT); // tmpUb = softmax

            // sigmoid(tmpUb) into accUb: 1 / (1 + exp(-p))
            AscendC::Muls(accUb, tmpUb, -1.0f, COUT);  // accUb = -p
            AscendC::Exp(accUb, accUb, COUT);          // accUb = exp(-p)
            AscendC::Adds(accUb, accUb, 1.0f, COUT);   // accUb = 1 + exp(-p)
            AscendC::Reciprocal(tmpUb, accUb, COUT);   // tmpUb = sigmoid(p)

            // write out y for this spatial position
            const int64_t yBaseN = (int64_t)n * (int64_t)COUT * Y_STRIDE_C;
            const int64_t ySpatial = (int64_t)od * Y_STRIDE_D + (int64_t)oh * Y_STRIDE_H + (int64_t)ow;
#pragma unroll
            for (int32_t co = 0; co < COUT; ++co) {
                const int64_t yIdx = yBaseN + (int64_t)co * Y_STRIDE_C + ySpatial;
                yGm.SetValue((uint64_t)yIdx, tmpUb.GetValue((uint32_t)co));
            }
        }

        tmpQ_.FreeTensor(tmpUb);
        accQ_.FreeTensor(accUb);
    }

private:
    // Fixed specialization constants
    static constexpr int32_t N = 16;
    static constexpr int32_t CIN = 32;
    static constexpr int32_t DIN = 16;
    static constexpr int32_t HIN = 32;
    static constexpr int32_t WIN = 32;

    static constexpr int32_t COUT = 64;
    static constexpr int32_t KD = 3;
    static constexpr int32_t KH = 3;
    static constexpr int32_t KW = 3;

    static constexpr int32_t STR = 2;
    static constexpr int32_t PAD = 1;
    static constexpr int32_t DIL = 1;

    static constexpr int32_t DOUT = 32;
    static constexpr int32_t HOUT = 64;
    static constexpr int32_t WOUT = 64;

    // NCDHW contiguous strides
    static constexpr int64_t X_STRIDE_C = (int64_t)DIN * HIN * WIN;
    static constexpr int64_t X_STRIDE_D = (int64_t)HIN * WIN;
    static constexpr int64_t X_STRIDE_H = (int64_t)WIN;

    static constexpr int64_t Y_STRIDE_C = (int64_t)DOUT * HOUT * WOUT;
    static constexpr int64_t Y_STRIDE_D = (int64_t)HOUT * WOUT;
    static constexpr int64_t Y_STRIDE_H = (int64_t)WOUT;

    // Weight layout [ci, co, kd, kh, kw] contiguous
    static constexpr int64_t W_STRIDE_CI = (int64_t)COUT * KD * KH * KW;
    static constexpr int64_t W_STRIDE_CO = (int64_t)KD * KH * KW;
    static constexpr int64_t W_STRIDE_KD = (int64_t)KH * KW;
    static constexpr int64_t W_STRIDE_KH = (int64_t)KW;

    uint32_t blockDim_{1};
    uint32_t totalRows_{1};

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> accQ_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> tmpQ_;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;
};

extern "C" __global__ __aicore__ void conv_transpose3d_softmax_sigmoid_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);
    KernelConvTranspose3dSoftmaxSigmoidCustom op;
    op.Init(x, weight, bias, y, t.blockDim, t.totalRows);
    op.Process();
}
