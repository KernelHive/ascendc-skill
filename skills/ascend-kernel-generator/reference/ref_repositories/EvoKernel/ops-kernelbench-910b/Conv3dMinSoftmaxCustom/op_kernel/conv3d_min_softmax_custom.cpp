
#include "kernel_operator.h"

// Specialized fused kernel for benchmark contract:
// x: [128, 3, 24, 32, 32] float32 NCDHW
// w: [24, 3, 3, 3, 3] float32 [Cout,Cin,Kd,Kh,Kw]
// b: [24] float32
// conv out: [128, 24, 22, 30, 30]
// min over depth (dim=2): [128, 24, 30, 30]
// softmax over channel dim=1: y [128, 24, 30, 30]
//
// This round improvements (keeping stable BlockDim=32):
// - Stage the full input patch for a given output depth do_ (CIN*KD*KH*KW=81 floats)
//   into UB once and reuse across all 24 output channels (reduces scalar GM reads of x ~24x).
// - Incremental (n,ho,wo) update inside block chunk to avoid repeated div/mod per pixel.
// - All UB tensors allocated once per block and reused; weights/bias prefetched once per block.

class KernelConv3dMinSoftmaxCustom {
public:
    __aicore__ inline KernelConv3dMinSoftmaxCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t totalY, uint32_t totalX, uint32_t totalW, uint32_t totalB)
    {
        (void)totalX;
        (void)totalW;
        (void)totalB;
        totalY_ = totalY;

        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        bGm.SetGlobalBuffer((__gm__ float*)b);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        pipe.InitBuffer(qW,     1, W_UB_ELEMS    * sizeof(float));
        pipe.InitBuffer(qB,     1, B_UB_ELEMS    * sizeof(float));
        pipe.InitBuffer(qPatch, 1, PATCH_ELEMS   * sizeof(float));
        pipe.InitBuffer(qMin,   1, VEC_ELEMS     * sizeof(float));
        pipe.InitBuffer(qExp,   1, VEC_ELEMS     * sizeof(float));
        pipe.InitBuffer(qTmp,   1, VEC_ELEMS     * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 128;
        constexpr int CIN = 3;
        constexpr int DIN = 24;
        constexpr int HIN = 32;
        constexpr int WIN = 32;

        constexpr int COUT = 24;
        constexpr int KD = 3;
        constexpr int KH = 3;
        constexpr int KW = 3;

        constexpr int DOUT = 22;
        constexpr int HOUT = 30;
        constexpr int WOUT = 30;

        constexpr uint64_t IN_HW = (uint64_t)HIN * (uint64_t)WIN;
        constexpr uint64_t IN_DHW = (uint64_t)DIN * IN_HW;

        constexpr uint32_t W_ELEMS = (uint32_t)(COUT * CIN * KD * KH * KW); // 1944
        constexpr uint32_t CO_STRIDE = (uint32_t)(CIN * KD * KH * KW);      // 81

        const int64_t pixelsTotal = (int64_t)N * (int64_t)HOUT * (int64_t)WOUT;
        const int64_t blockNum = (int64_t)AscendC::GetBlockNum();
        const int64_t blockIdx = (int64_t)AscendC::GetBlockIdx();

        const int64_t chunk = (pixelsTotal + blockNum - 1) / blockNum;
        int64_t start = blockIdx * chunk;
        int64_t end = start + chunk;
        if (end > pixelsTotal) end = pixelsTotal;
        if (start >= end) return;

        AscendC::LocalTensor<float> wUb   = qW.AllocTensor<float>();
        AscendC::LocalTensor<float> bUb   = qB.AllocTensor<float>();
        AscendC::LocalTensor<float> patch = qPatch.AllocTensor<float>();
        AscendC::LocalTensor<float> minV  = qMin.AllocTensor<float>();
        AscendC::LocalTensor<float> expV  = qExp.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpV  = qTmp.AllocTensor<float>();

        // Preload weights and bias once per block.
        AscendC::DataCopy(wUb, wGm, W_ELEMS);
        AscendC::DataCopy(bUb, bGm, (uint32_t)COUT);
        #pragma unroll
        for (uint32_t i = (uint32_t)COUT; i < B_UB_ELEMS; ++i) {
            bUb.SetValue(i, 0.0f);
        }

        // Decode start pixel once, then increment (wo,ho,n) cheaply.
        int64_t t = start;
        int wo = (int)(t % WOUT); t /= WOUT;
        int ho = (int)(t % HOUT); t /= HOUT;
        int n  = (int)t;

        for (int64_t p = start; p < end; ++p) {
            const int hi0 = ho;
            const int wi0 = wo;

            AscendC::Duplicate(minV, POS_INF, (int32_t)VEC_ELEMS);

            for (int do_ = 0; do_ < DOUT; ++do_) {

                // Stage input patch for this (n,hi0,wi0,do_) into UB: 81 floats.
                #pragma unroll
                for (int ci = 0; ci < CIN; ++ci) {
                    const uint64_t xNcBase = ((uint64_t)n * (uint64_t)CIN + (uint64_t)ci) * IN_DHW;

                    #pragma unroll
                    for (int kd = 0; kd < KD; ++kd) {
                        const int di = do_ + kd;
                        const uint64_t xZBase = xNcBase + (uint64_t)di * IN_HW;

                        #pragma unroll
                        for (int kh = 0; kh < KH; ++kh) {
                            const int hi = hi0 + kh;
                            const uint64_t xYBase = xZBase + (uint64_t)hi * (uint64_t)WIN;

                            #pragma unroll
                            for (int kw = 0; kw < KW; ++kw) {
                                const int wi = wi0 + kw;
                                const uint64_t xIdx = xYBase + (uint64_t)wi;
                                const uint32_t pIdx = (uint32_t)ci * 27u + (uint32_t)kd * 9u + (uint32_t)kh * 3u + (uint32_t)kw;
                                patch.SetValue(pIdx, xGm.GetValue(xIdx));
                            }
                        }
                    }
                }

                // For each output channel: dot(patch[81], wUb[co*81..co*81+80]) + bias, then min-reduce over do_.
                #pragma unroll
                for (int co = 0; co < COUT; ++co) {
                    float acc = bUb.GetValue((uint32_t)co);
                    const uint32_t wBase = (uint32_t)co * CO_STRIDE;

                    #pragma unroll
                    for (uint32_t i = 0; i < CO_STRIDE; ++i) {
                        acc += patch.GetValue(i) * wUb.GetValue(wBase + i);
                    }

                    const float prev = minV.GetValue((uint32_t)co);
                    if (acc < prev) {
                        minV.SetValue((uint32_t)co, acc);
                    }
                }
            }

            // Softmax over channel dim (24).
            float maxv = NEG_INF;
            #pragma unroll
            for (int co = 0; co < COUT; ++co) {
                float v = minV.GetValue((uint32_t)co);
                if (v > maxv) maxv = v;
            }

            AscendC::Duplicate(tmpV, maxv, (int32_t)VEC_ELEMS);
            AscendC::Sub(expV, minV, tmpV, (int32_t)VEC_ELEMS);
            AscendC::Exp(expV, expV, (int32_t)VEC_ELEMS);

            float sum = 0.0f;
            #pragma unroll
            for (int co = 0; co < COUT; ++co) {
                sum += expV.GetValue((uint32_t)co);
            }
            if (sum == 0.0f) sum = 1.0f;
            const float invSum = 1.0f / sum;

            AscendC::Duplicate(tmpV, invSum, (int32_t)VEC_ELEMS);
            AscendC::Mul(expV, expV, tmpV, (int32_t)VEC_ELEMS);

            // Store y in NCHW (channel-strided).
            const uint64_t yBase = (((uint64_t)n * (uint64_t)COUT) * (uint64_t)HOUT + (uint64_t)ho) * (uint64_t)WOUT + (uint64_t)wo;
            #pragma unroll
            for (int co = 0; co < COUT; ++co) {
                const uint64_t yIdx = yBase + (uint64_t)co * (uint64_t)HOUT * (uint64_t)WOUT;
                yGm.SetValue(yIdx, expV.GetValue((uint32_t)co));
            }

            // Increment (wo, ho, n) without div/mod for next pixel.
            wo += 1;
            if (wo == WOUT) {
                wo = 0;
                ho += 1;
                if (ho == HOUT) {
                    ho = 0;
                    n += 1;
                }
            }
        }

        qW.FreeTensor(wUb);
        qB.FreeTensor(bUb);
        qPatch.FreeTensor(patch);
        qMin.FreeTensor(minV);
        qExp.FreeTensor(expV);
        qTmp.FreeTensor(tmpV);

        (void)totalY_;
    }

private:
    static constexpr float POS_INF = 3.402823466e+38f;
    static constexpr float NEG_INF = -3.402823466e+38f;

    static constexpr uint32_t VEC_ELEMS   = 32;    // vector lanes (>=24)
    static constexpr uint32_t B_UB_ELEMS  = 32;    // padded bias
    static constexpr uint32_t W_UB_ELEMS  = 2048;  // padded weights (>=1944)
    static constexpr uint32_t PATCH_ELEMS = 81;    // CIN*KD*KH*KW

    uint32_t totalY_{0};

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qW;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qB;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qPatch;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qMin;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qExp;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qTmp;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;
};

extern "C" __global__ __aicore__ void conv3d_min_softmax_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dMinSoftmaxCustom op;
    op.Init(x, weight, bias, y,
            tiling_data.totalY, tiling_data.totalX, tiling_data.totalW, tiling_data.totalB);
    op.Process();
}
