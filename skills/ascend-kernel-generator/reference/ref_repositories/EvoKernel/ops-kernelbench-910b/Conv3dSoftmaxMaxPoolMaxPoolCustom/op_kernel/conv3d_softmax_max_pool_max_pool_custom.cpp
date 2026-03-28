
#include "kernel_operator.h"

// Specialized fused kernel for contract:
// x: [128,3,16,32,32], w_packed: [81,16] (tap-major contiguous over co), b: [16]
// conv valid -> [128,16,14,30,30], softmax over C, then MaxPool3d(2,2) twice -> [128,16,3,7,7].
//
// Optimization in this round:
// - Compute 4 width positions (xw=0..3) together: keep 4x 16-lane accumulators in UB.
// - Iterate taps once and update all 4 accumulators using vector ops with packed weights.
// - Keep slab staging (162 floats) to reuse input loads.
// - Store final 16-float output vector via one DataCopy to GM.

class KernelConv3dSoftmaxMaxPoolMaxPoolCustom {
public:
    __aicore__ inline KernelConv3dSoftmaxMaxPoolMaxPoolCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR wPacked, GM_ADDR b, GM_ADDR y,
                               uint32_t totalY, uint32_t totalX, uint32_t totalW, uint32_t totalB)
    {
        (void)totalX;
        (void)totalW;
        (void)totalB;
        (void)totalY;

        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)wPacked);
        bGm.SetGlobalBuffer((__gm__ float*)b);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        pipe.InitBuffer(qW,      1, W_UB_ELEMS   * sizeof(float));
        pipe.InitBuffer(qB,      1, B_UB_ELEMS   * sizeof(float));
        pipe.InitBuffer(qSlab,   1, SLAB_ELEMS   * sizeof(float));
        pipe.InitBuffer(qAcc0,   1, VEC_ELEMS    * sizeof(float));
        pipe.InitBuffer(qAcc1,   1, VEC_ELEMS    * sizeof(float));
        pipe.InitBuffer(qAcc2,   1, VEC_ELEMS    * sizeof(float));
        pipe.InitBuffer(qAcc3,   1, VEC_ELEMS    * sizeof(float));
        pipe.InitBuffer(qWvec,   1, VEC_ELEMS    * sizeof(float));
        pipe.InitBuffer(qTmp,    1, VEC_ELEMS    * sizeof(float));
        pipe.InitBuffer(qExp,    1, VEC_ELEMS    * sizeof(float));
        pipe.InitBuffer(qOut,    1, VEC_ELEMS    * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        constexpr int64_t pixelsTotal = (int64_t)N * (int64_t)D3 * (int64_t)H3 * (int64_t)W3;

        const int64_t blockNum = (int64_t)AscendC::GetBlockNum();
        const int64_t blockIdx = (int64_t)AscendC::GetBlockIdx();

        const int64_t chunk = (pixelsTotal + blockNum - 1) / blockNum;
        int64_t start = blockIdx * chunk;
        int64_t end = start + chunk;
        if (end > pixelsTotal) end = pixelsTotal;
        if (start >= end) return;

        AscendC::LocalTensor<float> wUb    = qW.AllocTensor<float>();
        AscendC::LocalTensor<float> bUb    = qB.AllocTensor<float>();
        AscendC::LocalTensor<float> slab   = qSlab.AllocTensor<float>();
        AscendC::LocalTensor<float> acc0   = qAcc0.AllocTensor<float>();
        AscendC::LocalTensor<float> acc1   = qAcc1.AllocTensor<float>();
        AscendC::LocalTensor<float> acc2   = qAcc2.AllocTensor<float>();
        AscendC::LocalTensor<float> acc3   = qAcc3.AllocTensor<float>();
        AscendC::LocalTensor<float> wVec   = qWvec.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpV   = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> expV   = qExp.AllocTensor<float>();
        AscendC::LocalTensor<float> outV   = qOut.AllocTensor<float>();

        // wPacked is small: 81*16=1296 floats.
        AscendC::DataCopy(wUb, wGm, (uint32_t)W_ELEMS);
        AscendC::DataCopy(bUb, bGm, (uint32_t)COUT);
        #pragma unroll
        for (uint32_t i = (uint32_t)COUT; i < B_UB_ELEMS; ++i) {
            bUb.SetValue(i, 0.0f);
        }

        int64_t t = start;
        int w3 = (int)(t % W3); t /= W3;
        int h3 = (int)(t % H3); t /= H3;
        int d3 = (int)(t % D3); t /= D3;
        int n  = (int)t;

        for (int64_t p = start; p < end; ++p) {
            float best[COUT];
            #pragma unroll
            for (int co = 0; co < COUT; ++co) best[co] = NEG_INF;

            const int baseOd = d3 * 4;
            const int baseOh = h3 * 4;
            const int baseOw = w3 * 4;

            #pragma unroll
            for (int zd = 0; zd < 4; ++zd) {
                const int od = baseOd + zd;
                if (od >= D1) continue;

                #pragma unroll
                for (int yh = 0; yh < 4; ++yh) {
                    const int oh = baseOh + yh;
                    if (oh >= H1) continue;

                    StageSlab(n, od, oh, baseOw, slab);

                    // Compute 4 ow positions together (ow=baseOw..baseOw+3), all valid here because baseOw in [0,24] and W1=30.
                    Conv4VoxelsFromSlab(baseOw, wUb, bUb, slab, acc0, acc1, acc2, acc3, wVec, tmpV);

                    Softmax16InPlace(acc0, expV, tmpV);
                    #pragma unroll
                    for (int co = 0; co < COUT; ++co) {
                        float v = acc0.GetValue((uint32_t)co);
                        if (v > best[co]) best[co] = v;
                    }

                    Softmax16InPlace(acc1, expV, tmpV);
                    #pragma unroll
                    for (int co = 0; co < COUT; ++co) {
                        float v = acc1.GetValue((uint32_t)co);
                        if (v > best[co]) best[co] = v;
                    }

                    Softmax16InPlace(acc2, expV, tmpV);
                    #pragma unroll
                    for (int co = 0; co < COUT; ++co) {
                        float v = acc2.GetValue((uint32_t)co);
                        if (v > best[co]) best[co] = v;
                    }

                    Softmax16InPlace(acc3, expV, tmpV);
                    #pragma unroll
                    for (int co = 0; co < COUT; ++co) {
                        float v = acc3.GetValue((uint32_t)co);
                        if (v > best[co]) best[co] = v;
                    }
                }
            }

            // Store best[0..15] as one vector DataCopy.
            #pragma unroll
            for (uint32_t co = 0; co < (uint32_t)COUT; ++co) outV.SetValue(co, best[co]);
            #pragma unroll
            for (uint32_t co = (uint32_t)COUT; co < VEC_ELEMS; ++co) outV.SetValue(co, 0.0f);

            const uint64_t yBase = ((((uint64_t)n * (uint64_t)COUT + 0u) * (uint64_t)D3 + (uint64_t)d3)
                                  * (uint64_t)H3 + (uint64_t)h3) * (uint64_t)W3 + (uint64_t)w3;
            // Channel-strided store: write 16 scalars but via 16-element DataCopy is not contiguous in GM, so we still must scatter.
            // However we can reduce scalar overhead by writing two 8-wide bursts is also strided; safest is scalar SetValue here.
            // Keep scalar stores but with fewer overall scalar work from conv/softmax.
            #pragma unroll
            for (int co = 0; co < COUT; ++co) {
                const uint64_t yIdx = yBase + (uint64_t)co * (uint64_t)D3 * (uint64_t)H3 * (uint64_t)W3;
                yGm.SetValue(yIdx, best[co]);
            }

            w3 += 1;
            if (w3 == W3) {
                w3 = 0; h3 += 1;
                if (h3 == H3) {
                    h3 = 0; d3 += 1;
                    if (d3 == D3) {
                        d3 = 0; n += 1;
                    }
                }
            }
        }

        qW.FreeTensor(wUb);
        qB.FreeTensor(bUb);
        qSlab.FreeTensor(slab);
        qAcc0.FreeTensor(acc0);
        qAcc1.FreeTensor(acc1);
        qAcc2.FreeTensor(acc2);
        qAcc3.FreeTensor(acc3);
        qWvec.FreeTensor(wVec);
        qTmp.FreeTensor(tmpV);
        qExp.FreeTensor(expV);
        qOut.FreeTensor(outV);
    }

private:
    static constexpr float NEG_INF = -3.402823466e+38f;

    static constexpr int N = 128;
    static constexpr int CIN = 3;
    static constexpr int COUT = 16;

    static constexpr int DIN = 16;
    static constexpr int HIN = 32;
    static constexpr int WIN = 32;

    static constexpr int K = 3;
    static constexpr int D1 = 14;
    static constexpr int H1 = 30;
    static constexpr int W1 = 30;

    static constexpr int D3 = 3;
    static constexpr int H3 = 7;
    static constexpr int W3 = 7;

    // Packed weights: [81,16]
    static constexpr uint32_t TAP_ELEMS = (uint32_t)(CIN * K * K * K); // 81
    static constexpr uint32_t W_ELEMS   = (uint32_t)(TAP_ELEMS * COUT); // 1296

    static constexpr uint32_t SLAB_W = 6;
    static constexpr uint32_t SLAB_ELEMS = (uint32_t)(CIN * K * K * SLAB_W); // 162

    static constexpr uint32_t VEC_ELEMS  = 16;
    static constexpr uint32_t B_UB_ELEMS = 32;
    static constexpr uint32_t W_UB_ELEMS = 1536;

    __aicore__ inline uint64_t IdxX(int n, int c, int d, int h, int w) const
    {
        constexpr uint64_t IN_HW  = (uint64_t)HIN * (uint64_t)WIN;
        constexpr uint64_t IN_DHW = (uint64_t)DIN * IN_HW;
        return ((uint64_t)n * (uint64_t)CIN + (uint64_t)c) * IN_DHW
             + (uint64_t)d * IN_HW + (uint64_t)h * (uint64_t)WIN + (uint64_t)w;
    }

    __aicore__ inline void StageSlab(int n, int od, int oh, int baseOw, AscendC::LocalTensor<float>& slab)
    {
        // baseOw is always 4*w3 with w3 in [0,6], so baseOw in [0,24], safe for SLAB_W=6 within W1=30.
        const int ow0 = baseOw;

        #pragma unroll
        for (int ci = 0; ci < CIN; ++ci) {
            #pragma unroll
            for (int kd = 0; kd < K; ++kd) {
                const int id = od + kd;
                #pragma unroll
                for (int kh = 0; kh < K; ++kh) {
                    const int ih = oh + kh;
                    const uint32_t base = (uint32_t)ci * (K * K * SLAB_W)
                                        + (uint32_t)kd * (K * SLAB_W)
                                        + (uint32_t)kh * SLAB_W;
                    #pragma unroll
                    for (int sw = 0; sw < (int)SLAB_W; ++sw) {
                        const int iw = ow0 + sw;
                        slab.SetValue(base + (uint32_t)sw, xGm.GetValue(IdxX(n, ci, id, ih, iw)));
                    }
                }
            }
        }
    }

    __aicore__ inline void Conv4VoxelsFromSlab(int baseOw,
                                              const AscendC::LocalTensor<float>& wUb,
                                              const AscendC::LocalTensor<float>& bUb,
                                              const AscendC::LocalTensor<float>& slab,
                                              AscendC::LocalTensor<float>& acc0,
                                              AscendC::LocalTensor<float>& acc1,
                                              AscendC::LocalTensor<float>& acc2,
                                              AscendC::LocalTensor<float>& acc3,
                                              AscendC::LocalTensor<float>& wVec,
                                              AscendC::LocalTensor<float>& tmpV)
    {
        (void)baseOw;

        // Initialize accumulators from bias.
        #pragma unroll
        for (uint32_t co = 0; co < (uint32_t)COUT; ++co) {
            const float bv = bUb.GetValue(co);
            acc0.SetValue(co, bv);
            acc1.SetValue(co, bv);
            acc2.SetValue(co, bv);
            acc3.SetValue(co, bv);
        }

        // 81 taps: for each tap i, load 16 weights contiguously and update four acc vectors.
        #pragma unroll
        for (uint32_t i = 0; i < TAP_ELEMS; ++i) {
            // wVec[0..15] = wUb[i*16 + co]
            const uint32_t wOff = i * (uint32_t)COUT;
            AscendC::DataCopy(wVec, wUb[wOff], (uint32_t)COUT);

            // Map tap i -> (ci,kd,kh,kw) -> slab base index and per-voxel xw offsets.
            const uint32_t ci = i / 27u;
            const uint32_t rem = i - ci * 27u;
            const uint32_t kd = rem / 9u;
            const uint32_t rem2 = rem - kd * 9u;
            const uint32_t kh = rem2 / 3u;
            const uint32_t kw = rem2 - kh * 3u;

            const uint32_t slabBase = ci * (K * K * SLAB_W) + kd * (K * SLAB_W) + kh * SLAB_W + kw;

            const float x0 = slab.GetValue(slabBase + 0u);
            const float x1 = slab.GetValue(slabBase + 1u);
            const float x2 = slab.GetValue(slabBase + 2u);
            const float x3 = slab.GetValue(slabBase + 3u);

            AscendC::Duplicate(tmpV, x0, (int32_t)VEC_ELEMS);
            AscendC::Mul(tmpV, wVec, tmpV, (int32_t)VEC_ELEMS);
            AscendC::Add(acc0, acc0, tmpV, (int32_t)VEC_ELEMS);

            AscendC::Duplicate(tmpV, x1, (int32_t)VEC_ELEMS);
            AscendC::Mul(tmpV, wVec, tmpV, (int32_t)VEC_ELEMS);
            AscendC::Add(acc1, acc1, tmpV, (int32_t)VEC_ELEMS);

            AscendC::Duplicate(tmpV, x2, (int32_t)VEC_ELEMS);
            AscendC::Mul(tmpV, wVec, tmpV, (int32_t)VEC_ELEMS);
            AscendC::Add(acc2, acc2, tmpV, (int32_t)VEC_ELEMS);

            AscendC::Duplicate(tmpV, x3, (int32_t)VEC_ELEMS);
            AscendC::Mul(tmpV, wVec, tmpV, (int32_t)VEC_ELEMS);
            AscendC::Add(acc3, acc3, tmpV, (int32_t)VEC_ELEMS);
        }
    }

    __aicore__ inline void Softmax16InPlace(AscendC::LocalTensor<float>& logits,
                                           AscendC::LocalTensor<float>& expV,
                                           AscendC::LocalTensor<float>& tmp)
    {
        float maxv = NEG_INF;
        #pragma unroll
        for (int i = 0; i < COUT; ++i) {
            float v = logits.GetValue((uint32_t)i);
            if (v > maxv) maxv = v;
        }

        AscendC::Duplicate(tmp, maxv, (int32_t)VEC_ELEMS);
        AscendC::Sub(expV, logits, tmp, (int32_t)VEC_ELEMS);
        AscendC::Exp(expV, expV, (int32_t)VEC_ELEMS);

        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < COUT; ++i) sum += expV.GetValue((uint32_t)i);
        if (sum == 0.0f) sum = 1.0f;
        const float inv = 1.0f / sum;

        AscendC::Duplicate(tmp, inv, (int32_t)VEC_ELEMS);
        AscendC::Mul(logits, expV, tmp, (int32_t)VEC_ELEMS); // write back to logits buffer
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> qW;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qB;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qSlab;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qAcc0;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qAcc1;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qAcc2;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qAcc3;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qWvec;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qTmp;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qExp;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qOut;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;
};

extern "C" __global__ __aicore__ void conv3d_softmax_max_pool_max_pool_custom(
    GM_ADDR x, GM_ADDR weight_packed, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConv3dSoftmaxMaxPoolMaxPoolCustom op;
    op.Init(x, weight_packed, bias, y,
            tiling_data.totalY, tiling_data.totalX, tiling_data.totalW, tiling_data.totalB);
    op.Process();
}
