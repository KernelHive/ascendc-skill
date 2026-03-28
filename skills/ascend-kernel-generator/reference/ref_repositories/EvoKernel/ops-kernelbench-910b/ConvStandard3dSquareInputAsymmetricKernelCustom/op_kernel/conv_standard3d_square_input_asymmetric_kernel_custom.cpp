
#include "kernel_operator.h"

// Specialized benchmark contract (logical ND contiguous):
// x:      [16, 3, 64, 64, 64]   (N,C,W,H,D)
// weight: [64, 3, 3, 5, 7]      (Cout,Cin,Kw,Kh,Kd)
// y:      [16, 64, 62, 60, 58]  (N,Cout,Wout,Hout,Dout)

class KernelConvStandard3dSquareInputAsymV2
{
public:
    __aicore__ inline KernelConvStandard3dSquareInputAsymV2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t blockDim)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        blockDim_ = (blockDim == 0) ? 1 : blockDim;
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t N = 16;
        constexpr uint32_t CIN = 3;
        constexpr uint32_t W = 64;
        constexpr uint32_t H = 64;
        constexpr uint32_t D = 64;

        constexpr uint32_t COUT = 64;
        constexpr uint32_t KW = 3;
        constexpr uint32_t KH = 5;
        constexpr uint32_t KD = 7;

        constexpr uint32_t WOUT = 62;
        constexpr uint32_t HOUT = 60;
        constexpr uint32_t DOUT = 58;

        // Strides for x: [N, CIN, W, H, D] contiguous
        constexpr uint32_t XsD  = 1;
        constexpr uint32_t XsH  = D;
        constexpr uint32_t XsW  = H * D;
        constexpr uint32_t XsCI = W * H * D;
        constexpr uint32_t XsN  = CIN * W * H * D;

        // Strides for y: [N, COUT, WOUT, HOUT, DOUT] contiguous
        constexpr uint32_t YsD  = 1;
        constexpr uint32_t YsH  = DOUT;
        constexpr uint32_t YsW  = HOUT * DOUT;
        constexpr uint32_t YsCO = WOUT * HOUT * DOUT;
        constexpr uint32_t YsN  = COUT * WOUT * HOUT * DOUT;

        // Weights: [COUT, CIN, KW, KH, KD] contiguous
        constexpr uint32_t WsCI = KW * KH * KD;      // 105
        constexpr uint32_t WsCO = CIN * WsCI;        // 315

        // Map work by (n, wo, coGroup). Each work item computes:
        // for co in [coBase..coBase+COG-1], for all ho, for all do.
        constexpr uint32_t COG = 2;
        constexpr uint32_t CGROUPS = COUT / COG;          // 32
        constexpr uint32_t TOTAL_ITEMS = N * WOUT * CGROUPS;

        uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        uint32_t bdim = blockDim_;

        uint32_t itemsPerBlock = (TOTAL_ITEMS + bdim - 1) / bdim;
        uint32_t itemBegin = bid * itemsPerBlock;
        uint32_t itemEnd = itemBegin + itemsPerBlock;
        if (itemEnd > TOTAL_ITEMS) itemEnd = TOTAL_ITEMS;

        for (uint32_t item = itemBegin; item < itemEnd; ++item) {
            uint32_t t = item;
            uint32_t n = t / (WOUT * CGROUPS);
            t -= n * (WOUT * CGROUPS);
            uint32_t wo = t / CGROUPS;
            uint32_t cg = t - wo * CGROUPS;

            uint32_t coBase = cg * COG;

            uint32_t xBaseN = n * XsN;
            uint32_t yBaseN = n * YsN;

            // Precompute base for wi=wo (stride=1, pad=0)
            uint32_t xWiBaseN = xBaseN + wo * XsW;
            uint32_t yWoBaseN = yBaseN + wo * YsW;

            // Load weights once for coBase and coBase+1 into registers.
            // Layout in GM: wBase + ci*WsCI + k (k = kw*KH*KD + kh*KD + kd)
            float wCo0_c0[WsCI];
            float wCo0_c1[WsCI];
            float wCo0_c2[WsCI];
            float wCo1_c0[WsCI];
            float wCo1_c1[WsCI];
            float wCo1_c2[WsCI];

            uint32_t wBase0 = (coBase + 0) * WsCO;
            uint32_t wBase1 = (coBase + 1) * WsCO;

#pragma unroll
            for (uint32_t k = 0; k < WsCI; ++k) {
                wCo0_c0[k] = wGm.GetValue(wBase0 + 0 * WsCI + k);
                wCo0_c1[k] = wGm.GetValue(wBase0 + 1 * WsCI + k);
                wCo0_c2[k] = wGm.GetValue(wBase0 + 2 * WsCI + k);

                wCo1_c0[k] = wGm.GetValue(wBase1 + 0 * WsCI + k);
                wCo1_c1[k] = wGm.GetValue(wBase1 + 1 * WsCI + k);
                wCo1_c2[k] = wGm.GetValue(wBase1 + 2 * WsCI + k);
            }

            // Iterate ho, do as inner contiguous reduction (do is contiguous in memory).
            for (uint32_t ho = 0; ho < HOUT; ++ho) {
                // For pad=0, stride=1: hi0=ho
                uint32_t hi0 = ho;
                uint32_t yBaseCo0 = yWoBaseN + (coBase + 0) * YsCO + ho * YsH;
                uint32_t yBaseCo1 = yWoBaseN + (coBase + 1) * YsCO + ho * YsH;

                // Precompute x channel bases for this (n,wo,ho)
                uint32_t xCi0Base = xWiBaseN + 0 * XsCI + hi0 * XsH;
                uint32_t xCi1Base = xWiBaseN + 1 * XsCI + hi0 * XsH;
                uint32_t xCi2Base = xWiBaseN + 2 * XsCI + hi0 * XsH;

                for (uint32_t do_ = 0; do_ < DOUT; ++do_) {
                    uint32_t di0 = do_;

                    float acc0 = 0.0f;
                    float acc1 = 0.0f;

                    // Flattened k index to avoid kw/kh/kd loop-carried address recompute.
                    // For each (kw,kh,kd): x offset = kw*XsW + kh*XsH + (di0+kd)
                    uint32_t k = 0;

#pragma unroll
                    for (uint32_t kw = 0; kw < KW; ++kw) {
                        uint32_t xKw0 = xCi0Base + kw * XsW;
                        uint32_t xKw1 = xCi1Base + kw * XsW;
                        uint32_t xKw2 = xCi2Base + kw * XsW;
#pragma unroll
                        for (uint32_t kh = 0; kh < KH; ++kh) {
                            uint32_t xKh0 = xKw0 + kh * XsH + di0;
                            uint32_t xKh1 = xKw1 + kh * XsH + di0;
                            uint32_t xKh2 = xKw2 + kh * XsH + di0;
#pragma unroll
                            for (uint32_t kd = 0; kd < KD; ++kd) {
                                float xv0 = xGm.GetValue(xKh0 + kd);
                                float xv1 = xGm.GetValue(xKh1 + kd);
                                float xv2 = xGm.GetValue(xKh2 + kd);

                                acc0 += xv0 * wCo0_c0[k];
                                acc0 += xv1 * wCo0_c1[k];
                                acc0 += xv2 * wCo0_c2[k];

                                acc1 += xv0 * wCo1_c0[k];
                                acc1 += xv1 * wCo1_c1[k];
                                acc1 += xv2 * wCo1_c2[k];

                                ++k;
                            }
                        }
                    }

                    yGm.SetValue(yBaseCo0 + do_, acc0);
                    yGm.SetValue(yBaseCo1 + do_, acc1);
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t blockDim_{1};
};

extern "C" __global__ __aicore__ void conv_standard3d_square_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvStandard3dSquareInputAsymV2 op;
    op.Init(x, weight, y, tiling_data.blockDim);
    op.Process();
}
