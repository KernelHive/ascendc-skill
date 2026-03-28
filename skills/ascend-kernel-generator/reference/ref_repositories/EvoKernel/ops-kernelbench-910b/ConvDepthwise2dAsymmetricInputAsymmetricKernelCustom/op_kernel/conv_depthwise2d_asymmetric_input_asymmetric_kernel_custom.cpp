
#include "kernel_operator.h"

// Specialized Depthwise Conv2d:
// x: [N=32, C=128, H=128, W=256]
// weight: [C=128, 1, Kh=3, Kw=7]
// stride=(1,1), pad=(0,0), dilation=(1,1), groups=C, bias=False
// y: [32, 128, Ho=126, Wo=250]

class KernelConvDepthwise2DAsymAsym {
public:
    __aicore__ inline KernelConvDepthwise2DAsymAsym() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t totalY, uint32_t totalX, uint32_t totalW)
    {
        (void)totalY; (void)totalX; (void)totalW;
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 32;
        constexpr int C = 128;
        constexpr int H = 128;
        constexpr int W = 256;

        constexpr int KH = 3;
        constexpr int KW = 7;

        constexpr int STRIDE_H = 1;
        constexpr int STRIDE_W = 1;
        constexpr int PAD_H = 0;
        constexpr int PAD_W = 0;
        constexpr int DIL_H = 1;
        constexpr int DIL_W = 1;

        constexpr int HO = (H + 2 * PAD_H - DIL_H * (KH - 1) - 1) / STRIDE_H + 1; // 126
        constexpr int WO = (W + 2 * PAD_W - DIL_W * (KW - 1) - 1) / STRIDE_W + 1; // 250

        // Single-core direct convolution.
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                const int wBase = ((c * 1) * KH) * KW; // weight[c,0,kh,kw]
                for (int oh = 0; oh < HO; ++oh) {
                    const int ih0 = oh * STRIDE_H - PAD_H;
                    for (int ow = 0; ow < WO; ++ow) {
                        const int iw0 = ow * STRIDE_W - PAD_W;

                        float acc = 0.0f;

                        for (int kh = 0; kh < KH; ++kh) {
                            const int ih = ih0 + kh * DIL_H;
                            if ((unsigned)ih >= (unsigned)H) continue;
                            for (int kw = 0; kw < KW; ++kw) {
                                const int iw = iw0 + kw * DIL_W;
                                if ((unsigned)iw >= (unsigned)W) continue;

                                // x[n,c,ih,iw] index: (((n*C + c)*H + ih)*W + iw)
                                const int xIdx = (((n * C + c) * H + ih) * W + iw);
                                // w[c,0,kh,kw] index: (((c*1 + 0)*KH + kh)*KW + kw)
                                const int wIdx = wBase + kh * KW + kw;

                                acc += xGm.GetValue(xIdx) * wGm.GetValue(wIdx);
                            }
                        }

                        // y[n,c,oh,ow] index: (((n*C + c)*HO + oh)*WO + ow)
                        const int yIdx = (((n * C + c) * HO + oh) * WO + ow);
                        yGm.SetValue(yIdx, acc);
                    }
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
};

extern "C" __global__ __aicore__ void conv_depthwise2d_asymmetric_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvDepthwise2DAsymAsym op;
    op.Init(x, weight, y, tiling_data.totalY, tiling_data.totalX, tiling_data.totalW);
    op.Process();
}
