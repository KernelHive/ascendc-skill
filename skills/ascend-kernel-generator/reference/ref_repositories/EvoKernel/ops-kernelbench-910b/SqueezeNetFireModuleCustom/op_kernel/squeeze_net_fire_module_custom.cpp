
#include "kernel_operator.h"

static constexpr int kN = 128;
static constexpr int kCin = 3;
static constexpr int kH = 256;
static constexpr int kW = 256;
static constexpr int kHW = kH * kW;

static constexpr int kCs = 6;
static constexpr int kC1 = 64;
static constexpr int kC3 = 64;
static constexpr int kOutC = kC1 + kC3;

static constexpr int kTileC = 8;

class KernelSqueezeNetFireModuleCustom {
public:
    __aicore__ inline KernelSqueezeNetFireModuleCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR w_squeeze, GM_ADDR b_squeeze,
        GM_ADDR w_expand1, GM_ADDR b_expand1,
        GM_ADDR w_expand3, GM_ADDR b_expand3,
        GM_ADDR y,
        uint32_t totalX, uint32_t totalWs, uint32_t totalBs,
        uint32_t totalW1, uint32_t totalB1,
        uint32_t totalW3, uint32_t totalB3,
        uint32_t totalY,
        uint32_t rowsNH, uint32_t outcTiles, uint32_t tileC)
    {
        (void)totalX; (void)totalWs; (void)totalBs;
        (void)totalW1; (void)totalB1; (void)totalW3; (void)totalB3;
        (void)totalY;

        xGm.SetGlobalBuffer((__gm__ float*)x);
        wsGm.SetGlobalBuffer((__gm__ float*)w_squeeze);
        bsGm.SetGlobalBuffer((__gm__ float*)b_squeeze);
        w1Gm.SetGlobalBuffer((__gm__ float*)w_expand1);
        b1Gm.SetGlobalBuffer((__gm__ float*)b_expand1);
        w3Gm.SetGlobalBuffer((__gm__ float*)w_expand3);
        b3Gm.SetGlobalBuffer((__gm__ float*)b_expand3);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        rowsNH_ = rowsNH;
        outcTiles_ = outcTiles;
        tileC_ = tileC;
    }

    __aicore__ inline float ReluF(float v) { return v > 0.0f ? v : 0.0f; }

    __aicore__ inline void LoadSqueezeParamsOnce()
    {
#pragma unroll
        for (int cs = 0; cs < kCs; ++cs) {
            bsReg_[cs] = bsGm.GetValue(cs);
            const int wsBase = cs * kCin;
            wsReg_[cs][0] = wsGm.GetValue(wsBase + 0);
            wsReg_[cs][1] = wsGm.GetValue(wsBase + 1);
            wsReg_[cs][2] = wsGm.GetValue(wsBase + 2);
        }
    }

    __aicore__ inline void ComputeSqueezeAtPix(const int xPixBase, float sOut[kCs])
    {
        const float x0 = xGm.GetValue(xPixBase + 0 * kHW);
        const float x1 = xGm.GetValue(xPixBase + 1 * kHW);
        const float x2 = xGm.GetValue(xPixBase + 2 * kHW);
#pragma unroll
        for (int cs = 0; cs < kCs; ++cs) {
            float s = bsReg_[cs]
                    + x0 * wsReg_[cs][0]
                    + x1 * wsReg_[cs][1]
                    + x2 * wsReg_[cs][2];
            sOut[cs] = ReluF(s);
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t bdim = (uint32_t)AscendC::GetBlockNum();

        const uint32_t blocksTotal = rowsNH_ * outcTiles_;
        if (blocksTotal == 0) return;

        for (uint32_t logical = bid; logical < blocksTotal; logical += bdim) {
            const uint32_t rowNH = logical / outcTiles_;
            const uint32_t tileIdx = logical - rowNH * outcTiles_;

            const uint32_t n = rowNH / (uint32_t)kH;
            const uint32_t h0 = rowNH - n * (uint32_t)kH;

            const uint32_t outcStart = tileIdx * tileC_;
            const uint32_t outcEnd = outcStart + tileC_;

            const int xNBase = (int)n * (kCin * kHW);
            const int yNBase = (int)n * (kOutC * kHW);
            const int xRowBase = xNBase + (int)h0 * kW;
            const int yRowBase = yNBase + (int)h0 * kW;

            LoadSqueezeParamsOnce();

            if (outcEnd <= (uint32_t)kC1) {
                float w1Tile[kTileC][kCs];
                float b1Tile[kTileC];
#pragma unroll
                for (int t = 0; t < kTileC; ++t) {
                    const int co = (int)outcStart + t;
                    b1Tile[t] = b1Gm.GetValue(co);
                    const int w1Base = co * kCs;
#pragma unroll
                    for (int cs = 0; cs < kCs; ++cs) {
                        w1Tile[t][cs] = w1Gm.GetValue(w1Base + cs);
                    }
                }

                // Process two pixels per iteration to reduce scalar overhead.
                int w0 = 0;
                for (; w0 + 1 < kW; w0 += 2) {
                    float s0[kCs];
                    float s1[kCs];
                    const int xPix0 = xRowBase + w0;
                    const int xPix1 = xRowBase + (w0 + 1);
                    ComputeSqueezeAtPix(xPix0, s0);
                    ComputeSqueezeAtPix(xPix1, s1);

#pragma unroll
                    for (int t = 0; t < kTileC; ++t) {
                        float acc0 = b1Tile[t];
                        float acc1 = b1Tile[t];
#pragma unroll
                        for (int cs = 0; cs < kCs; ++cs) {
                            const float wv = w1Tile[t][cs];
                            acc0 += s0[cs] * wv;
                            acc1 += s1[cs] * wv;
                        }
                        acc0 = ReluF(acc0);
                        acc1 = ReluF(acc1);
                        const int outc = (int)outcStart + t;
                        const int yBase = yRowBase + outc * kHW;
                        yGm.SetValue(yBase + w0, acc0);
                        yGm.SetValue(yBase + (w0 + 1), acc1);
                    }
                }

                if (w0 < kW) {
                    float s[kCs];
                    const int xPix = xRowBase + w0;
                    ComputeSqueezeAtPix(xPix, s);
#pragma unroll
                    for (int t = 0; t < kTileC; ++t) {
                        float acc = b1Tile[t];
#pragma unroll
                        for (int cs = 0; cs < kCs; ++cs) {
                            acc += s[cs] * w1Tile[t][cs];
                        }
                        acc = ReluF(acc);
                        const int outc = (int)outcStart + t;
                        const int yBase = yRowBase + outc * kHW;
                        yGm.SetValue(yBase + w0, acc);
                    }
                }
            } else {
                const uint32_t co3Start = outcStart - (uint32_t)kC1;

                float w3Tile[kTileC][kCs][9];
                float b3Tile[kTileC];
#pragma unroll
                for (int t = 0; t < kTileC; ++t) {
                    const int co3 = (int)co3Start + t;
                    b3Tile[t] = b3Gm.GetValue(co3);
                    const int w3Base = co3 * (kCs * 9);
#pragma unroll
                    for (int cs = 0; cs < kCs; ++cs) {
#pragma unroll
                        for (int tap = 0; tap < 9; ++tap) {
                            w3Tile[t][cs][tap] = w3Gm.GetValue(w3Base + cs * 9 + tap);
                        }
                    }
                }

                float sL[3][kCs];
                float sM[3][kCs];
                float sR[3][kCs];

#pragma unroll
                for (int r = 0; r < 3; ++r) {
#pragma unroll
                    for (int cs = 0; cs < kCs; ++cs) {
                        sL[r][cs] = 0.0f;
                        sM[r][cs] = 0.0f;
                        sR[r][cs] = 0.0f;
                    }
                }

#pragma unroll
                for (int r = 0; r < 3; ++r) {
                    const int ih = (int)h0 + r - 1;
                    if ((unsigned)ih < (unsigned)kH) {
                        const int xRow = xNBase + ih * kW;
                        ComputeSqueezeAtPix(xRow + 0, sM[r]);
                        if (kW > 1) {
                            ComputeSqueezeAtPix(xRow + 1, sR[r]);
                        }
                    }
                }

                // Use running offsets to reduce address recomputation.
                int yBaseTile[kTileC];
#pragma unroll
                for (int t = 0; t < kTileC; ++t) {
                    const int outc = (int)outcStart + t;
                    yBaseTile[t] = yRowBase + outc * kHW;
                }

                for (int w0 = 0; w0 < kW; ++w0) {
#pragma unroll
                    for (int t = 0; t < kTileC; ++t) {
                        float acc = b3Tile[t];
#pragma unroll
                        for (int kh = 0; kh < 3; ++kh) {
#pragma unroll
                            for (int cs = 0; cs < kCs; ++cs) {
                                acc += sL[kh][cs] * w3Tile[t][cs][kh * 3 + 0];
                                acc += sM[kh][cs] * w3Tile[t][cs][kh * 3 + 1];
                                acc += sR[kh][cs] * w3Tile[t][cs][kh * 3 + 2];
                            }
                        }
                        acc = ReluF(acc);
                        yGm.SetValue(yBaseTile[t] + w0, acc);
                    }

#pragma unroll
                    for (int r = 0; r < 3; ++r) {
#pragma unroll
                        for (int cs = 0; cs < kCs; ++cs) {
                            sL[r][cs] = sM[r][cs];
                            sM[r][cs] = sR[r][cs];
                        }
                    }

                    const int wNextR = w0 + 2;
                    if (wNextR < kW) {
#pragma unroll
                        for (int r = 0; r < 3; ++r) {
                            const int ih = (int)h0 + r - 1;
                            if ((unsigned)ih < (unsigned)kH) {
                                const int xRow = xNBase + ih * kW;
                                ComputeSqueezeAtPix(xRow + wNextR, sR[r]);
                            } else {
#pragma unroll
                                for (int cs = 0; cs < kCs; ++cs) sR[r][cs] = 0.0f;
                            }
                        }
                    } else {
#pragma unroll
                        for (int r = 0; r < 3; ++r) {
#pragma unroll
                            for (int cs = 0; cs < kCs; ++cs) sR[r][cs] = 0.0f;
                        }
                    }
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wsGm;
    AscendC::GlobalTensor<float> bsGm;
    AscendC::GlobalTensor<float> w1Gm;
    AscendC::GlobalTensor<float> b1Gm;
    AscendC::GlobalTensor<float> w3Gm;
    AscendC::GlobalTensor<float> b3Gm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t rowsNH_{0};
    uint32_t outcTiles_{0};
    uint32_t tileC_{0};

    float bsReg_[kCs];
    float wsReg_[kCs][kCin];
};

extern "C" __global__ __aicore__ void squeeze_net_fire_module_custom(
    GM_ADDR x,
    GM_ADDR w_squeeze, GM_ADDR b_squeeze,
    GM_ADDR w_expand1, GM_ADDR b_expand1,
    GM_ADDR w_expand3, GM_ADDR b_expand3,
    GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelSqueezeNetFireModuleCustom op;
    op.Init(x,
            w_squeeze, b_squeeze,
            w_expand1, b_expand1,
            w_expand3, b_expand3,
            y,
            tiling_data.totalX, tiling_data.totalWs, tiling_data.totalBs,
            tiling_data.totalW1, tiling_data.totalB1,
            tiling_data.totalW3, tiling_data.totalB3,
            tiling_data.totalY,
            tiling_data.rowsNH, tiling_data.outcTiles, tiling_data.tileC);
    op.Process();
}
