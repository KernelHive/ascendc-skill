
#include "kernel_operator.h"

// Optimized coBlock=2 kernel with branch-free interior tiles and reduced vertical branching.
class KernelConvStandard2dSqInAsymKerDilPad_CoBlock2_TiledOW_InteriorFast {
public:
    __aicore__ inline KernelConvStandard2dSqInAsymKerDilPad_CoBlock2_TiledOW_InteriorFast() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t tasks, uint32_t OW, uint32_t OH,
                               uint32_t W, uint32_t H,
                               uint32_t CIN, uint32_t COUT,
                               uint32_t tileOw, uint32_t owTiles,
                               uint32_t coBlock, uint32_t coBlocks,
                               uint32_t interiorTileBegin, uint32_t interiorTileEnd)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        tasks_ = tasks;
        OW_ = OW; OH_ = OH;
        W_ = W; H_ = H;
        CIN_ = CIN; COUT_ = COUT;
        tileOw_ = tileOw;
        owTiles_ = owTiles;
        coBlock_ = coBlock;
        coBlocks_ = coBlocks;
        interiorTileBegin_ = interiorTileBegin;
        interiorTileEnd_ = interiorTileEnd;
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId  = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();

        const uint32_t tasksPerCore = (tasks_ + coreNum - 1U) / coreNum;
        uint32_t tStart = coreId * tasksPerCore;
        uint32_t tEnd = tStart + tasksPerCore;
        if (tEnd > tasks_) tEnd = tasks_;

        constexpr uint32_t N = 8;
        constexpr uint32_t KH = 5;
        constexpr uint32_t KW = 9;

        constexpr int32_t STRIDE_H = 1;
        constexpr int32_t STRIDE_W = 1;
        constexpr int32_t PAD_H = 2;
        constexpr int32_t PAD_W = 4;
        constexpr int32_t DIL_H = 2;
        constexpr int32_t DIL_W = 3;

        // task index order: ((((n*OH + oh)*coBlocks + coBlk)*owTiles) + tile)
        for (uint32_t t = tStart; t < tEnd; ++t) {
            uint32_t tmp = t;
            const uint32_t tile = tmp % owTiles_;
            tmp /= owTiles_;
            const uint32_t coBlk = tmp % coBlocks_;
            tmp /= coBlocks_;
            const uint32_t oh = tmp % OH_;
            const uint32_t n  = tmp / OH_;
            if (n >= N) continue;

            const uint32_t ow0 = tile * tileOw_;
            uint32_t owLen = tileOw_;
            if (ow0 + owLen > OW_) owLen = OW_ - ow0;

            const uint32_t co0 = coBlk * coBlock_; // 2
            if (co0 + 1 >= COUT_) continue;

            // Fast-path decision is a single range check on tile index
            const bool interior = (tile >= interiorTileBegin_) && (tile < interiorTileEnd_);

            // Precompute vertical valid r range to avoid per-r branch:
            // ih = oh - PAD_H + r*DIL_H, valid if 0 <= ih < H
            // => (PAD_H - oh) / DIL_H <= r <= (H-1 + PAD_H - oh) / DIL_H
            int32_t rBegin = (PAD_H - (int32_t)oh + DIL_H - 1) / DIL_H; // ceil
            int32_t rEnd = ((int32_t)H_ - 1 + PAD_H - (int32_t)oh) / DIL_H + 1; // exclusive
            if (rBegin < 0) rBegin = 0;
            if (rEnd > (int32_t)KH) rEnd = (int32_t)KH;
            if (rBegin >= rEnd) {
                // All rows invalid -> output is zero
                const uint32_t yBase0 = ((n * COUT_ + (co0 + 0)) * OH_ + oh) * OW_ + ow0;
                const uint32_t yBase1 = ((n * COUT_ + (co0 + 1)) * OH_ + oh) * OW_ + ow0;
                if (owLen > 0) { yGm.SetValue(yBase0 + 0, 0.f); yGm.SetValue(yBase1 + 0, 0.f); }
                if (owLen > 1) { yGm.SetValue(yBase0 + 1, 0.f); yGm.SetValue(yBase1 + 1, 0.f); }
                if (owLen > 2) { yGm.SetValue(yBase0 + 2, 0.f); yGm.SetValue(yBase1 + 2, 0.f); }
                if (owLen > 3) { yGm.SetValue(yBase0 + 3, 0.f); yGm.SetValue(yBase1 + 3, 0.f); }
                if (owLen > 4) { yGm.SetValue(yBase0 + 4, 0.f); yGm.SetValue(yBase1 + 4, 0.f); }
                if (owLen > 5) { yGm.SetValue(yBase0 + 5, 0.f); yGm.SetValue(yBase1 + 5, 0.f); }
                if (owLen > 6) { yGm.SetValue(yBase0 + 6, 0.f); yGm.SetValue(yBase1 + 6, 0.f); }
                if (owLen > 7) { yGm.SetValue(yBase0 + 7, 0.f); yGm.SetValue(yBase1 + 7, 0.f); }
                continue;
            }

            // Accumulators: 2 output channels * 8 lanes
            float a0_0=0.f,a0_1=0.f,a0_2=0.f,a0_3=0.f,a0_4=0.f,a0_5=0.f,a0_6=0.f,a0_7=0.f;
            float a1_0=0.f,a1_1=0.f,a1_2=0.f,a1_3=0.f,a1_4=0.f,a1_5=0.f,a1_6=0.f,a1_7=0.f;

            const uint32_t xNBase = n * CIN_ * H_ * W_;
            const uint32_t wCo0Base = (co0 + 0) * CIN_ * KH * KW;
            const uint32_t wCo1Base = (co0 + 1) * CIN_ * KH * KW;

            const int32_t ih0_base = (int32_t)oh * STRIDE_H - PAD_H;
            const int32_t iw0_base_lane0 = (int32_t)ow0 * STRIDE_W - PAD_W;

            if (interior) {
                // Branch-free horizontal taps; vertical already range-bounded by rBegin/rEnd.
                // Precompute base indices for lanes once (iw0 differs only by +lane).
                const uint32_t base0 = (uint32_t)(iw0_base_lane0 + 0);
                const uint32_t base1 = (uint32_t)(iw0_base_lane0 + 1);
                const uint32_t base2 = (uint32_t)(iw0_base_lane0 + 2);
                const uint32_t base3 = (uint32_t)(iw0_base_lane0 + 3);
                const uint32_t base4 = (uint32_t)(iw0_base_lane0 + 4);
                const uint32_t base5 = (uint32_t)(iw0_base_lane0 + 5);
                const uint32_t base6 = (uint32_t)(iw0_base_lane0 + 6);
                const uint32_t base7 = (uint32_t)(iw0_base_lane0 + 7);

                // Constant offsets s*DIL_W
                constexpr uint32_t o0 = 0;
                constexpr uint32_t o1 = 3;
                constexpr uint32_t o2 = 6;
                constexpr uint32_t o3 = 9;
                constexpr uint32_t o4 = 12;
                constexpr uint32_t o5 = 15;
                constexpr uint32_t o6 = 18;
                constexpr uint32_t o7 = 21;
                constexpr uint32_t o8 = 24;

                for (uint32_t ci = 0; ci < CIN_; ++ci) {
                    const uint32_t xNcBase = xNBase + ci * H_ * W_;
                    const uint32_t w0CiBase = wCo0Base + ci * KH * KW;
                    const uint32_t w1CiBase = wCo1Base + ci * KH * KW;

                    for (int32_t r = rBegin; r < rEnd; ++r) {
                        const uint32_t ih = (uint32_t)(ih0_base + r * DIL_H);
                        const uint32_t xRowBase = xNcBase + ih * W_;
                        const uint32_t wRow0 = w0CiBase + (uint32_t)r * KW;
                        const uint32_t wRow1 = w1CiBase + (uint32_t)r * KW;

                        // Load weights once per row for both channels
                        const float w00 = wGm.GetValue(wRow0 + 0);
                        const float w01 = wGm.GetValue(wRow0 + 1);
                        const float w02 = wGm.GetValue(wRow0 + 2);
                        const float w03 = wGm.GetValue(wRow0 + 3);
                        const float w04 = wGm.GetValue(wRow0 + 4);
                        const float w05 = wGm.GetValue(wRow0 + 5);
                        const float w06 = wGm.GetValue(wRow0 + 6);
                        const float w07 = wGm.GetValue(wRow0 + 7);
                        const float w08 = wGm.GetValue(wRow0 + 8);

                        const float w10 = wGm.GetValue(wRow1 + 0);
                        const float w11 = wGm.GetValue(wRow1 + 1);
                        const float w12 = wGm.GetValue(wRow1 + 2);
                        const float w13 = wGm.GetValue(wRow1 + 3);
                        const float w14 = wGm.GetValue(wRow1 + 4);
                        const float w15 = wGm.GetValue(wRow1 + 5);
                        const float w16 = wGm.GetValue(wRow1 + 6);
                        const float w17 = wGm.GetValue(wRow1 + 7);
                        const float w18 = wGm.GetValue(wRow1 + 8);

                        // Lane 0
                        {
                            const float x0 = xGm.GetValue(xRowBase + base0 + o0);
                            const float x1 = xGm.GetValue(xRowBase + base0 + o1);
                            const float x2 = xGm.GetValue(xRowBase + base0 + o2);
                            const float x3 = xGm.GetValue(xRowBase + base0 + o3);
                            const float x4 = xGm.GetValue(xRowBase + base0 + o4);
                            const float x5 = xGm.GetValue(xRowBase + base0 + o5);
                            const float x6 = xGm.GetValue(xRowBase + base0 + o6);
                            const float x7 = xGm.GetValue(xRowBase + base0 + o7);
                            const float x8 = xGm.GetValue(xRowBase + base0 + o8);
                            a0_0 += x0*w00 + x1*w01 + x2*w02 + x3*w03 + x4*w04 + x5*w05 + x6*w06 + x7*w07 + x8*w08;
                            a1_0 += x0*w10 + x1*w11 + x2*w12 + x3*w13 + x4*w14 + x5*w15 + x6*w16 + x7*w17 + x8*w18;
                        }
                        // Lane 1
                        {
                            const float x0 = xGm.GetValue(xRowBase + base1 + o0);
                            const float x1 = xGm.GetValue(xRowBase + base1 + o1);
                            const float x2 = xGm.GetValue(xRowBase + base1 + o2);
                            const float x3 = xGm.GetValue(xRowBase + base1 + o3);
                            const float x4 = xGm.GetValue(xRowBase + base1 + o4);
                            const float x5 = xGm.GetValue(xRowBase + base1 + o5);
                            const float x6 = xGm.GetValue(xRowBase + base1 + o6);
                            const float x7 = xGm.GetValue(xRowBase + base1 + o7);
                            const float x8 = xGm.GetValue(xRowBase + base1 + o8);
                            a0_1 += x0*w00 + x1*w01 + x2*w02 + x3*w03 + x4*w04 + x5*w05 + x6*w06 + x7*w07 + x8*w08;
                            a1_1 += x0*w10 + x1*w11 + x2*w12 + x3*w13 + x4*w14 + x5*w15 + x6*w16 + x7*w17 + x8*w18;
                        }
                        // Lane 2
                        {
                            const float x0 = xGm.GetValue(xRowBase + base2 + o0);
                            const float x1 = xGm.GetValue(xRowBase + base2 + o1);
                            const float x2 = xGm.GetValue(xRowBase + base2 + o2);
                            const float x3 = xGm.GetValue(xRowBase + base2 + o3);
                            const float x4 = xGm.GetValue(xRowBase + base2 + o4);
                            const float x5 = xGm.GetValue(xRowBase + base2 + o5);
                            const float x6 = xGm.GetValue(xRowBase + base2 + o6);
                            const float x7 = xGm.GetValue(xRowBase + base2 + o7);
                            const float x8 = xGm.GetValue(xRowBase + base2 + o8);
                            a0_2 += x0*w00 + x1*w01 + x2*w02 + x3*w03 + x4*w04 + x5*w05 + x6*w06 + x7*w07 + x8*w08;
                            a1_2 += x0*w10 + x1*w11 + x2*w12 + x3*w13 + x4*w14 + x5*w15 + x6*w16 + x7*w17 + x8*w18;
                        }
                        // Lane 3
                        {
                            const float x0 = xGm.GetValue(xRowBase + base3 + o0);
                            const float x1 = xGm.GetValue(xRowBase + base3 + o1);
                            const float x2 = xGm.GetValue(xRowBase + base3 + o2);
                            const float x3 = xGm.GetValue(xRowBase + base3 + o3);
                            const float x4 = xGm.GetValue(xRowBase + base3 + o4);
                            const float x5 = xGm.GetValue(xRowBase + base3 + o5);
                            const float x6 = xGm.GetValue(xRowBase + base3 + o6);
                            const float x7 = xGm.GetValue(xRowBase + base3 + o7);
                            const float x8 = xGm.GetValue(xRowBase + base3 + o8);
                            a0_3 += x0*w00 + x1*w01 + x2*w02 + x3*w03 + x4*w04 + x5*w05 + x6*w06 + x7*w07 + x8*w08;
                            a1_3 += x0*w10 + x1*w11 + x2*w12 + x3*w13 + x4*w14 + x5*w15 + x6*w16 + x7*w17 + x8*w18;
                        }
                        // Lane 4
                        {
                            const float x0 = xGm.GetValue(xRowBase + base4 + o0);
                            const float x1 = xGm.GetValue(xRowBase + base4 + o1);
                            const float x2 = xGm.GetValue(xRowBase + base4 + o2);
                            const float x3 = xGm.GetValue(xRowBase + base4 + o3);
                            const float x4 = xGm.GetValue(xRowBase + base4 + o4);
                            const float x5 = xGm.GetValue(xRowBase + base4 + o5);
                            const float x6 = xGm.GetValue(xRowBase + base4 + o6);
                            const float x7 = xGm.GetValue(xRowBase + base4 + o7);
                            const float x8 = xGm.GetValue(xRowBase + base4 + o8);
                            a0_4 += x0*w00 + x1*w01 + x2*w02 + x3*w03 + x4*w04 + x5*w05 + x6*w06 + x7*w07 + x8*w08;
                            a1_4 += x0*w10 + x1*w11 + x2*w12 + x3*w13 + x4*w14 + x5*w15 + x6*w16 + x7*w17 + x8*w18;
                        }
                        // Lane 5
                        {
                            const float x0 = xGm.GetValue(xRowBase + base5 + o0);
                            const float x1 = xGm.GetValue(xRowBase + base5 + o1);
                            const float x2 = xGm.GetValue(xRowBase + base5 + o2);
                            const float x3 = xGm.GetValue(xRowBase + base5 + o3);
                            const float x4 = xGm.GetValue(xRowBase + base5 + o4);
                            const float x5 = xGm.GetValue(xRowBase + base5 + o5);
                            const float x6 = xGm.GetValue(xRowBase + base5 + o6);
                            const float x7 = xGm.GetValue(xRowBase + base5 + o7);
                            const float x8 = xGm.GetValue(xRowBase + base5 + o8);
                            a0_5 += x0*w00 + x1*w01 + x2*w02 + x3*w03 + x4*w04 + x5*w05 + x6*w06 + x7*w07 + x8*w08;
                            a1_5 += x0*w10 + x1*w11 + x2*w12 + x3*w13 + x4*w14 + x5*w15 + x6*w16 + x7*w17 + x8*w18;
                        }
                        // Lane 6
                        {
                            const float x0 = xGm.GetValue(xRowBase + base6 + o0);
                            const float x1 = xGm.GetValue(xRowBase + base6 + o1);
                            const float x2 = xGm.GetValue(xRowBase + base6 + o2);
                            const float x3 = xGm.GetValue(xRowBase + base6 + o3);
                            const float x4 = xGm.GetValue(xRowBase + base6 + o4);
                            const float x5 = xGm.GetValue(xRowBase + base6 + o5);
                            const float x6 = xGm.GetValue(xRowBase + base6 + o6);
                            const float x7 = xGm.GetValue(xRowBase + base6 + o7);
                            const float x8 = xGm.GetValue(xRowBase + base6 + o8);
                            a0_6 += x0*w00 + x1*w01 + x2*w02 + x3*w03 + x4*w04 + x5*w05 + x6*w06 + x7*w07 + x8*w08;
                            a1_6 += x0*w10 + x1*w11 + x2*w12 + x3*w13 + x4*w14 + x5*w15 + x6*w16 + x7*w17 + x8*w18;
                        }
                        // Lane 7
                        {
                            const float x0 = xGm.GetValue(xRowBase + base7 + o0);
                            const float x1 = xGm.GetValue(xRowBase + base7 + o1);
                            const float x2 = xGm.GetValue(xRowBase + base7 + o2);
                            const float x3 = xGm.GetValue(xRowBase + base7 + o3);
                            const float x4 = xGm.GetValue(xRowBase + base7 + o4);
                            const float x5 = xGm.GetValue(xRowBase + base7 + o5);
                            const float x6 = xGm.GetValue(xRowBase + base7 + o6);
                            const float x7 = xGm.GetValue(xRowBase + base7 + o7);
                            const float x8 = xGm.GetValue(xRowBase + base7 + o8);
                            a0_7 += x0*w00 + x1*w01 + x2*w02 + x3*w03 + x4*w04 + x5*w05 + x6*w06 + x7*w07 + x8*w08;
                            a1_7 += x0*w10 + x1*w11 + x2*w12 + x3*w13 + x4*w14 + x5*w15 + x6*w16 + x7*w17 + x8*w18;
                        }
                    } // r
                } // ci
            } else {
                // Boundary path: keep W checks; vertical already r-bounded.
                for (uint32_t ci = 0; ci < CIN_; ++ci) {
                    const uint32_t xNcBase = xNBase + ci * H_ * W_;
                    const uint32_t w0CiBase = wCo0Base + ci * KH * KW;
                    const uint32_t w1CiBase = wCo1Base + ci * KH * KW;

                    for (int32_t r = rBegin; r < rEnd; ++r) {
                        const uint32_t ih = (uint32_t)(ih0_base + r * DIL_H);
                        const uint32_t xRowBase = xNcBase + ih * W_;
                        const uint32_t wRow0 = w0CiBase + (uint32_t)r * KW;
                        const uint32_t wRow1 = w1CiBase + (uint32_t)r * KW;

                        const float w0[9] = {
                            wGm.GetValue(wRow0 + 0), wGm.GetValue(wRow0 + 1), wGm.GetValue(wRow0 + 2),
                            wGm.GetValue(wRow0 + 3), wGm.GetValue(wRow0 + 4), wGm.GetValue(wRow0 + 5),
                            wGm.GetValue(wRow0 + 6), wGm.GetValue(wRow0 + 7), wGm.GetValue(wRow0 + 8)
                        };
                        const float w1[9] = {
                            wGm.GetValue(wRow1 + 0), wGm.GetValue(wRow1 + 1), wGm.GetValue(wRow1 + 2),
                            wGm.GetValue(wRow1 + 3), wGm.GetValue(wRow1 + 4), wGm.GetValue(wRow1 + 5),
                            wGm.GetValue(wRow1 + 6), wGm.GetValue(wRow1 + 7), wGm.GetValue(wRow1 + 8)
                        };

                        // lanes 0..7
                        for (int lane = 0; lane < 8; ++lane) {
                            const int32_t iw0 = iw0_base_lane0 + lane;
                            float s0 = 0.f, s1 = 0.f;
#pragma unroll
                            for (int k = 0; k < 9; ++k) {
                                const int32_t ix = iw0 + k * DIL_W;
                                if ((uint32_t)ix < W_) {
                                    const float xv = xGm.GetValue(xRowBase + (uint32_t)ix);
                                    s0 += xv * w0[k];
                                    s1 += xv * w1[k];
                                }
                            }
                            if (lane == 0) { a0_0 += s0; a1_0 += s1; }
                            else if (lane == 1) { a0_1 += s0; a1_1 += s1; }
                            else if (lane == 2) { a0_2 += s0; a1_2 += s1; }
                            else if (lane == 3) { a0_3 += s0; a1_3 += s1; }
                            else if (lane == 4) { a0_4 += s0; a1_4 += s1; }
                            else if (lane == 5) { a0_5 += s0; a1_5 += s1; }
                            else if (lane == 6) { a0_6 += s0; a1_6 += s1; }
                            else { a0_7 += s0; a1_7 += s1; }
                        }
                    } // r
                } // ci
            }

            const uint32_t yBase0 = ((n * COUT_ + (co0 + 0)) * OH_ + oh) * OW_ + ow0;
            const uint32_t yBase1 = ((n * COUT_ + (co0 + 1)) * OH_ + oh) * OW_ + ow0;

            if (owLen > 0) { yGm.SetValue(yBase0 + 0, a0_0); yGm.SetValue(yBase1 + 0, a1_0); }
            if (owLen > 1) { yGm.SetValue(yBase0 + 1, a0_1); yGm.SetValue(yBase1 + 1, a1_1); }
            if (owLen > 2) { yGm.SetValue(yBase0 + 2, a0_2); yGm.SetValue(yBase1 + 2, a1_2); }
            if (owLen > 3) { yGm.SetValue(yBase0 + 3, a0_3); yGm.SetValue(yBase1 + 3, a1_3); }
            if (owLen > 4) { yGm.SetValue(yBase0 + 4, a0_4); yGm.SetValue(yBase1 + 4, a1_4); }
            if (owLen > 5) { yGm.SetValue(yBase0 + 5, a0_5); yGm.SetValue(yBase1 + 5, a1_5); }
            if (owLen > 6) { yGm.SetValue(yBase0 + 6, a0_6); yGm.SetValue(yBase1 + 6, a1_6); }
            if (owLen > 7) { yGm.SetValue(yBase0 + 7, a0_7); yGm.SetValue(yBase1 + 7, a1_7); }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t tasks_{0};
    uint32_t OW_{0}, OH_{0};
    uint32_t W_{0}, H_{0};
    uint32_t CIN_{0}, COUT_{0};
    uint32_t tileOw_{0}, owTiles_{0};
    uint32_t coBlock_{0}, coBlocks_{0};
    uint32_t interiorTileBegin_{0}, interiorTileEnd_{0};
};

extern "C" __global__ __aicore__ void conv_standard2d_square_input_asymmetric_kernel_dilated_padded_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvStandard2dSqInAsymKerDilPad_CoBlock2_TiledOW_InteriorFast op;
    op.Init(x, weight, y,
            tiling_data.tasks, tiling_data.ow, tiling_data.oh,
            tiling_data.w, tiling_data.h,
            tiling_data.cin, tiling_data.cout,
            tiling_data.tile_ow, tiling_data.ow_tiles,
            tiling_data.coblock, tiling_data.co_blocks,
            tiling_data.interior_tile_begin, tiling_data.interior_tile_end);
    op.Process();
}
