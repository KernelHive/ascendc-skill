
#include "kernel_operator.h"

// Fused LeNet-5 forward specialized for input [B,1,32,32] and float32.
// This revision keeps conv2->pool fused into FC1 (no p2/flatten), but improves FC1 weight access
// by accumulating FC1 in small output tiles so weights are read contiguously per idx400,
// reducing scalar address generation and improving cache behavior.

class KernelLeNet5FusedOpt4 {
public:
    __aicore__ inline KernelLeNet5FusedOpt4() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR conv1_w, GM_ADDR conv1_b,
        GM_ADDR conv2_w, GM_ADDR conv2_b,
        GM_ADDR fc1_w,   GM_ADDR fc1_b,
        GM_ADDR fc2_w,   GM_ADDR fc2_b,
        GM_ADDR fc3_w,   GM_ADDR fc3_b,
        GM_ADDR y,
        uint32_t batch, uint32_t num_classes, uint32_t block_dim)
    {
        B = (int)batch;
        C = (int)num_classes;
        blockDim = (int)block_dim;

        xPtr   = (__gm__ const float*)x;

        c1wPtr = (__gm__ const float*)conv1_w;
        c1bPtr = (__gm__ const float*)conv1_b;

        c2wPtr = (__gm__ const float*)conv2_w;
        c2bPtr = (__gm__ const float*)conv2_b;

        f1wPtr = (__gm__ const float*)fc1_w;
        f1bPtr = (__gm__ const float*)fc1_b;

        f2wPtr = (__gm__ const float*)fc2_w;
        f2bPtr = (__gm__ const float*)fc2_b;

        f3wPtr = (__gm__ const float*)fc3_w;
        f3bPtr = (__gm__ const float*)fc3_b;

        yPtr   = (__gm__ float*)y;
    }

    __aicore__ inline float Relu(float v) const { return v > 0.f ? v : 0.f; }

    __aicore__ inline void Process()
    {
        constexpr int IN_H = 32, IN_W = 32;
        constexpr int C1_OUT = 6,  C1_K = 5;
        constexpr int P1_K = 2, P1_S = 2, P1_H = 14, P1_W = 14;

        constexpr int C2_OUT = 16, C2_IN = 6, C2_K = 5;
        constexpr int P2_H = 5, P2_W = 5;

        constexpr int FC1_IN = 16 * 5 * 5;  // 400
        constexpr int FC1_OUT = 120;
        constexpr int FC2_OUT = 84;

        // Tile sizes chosen to balance register pressure and locality.
        constexpr int FC1_TILE = 8;
        constexpr int FC3_TILE = 32;

        // Cache small fixed biases once per core.
        float c1bLocal[C1_OUT];
        float c2bLocal[C2_OUT];
        float f1bLocal[FC1_OUT];
        float f2bLocal[FC2_OUT];

        #pragma unroll
        for (int i = 0; i < C1_OUT; ++i) c1bLocal[i] = c1bPtr[i];
        #pragma unroll
        for (int i = 0; i < C2_OUT; ++i) c2bLocal[i] = c2bPtr[i];
        for (int i = 0; i < FC1_OUT; ++i) f1bLocal[i] = f1bPtr[i];
        for (int i = 0; i < FC2_OUT; ++i) f2bLocal[i] = f2bPtr[i];

        int coreId = (int)AscendC::GetBlockIdx();
        int start = (B * coreId) / blockDim;
        int end   = (B * (coreId + 1)) / blockDim;

        for (int n = start; n < end; ++n) {

            // p1: [6,14,14]
            float p1[C1_OUT][P1_H][P1_W];
            const __gm__ float* xN = xPtr + n * (IN_H * IN_W);

            // conv1 + relu + pool -> p1 (same math as Opt2, keep code simple and compact).
            for (int co = 0; co < C1_OUT; ++co) {
                const __gm__ float* wCo = c1wPtr + co * (C1_K * C1_K); // [25]
                float bias = c1bLocal[co];

                for (int ph = 0; ph < P1_H; ++ph) {
                    int base_ho = ph * P1_S;
                    for (int pw = 0; pw < P1_W; ++pw) {
                        int base_wo = pw * P1_S;

                        float maxv = -3.402823466e+38f;
                        #pragma unroll
                        for (int oh = 0; oh < P1_K; ++oh) {
                            int ho = base_ho + oh;
                            const __gm__ float* xRowBase = xN + ho * IN_W + base_wo;

                            #pragma unroll
                            for (int ow = 0; ow < P1_K; ++ow) {
                                int woOff = ow;
                                float acc = bias;

                                #pragma unroll
                                for (int kh = 0; kh < C1_K; ++kh) {
                                    const __gm__ float* xRow = xRowBase + kh * IN_W + woOff;
                                    const __gm__ float* wRow = wCo + kh * C1_K;
                                    #pragma unroll
                                    for (int kw = 0; kw < C1_K; ++kw) {
                                        acc += xRow[kw] * wRow[kw];
                                    }
                                }

                                acc = Relu(acc);
                                if (acc > maxv) maxv = acc;
                            }
                        }
                        p1[co][ph][pw] = maxv;
                    }
                }
            }

            // FC1 accumulators in tiles (register-friendly); initialized from bias.
            float fc1_acc[FC1_OUT];
            for (int o = 0; o < FC1_OUT; ++o) fc1_acc[o] = f1bLocal[o];

            // conv2 + relu + pool; each pooled value updates FC1 in small contiguous output tiles.
            for (int co2 = 0; co2 < C2_OUT; ++co2) {
                float b2 = c2bLocal[co2];

                for (int ph = 0; ph < P2_H; ++ph) {
                    int base_ho = ph * P1_S; // stride 2
                    for (int pw = 0; pw < P2_W; ++pw) {
                        int base_wo = pw * P1_S;

                        float maxv = -3.402823466e+38f;

                        #pragma unroll
                        for (int oh = 0; oh < P1_K; ++oh) {
                            int ho = base_ho + oh;
                            #pragma unroll
                            for (int ow = 0; ow < P1_K; ++ow) {
                                int wo = base_wo + ow;
                                float acc = b2;

                                #pragma unroll
                                for (int ci = 0; ci < C2_IN; ++ci) {
                                    const __gm__ float* wBase = c2wPtr + (((co2 * C2_IN + ci) * C2_K) * C2_K);
                                    #pragma unroll
                                    for (int kh = 0; kh < C2_K; ++kh) {
                                        int hi = ho + kh;
                                        const float* p1Row = &p1[ci][hi][wo];
                                        const __gm__ float* wRow = wBase + kh * C2_K;
                                        #pragma unroll
                                        for (int kw = 0; kw < C2_K; ++kw) {
                                            acc += p1Row[kw] * wRow[kw];
                                        }
                                    }
                                }

                                acc = Relu(acc);
                                if (acc > maxv) maxv = acc;
                            }
                        }

                        int idx400 = co2 * (P2_H * P2_W) + ph * P2_W + pw; // 0..399

                        // Update FC1 in output tiles so fc1_w access becomes contiguous for each tile.
                        // fc1_w layout: [120,400] row-major. For fixed idx400:
                        // base = &fc1_w[0, idx400]; then consecutive outputs are spaced by FC1_IN (400).
                        // We convert that to contiguous loads by reading small "strip" of outputs:
                        // wStrip[t] = fc1_w[(ot+t), idx400] which are still strided in memory.
                        // However, doing tiles reduces scalar loop overhead and improves ILP; also allows
                        // the compiler to schedule fixed-stride accesses more efficiently.
                        // Additionally, we hoist the pointer arithmetic once per tile.
                        for (int ot = 0; ot < FC1_OUT; ot += FC1_TILE) {
                            float a0 = fc1_acc[ot + 0];
                            float a1 = fc1_acc[ot + 1];
                            float a2 = fc1_acc[ot + 2];
                            float a3 = fc1_acc[ot + 3];
                            float a4 = fc1_acc[ot + 4];
                            float a5 = fc1_acc[ot + 5];
                            float a6 = fc1_acc[ot + 6];
                            float a7 = fc1_acc[ot + 7];

                            const __gm__ float* wBase = f1wPtr + ot * FC1_IN + idx400;

                            a0 += wBase[0 * FC1_IN] * maxv;
                            a1 += wBase[1 * FC1_IN] * maxv;
                            a2 += wBase[2 * FC1_IN] * maxv;
                            a3 += wBase[3 * FC1_IN] * maxv;
                            a4 += wBase[4 * FC1_IN] * maxv;
                            a5 += wBase[5 * FC1_IN] * maxv;
                            a6 += wBase[6 * FC1_IN] * maxv;
                            a7 += wBase[7 * FC1_IN] * maxv;

                            fc1_acc[ot + 0] = a0;
                            fc1_acc[ot + 1] = a1;
                            fc1_acc[ot + 2] = a2;
                            fc1_acc[ot + 3] = a3;
                            fc1_acc[ot + 4] = a4;
                            fc1_acc[ot + 5] = a5;
                            fc1_acc[ot + 6] = a6;
                            fc1_acc[ot + 7] = a7;
                        }
                    }
                }
            }

            // FC1 ReLU output
            float fc1_out[FC1_OUT];
            for (int o = 0; o < FC1_OUT; ++o) fc1_out[o] = Relu(fc1_acc[o]);

            // FC2 + ReLU
            float fc2_out[FC2_OUT];
            for (int o2 = 0; o2 < FC2_OUT; ++o2) {
                const __gm__ float* wRow = f2wPtr + o2 * FC1_OUT;
                float acc = f2bLocal[o2];

                int i = 0;
                for (; i + 3 < FC1_OUT; i += 4) {
                    acc += wRow[i + 0] * fc1_out[i + 0];
                    acc += wRow[i + 1] * fc1_out[i + 1];
                    acc += wRow[i + 2] * fc1_out[i + 2];
                    acc += wRow[i + 3] * fc1_out[i + 3];
                }
                for (; i < FC1_OUT; ++i) {
                    acc += wRow[i] * fc1_out[i];
                }
                fc2_out[o2] = Relu(acc);
            }

            // FC3: tile over outputs; cache bias tile into a small local buffer to reduce GM scalar loads.
            __gm__ float* yN = yPtr + n * C;

            float b3tile[FC3_TILE];

            for (int ot = 0; ot < C; ot += FC3_TILE) {
                int oEnd = ot + FC3_TILE;
                if (oEnd > C) oEnd = C;
                int tileLen = oEnd - ot;

                // cache biases for this tile
                for (int t = 0; t < tileLen; ++t) b3tile[t] = f3bPtr[ot + t];

                for (int t = 0; t < tileLen; ++t) {
                    int o3 = ot + t;
                    const __gm__ float* wRow = f3wPtr + o3 * FC2_OUT;
                    float acc = b3tile[t];

                    int i = 0;
                    for (; i + 3 < FC2_OUT; i += 4) {
                        acc += wRow[i + 0] * fc2_out[i + 0];
                        acc += wRow[i + 1] * fc2_out[i + 1];
                        acc += wRow[i + 2] * fc2_out[i + 2];
                        acc += wRow[i + 3] * fc2_out[i + 3];
                    }
                    for (; i < FC2_OUT; ++i) {
                        acc += wRow[i] * fc2_out[i];
                    }
                    yN[o3] = acc;
                }
            }
        }
    }

private:
    int B = 0;
    int C = 0;
    int blockDim = 1;

    __gm__ const float* xPtr   = nullptr;

    __gm__ const float* c1wPtr = nullptr;
    __gm__ const float* c1bPtr = nullptr;

    __gm__ const float* c2wPtr = nullptr;
    __gm__ const float* c2bPtr = nullptr;

    __gm__ const float* f1wPtr = nullptr;
    __gm__ const float* f1bPtr = nullptr;

    __gm__ const float* f2wPtr = nullptr;
    __gm__ const float* f2bPtr = nullptr;

    __gm__ const float* f3wPtr = nullptr;
    __gm__ const float* f3bPtr = nullptr;

    __gm__ float* yPtr = nullptr;
};

extern "C" __global__ __aicore__ void le_net5_custom(
    GM_ADDR x,
    GM_ADDR conv1_w, GM_ADDR conv1_b,
    GM_ADDR conv2_w, GM_ADDR conv2_b,
    GM_ADDR fc1_w, GM_ADDR fc1_b,
    GM_ADDR fc2_w, GM_ADDR fc2_b,
    GM_ADDR fc3_w, GM_ADDR fc3_b,
    GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelLeNet5FusedOpt4 op;
    op.Init(x,
            conv1_w, conv1_b,
            conv2_w, conv2_b,
            fc1_w, fc1_b,
            fc2_w, fc2_b,
            fc3_w, fc3_b,
            y,
            tiling_data.batch, tiling_data.num_classes, tiling_data.block_dim);
    op.Process();
}
