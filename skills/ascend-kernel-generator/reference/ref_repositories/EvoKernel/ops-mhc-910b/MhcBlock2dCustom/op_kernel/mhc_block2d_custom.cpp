
#include "kernel_operator.h"

using F16 = half;
using F32 = float;
using I32 = int32_t;
using U32 = uint32_t;

class KernelMhcBlock2dCustom {
public:
    __aicore__ inline KernelMhcBlock2dCustom() {}

    static constexpr U32 C = 64;
    static constexpr U32 H = 32;
    static constexpr U32 W = 32;
    static constexpr U32 HW = 1024;
    static constexpr U32 NS = 4;
    static constexpr U32 CS = 16;         // channels per stream
    static constexpr U32 TILE = 256;      // elements per tile within a channel plane

    __aicore__ inline void Init(
        GM_ADDR x_fp16, GM_ADDR out_fp16,
        GM_ADDR map_w, GM_ADDR map_bias,
        GM_ADDR num_streams, GM_ADDR sinkhorn_iter, GM_ADDR sinkhorn_eps, GM_ADDR sinkhorn_temp,
        GM_ADDR in_channels, GM_ADDR out_channels, GM_ADDR height, GM_ADDR width,
        GM_ADDR y_fp16,
        U32 B_, U32 C_, U32 H_, U32 W_)
    {
        B = B_; Cin = C_; Hin = H_; Win = W_;

        const U32 blockNum = AscendC::GetBlockNum();
        const U32 blockIdx = AscendC::GetBlockIdx();
        const U32 bPerBlock = (B + blockNum - 1u) / blockNum;
        bStart = blockIdx * bPerBlock;
        U32 bEnd0 = bStart + bPerBlock;
        if (bEnd0 > B) bEnd0 = B;
        bEnd = bEnd0;

        const uint64_t elems = (uint64_t)B * (uint64_t)Cin * (uint64_t)Hin * (uint64_t)Win;
        xGm.SetGlobalBuffer((__gm__ F16*)x_fp16, elems);
        outGm.SetGlobalBuffer((__gm__ F16*)out_fp16, elems);
        yGm.SetGlobalBuffer((__gm__ F16*)y_fp16, elems);

        mapWGm.SetGlobalBuffer((__gm__ F32*)map_w, 16ull * 64ull);
        mapBGm.SetGlobalBuffer((__gm__ F32*)map_bias, 16ull);

        nsGm.SetGlobalBuffer((__gm__ I32*)num_streams, 1ull);
        itGm.SetGlobalBuffer((__gm__ I32*)sinkhorn_iter, 1ull);
        seGm.SetGlobalBuffer((__gm__ F32*)sinkhorn_eps, 1ull);
        stGm.SetGlobalBuffer((__gm__ F32*)sinkhorn_temp, 1ull);

        icGm.SetGlobalBuffer((__gm__ I32*)in_channels, 1ull);
        ocGm.SetGlobalBuffer((__gm__ I32*)out_channels, 1ull);
        hGm.SetGlobalBuffer((__gm__ I32*)height, 1ull);
        wGm.SetGlobalBuffer((__gm__ I32*)width, 1ull);

        // UB buffers: x tile fp16, 4 input tiles fp16, acc tile fp32.
        pipe.InitBuffer(xTileBuf, (U32)(TILE * sizeof(F16)));
        pipe.InitBuffer(in0Buf,    (U32)(TILE * sizeof(F16)));
        pipe.InitBuffer(in1Buf,    (U32)(TILE * sizeof(F16)));
        pipe.InitBuffer(in2Buf,    (U32)(TILE * sizeof(F16)));
        pipe.InitBuffer(in3Buf,    (U32)(TILE * sizeof(F16)));
        pipe.InitBuffer(accBuf,    (U32)(TILE * sizeof(F32)));

        supported = (Cin == C && Hin == H && Win == W);
    }

    __aicore__ inline void Process()
    {
        if (!supported) return;
        if (bStart >= bEnd) return;

        const I32 ns = nsGm.GetValue(0);
        const I32 ic = icGm.GetValue(0);
        const I32 oc = ocGm.GetValue(0);
        const I32 hh = hGm.GetValue(0);
        const I32 ww = wGm.GetValue(0);
        if (ns != (I32)NS || ic != (I32)C || oc != (I32)C || hh != (I32)H || ww != (I32)W) return;

        I32 iters = itGm.GetValue(0);
        if (iters < 0) iters = 0;
        if (iters > 50) iters = 50;

        F32 eps = seGm.GetValue(0);
        if (!(eps > (F32)0.0f)) eps = (F32)1e-8f;

        F32 temp = stGm.GetValue(0);
        if (temp == (F32)0.0f) temp = (F32)1.0f;

        // stage bias and weights into registers (small)
        F32 bias16[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) bias16[i] = mapBGm.GetValue((uint64_t)i);

        // weights [16,64]
        F32 w16x64[16][64];
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const uint64_t base = (uint64_t)r * 64ull;
            #pragma unroll
            for (int c = 0; c < 64; ++c) {
                w16x64[r][c] = mapWGm.GetValue(base + (uint64_t)c);
            }
        }

        auto xTile = xTileBuf.Get<F16>();
        auto in0 = in0Buf.Get<F16>();
        auto in1 = in1Buf.Get<F16>();
        auto in2 = in2Buf.Get<F16>();
        auto in3 = in3Buf.Get<F16>();
        auto acc = accBuf.Get<F32>();

        for (U32 b = bStart; b < bEnd; ++b) {
            // 1) avgpool x over H*W to pooled[64] in fp32, using tiled DataCopy
            F32 pooled[64];
            #pragma unroll
            for (int c = 0; c < 64; ++c) pooled[c] = (F32)0.0f;

            const uint64_t xBase = ((uint64_t)b * 64ull) * 1024ull;

            for (U32 c = 0; c < 64u; ++c) {
                const uint64_t chanBase = xBase + (uint64_t)c * 1024ull;
                F32 sum = (F32)0.0f;
                // 4 tiles of 256
                for (U32 t = 0; t < 1024u; t += TILE) {
                    AscendC::DataCopy(xTile, xGm[chanBase + (uint64_t)t], TILE);
                    // accumulate tile scalar (256) into sum; kept small vs 1024*64 total.
                    #pragma unroll
                    for (U32 i = 0; i < TILE; ++i) {
                        sum += (F32)xTile.GetValue(i);
                    }
                }
                pooled[c] = sum * ((F32)1.0f / (F32)1024.0f);
            }

            // 2) linear 64->16 with bias
            F32 logits16[16];
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                F32 s = bias16[r];
                #pragma unroll
                for (int c = 0; c < 64; ++c) s += w16x64[r][c] * pooled[c];
                logits16[r] = s;
            }

            // 3) sinkhorn on 4x4
            F32 mat[16];
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                F32 x = logits16[i];
                if (x > (F32)50.0f) x = (F32)50.0f;
                if (x < (F32)-50.0f) x = (F32)-50.0f;
                mat[i] = ExpApprox(x / temp);
            }
            for (I32 it = 0; it < iters; ++it) {
                // row norm
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int base = r * 4;
                    F32 s = mat[base+0] + mat[base+1] + mat[base+2] + mat[base+3];
                    const F32 inv = (F32)1.0f / (s + eps);
                    mat[base+0] *= inv; mat[base+1] *= inv; mat[base+2] *= inv; mat[base+3] *= inv;
                }
                // col norm
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    F32 s = mat[0*4+c] + mat[1*4+c] + mat[2*4+c] + mat[3*4+c];
                    const F32 inv = (F32)1.0f / (s + eps);
                    mat[0*4+c] *= inv; mat[1*4+c] *= inv; mat[2*4+c] *= inv; mat[3*4+c] *= inv;
                }
            }

            // 4) apply mapping to out: out is (B,64,32,32). Interpret as streams (4,16,HW).
            const uint64_t oBase = ((uint64_t)b * 64ull) * 1024ull;
            const uint64_t yBase = oBase;

            // iterate over stream output and channel-in-stream
            for (U32 sOut = 0; sOut < 4u; ++sOut) {
                for (U32 cIn = 0; cIn < 16u; ++cIn) {
                    const U32 outChan = sOut * 16u + cIn;

                    for (U32 t = 0; t < 1024u; t += TILE) {
                        // load 4 input stream tiles for same channel index cIn
                        const U32 inChan0 = 0u * 16u + cIn;
                        const U32 inChan1 = 1u * 16u + cIn;
                        const U32 inChan2 = 2u * 16u + cIn;
                        const U32 inChan3 = 3u * 16u + cIn;

                        AscendC::DataCopy(in0, outGm[oBase + (uint64_t)inChan0 * 1024ull + (uint64_t)t], TILE);
                        AscendC::DataCopy(in1, outGm[oBase + (uint64_t)inChan1 * 1024ull + (uint64_t)t], TILE);
                        AscendC::DataCopy(in2, outGm[oBase + (uint64_t)inChan2 * 1024ull + (uint64_t)t], TILE);
                        AscendC::DataCopy(in3, outGm[oBase + (uint64_t)inChan3 * 1024ull + (uint64_t)t], TILE);

                        const F32 m0 = mat[sOut * 4u + 0u];
                        const F32 m1 = mat[sOut * 4u + 1u];
                        const F32 m2 = mat[sOut * 4u + 2u];
                        const F32 m3 = mat[sOut * 4u + 3u];

                        // compute acc in fp32 and write fp16
                        #pragma unroll
                        for (U32 i = 0; i < TILE; ++i) {
                            const F32 v0 = (F32)in0.GetValue(i);
                            const F32 v1 = (F32)in1.GetValue(i);
                            const F32 v2 = (F32)in2.GetValue(i);
                            const F32 v3 = (F32)in3.GetValue(i);
                            const F32 yv = v0 * m0 + v1 * m1 + v2 * m2 + v3 * m3;
                            xTile.SetValue(i, (F16)yv); // reuse xTile buffer as output tile
                        }

                        AscendC::DataCopy(yGm[yBase + (uint64_t)outChan * 1024ull + (uint64_t)t], xTile, TILE);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline F32 ExpApprox(F32 x)
    {
        if (x > (F32)10.0f) x = (F32)10.0f;
        if (x < (F32)-10.0f) x = (F32)-10.0f;
        const F32 x2 = x * x;
        const F32 x3 = x2 * x;
        const F32 x4 = x2 * x2;
        const F32 x5 = x4 * x;
        return (F32)1.0f + x + x2 * (F32)0.5f + x3 * (F32)(1.0f/6.0f) + x4 * (F32)(1.0f/24.0f) + x5 * (F32)(1.0f/120.0f);
    }

private:
    AscendC::GlobalTensor<F16> xGm, outGm, yGm;
    AscendC::GlobalTensor<F32> mapWGm, mapBGm;

    AscendC::GlobalTensor<I32> nsGm, itGm, icGm, ocGm, hGm, wGm;
    AscendC::GlobalTensor<F32> seGm, stGm;

    U32 B, Cin, Hin, Win;
    U32 bStart, bEnd;
    bool supported;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xTileBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> in0Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> in1Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> in2Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> in3Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf;
};

extern "C" __global__ __aicore__ void mhc_block2d_custom(
    GM_ADDR x_fp16, GM_ADDR out_fp16,
    GM_ADDR map_w, GM_ADDR map_bias,
    GM_ADDR num_streams, GM_ADDR sinkhorn_iter, GM_ADDR sinkhorn_eps, GM_ADDR sinkhorn_temp,
    GM_ADDR in_channels, GM_ADDR out_channels, GM_ADDR height, GM_ADDR width,
    GM_ADDR y_fp16,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelMhcBlock2dCustom op;
    op.Init(x_fp16, out_fp16,
            map_w, map_bias,
            num_streams, sinkhorn_iter, sinkhorn_eps, sinkhorn_temp,
            in_channels, out_channels, height, width,
            y_fp16,
            td.B, td.C, td.H, td.W);
    op.Process();
}
