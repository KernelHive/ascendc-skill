
#include "kernel_operator.h"

using F16 = AscendC::half;
using F32 = float;
using I32 = int32_t;

static __aicore__ inline F32 ClampF(F32 v, F32 lo, F32 hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// NOTE: sinkhorn operates on small SxS in UB, scalar loops are OK.
// Large tensors use DataCopy + vector ops (Axpy/Add/Relu) to avoid GM scalars.
class KernelMhcBlockBottleneck2dCustom {
public:
    __aicore__ inline KernelMhcBlockBottleneck2dCustom() {}

    __aicore__ inline void Init(
        GM_ADDR out_bn3, GM_ADDR identity, GM_ADDR mapping_logits,
        GM_ADDR sinkhorn_iter, GM_ADDR sinkhorn_eps, GM_ADDR sinkhorn_temperature,
        GM_ADDR y,
        uint32_t B, uint32_t C, uint32_t H, uint32_t W,
        uint32_t S, uint32_t Cps, uint32_t K,
        uint32_t tileK, uint32_t Kpad)
    {
        this->B = B; this->C = C; this->H = H; this->W = W;
        this->S = S; this->Cps = Cps; this->K = K;
        this->tileK = tileK; this->Kpad = Kpad;

        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t bPerBlock = (B + blockNum - 1u) / blockNum;
        bStart = blockIdx * bPerBlock;
        uint32_t bEnd = bStart + bPerBlock;
        if (bEnd > B) bEnd = B;
        bCount = (bEnd > bStart) ? (bEnd - bStart) : 0;

        const uint64_t HW = (uint64_t)H * (uint64_t)W;
        const uint64_t total = (uint64_t)B * (uint64_t)C * HW;

        outGm.SetGlobalBuffer((__gm__ F16*)out_bn3, total);
        idGm.SetGlobalBuffer((__gm__ F16*)identity, total);
        yGm.SetGlobalBuffer((__gm__ F16*)y, total);
        mapGm.SetGlobalBuffer((__gm__ F16*)mapping_logits, (uint64_t)B * (uint64_t)S * (uint64_t)S);

        iterGm.SetGlobalBuffer((__gm__ I32*)sinkhorn_iter, 1);
        epsGm.SetGlobalBuffer((__gm__ F32*)sinkhorn_eps, 1);
        tempGm.SetGlobalBuffer((__gm__ F32*)sinkhorn_temperature, 1);

        // UB buffers:
        // mapH: fp16 S*S (aligned to 16)
        // mapF: fp32 S*S (aligned to 8)
        // sums: fp32 2*S (aligned to 8)
        // xTile: fp16 Kpad
        // idTile: fp16 Kpad
        // yTile: fp16 Kpad (output tile for one i)
        // acc: fp16 Kpad (accumulator in fp16 via Axpy; good enough for inference-style speed)
        // tmp: u8 for vector ops requiring temp
        mapHBytes = AlignUp(S * S, 16u) * (uint32_t)sizeof(F16);
        mapFBytes = AlignUp(S * S, 8u)  * (uint32_t)sizeof(F32);
        sumBytes  = AlignUp(2u * S, 8u) * (uint32_t)sizeof(F32);

        pipe.InitBuffer(mapHBuf, mapHBytes);
        pipe.InitBuffer(mapFBuf, mapFBytes);
        pipe.InitBuffer(sumBuf,  sumBytes);

        pipe.InitBuffer(xBuf,    (uint32_t)(Kpad * sizeof(F16)));
        pipe.InitBuffer(idBuf,   (uint32_t)(Kpad * sizeof(F16)));
        pipe.InitBuffer(yBuf,    (uint32_t)(Kpad * sizeof(F16)));
        pipe.InitBuffer(accBuf,  (uint32_t)(Kpad * sizeof(F16)));
        pipe.InitBuffer(tmpBuf,  2048); // shared tmp; enough for common vector unary ops

        sinkIters = iterGm.GetValue(0);
        if (sinkIters < 0) sinkIters = 0;
        if (sinkIters > 256) sinkIters = 256; // cap tighter for kernel time safety
        sinkEps = epsGm.GetValue(0);
        if (sinkEps < (F32)0) sinkEps = (F32)0;
        sinkTemp = tempGm.GetValue(0);
        if (sinkTemp <= (F32)0) sinkTemp = (F32)1;
    }

    __aicore__ inline void Process()
    {
        if (bCount == 0) return;

        const uint64_t HW = (uint64_t)H * (uint64_t)W;
        const uint64_t streamStride = (uint64_t)Cps * HW; // K

        AscendC::LocalTensor<uint8_t> tmp = tmpBuf.Get<uint8_t>();

        for (uint32_t bi = 0; bi < bCount; ++bi) {
            const uint32_t b = bStart + bi;
            const uint64_t baseB = (uint64_t)b * (uint64_t)C * HW;

            // Load logits and run sinkhorn -> mapF (fp32), then cast to fp16 mapA for Axpy scaling.
            AscendC::LocalTensor<F16> mapH = mapHBuf.Get<F16>();
            LoadMap(b, mapH);

            AscendC::LocalTensor<F32> mapF = mapFBuf.Get<F32>();
            LogitsToExpF32(mapH, mapF);

            SinkhornInPlace(mapF);

            // Cast mapF -> mapH (reuse mapH as fp16 weights)
            const uint32_t P = S * S;
            for (uint32_t p = 0; p < P; ++p) {
                mapH.SetValue(p, (F16)mapF.GetValue(p));
            }

            // K tiled over flattened (Cps*H*W)
            for (uint32_t k0 = 0; k0 < K; k0 += tileK) {
                const uint32_t curK = ((K - k0) > tileK) ? tileK : (K - k0);

                for (uint32_t i = 0; i < S; ++i) {
                    AscendC::LocalTensor<F16> acc = accBuf.Get<F16>();
                    AscendC::Duplicate(acc, (F16)0, Kpad);

                    // acc += sum_j map[i,j] * x_j
                    for (uint32_t j = 0; j < S; ++j) {
                        const F16 a = mapH.GetValue(i * S + j);

                        AscendC::LocalTensor<F16> xTile = xBuf.Get<F16>();
                        const uint64_t xBase = baseB + (uint64_t)j * streamStride + (uint64_t)k0;
                        CopyGmToUbPad(xTile, outGm, xBase, curK, Kpad);

                        AscendC::Axpy(acc, xTile, a, Kpad);
                    }

                    // yTile = acc + id
                    AscendC::LocalTensor<F16> yTile = yBuf.Get<F16>();
                    AscendC::DataCopy(yTile, acc, Kpad);

                    AscendC::LocalTensor<F16> idTile = idBuf.Get<F16>();
                    const uint64_t outBase = baseB + (uint64_t)i * streamStride + (uint64_t)k0;
                    CopyGmToUbPad(idTile, idGm, outBase, curK, Kpad);

                    AscendC::Add(yTile, yTile, idTile, Kpad);
                    AscendC::Relu(yTile, yTile, (int32_t)Kpad);

                    // store only valid
                    if (curK > 0) {
                        AscendC::DataCopy(yGm[outBase], yTile, curK);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t a) { return ((x + a - 1u) / a) * a; }

    __aicore__ inline void LoadMap(uint32_t b, const AscendC::LocalTensor<F16>& mapH)
    {
        const uint32_t P = S * S;
        const uint64_t base = (uint64_t)b * (uint64_t)P;
        AscendC::DataCopy(mapH, mapGm[base], P);
        // pad tail to 16 for safety
        const uint32_t padN = AlignUp(P, 16u);
        if (padN > P) AscendC::Duplicate(mapH[P], (F16)0, padN - P);
    }

    __aicore__ inline void LogitsToExpF32(const AscendC::LocalTensor<F16>& mapH,
                                         const AscendC::LocalTensor<F32>& mapF)
    {
        const uint32_t P = S * S;
        for (uint32_t p = 0; p < P; ++p) {
            F32 v = (F32)mapH.GetValue(p);
            v = ClampF(v, (F32)-50.0f, (F32)50.0f);
            v = v / sinkTemp;
            v = AscendC::Exp(v);
            mapF.SetValue(p, v);
        }
    }

    __aicore__ inline void SinkhornInPlace(const AscendC::LocalTensor<F32>& mapF)
    {
        AscendC::LocalTensor<F32> sums = sumBuf.Get<F32>();

        for (int it = 0; it < sinkIters; ++it) {
            // row sums
            for (uint32_t r = 0; r < S; ++r) {
                F32 rs = (F32)0;
                const uint32_t base = r * S;
                for (uint32_t c = 0; c < S; ++c) rs += mapF.GetValue(base + c);
                rs += sinkEps;
                sums.SetValue(r, rs);
            }
            // row normalize
            for (uint32_t r = 0; r < S; ++r) {
                const F32 inv = (F32)1.0f / sums.GetValue(r);
                const uint32_t base = r * S;
                for (uint32_t c = 0; c < S; ++c) {
                    mapF.SetValue(base + c, mapF.GetValue(base + c) * inv);
                }
            }

            // col sums
            for (uint32_t c = 0; c < S; ++c) {
                F32 cs = (F32)0;
                for (uint32_t r = 0; r < S; ++r) cs += mapF.GetValue(r * S + c);
                cs += sinkEps;
                sums.SetValue(S + c, cs);
            }
            // col normalize
            for (uint32_t c = 0; c < S; ++c) {
                const F32 inv = (F32)1.0f / sums.GetValue(S + c);
                for (uint32_t r = 0; r < S; ++r) {
                    const uint32_t idx = r * S + c;
                    mapF.SetValue(idx, mapF.GetValue(idx) * inv);
                }
            }
        }
    }

    __aicore__ inline void CopyGmToUbPad(const AscendC::LocalTensor<F16>& dst,
                                        const AscendC::GlobalTensor<F16>& src,
                                        uint64_t srcOffset,
                                        uint32_t valid,
                                        uint32_t padded)
    {
        if (valid > 0) AscendC::DataCopy(dst, src[srcOffset], valid);
        if (padded > valid) AscendC::Duplicate(dst[valid], (F16)0, padded - valid);
    }

private:
    AscendC::TPipe pipe;

    AscendC::GlobalTensor<F16> outGm, idGm, yGm, mapGm;
    AscendC::GlobalTensor<I32> iterGm;
    AscendC::GlobalTensor<F32> epsGm, tempGm;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> mapHBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> mapFBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sumBuf;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> xBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> idBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> yBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;

    uint32_t mapHBytes{}, mapFBytes{}, sumBytes{};

    uint32_t B{}, C{}, H{}, W{}, S{}, Cps{}, K{}, tileK{}, Kpad{};
    uint32_t bStart{}, bCount{};
    int sinkIters{};
    F32 sinkEps{}, sinkTemp{};
};

extern "C" __global__ __aicore__ void mhc_block_bottleneck2d_custom(
    GM_ADDR out_bn3, GM_ADDR identity, GM_ADDR mapping_logits,
    GM_ADDR sinkhorn_iter, GM_ADDR sinkhorn_eps, GM_ADDR sinkhorn_temperature,
    GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelMhcBlockBottleneck2dCustom op;
    op.Init(out_bn3, identity, mapping_logits,
            sinkhorn_iter, sinkhorn_eps, sinkhorn_temperature,
            y,
            td.B, td.C, td.H, td.W,
            td.S, td.Cps, td.K,
            td.tileK, td.Kpad);
    op.Process();
}
