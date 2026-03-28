
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }
template <typename T>
__aicore__ inline T MinT(T a, T b) { return a < b ? a : b; }

class KernelMatmulDropoutSoftmax {
public:
    __aicore__ inline KernelMatmulDropoutSoftmax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR dropout_p, GM_ADDR training,
                               GM_ADDR y, GM_ADDR workspace,
                               uint32_t B, uint32_t K, uint32_t N,
                               uint32_t rowsPerCore, uint32_t tileN, uint32_t kChunkK)
    {
        B_ = B; K_ = K; N_ = N;
        rowsPerCore_ = rowsPerCore;
        tileN_ = (tileN == 0) ? 1u : tileN;
        if (N_ > 0 && tileN_ > N_) tileN_ = N_;
        if (N_ == 0) tileN_ = 1;
        kChunkK_ = (kChunkK == 0) ? 1u : kChunkK;
        if (K_ > 0 && kChunkK_ > K_) kChunkK_ = K_;
        if (K_ == 0) kChunkK_ = 1;

        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        startRow_ = blockIdx * rowsPerCore_;
        uint32_t endRow = startRow_ + rowsPerCore_;
        if (endRow > B_) endRow = B_;
        rowCount_ = (startRow_ >= B_) ? 0u : (endRow - startRow_);

        const uint64_t xSize = static_cast<uint64_t>(B_) * static_cast<uint64_t>(K_);
        const uint64_t wSize = static_cast<uint64_t>(N_) * static_cast<uint64_t>(K_);
        const uint64_t ySize = static_cast<uint64_t>(B_) * static_cast<uint64_t>(N_);

        xGm_.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm_.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm_.SetGlobalBuffer((__gm__ float*)bias, static_cast<uint64_t>(N_));
        dpGm_.SetGlobalBuffer((__gm__ float*)dropout_p, 1);
        trGm_.SetGlobalBuffer((__gm__ int32_t*)training, 1);
        yGm_.SetGlobalBuffer((__gm__ float*)y, ySize);

        // workspace layout from host:
        // ws0: logits [B*N]
        // ws1: rowMax [B]
        // ws2: rowSum [B]
        __gm__ uint8_t* wsBase = (__gm__ uint8_t*)workspace;
        const uint64_t wsLogitsBytes = static_cast<uint64_t>(B_) * static_cast<uint64_t>(N_) * sizeof(float);
        const uint64_t wsRowBytes    = static_cast<uint64_t>(B_) * sizeof(float);
        logitsWs_.SetGlobalBuffer((__gm__ float*)wsBase, static_cast<uint64_t>(B_) * static_cast<uint64_t>(N_));
        rowMaxWs_.SetGlobalBuffer((__gm__ float*)(wsBase + wsLogitsBytes), static_cast<uint64_t>(B_));
        rowSumWs_.SetGlobalBuffer((__gm__ float*)(wsBase + wsLogitsBytes + wsRowBytes), static_cast<uint64_t>(B_));

        pipe_.InitBuffer(qX_, BUFFER_NUM, K_ * sizeof(float));
        pipe_.InitBuffer(qIn_, BUFFER_NUM, tileN_ * sizeof(float));   // generic VECIN for tile vectors
        pipe_.InitBuffer(qDp_, 1, 1 * sizeof(float));
        pipe_.InitBuffer(qTr_, 1, 1 * sizeof(int32_t));
        pipe_.InitBuffer(qOut_, BUFFER_NUM, tileN_ * sizeof(float));  // VECOUT only for writes

        // VECCALC partition (floats), strictly non-overlapping:
        // logits(tileN) | exp(tileN) | workW(kChunkK) | workX(kChunkK) | redWork(kChunkK) | scalar(256)
        tmpLogitsOff_ = 0;
        tmpExpOff_ = tmpLogitsOff_ + tileN_;
        tmpWOff_ = tmpExpOff_ + tileN_;
        tmpXOff_ = tmpWOff_ + kChunkK_;
        tmpRedOff_ = tmpXOff_ + kChunkK_;
        tmpScalarOff_ = tmpRedOff_ + kChunkK_;
        tmpFloats_ = tmpScalarOff_ + 256u;
        pipe_.InitBuffer(tmpCalc_, tmpFloats_ * sizeof(float));

        // load dropout_p/training scalars
        AscendC::LocalTensor<float> dpL = qDp_.AllocTensor<float>();
        AscendC::DataCopy(dpL, dpGm_[0], 1);
        qDp_.EnQue<float>(dpL);
        AscendC::LocalTensor<float> dpV = qDp_.DeQue<float>();
        dropoutP_ = dpV(0);
        qDp_.FreeTensor(dpV);

        AscendC::LocalTensor<int32_t> trL = qTr_.AllocTensor<int32_t>();
        AscendC::DataCopy(trL, trGm_[0], 1);
        qTr_.EnQue<int32_t>(trL);
        AscendC::LocalTensor<int32_t> trV = qTr_.DeQue<int32_t>();
        training_ = (trV(0) != 0);
        qTr_.FreeTensor(trV);

        if (dropoutP_ < 0.0f) dropoutP_ = 0.0f;
        if (dropoutP_ > 1.0f) dropoutP_ = 1.0f;
        keepP_ = 1.0f - dropoutP_;
        scale_ = (keepP_ > 0.0f) ? (1.0f / keepP_) : 0.0f;
        period_ = 5u;
    }

    __aicore__ inline void Process()
    {
        if (rowCount_ == 0 || B_ == 0 || N_ == 0 || K_ == 0) return;

        const uint32_t tilesN = CeilDivU32(N_, tileN_);

        for (uint32_t r = 0; r < rowCount_; ++r) {
            const uint32_t row = startRow_ + r;

            // load x row
            AscendC::LocalTensor<float> xRow = qX_.AllocTensor<float>();
            const uint64_t xBase = static_cast<uint64_t>(row) * static_cast<uint64_t>(K_);
            AscendC::DataCopy(xRow, xGm_[xBase], K_);
            qX_.EnQue<float>(xRow);
            AscendC::LocalTensor<float> xV = qX_.DeQue<float>();

            // pass1: compute logits tiles once and store to ws, accumulate max
            float rowMax = -3.402823466e+38f;
            for (uint32_t tN = 0; tN < tilesN; ++tN) {
                const uint32_t nOff = tN * tileN_;
                const uint32_t nLen = MinT<uint32_t>(tileN_, N_ - nOff);

                AscendC::LocalTensor<float> logits = Logits();
                ComputeLogitsTileRowStationary(xV, nOff, nLen, logits);
                ApplyDeterministicDropoutInplace(logits, row, nOff, nLen);

                // write logits to workspace (use VECOUT queue only for writes)
                const uint64_t wsBase = static_cast<uint64_t>(row) * static_cast<uint64_t>(N_) + static_cast<uint64_t>(nOff);
                AscendC::LocalTensor<float> out = qOut_.AllocTensor<float>();
                AscendC::DataCopy(out, logits, nLen);
                qOut_.EnQue<float>(out);
                AscendC::LocalTensor<float> outV = qOut_.DeQue<float>();
                AscendC::DataCopy(logitsWs_[wsBase], outV, nLen);
                qOut_.FreeTensor(outV);

                // reduce max
                AscendC::LocalTensor<float> work = RedWork();
                AscendC::LocalTensor<float> s = ScalarBuf();
                AscendC::ReduceMax<float>(s, logits, work, static_cast<int32_t>(nLen), false);
                float tmax = s(0);
                if (tmax > rowMax) rowMax = tmax;
            }

            // store rowMax
            StoreScalarToGm(rowMaxWs_, row, rowMax);

            // pass2: read logits ws and accumulate exp sum
            float rowSum = 0.0f;
            for (uint32_t tN = 0; tN < tilesN; ++tN) {
                const uint32_t nOff = tN * tileN_;
                const uint32_t nLen = MinT<uint32_t>(tileN_, N_ - nOff);
                const uint64_t wsBase = static_cast<uint64_t>(row) * static_cast<uint64_t>(N_) + static_cast<uint64_t>(nOff);

                // read logits tile using VECIN queue
                AscendC::LocalTensor<float> in = qIn_.AllocTensor<float>();
                AscendC::DataCopy(in, logitsWs_[wsBase], nLen);
                qIn_.EnQue<float>(in);
                AscendC::LocalTensor<float> lV = qIn_.DeQue<float>();

                AscendC::Adds(lV, lV, -rowMax, static_cast<int32_t>(nLen));
                AscendC::LocalTensor<float> expv = ExpVec();
                AscendC::Exp(expv, lV, static_cast<int32_t>(nLen));

                AscendC::LocalTensor<float> work = RedWork();
                AscendC::LocalTensor<float> s = ScalarBuf();
                AscendC::ReduceSum<float>(s, expv, work, static_cast<int32_t>(nLen));
                rowSum += s(0);

                qIn_.FreeTensor(lV);
            }

            StoreScalarToGm(rowSumWs_, row, rowSum);

            if (!(rowSum > 0.0f)) {
                WriteRowZeros(row, tilesN);
                qX_.FreeTensor(xV);
                continue;
            }
            float invSum = ReciprocalScalar(rowSum);

            // pass3: read logits ws again, exp+normalize+write y
            for (uint32_t tN = 0; tN < tilesN; ++tN) {
                const uint32_t nOff = tN * tileN_;
                const uint32_t nLen = MinT<uint32_t>(tileN_, N_ - nOff);
                const uint64_t wsBase = static_cast<uint64_t>(row) * static_cast<uint64_t>(N_) + static_cast<uint64_t>(nOff);
                const uint64_t yBase  = static_cast<uint64_t>(row) * static_cast<uint64_t>(N_) + static_cast<uint64_t>(nOff);

                AscendC::LocalTensor<float> in = qIn_.AllocTensor<float>();
                AscendC::DataCopy(in, logitsWs_[wsBase], nLen);
                qIn_.EnQue<float>(in);
                AscendC::LocalTensor<float> lV = qIn_.DeQue<float>();

                AscendC::Adds(lV, lV, -rowMax, static_cast<int32_t>(nLen));
                AscendC::LocalTensor<float> expv = ExpVec();
                AscendC::Exp(expv, lV, static_cast<int32_t>(nLen));
                AscendC::Muls(expv, expv, invSum, static_cast<int32_t>(nLen));

                AscendC::LocalTensor<float> out = qOut_.AllocTensor<float>();
                AscendC::DataCopy(out, expv, nLen);
                qOut_.EnQue<float>(out);
                AscendC::LocalTensor<float> outV = qOut_.DeQue<float>();
                AscendC::DataCopy(yGm_[yBase], outV, nLen);
                qOut_.FreeTensor(outV);

                qIn_.FreeTensor(lV);
            }

            qX_.FreeTensor(xV);
        }
    }

private:
    __aicore__ inline AscendC::LocalTensor<float> TmpBase() { return tmpCalc_.Get<float>(); }
    __aicore__ inline AscendC::LocalTensor<float> Logits() { return TmpBase()[static_cast<int32_t>(tmpLogitsOff_)]; }
    __aicore__ inline AscendC::LocalTensor<float> ExpVec() { return TmpBase()[static_cast<int32_t>(tmpExpOff_)]; }
    __aicore__ inline AscendC::LocalTensor<float> WChunk() { return TmpBase()[static_cast<int32_t>(tmpWOff_)]; }
    __aicore__ inline AscendC::LocalTensor<float> XChunk() { return TmpBase()[static_cast<int32_t>(tmpXOff_)]; }
    __aicore__ inline AscendC::LocalTensor<float> RedWork() { return TmpBase()[static_cast<int32_t>(tmpRedOff_)]; }
    __aicore__ inline AscendC::LocalTensor<float> ScalarBuf() { return TmpBase()[static_cast<int32_t>(tmpScalarOff_)]; }

    __aicore__ inline float ReciprocalScalar(float v)
    {
        AscendC::LocalTensor<float> s = ScalarBuf();
        s(0) = v;
        AscendC::Reciprocal(s, s, 1);
        return s(0);
    }

    __aicore__ inline void StoreScalarToGm(AscendC::GlobalTensor<float>& gm, uint32_t idx, float v)
    {
        AscendC::LocalTensor<float> s = ScalarBuf();
        s(0) = v;
        AscendC::LocalTensor<float> out = qOut_.AllocTensor<float>();
        out(0) = s(0);
        qOut_.EnQue<float>(out);
        AscendC::LocalTensor<float> outV = qOut_.DeQue<float>();
        AscendC::DataCopy(gm[static_cast<uint64_t>(idx)], outV, 1);
        qOut_.FreeTensor(outV);
    }

    __aicore__ inline void ComputeLogitsTileRowStationary(const AscendC::LocalTensor<float>& xRow,
                                                         uint32_t nOff, uint32_t nLen,
                                                         AscendC::LocalTensor<float>& logitsOut)
    {
        // init logits with bias
        AscendC::DataCopy(logitsOut, bGm_[static_cast<uint64_t>(nOff)], nLen);

        // K chunk loop: for each chunk, load x chunk once and stream weights for all nLen outputs
        uint32_t kOff = 0;
        while (kOff < K_) {
            const uint32_t kLen = MinT<uint32_t>(kChunkK_, K_ - kOff);

            AscendC::LocalTensor<float> xC = XChunk();
            AscendC::DataCopy(xC, xRow[static_cast<int32_t>(kOff)], kLen);

            // for each output in tile, accumulate dot over this chunk using vector ops
            AscendC::LocalTensor<float> wC = WChunk();
            AscendC::LocalTensor<float> mulTmp = RedWork(); // reuse as mul buffer of length kLen

            for (uint32_t j = 0; j < nLen; ++j) {
                const uint32_t nIdx = nOff + j;
                const uint64_t wBase = static_cast<uint64_t>(nIdx) * static_cast<uint64_t>(K_) + static_cast<uint64_t>(kOff);
                AscendC::DataCopy(wC, wGm_[wBase], kLen);

                AscendC::Mul(mulTmp, wC, xC, static_cast<int32_t>(kLen));
                AscendC::LocalTensor<float> s = ScalarBuf();
                // reduce sum over kLen into s(0)
                AscendC::ReduceSum<float>(s, mulTmp, WChunk(), static_cast<int32_t>(kLen));

                logitsOut(static_cast<int32_t>(j)) = logitsOut(static_cast<int32_t>(j)) + s(0);
            }

            kOff += kLen;
        }
    }

    __aicore__ inline void ApplyDeterministicDropoutInplace(AscendC::LocalTensor<float>& logits,
                                                           uint32_t row, uint32_t nOff, uint32_t nLen)
    {
        if (!training_ || dropoutP_ <= 0.0f) return;
        if (keepP_ <= 0.0f) {
            AscendC::Duplicate(logits, 0.0f, static_cast<int32_t>(nLen));
            return;
        }
        for (uint32_t i = 0; i < nLen; ++i) {
            const uint32_t idx = row + nOff + i;
            const bool keep = ((idx % period_) != 0u);
            const float m = keep ? scale_ : 0.0f;
            logits(static_cast<int32_t>(i)) = logits(static_cast<int32_t>(i)) * m;
        }
    }

    __aicore__ inline void WriteRowZeros(uint32_t row, uint32_t tilesN)
    {
        for (uint32_t tN = 0; tN < tilesN; ++tN) {
            const uint32_t nOff = tN * tileN_;
            const uint32_t nLen = MinT<uint32_t>(tileN_, N_ - nOff);
            const uint64_t yBase = static_cast<uint64_t>(row) * static_cast<uint64_t>(N_) + static_cast<uint64_t>(nOff);

            AscendC::LocalTensor<float> out = qOut_.AllocTensor<float>();
            AscendC::Duplicate(out, 0.0f, static_cast<int32_t>(nLen));
            qOut_.EnQue<float>(out);
            AscendC::LocalTensor<float> outV = qOut_.DeQue<float>();
            AscendC::DataCopy(yGm_[yBase], outV, nLen);
            qOut_.FreeTensor(outV);
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> qX_;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> qIn_;
    AscendC::TQue<AscendC::TPosition::VECIN,  1>         qDp_;
    AscendC::TQue<AscendC::TPosition::VECIN,  1>         qTr_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> qOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpCalc_;

    AscendC::GlobalTensor<float> xGm_, wGm_, bGm_, dpGm_, yGm_;
    AscendC::GlobalTensor<int32_t> trGm_;

    AscendC::GlobalTensor<float> logitsWs_, rowMaxWs_, rowSumWs_;

    uint32_t B_{0}, K_{0}, N_{0};
    uint32_t rowsPerCore_{1};
    uint32_t tileN_{1};
    uint32_t kChunkK_{256};
    uint32_t startRow_{0}, rowCount_{0};

    uint32_t tmpLogitsOff_{0}, tmpExpOff_{0}, tmpWOff_{0}, tmpXOff_{0}, tmpRedOff_{0}, tmpScalarOff_{0};
    uint32_t tmpFloats_{0};

    float dropoutP_{0.0f}, keepP_{1.0f}, scale_{1.0f};
    bool training_{false};
    uint32_t period_{5};
};

extern "C" __global__ __aicore__ void matmul_dropout_softmax_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR dropout_p, GM_ADDR training,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    KernelMatmulDropoutSoftmax op;
    op.Init(x, weight, bias, dropout_p, training, y, workspace, t.B, t.K, t.N, t.rowsPerCore, t.tileN, t.kChunkK);
    op.Process();
}
