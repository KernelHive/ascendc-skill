
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }
template <typename T>
__aicore__ inline T MinT(T a, T b) { return a < b ? a : b; }

class KernelMatmulGeluSoftmax {
public:
    __aicore__ inline KernelMatmulGeluSoftmax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR y,
                               uint32_t B, uint32_t K, uint32_t N,
                               uint32_t rowsPerCore, uint32_t tileN)
    {
        B_ = B; K_ = K; N_ = N;
        rowsPerCore_ = (rowsPerCore == 0) ? 1u : rowsPerCore;

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
        yGm_.SetGlobalBuffer((__gm__ float*)y, ySize);

        tileN_ = (tileN == 0) ? 1u : tileN;
        if (N_ > 0 && tileN_ > N_) tileN_ = N_;
        if (N_ == 0) tileN_ = 1;

        pipe_.InitBuffer(qX_, BUFFER_NUM, K_ * sizeof(float));
        pipe_.InitBuffer(qY_, BUFFER_NUM, tileN_ * sizeof(float));

        // Single VECCALC scratch with explicit, non-overlapping layout:
        // logits[tileN] | exp[tileN] | geluTmp0[tileN] | geluTmp1[tileN] |
        // dotTmp[3*kChunkK] | reduceTmp[max(tileN,kChunkK)] | scalar[256]
        constexpr uint32_t kChunkK = 1024;
        constexpr uint32_t kScalarFloats = 256;
        const uint32_t reduceFloats = (tileN_ > kChunkK) ? tileN_ : kChunkK;
        tmpFloats_ = (4u * tileN_) + (3u * kChunkK) + reduceFloats + kScalarFloats;
        pipe_.InitBuffer(tmpCalc_, tmpFloats_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (rowCount_ == 0 || B_ == 0 || N_ == 0 || K_ == 0) return;

        const uint32_t tilesN = CeilDivU32(N_, tileN_);

        for (uint32_t r = 0; r < rowCount_; ++r) {
            const uint32_t row = startRow_ + r;

            AscendC::LocalTensor<float> xRow = qX_.AllocTensor<float>();
            const uint64_t xBase = static_cast<uint64_t>(row) * static_cast<uint64_t>(K_);
            AscendC::DataCopy(xRow, xGm_[xBase], K_);
            qX_.EnQue<float>(xRow);
            AscendC::LocalTensor<float> xV = qX_.DeQue<float>();

            // Pass1: rowMax over GELU(logits)
            float rowMax = -3.402823466e+38f;
            for (uint32_t tN = 0; tN < tilesN; ++tN) {
                const uint32_t nOff = tN * tileN_;
                const uint32_t nLen = MinT<uint32_t>(tileN_, N_ - nOff);

                AscendC::LocalTensor<float> logits = Logits();
                ComputeLogitsTile(xV, nOff, nLen, logits);
                GeluInplace(logits, nLen);

                AscendC::LocalTensor<float> red = ReduceTmp();
                AscendC::LocalTensor<float> s = ScalarBuf();
                AscendC::ReduceMax<float>(s, logits, red, static_cast<int32_t>(nLen), false);
                const float tmax = s(0);
                if (tmax > rowMax) rowMax = tmax;
            }

            // Pass2: sumexp over GELU(logits)-rowMax
            float rowSum = 0.0f;
            for (uint32_t tN = 0; tN < tilesN; ++tN) {
                const uint32_t nOff = tN * tileN_;
                const uint32_t nLen = MinT<uint32_t>(tileN_, N_ - nOff);

                AscendC::LocalTensor<float> logits = Logits();
                ComputeLogitsTile(xV, nOff, nLen, logits);
                GeluInplace(logits, nLen);

                AscendC::Adds(logits, logits, -rowMax, static_cast<int32_t>(nLen));
                AscendC::LocalTensor<float> expv = ExpVec();
                AscendC::Exp(expv, logits, static_cast<int32_t>(nLen));

                AscendC::LocalTensor<float> red = ReduceTmp();
                AscendC::LocalTensor<float> s = ScalarBuf();
                AscendC::ReduceSum<float>(s, expv, red, static_cast<int32_t>(nLen));
                rowSum += s(0);
            }

            // Guard: invalid sum -> write zeros (handles <=0, NaN, Inf)
            if (!(rowSum > 0.0f) || !(rowSum < 3.402823466e+38f)) {
                WriteRowZeros(row, tilesN);
                qX_.FreeTensor(xV);
                continue;
            }

            const float invSum = ReciprocalScalar(rowSum);

            // Pass3: write softmax = exp(GELU(logits)-rowMax) * invSum
            for (uint32_t tN = 0; tN < tilesN; ++tN) {
                const uint32_t nOff = tN * tileN_;
                const uint32_t nLen = MinT<uint32_t>(tileN_, N_ - nOff);
                const uint64_t yBase = static_cast<uint64_t>(row) * static_cast<uint64_t>(N_) + static_cast<uint64_t>(nOff);

                AscendC::LocalTensor<float> logits = Logits();
                ComputeLogitsTile(xV, nOff, nLen, logits);
                GeluInplace(logits, nLen);

                AscendC::Adds(logits, logits, -rowMax, static_cast<int32_t>(nLen));
                AscendC::LocalTensor<float> expv = ExpVec();
                AscendC::Exp(expv, logits, static_cast<int32_t>(nLen));
                AscendC::Muls(expv, expv, invSum, static_cast<int32_t>(nLen));

                AscendC::LocalTensor<float> yOut = qY_.AllocTensor<float>();
                AscendC::DataCopy(yOut, expv, nLen);
                qY_.EnQue<float>(yOut);
                AscendC::LocalTensor<float> yV = qY_.DeQue<float>();
                AscendC::DataCopy(yGm_[yBase], yV, nLen);
                qY_.FreeTensor(yV);
            }

            qX_.FreeTensor(xV);
        }
    }

private:
    __aicore__ inline AscendC::LocalTensor<float> TmpBase() { return tmpCalc_.Get<float>(); }
    __aicore__ inline AscendC::LocalTensor<float> Logits() { return TmpBase(); }                    // [0:tileN)
    __aicore__ inline AscendC::LocalTensor<float> ExpVec() { return TmpBase()[tileN_]; }            // [tileN:2*tileN)
    __aicore__ inline AscendC::LocalTensor<float> GeluTmp0() { return TmpBase()[2u * tileN_]; }     // [2*tileN:3*tileN)
    __aicore__ inline AscendC::LocalTensor<float> GeluTmp1() { return TmpBase()[3u * tileN_]; }     // [3*tileN:4*tileN)

    __aicore__ inline AscendC::LocalTensor<float> DotTmpBase()
    {
        constexpr uint32_t kChunkK = 1024;
        return TmpBase()[4u * tileN_]; // size 3*kChunkK
    }

    __aicore__ inline AscendC::LocalTensor<float> ReduceTmp()
    {
        constexpr uint32_t kChunkK = 1024;
        return TmpBase()[4u * tileN_ + 3u * kChunkK];
    }

    __aicore__ inline AscendC::LocalTensor<float> ScalarBuf()
    {
        constexpr uint32_t kChunkK = 1024;
        const uint32_t reduceFloats = (tileN_ > kChunkK) ? tileN_ : kChunkK;
        return TmpBase()[4u * tileN_ + 3u * kChunkK + reduceFloats];
    }

    __aicore__ inline float ReciprocalScalar(float v)
    {
        AscendC::LocalTensor<float> s = ScalarBuf();
        s(0) = v;
        AscendC::Reciprocal(s, s, 1);
        return s(0);
    }

    __aicore__ inline void ComputeLogitsTile(const AscendC::LocalTensor<float>& xRow,
                                            uint32_t nOff, uint32_t nLen,
                                            AscendC::LocalTensor<float>& logitsOut)
    {
        // logits = bias
        AscendC::DataCopy(logitsOut, bGm_[static_cast<uint64_t>(nOff)], nLen);

        constexpr uint32_t kChunkK = 1024;
        AscendC::LocalTensor<float> dot = DotTmpBase();
        AscendC::LocalTensor<float> wChunk = dot;                 // [0:kChunkK]
        AscendC::LocalTensor<float> xChunk = dot[kChunkK];        // [kChunkK:2*kChunkK]
        AscendC::LocalTensor<float> redWork = dot[2u * kChunkK];  // [2*kChunkK:3*kChunkK]
        AscendC::LocalTensor<float> s = ScalarBuf();

        for (uint32_t j = 0; j < nLen; ++j) {
            const uint32_t nIdx = nOff + j;
            float acc = 0.0f;
            uint32_t kOff = 0;
            while (kOff < K_) {
                const uint32_t kLen = MinT<uint32_t>(kChunkK, K_ - kOff);
                const uint64_t wBase = static_cast<uint64_t>(nIdx) * static_cast<uint64_t>(K_) + static_cast<uint64_t>(kOff);

                AscendC::DataCopy(wChunk, wGm_[wBase], kLen);
                AscendC::DataCopy(xChunk, xRow[static_cast<int32_t>(kOff)], kLen);

                AscendC::Mul(wChunk, wChunk, xChunk, static_cast<int32_t>(kLen));
                AscendC::ReduceSum<float>(s, wChunk, redWork, static_cast<int32_t>(kLen));
                acc += s(0);

                kOff += kLen;
            }
            logitsOut(static_cast<int32_t>(j)) = logitsOut(static_cast<int32_t>(j)) + acc;
        }
    }

    // GELU approximation: 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
    __aicore__ inline void GeluInplace(AscendC::LocalTensor<float>& x, uint32_t n)
    {
        const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
        const float k1 = 0.044715f;

        AscendC::LocalTensor<float> t0 = GeluTmp0();
        AscendC::LocalTensor<float> t1 = GeluTmp1();
        AscendC::LocalTensor<float> red = ReduceTmp();

        // t0 = x^3
        AscendC::Mul(t0, x, x, static_cast<int32_t>(n));             // x^2
        AscendC::Mul(t0, t0, x, static_cast<int32_t>(n));            // x^3
        AscendC::Muls(t0, t0, k1, static_cast<int32_t>(n));          // 0.044715*x^3
        AscendC::Add(t0, t0, x, static_cast<int32_t>(n));            // x + ...
        AscendC::Muls(t0, t0, k0, static_cast<int32_t>(n));          // z

        // tanh(z) = 2*sigmoid(2z) - 1 ; sigmoid(u)=1/(1+exp(-u))
        AscendC::Muls(t0, t0, 2.0f, static_cast<int32_t>(n));        // u = 2z
        AscendC::Muls(t1, t0, -1.0f, static_cast<int32_t>(n));       // -u
        AscendC::Exp(t1, t1, static_cast<int32_t>(n));               // exp(-u)
        AscendC::Adds(t1, t1, 1.0f, static_cast<int32_t>(n));        // 1+exp(-u)
        AscendC::Reciprocal(t1, t1, static_cast<int32_t>(n));        // sigmoid(u)
        AscendC::Muls(t1, t1, 2.0f, static_cast<int32_t>(n));        // 2*sigmoid
        AscendC::Adds(t1, t1, -1.0f, static_cast<int32_t>(n));       // tanh(z)

        // x = 0.5*x*(1+tanh)
        AscendC::Adds(t1, t1, 1.0f, static_cast<int32_t>(n));
        AscendC::Mul(x, x, t1, static_cast<int32_t>(n));
        AscendC::Muls(x, x, 0.5f, static_cast<int32_t>(n));

        (void)red; // keep explicit view usage consistent; red is used elsewhere.
    }

    __aicore__ inline void WriteRowZeros(uint32_t row, uint32_t tilesN)
    {
        for (uint32_t tN = 0; tN < tilesN; ++tN) {
            const uint32_t nOff = tN * tileN_;
            const uint32_t nLen = MinT<uint32_t>(tileN_, N_ - nOff);
            const uint64_t yBase = static_cast<uint64_t>(row) * static_cast<uint64_t>(N_) + static_cast<uint64_t>(nOff);

            AscendC::LocalTensor<float> yOut = qY_.AllocTensor<float>();
            AscendC::Duplicate(yOut, 0.0f, static_cast<int32_t>(nLen));
            qY_.EnQue<float>(yOut);
            AscendC::LocalTensor<float> yV = qY_.DeQue<float>();
            AscendC::DataCopy(yGm_[yBase], yV, nLen);
            qY_.FreeTensor(yV);
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> qX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> qY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpCalc_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t B_{0}, K_{0}, N_{0};
    uint32_t rowsPerCore_{1};
    uint32_t tileN_{1};
    uint32_t startRow_{0}, rowCount_{0};
    uint32_t tmpFloats_{0};
};

extern "C" __global__ __aicore__ void matmul_gelu_softmax_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);
    KernelMatmulGeluSoftmax op;
    op.Init(x, weight, bias, y, t.B, t.K, t.N, t.rowsPerCore, t.tileN);
    op.Process();
}
