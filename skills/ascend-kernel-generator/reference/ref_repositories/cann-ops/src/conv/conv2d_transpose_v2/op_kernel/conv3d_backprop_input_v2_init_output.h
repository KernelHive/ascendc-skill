/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_backprop_input_v2_init_output.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_INIT_OUTPUT_H
#define CONV3D_BACKPROP_INPUT_V2_INIT_OUTPUT_H

namespace AscendC {
#if __CCE_AICORE__ == 220
using FixpipeParamsNZ = AscendC::FixpipeParamsV220;
using FixpipeParamsRowMajor = AscendC::FixpipeParamsV220;
#endif
constexpr int32_t L0C_M = TOTAL_L0C_SIZE / 64;  // TOTAL_LOC_SIZE / (BLOCK_CUBE * sizeof(float))
constexpr int32_t L0C_ELEMENTS = L0C_M * BLOCK_CUBE;
constexpr int32_t BEST_FIXPIPE_ELEMENTS = 32;  // 128B/sizeof(float)
constexpr int32_t CUBE_BLOCK_LEN_BITS = 4;  // 16 = 2^4
constexpr int32_t FRACTAL_LEN_BITS = 9;  // 512 = 2^9

enum class InitOutputFlag {
    NO_INIT = 0,
    L0_INIT = 1,
    L1_INIT = 2,
};

template <typename yType, int yFormat, InitOutputFlag initOutputFlag>
class Conv3dDxInitOutput {
public:
    __aicore__ inline Conv3dDxInitOutput() {}
    __aicore__ inline void Init(GM_ADDR y, const Conv3DBackpropInputV2TilingData *tilingData)
    {
        InitTilingData(tilingData);
        // init global buffer
        yGm_.SetGlobalBuffer((__gm__ yType *)y);
        if constexpr (initOutputFlag == InitOutputFlag::L1_INIT) {
            pipe_.InitBuffer(localBuffer_, TOTAL_L1_SIZE);
        } else if constexpr (initOutputFlag == InitOutputFlag::L0_INIT) {
            pipe_.InitBuffer(l0A_, TOTAL_L0A_SIZE);
            pipe_.InitBuffer(l0B_, TOTAL_L0B_SIZE);
            pipe_.InitBuffer(l0C_, 1, TOTAL_L0C_SIZE);
        }
    }

    /** main logical function
     */

    __aicore__ inline uint64_t Ceil(uint64_t a, uint32_t b)
    {
        return (a + b - 1) / b;
    }

    __aicore__ inline void SetL0CZero() {
        if ASCEND_IS_AIC {
            LocalTensor<half> l0a = l0A_.template Get<half>();
            LocalTensor<half> l0b = l0B_.template Get<half>();
            LocalTensor<float> l0c = l0C_.template AllocTensor<float>();
            InitConstValue(l0a, {1, static_cast<uint16_t>(TOTAL_L0A_SIZE >> FRACTAL_LEN_BITS), 0, 0U});
            InitConstValue(l0b, {1, static_cast<uint16_t>(TOTAL_L0B_SIZE >> FRACTAL_LEN_BITS), 0, 0U});
            MmadParams mmadParams;
            mmadParams.m = L0C_M;
            mmadParams.n = BLOCK_CUBE;
            mmadParams.k = 1;
            mmadParams.cmatrixInitVal = true;
            TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE1_M>();
            SetFlag<HardEvent::MTE1_M>(eventId);
            WaitFlag<HardEvent::MTE1_M>(eventId);

            MmadImpl(l0c, l0a, l0b, mmadParams);  // 由于m*n=L0C_M * BLOCK_CUBE 一定会超过2560, 所以无需pipe_barrier
            l0C_.EnQue(l0c);
        }
    }

    __aicore__ inline void ProcessWithL1()
    {
        uint32_t usedCoreNum = GetBlockNum();
        uint64_t clearSizePerCore = AlignUp(Ceil(outputSize_, usedCoreNum), BLOCK_CUBE * BLOCK_CUBE);
        usedCoreNum = Ceil(outputSize_, clearSizePerCore);
        if (GetBlockIdx() >= usedCoreNum) {
            SyncAllCores();
            return;
        }

        uint64_t offset = clearSizePerCore * GetBlockIdx();
        uint64_t realClearSize = outputSize_ - offset;
        realClearSize = realClearSize > clearSizePerCore ? clearSizePerCore : realClearSize;

        LocalTensor<yType> popBuffer = localBuffer_.template Get<yType>();
        uint32_t localSize = static_cast<uint64_t>(popBuffer.GetSize()) < realClearSize
                                 ? popBuffer.GetSize()
                                 : static_cast<uint32_t>(realClearSize);
        constexpr uint16_t MAX_BLOCK_NUM = (1 << 15) - 1;
        uint16_t repeatTime = Ceil(localSize * sizeof(yType) / ONE_BLK_SIZE, MAX_BLOCK_NUM);
        uint16_t blockNum = Ceil(localSize * sizeof(yType) / ONE_BLK_SIZE, repeatTime);
        localSize = static_cast<uint32_t>(blockNum) * repeatTime * ONE_BLK_SIZE / sizeof(yType);
        InitConstValue(popBuffer, {repeatTime, blockNum, 0, 0U});

        TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE2_MTE3>();
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);

        uint64_t round = realClearSize / localSize;
        uint32_t tail = realClearSize % localSize;
        // set 2d block num 15bit, move_L1_to_out support 16 bit
        uint16_t blockLen = localSize * sizeof(yType) / ONE_BLK_SIZE;
        struct DataCopyParams dataCopyParams(1, blockLen, 0, 0);
        for (uint64_t idx = 0; idx < round; ++idx) {
            DataCopy(yGm_[offset], popBuffer, dataCopyParams);
            offset += localSize;
        }

        if (tail != 0) {
            // when out NCDHW, tail may not 32byte align, need ceil
            dataCopyParams.blockLen = (tail * sizeof(yType) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE;
            DataCopy(yGm_[offset], popBuffer, dataCopyParams);
        }

        SyncAllCores();
    }

    __aicore__ inline void ProcessWithL0()
    {
        uint32_t usedCoreNum = GetBlockNum();
        uint64_t clearSizePerCore = AlignUp(Ceil(outputSize_, usedCoreNum), BEST_FIXPIPE_ELEMENTS);
        usedCoreNum = Ceil(outputSize_, clearSizePerCore);
        if (GetBlockIdx() >= usedCoreNum) {
            SyncAllCores();
            return;
        }

        SetL0CZero();
        LocalTensor<float> l0c = l0C_.template DeQue<float>();
        uint64_t offset = clearSizePerCore * GetBlockIdx();
        uint64_t tail = outputSize_ - offset;
        tail = tail > clearSizePerCore ? clearSizePerCore : tail;
        uint64_t loop = 0;
        if (tail > L0C_ELEMENTS) {
            loop = tail / L0C_ELEMENTS;
            tail = tail - L0C_ELEMENTS * loop;
        } else if (tail == L0C_ELEMENTS) {
            loop = 1;
            tail = 0;
        }
        QuantMode_t quantMode = QuantMode_t::F322BF16;
        if constexpr(std::is_same<yType, half>::value) {
            quantMode = QuantMode_t::F322F16;
        } else if constexpr(std::is_same<yType, float>::value) {
            quantMode = QuantMode_t::NoQuant;
        }
        FixpipeParamsNZ nzParams;
        nzParams.nSize = BLOCK_CUBE;
        nzParams.mSize = L0C_M;
        nzParams.srcStride = 1;
        nzParams.dstStride = 1;
        nzParams.quantPre = quantMode;
        for (uint64_t i = 0; i < loop; ++i) {
            AscendC::Fixpipe<yType, float, CFG_NZ>(yGm_[offset], l0c, nzParams);
            offset += L0C_ELEMENTS;
        }
        if (tail != 0) {
            nzParams.mSize = tail >> CUBE_BLOCK_LEN_BITS;
            if (nzParams.mSize > 0) {
                AscendC::Fixpipe<yType, float, CFG_NZ>(yGm_[offset], l0c, nzParams);
            }
            uint32_t tailBlockSize = nzParams.mSize << CUBE_BLOCK_LEN_BITS;
            uint16_t remainNSize = tail - tailBlockSize;
            // tail大于16和tail对16取余不等于0 可以同时存在，也可单独存在
            if (remainNSize != 0) {
                offset += tailBlockSize;
                FixpipeParamsRowMajor rowMajorParams;
                rowMajorParams.nSize = remainNSize;
                rowMajorParams.mSize = 1;
                rowMajorParams.srcStride = 1;
                rowMajorParams.dstStride = 1;
                rowMajorParams.quantPre = quantMode;
                AscendC::Fixpipe<yType, float, CFG_ROW_MAJOR>(yGm_[offset], l0c, rowMajorParams);
            }
        }
        l0C_.FreeTensor(l0c);
        SyncAllCores();
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            return;
        }

        if constexpr (initOutputFlag == InitOutputFlag::L1_INIT) {
            ProcessWithL1();
        } else if constexpr (initOutputFlag == InitOutputFlag::L0_INIT) {
            ProcessWithL0();
        }
    }

    __aicore__ inline void SyncAllCores()
    {
#if __CCE_AICORE__ == 220
        if constexpr (initOutputFlag == InitOutputFlag::L1_INIT) {
            CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_AIC_FLAG);
        } else if constexpr (initOutputFlag == InitOutputFlag::L0_INIT) {
            CrossCoreSetFlag<0, PIPE_FIX>(SYNC_AIC_FLAG);
        }
        CrossCoreWaitFlag(SYNC_AIC_FLAG);
#endif
    }

    __aicore__ inline void Destroy()
    {
        pipe_.Destroy();
    }

protected:
    uint64_t outputSize_;
    TPipe pipe_;
    TBuf<TPosition::TSCM> localBuffer_;
    TBuf<TPosition::A2> l0A_;
    TBuf<TPosition::B2> l0B_;
    TQue<TPosition::CO1, 1> l0C_;
    GlobalTensor<yType> yGm_;

    __aicore__ inline void InitTilingData(const Conv3DBackpropInputV2TilingData *tilingData)
    {
        uint64_t mSize = static_cast<uint64_t>(tilingData->conv3DDxTiling.hi) * tilingData->conv3DDxTiling.wi;
        if constexpr (yFormat == FORMAT_NCDHW) {
            outputSize_ = mSize * tilingData->conv3DDxTiling.cin * tilingData->conv3DDxTiling.di *
                tilingData->conv3DDxTiling.batch;
        } else if constexpr (yFormat == FORMAT_NDC1HWC0) {
            if constexpr (std::is_same<yType, float>::value) {
                outputSize_ = mSize * tilingData->conv3DDxTiling.cin1 * B32_DATA_NUM_PER_BLOCK *
                    tilingData->conv3DDxTiling.di * tilingData->conv3DDxTiling.batch;
            } else {
                outputSize_ = mSize * tilingData->conv3DDxTiling.cin1 * B16_DATA_NUM_PER_BLOCK *
                    tilingData->conv3DDxTiling.di * tilingData->conv3DDxTiling.batch;
            }
        }
    }
};
}  // namespace AscendC

#endif  // CONV3D_BACKPROP_INPUT_V2_INIT_OUTPUT_H