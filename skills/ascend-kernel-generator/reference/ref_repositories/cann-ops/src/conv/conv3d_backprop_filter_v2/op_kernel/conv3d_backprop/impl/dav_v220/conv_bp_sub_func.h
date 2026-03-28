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
 * \file conv_bp_sub_func.h
 * \brief
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_H
#define CONV3D_BP_FILTER_SUB_FUNC_H

namespace ConvolutionBackpropFunc {
const int32_t MAX_BLOCK_COUNT = 4095;
const int32_t MIN_BLOCK_COUNT = 1;
const int32_t MAX_BLOCK_LEN = 65535;
const int32_t MAX_16BITS_STRIDE = 65535;
const int32_t MAX_L1W = 32767;

struct Out2L1ScalarParams {
    // to L1A
    bool isLoad2L1A{false};
    bool isLastMAL1{false};
    bool isFreeAL1{false};
    uint64_t out2A1SrcAddr{0};

    // to L1B
    bool isLoad2L1B{false};
    bool isFreeBL1{false};
    uint64_t out2B1SrcAddr{0};
    uint64_t singleShapeHi{0};
    uint32_t bL1cin1CopyLen{0};
};

template <class Intf>
static __aicore__ inline void CalOut2L1ScalarParams(Intf *self, Out2L1ScalarParams& params) {

    // to L1A
    if (params.isLoad2L1A) {
        params.out2A1SrcAddr = static_cast<uint64_t>(self->ctx.curML1Idx_) * self->ctx.tiling_->baseM * self->ctx.hwO_;
        params.isLastMAL1 =
            DivStepM((self->ctx.mIter_ - 1), self->ctx.tiling_->stepM) == DivStepM(self->ctx.curML0Idx_, self->ctx.tiling_->stepM);
    }

    // to L1B
    // 计算L1上cin起始idx, 去掉cin1HkWkCin里的HkWk，fp32场景下GM对齐到8
    if (params.isLoad2L1B) {
        uint64_t localN = ShiftDivChannelSize<Intf>(self->ctx.tiling_->baseN, self->ctx.tiling_->channelSize);
        uint64_t b1SrCin = DivHkWk(self->ctx.curNL1Idx_ * localN, self->ctx.hwK_);
        uint64_t singleShapeHi = self->ctx.singleShapeHo_ * self->ctx.tiling_->strideH + self->ctx.strideKernelDilationH;
        params.singleShapeHi = self->ctx.tiling_->hi > singleShapeHi ? singleShapeHi : self->ctx.tiling_->hi;
        uint64_t singleCoreHi = self->ctx.tiling_->singleCoreHo * self->ctx.tiling_->strideH + self->ctx.strideKernelDilationH;
        singleCoreHi = self->ctx.tiling_->hi > singleCoreHi ? singleCoreHi : self->ctx.tiling_->hi;
        if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) { // Transdata merge x input
            params.out2B1SrcAddr = b1SrCin * singleCoreHi * self->ctx.tiling_->wi;
        } else {
            params.out2B1SrcAddr = b1SrCin * self->ctx.hwI_;
        }
        uint32_t bL1N = self->ctx.curStepN_ * localN;
        uint32_t bL1cin1CopyLen = CeilHkWk(bL1N, self->ctx.hwK_);
        if (self->ctx.hwK_ > bL1N) {
            if (self->ctx.hwK_ % bL1N != 0) {
                ++bL1cin1CopyLen; // cin1搬一行就大于基本块大小, 多搬一行即可
            }
        } else if (RemainderOfHkWk(2 * bL1N, self->ctx.hwK_) != 0) {
            ++bL1cin1CopyLen; // 除非尾块正好0.5, 考虑拖尾都要再搬一行
        }
        uint64_t cin1RemainLen = ShiftDivChannelSize<Intf>(self->ctx.singleShapeCin_, self->ctx.tiling_->channelSize) - b1SrCin;
        params.bL1cin1CopyLen = cin1RemainLen > bL1cin1CopyLen ? bL1cin1CopyLen: cin1RemainLen;
    }
}

template <class Intf>
static __aicore__ inline void InitLoadToA2Params(Intf *self)
{
    self->ctx.dstL12L0aOffset_ = 0;
    self->ctx.srcL12L0aOffset_ = 0;
    // posK
    self->ctx.load3dA_.kStartPt = 0;
    // posM
    self->ctx.load3dA_.mStartPt = 0;
    self->ctx.load3dA_.strideW = 1;
    self->ctx.load3dA_.strideH = 1;
    self->ctx.load3dA_.filterW = 1;
    self->ctx.load3dA_.filterH = 1;
    self->ctx.load3dA_.dilationFilterW = 1;
    self->ctx.load3dA_.dilationFilterH = 1;
    self->ctx.load3dA_.filterSizeW = 0;
    self->ctx.load3dA_.filterSizeH = 0;
    self->ctx.load3dA_.enTranspose = 1;
    self->ctx.load3dA_.fMatrixCtrl = 0;
    self->ctx.load3dA_.channelSize = BLOCK_CUBE;
}

// 计算Load2B2的指令参数
template <class Intf>
static __aicore__ inline void InitLoadToB2Params(Intf *self) {
    // load3dStepK
    self->ctx.load3dB_.kExtension = self->ctx.tiling_->baseN;
    // load3dStepM
    self->ctx.load3dB_.mExtension = self->ctx.tiling_->baseM;
    // posK
    self->ctx.load3dB_.kStartPt = 0;
     // posM
    self->ctx.load3dB_.mStartPt = 0;
    self->ctx.load3dB_.strideW = self->ctx.tiling_->strideW;
    self->ctx.load3dB_.strideH = self->ctx.tiling_->strideH;
    self->ctx.load3dB_.filterW = self->ctx.tiling_->wk;
    self->ctx.load3dB_.filterH = self->ctx.tiling_->hk;
    self->ctx.load3dB_.dilationFilterW = self->ctx.tiling_->dilationW;
    self->ctx.load3dB_.dilationFilterH = self->ctx.tiling_->dilationH;
    self->ctx.load3dB_.filterSizeW = (self->ctx.tiling_->wk >> 8) & 255;
    self->ctx.load3dB_.filterSizeH = (self->ctx.tiling_->hk >> 8) & 255;
    self->ctx.load3dB_.enTranspose = 0;
    self->ctx.load3dB_.fMatrixCtrl = 1;
    self->ctx.load3dB_.channelSize = 16;
}

template <class Intf>
static __aicore__ inline void InitSetFmatrixParams(Intf *self)
{
    self->ctx.load3dA_.padList[0] = 0;
    self->ctx.load3dA_.padList[1] = 0;
    self->ctx.load3dA_.padList[2] = 0;
    self->ctx.load3dA_.padList[3] = 0;

    // W
    self->ctx.load3dB_.l1W = self->ctx.tiling_->wi;
    // H
    self->ctx.load3dB_.l1H = 1;
    self->ctx.load3dB_.padList[0] = self->ctx.tiling_->padLeft;
    self->ctx.load3dB_.padList[1] = self->ctx.tiling_->padRight;
    // padUp now is set in Load BL1
    self->ctx.load3dB_.padList[3] = self->ctx.tiling_->padDown;
}

template <class Intf>
static __aicore__ inline void CalcParamsMmad(Intf *self, uint64_t kPos)
{
    self->ctx.mmad_.m = self->ctx.baseUseM_;
    self->ctx.mmad_.n = self->ctx.baseUseN_;
}

template <class Intf>
static __aicore__ inline void InitMmadParams(Intf *self)
{
    self->ctx.dstL0cOffset_ = 0;
    self->ctx.srcL0aOffset_ = 0;
    self->ctx.srcL0bOffset_ = 0;
    self->ctx.mmad_.m = self->ctx.tiling_->baseM;
    self->ctx.mmad_.k = self->ctx.tiling_->baseK;
    self->ctx.mmad_.n = self->ctx.tiling_->baseN;
    self->ctx.mmad_.unitFlag = 0;
    if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
        self->ctx.mmad_.kDirectionAlign = 1;
    } else {
        self->ctx.mmad_.kDirectionAlign = 0;
    }
    self->ctx.mmad_.cmatrixSource = 0;
    self->ctx.mmad_.cmatrixInitVal = 0;
}

template <class Intf>
static __aicore__ inline void LoadL12L0a(Intf *self, const LocalTensor<typename Intf::SrcT> &l1AMatrix,
                                         uint32_t kPos, LocalTensor<typename Intf::SrcT> &l0a,
                                         const Out2L1ScalarParams& params, uint64_t kaStepIdx)
{
    uint32_t kl1 = self->ctx.kal1_;
    if (self->ctx.stepKaRound == (kaStepIdx + 1)) {
        kl1 = self->ctx.singleShapeHo_ * self->ctx.tiling_->wo - kaStepIdx * self->ctx.kal1_;
    }

    if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
        self->ctx.load3dA_.l1W = kl1;
        self->ctx.load3dA_.l1H = 1;

        // load3dStepK
        self->ctx.load3dA_.kExtension = Ceil(self->ctx.baseUseM_, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
        // load3dStepM
        self->ctx.load3dA_.mExtension = Ceil(self->ctx.baseUseK_, self->ctx.tiling_->k0) * self->ctx.tiling_->k0;
        self->ctx.load3dA_.channelSize = Ceil(self->ctx.baseUseM_, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;

        LoadData(l0a, l1AMatrix[self->ctx.srcL12L0aOffset_], self->ctx.load3dA_);
    } else {
        if (kl1 <= MAX_L1W) {
            self->ctx.load3dA_.l1W = kl1;
            self->ctx.load3dA_.l1H = 1;
        } else {
            uint64_t a1SrcKAlign = kaStepIdx * self->ctx.kal1_;
            uint32_t a1SrcHo = a1SrcKAlign / self->ctx.tiling_->wo;
            uint32_t hoCopyLen = CalRows2Copy(kl1, self->ctx.tiling_->wo);
            int32_t hoRemain = self->ctx.singleShapeHo_ - a1SrcHo;
            hoCopyLen = hoRemain > hoCopyLen ? hoCopyLen : hoRemain;
            self->ctx.load3dA_.l1W = self->ctx.tiling_->wo;
            self->ctx.load3dA_.l1H = hoCopyLen;
        }
        LoadData(l0a, l1AMatrix[self->ctx.srcL12L0aOffset_], self->ctx.load3dA_);
    }
}

template <class Intf>
static __aicore__ inline void MmadLocal(Intf *self, const LocalTensor<typename Intf::SrcT> &l0a,
                                        const LocalTensor<typename Intf::SrcT> &l0b,
                                        LocalTensor<typename Intf::L0cT> &l0c)
{
    MmadImpl(l0c[self->ctx.dstL0cOffset_],
        l0a[self->ctx.srcL0aOffset_],
        l0b[self->ctx.srcL0bOffset_],
        self->ctx.mmad_);
    // MMAD计算量baseM*baseN小于一定阈值时需要添加PIPE_M同步,当前平台阈值为10*256
    if (self->ctx.mmad_.m * self->ctx.mmad_.n < 2560) {
        PipeBarrier<PIPE_M>();
    }
}

template <class Intf, class src0_T>
__aicore__ inline void LoadToA1(Intf *self, bool cachePosA1, uint64_t kaIdx, const Out2L1ScalarParams& params,
                                bool isLoadA1, uint64_t kaStepIdx)
{
    if (!isLoadA1) {
        return;
    }
    if (params.isLoad2L1A) {
        LocalTensor<typename Intf::SrcT> useA1Buf;
        if (cachePosA1) {
            useA1Buf = self->ctx.a1Ping_.template AllocTensor<typename Intf::SrcT>();
        } else {
            useA1Buf = self->ctx.a1Pong_.template AllocTensor<typename Intf::SrcT>();
        }
        uint64_t out2A1SrcAddrOffset =
            params.out2A1SrcAddr + kaStepIdx * self->ctx.kal1_ * self->ctx.tiling_->channelSize;

        DataCopyParams dataCopyParams;
        dataCopyParams.dstStride = 0;
        if (self->ctx.stepKaRound == (kaStepIdx + 1)) {
            // 最后一块kAL1，考虑tailK, 32表示32Byte
            dataCopyParams.blockLen =
                (self->ctx.singleShapeHo_ * self->ctx.tiling_->wo - kaIdx * self->ctx.tiling_->baseK);
        } else {
            dataCopyParams.blockLen = self->ctx.kal1_;
        }
        self->ctx.curLoadKal1_ = dataCopyParams.blockLen;

        uint32_t blockCount = 0;
        if (params.isLastMAL1) {
            // 最后一块mAL1，需要考虑tailM
            blockCount = ShiftDivChannelSize<Intf>(((self->ctx.curStepM_ - 1) * self->ctx.tiling_->baseM + self->ctx.tailM_),
                                        self->ctx.tiling_->channelSize);
        } else {
            blockCount = ShiftDivChannelSize<Intf>(self->ctx.curStepM_ * self->ctx.tiling_->baseM, self->ctx.tiling_->channelSize);
        }

        if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
            if (blockCount & 0x1) {
                InitConstValue(
                    useA1Buf[blockCount * dataCopyParams.blockLen * ONE_BLK_SIZE / sizeof(float)],
                    {1, dataCopyParams.blockLen, 0, 0U});
            }
        }

        // blockcout和blockLen关联L1, 溢出风险低
        uint64_t srcStride = self->ctx.hwO_ - dataCopyParams.blockLen;
        if (srcStride <= MAX_16BITS_STRIDE) {
            dataCopyParams.srcStride = srcStride;
            dataCopyParams.blockCount = blockCount;
            DataCopy(useA1Buf, self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], dataCopyParams);
        } else {
            dataCopyParams.srcStride = 0;
            dataCopyParams.blockCount = 1;
            uint64_t srcOffset = out2A1SrcAddrOffset;
            uint64_t dstOffset = 0;
            for (uint32_t idx = 0; idx < blockCount; ++idx) {
                DataCopy(useA1Buf[dstOffset], self->ctx.outBackPropGlobal_[srcOffset], dataCopyParams);
                srcOffset += self->ctx.hwO_ * self->ctx.tiling_->channelSize;
                dstOffset += dataCopyParams.blockLen * self->ctx.tiling_->channelSize;
            }
        }

        if (cachePosA1) {
            self->ctx.a1Ping_.EnQue(useA1Buf);
        } else {
            self->ctx.a1Pong_.EnQue(useA1Buf);
        }
    }
}

template <class Intf, class src1_T>
__aicore__ inline void LoadToB1(Intf *self, bool cachePosB1, uint64_t kbIdx, const Out2L1ScalarParams& params,
                                bool isLoadB1, uint64_t kbStepIdx)
{
    if (!isLoadB1) {
        return;
    }
    // 需要载入BL1的条件为，被计算的BL0块是BL1上的第一块数据，一次载入完整BL1大小
    // 此时满足以下条件之一需要载入BL1：
    // 1.BL1上无db，并且K方向需要多于一个buffer，每次都需要载入；BL1开db，并且K方向buffer数量小于等于2
    // 2.singleShapeK / stepKb > 2, 优先循环k方向，BL1上数据无法复用
    // 3.order_M时，L1上驻留AL1, BL1数据不复用
    // 4.order_N时，BL1驻留在L1上，且K <=
    // 2，即L1上可以栽下全部Kb，此时遍历M方向，BL1数据上数据不会被覆盖，只在M方向循环第一次时载入BL1
    if (params.isLoad2L1B) {
        LocalTensor<typename Intf::SrcT> useB1Buf;
        if (cachePosB1) {
            useB1Buf = self->ctx.b1Ping_.template AllocTensor<typename Intf::SrcT>();
        } else {
            useB1Buf = self->ctx.b1Pong_.template AllocTensor<typename Intf::SrcT>();
        }

        // L0shape到orgShape的对应关系，L0和L1是16对齐的，orgShape是Wi对齐的,先算Wo对齐再算Wi对齐
        // 先算L0B所在BL1块的起始地址，16对齐的
        uint64_t b1SrcKAlign = kbStepIdx * self->ctx.kbl1_;
        // load3d必须有完整Wo，做Wo对齐，计算起始地址所在的Ho
        uint32_t b1SrcHo = b1SrcKAlign / self->ctx.tiling_->wo;
        uint32_t b1SrcHoGm = b1SrcHo + self->ctx.hoStartIdx_;
        // 计算Ho对应的Hi，根据卷积原理
        int64_t b1SrcHiGm = static_cast<uint64_t>(b1SrcHoGm) * self->ctx.tiling_->strideH - self->ctx.tiling_->padUp;
        uint32_t b1SrcHi = 0;
        if (b1SrcHiGm > 0 && self->ctx.hiStartIdx_ > 0) {
            b1SrcHi = b1SrcHiGm - self->ctx.hiStartIdx_;
        } else if (b1SrcHiGm > 0) {
            b1SrcHi = b1SrcHiGm;
        }

        uint32_t kbl1 = self->ctx.kbl1_;
        if (self->ctx.stepKbRound == (kbStepIdx + 1)) {
            kbl1 = self->ctx.singleShapeHo_ * self->ctx.tiling_->wo - b1SrcKAlign;
        }
        uint32_t ho = CalRows2Copy(kbl1, self->ctx.tiling_->wo);
        uint32_t hiCopyLen = ho * self->ctx.tiling_->strideH + self->ctx.strideKernelDilationH;

        uint32_t padUp = 0;
        if (b1SrcHiGm < 0) {
            hiCopyLen = hiCopyLen + b1SrcHiGm;
            padUp = -b1SrcHiGm;
        }
        if (b1SrcHiGm + hiCopyLen > self->ctx.tiling_->hi) {
            hiCopyLen = self->ctx.tiling_->hi - b1SrcHiGm;
        }

        uint32_t hiRemainLen = params.singleShapeHi - b1SrcHi;
        hiCopyLen = hiRemainLen > hiCopyLen ? hiCopyLen: hiRemainLen;

        DataCopyParams dataCopyParams;
        dataCopyParams.dstStride = 0;

        uint64_t blockLen = static_cast<uint64_t>(hiCopyLen) * self->ctx.tiling_->wi;
        uint64_t srcStride = self->ctx.hwI_ - blockLen;
        uint64_t singleCoreHi = self->ctx.tiling_->singleCoreHo * self->ctx.tiling_->strideH + self->ctx.strideKernelDilationH;
        singleCoreHi = self->ctx.tiling_->hi > singleCoreHi ? singleCoreHi : self->ctx.tiling_->hi;
        if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) { // Transdata merge x input
            srcStride = singleCoreHi * self->ctx.tiling_->wi - blockLen;
        }

        // 得到gm的偏移量
        uint64_t srcOffset = (params.out2B1SrcAddr + static_cast<uint64_t>(b1SrcHi) * self->ctx.tiling_->wi) * self->ctx.tiling_->channelSize;
        uint64_t dstOffset = 0;
        if (blockLen <= MAX_BLOCK_LEN && srcStride <= MAX_16BITS_STRIDE) {
            dataCopyParams.srcStride = srcStride;
            dataCopyParams.blockLen = blockLen;
            dataCopyParams.blockCount = MAX_BLOCK_COUNT;

            uint32_t loop = params.bL1cin1CopyLen / MAX_BLOCK_COUNT;
            for (uint32_t idx = 0; idx < loop; ++idx) {
                DataCopy(useB1Buf[dstOffset], self->ctx.fmapGlobal_[srcOffset], dataCopyParams);
                if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) { // Transdata merge x input
                    srcOffset += MAX_BLOCK_COUNT * singleCoreHi * self->ctx.tiling_->wi * self->ctx.tiling_->channelSize;
                } else {
                    srcOffset += MAX_BLOCK_COUNT * self->ctx.hwI_ * self->ctx.tiling_->channelSize;
                }
                dstOffset += MAX_BLOCK_COUNT * blockLen * self->ctx.tiling_->channelSize;
            }

            dataCopyParams.blockCount = params.bL1cin1CopyLen - loop * MAX_BLOCK_COUNT;
            if (dataCopyParams.blockCount > 0) {
                DataCopy(useB1Buf[dstOffset], self->ctx.fmapGlobal_[srcOffset], dataCopyParams);
            }
        } else {
            dataCopyParams.srcStride = 0;
            dataCopyParams.blockCount = hiCopyLen;
            dataCopyParams.blockLen = self->ctx.tiling_->wi;
            for (uint32_t idx = 0; idx < params.bL1cin1CopyLen; ++idx) {
                DataCopy(useB1Buf[dstOffset], self->ctx.fmapGlobal_[srcOffset], dataCopyParams);
                if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) { // Transdata merge x input
                    srcOffset += singleCoreHi * self->ctx.tiling_->wi * self->ctx.tiling_->channelSize;
                } else {
                    srcOffset += self->ctx.hwI_ * self->ctx.tiling_->channelSize;
                }
                dstOffset += blockLen * self->ctx.tiling_->channelSize;
            }
        }

        if (cachePosB1) {
            self->ctx.bL1HiCopyLenPing = hiCopyLen;
            self->ctx.bL1PadUpPing = padUp;
            self->ctx.b1Ping_.EnQue(useB1Buf);
        } else {
            self->ctx.bL1HiCopyLenPong = hiCopyLen;
            self->ctx.bL1PadUpPong = padUp;
            self->ctx.b1Pong_.EnQue(useB1Buf);
        }
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2Gm(
    Intf *self, const GlobalTensor<typename Intf::DstT> &output, uint8_t enAtomic = 0, bool enSequentialWrite = false)
{
    LocalTensor<typename Intf::L0cT> l0c;
    if (self->ctx.l0cPingPongFlag_) {
        l0c = self->ctx.l0cPing_.template DeQue<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.l0cPong_.template DeQue<typename Intf::L0cT>();
    }

    if (enAtomic == 1) {
        SetAtomicAdd<typename Intf::DstT>();
    }
    if constexpr (Intf::Config::dType::format == ConvolutionBackprop::CubeFormat::FRACTAL_Z_3D) {
        if (!enSequentialWrite) {
            uint64_t alignCoutG = static_cast<uint64_t>(self->ctx.tiling_->cout1G) * self->ctx.tiling_->channelSize;
            uint64_t dstOffset =
                static_cast<uint64_t>(self->ctx.curNL0Idx_) * self->ctx.tiling_->baseN * alignCoutG +
                static_cast<uint64_t>(self->ctx.curML0Idx_) * self->ctx.tiling_->baseM * self->ctx.tiling_->channelSize;
            // bf16 c0_size is 16 * sizeof(float) , fp32 enable channel split, c0_size is 8 * sizeof(float)
            uint64_t dstStrideIn =
                alignCoutG * self->ctx.tiling_->channelSize * sizeof(typename Intf::DstT) / ONE_BLK_SIZE;
            FixpipeParamsV220 fixpipeParams(
                static_cast<uint16_t>(self->ctx.baseUseN_), static_cast<uint16_t>(self->ctx.baseUseM_),
                ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0) * self->ctx.tiling_->m0, dstStrideIn, 0);
            if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
                fixpipeParams.isChannelSplit = true;
            }
            Fixpipe<typename Intf::DstT, typename Intf::L0cT, CFG_NZ>(output[dstOffset], l0c, fixpipeParams);
        } else {
            uint64_t dstStrideIn =
                self->ctx.tiling_->baseM * self->ctx.tiling_->channelSize * sizeof(typename Intf::DstT) / ONE_BLK_SIZE;
            FixpipeParamsV220 fixpipeParams(
                static_cast<uint16_t>(self->ctx.baseUseN_), static_cast<uint16_t>(self->ctx.baseUseM_),
                ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0) * self->ctx.tiling_->m0, dstStrideIn, 0);
            if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
                fixpipeParams.isChannelSplit = true;
            }
            Fixpipe<typename Intf::DstT, typename Intf::L0cT, CFG_NZ>(output, l0c, fixpipeParams);
        }
    }
    if (enAtomic == 1) {
        SetAtomicNone();
    }
    if (self->ctx.l0cPingPongFlag_) {
        self->ctx.l0cPing_.FreeTensor(l0c);
    } else {
        self->ctx.l0cPong_.FreeTensor(l0c);
    }
    if (self->ctx.tiling_->cl0Pbuffer > 1) {
        self->ctx.l0cPingPongFlag_ = !self->ctx.l0cPingPongFlag_;
    }
}
}  // namespace ConvolutionBackpropFunc

#endif