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
 * \file conv3d_common_func.h
 * \brief
 */

#ifndef CONV3D_COMMON_FUNC_H
#define CONV3D_COMMON_FUNC_H

#include "conv3d_common_init_func.h"
#include "conv3d_common_set_func.h"

namespace Conv3dFunc {

CONV_DECLARE_REG_IMPL(Init);
CONV_DECLARE_REG_IMPL(SetOrgFmapShape);
CONV_DECLARE_REG_IMPL(SetOrgWeightShape);
CONV_DECLARE_REG_IMPL(SetOrgOutputShape);
CONV_DECLARE_REG_IMPL(SetSingleFmapShape);
CONV_DECLARE_REG_IMPL(SetSingleOutputShape);
CONV_DECLARE_REG_IMPL(SetFmapStartPosition);
CONV_DECLARE_REG_IMPL(BeginSetCrossFlag);
CONV_DECLARE_REG_IMPL(EndWaitCrossFlag);
CONV_DECLARE_REG_IMPL(SetWorkspace);
CONV_DECLARE_REG_IMPL(SetVecScale);
CONV_DECLARE_REG_IMPL(SetSubBlockIdx);
CONV_DECLARE_REG_IMPL(LoadTotalCoreChannel);
CONV_DECLARE_REG_IMPL(FreeTotalCoreChannel);
CONV_DECLARE_REG_IMPL(SetGroupOptInfo);
CONV_DECLARE_REG_IMPL(Iterate);
CONV_DECLARE_REG_IMPL(GetTensorC);
CONV_DECLARE_REG_IMPL(IterateAll);
CONV_DECLARE_REG_IMPL(VecCompute);

using TypeFalse = const struct {
    __uint128_t _[1024];
};

template <class Intf, uint32_t ImplType>
struct Iterate {
    template <bool sync = true>
    static __aicore__ inline bool call(Intf *self, bool enPartialSum = false)
    {
        return IterateImpl(self, enPartialSum);
    }

    static __aicore__ inline uint64_t CalcL0CurrentN(Intf *self)
    {
        uint64_t n = (self->ctx.nBL1Iter == self->ctx.maxNBL1Iter && self->ctx.nBL0Iter == self->ctx.maxNL0Iter)
                         ? self->ctx.nL0Tail
                         : self->ctx.conv3dTiling->nL0;
        return n;
    }

    static __aicore__ inline uint64_t CalcL0CurrentM(Intf *self)
    {
        uint64_t m = (self->ctx.mAL1Iter == self->ctx.maxMAL1Iter && self->ctx.mAL0Iter == self->ctx.maxML0Iter)
                         ? self->ctx.mAL0Tail
                         : self->ctx.conv3dTiling->mL0;
        return m;
    }

    static __aicore__ void inline FirstIterateImpl(Intf *self)
    {
        // 先更新index再load，就需要加第一次处理。
        self->ctx.nBL0Iter = 0;
        self->ctx.mAL0Iter = 0;
        self->ctx.mAL1Iter = 0;
        self->ctx.nBL1Iter = 0;
        self->ctx.dOutIter = 0;
        self->ctx.loadAL1Flag = true;
        self->ctx.loadBL1Flag = !Intf::bl1bypass;
        self->ctx.loadAL0Flag = true;
        self->ctx.loadBL0Flag = true;
        self->ctx.isFirstIterate = false;
        if constexpr (Intf::formatType != conv::ConvFormat::NCDHW) {
            self->ctx.loadAl1Ins.SetLoadData3DParams();
            if (self->ctx.conv3dTiling->mL0 % self->ctx.orgWo == 0) {
                self->ctx.mL0IsDivisibleByWo = true;
            }
        }
        if constexpr (Intf::outputOrder) {
            self->ctx.hoL1Iter = 0;
        }
    }

    static __aicore__ bool inline IterateMFirst(Intf *self)
    {
        // ReorderN: 先往M轴方向偏移再往N轴方向偏移。Fmap复用Weight。
        //    M
        //    |
        //    |
        //    |----------N-------->
        // ==================L0========================
        self->ctx.mAL0Iter++;
        if (self->ctx.mAL0Iter == self->ctx.l12l0LoopM) {
            self->ctx.mAL0Iter = 0;
            self->ctx.nBL0Iter++;
        }
        if (self->ctx.nBL0Iter == self->ctx.l12l0LoopN) {
            self->ctx.nBL0Iter = 0;
            self->ctx.mAL1Iter++;
            self->ctx.loadAL1Flag = true;
        }
        if constexpr (Intf::outputOrder) {
            if (self->ctx.mAL1Iter == self->ctx.ddr2l1LoopM) {
                self->ctx.mAL1Iter = 0;
                self->ctx.hoL1Iter++;
            }
            if (self->ctx.hoL1Iter == self->ctx.ddr2l1LoopHo) {
                self->ctx.hoL1Iter = 0;
                self->ctx.dOutIter++;
            }
        } else {
            if (self->ctx.mAL1Iter == self->ctx.ddr2l1LoopM) {
                self->ctx.mAL1Iter = 0;
                self->ctx.dOutIter++;
            }
        }
        if (self->ctx.dOutIter == self->ctx.ddr2l1LoopD) {
            self->ctx.dOutIter = 0;
            self->ctx.nBL1Iter++;
            self->ctx.loadBL1Flag = true;
        }
        if (self->ctx.nBL1Iter == self->ctx.ddr2l1LoopN) {
            return false;
        }
        return true;
    }

    static __aicore__ bool inline IterateNFirst(Intf *self)
    {
        // ReorderM: 先往N轴方向偏移再往M轴方向偏移。Weight复用Fmap。
        //    ----------N-------->
        //                       |
        //                       |
        //                       M
        //                       |
        //                       |
        // ==================L0========================
        self->ctx.nBL0Iter++;
        if (self->ctx.nBL0Iter == self->ctx.l12l0LoopN) {
            self->ctx.nBL0Iter = 0;
            self->ctx.mAL0Iter++;
        }
        if (self->ctx.mAL0Iter == self->ctx.l12l0LoopM) {
            self->ctx.mAL0Iter = 0;
            self->ctx.nBL1Iter++;
            self->ctx.loadBL1Flag = true;
        }
        if (self->ctx.nBL1Iter == self->ctx.ddr2l1LoopN) {
            self->ctx.nBL1Iter = 0;
            self->ctx.mAL1Iter++;
            self->ctx.loadAL1Flag = true;
        }
        if constexpr (Intf::outputOrder) {
            if (self->ctx.mAL1Iter == self->ctx.ddr2l1LoopM) {
                self->ctx.mAL1Iter = 0;
                self->ctx.hoL1Iter++;
            }
            if (self->ctx.hoL1Iter == self->ctx.ddr2l1LoopHo) {
                self->ctx.hoL1Iter = 0;
                self->ctx.dOutIter++;
            }
        } else {
            if (self->ctx.mAL1Iter == self->ctx.ddr2l1LoopM) {
                self->ctx.mAL1Iter = 0;
                self->ctx.dOutIter++;
            }
        }
        if (self->ctx.dOutIter == self->ctx.ddr2l1LoopD) {
            return false;
        }
        return true;
    }

    static __aicore__ void inline ReduceKNoPingPongBL1ByPass(Intf *self) {
        wait_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID0);
        self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
        self->ctx.loadAL0Ins.LoadAL0();
        set_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID0);

        wait_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID0);
        self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
        self->ctx.loadBL0Ins.LoadBL0();
        set_flag(PIPE_MTE2, PIPE_M, event_t::EVENT_ID1);

        wait_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_M, event_t::EVENT_ID1);

        self->ctx.madIns.Mad();
        set_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID0);
    }

    static __aicore__ void inline ReduceKNoPingPongBL1NoByPass(Intf *self) {
        wait_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID0);
        self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
        self->ctx.loadAL0Ins.LoadAL0();

        self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
        self->ctx.loadBL0Ins.LoadBL0();
        set_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID0);

        self->ctx.madIns.Mad();
        set_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID0);
    }

    static __aicore__ void inline ReduceKL0APingPongBL1ByPass(Intf *self, const uint16_t& l0aFlag) {
        wait_flag(PIPE_M, PIPE_MTE1, l0aFlag);
        self->ctx.al0 = l0aFlag ? self->ctx.al0Pong : self->ctx.al0Ping;
        self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
        self->ctx.loadAL0Ins.LoadAL0();
        set_flag(PIPE_MTE1, PIPE_M, l0aFlag);

        wait_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID2);
        if (self->ctx.loadBL0Flag) {
            self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
            self->ctx.loadBL0Ins.LoadBL0();
            set_flag(PIPE_MTE2, PIPE_M, event_t::EVENT_ID2);
        }

        wait_flag(PIPE_MTE1, PIPE_M, l0aFlag);
        if (self->ctx.loadBL0Flag) {
            wait_flag(PIPE_MTE2, PIPE_M, event_t::EVENT_ID2);
        }
        self->ctx.madIns.Mad();
        set_flag(PIPE_M, PIPE_MTE1, l0aFlag);
        set_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID2);
    }

    static __aicore__ void inline ReduceKL0APingPongBL1NoByPass(Intf *self, const uint16_t& l0aFlag) {
        wait_flag(PIPE_M, PIPE_MTE1, l0aFlag);
        self->ctx.al0 = l0aFlag ? self->ctx.al0Pong : self->ctx.al0Ping;
        self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
        self->ctx.loadAL0Ins.LoadAL0();

        if (self->ctx.loadBL0Flag) {
            self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
            self->ctx.loadBL0Ins.LoadBL0();
        }
        set_flag(PIPE_MTE1, PIPE_M, l0aFlag);
        wait_flag(PIPE_MTE1, PIPE_M, l0aFlag);
        self->ctx.madIns.Mad();
        set_flag(PIPE_M, PIPE_MTE1, l0aFlag);
    }

    static __aicore__ void inline ReduceKL0BPingPongBL1ByPass(Intf *self, const uint16_t& l0bFlag) {
        wait_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID2);
        if (self->ctx.loadAL0Flag) {
            self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
            self->ctx.loadAL0Ins.LoadAL0();
            set_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID2);
        }

        wait_flag(PIPE_M, PIPE_MTE2, l0bFlag);
        self->ctx.bl0 = l0bFlag ? self->ctx.bl0Pong : self->ctx.bl0Ping;
        self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
        self->ctx.loadBL0Ins.LoadBL0();
        set_flag(PIPE_MTE2, PIPE_M, l0bFlag);

        if (self->ctx.loadAL0Flag) {
            wait_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID2);
        }
        wait_flag(PIPE_MTE2, PIPE_M, l0bFlag);

        self->ctx.madIns.Mad();
        set_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE2, l0bFlag);
    }

    static __aicore__ void inline ReduceKL0BPingPongBL1NoByPass(Intf *self, const uint16_t& l0bFlag) {
        wait_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID2);
        if (self->ctx.loadAL0Flag) {
            self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;

            self->ctx.loadAL0Ins.LoadAL0();
            set_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID2);
        }

        wait_flag(PIPE_M, PIPE_MTE1, l0bFlag);
        self->ctx.bl0 = l0bFlag ? self->ctx.bl0Pong : self->ctx.bl0Ping;
        self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
        self->ctx.loadBL0Ins.LoadBL0();
        set_flag(PIPE_MTE1, PIPE_M, l0bFlag);

        if (self->ctx.loadAL0Flag) {
            wait_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID2);
        }
        wait_flag(PIPE_MTE1, PIPE_M, l0bFlag);

        self->ctx.madIns.Mad();
        set_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, l0bFlag);
    }

    static __aicore__ void inline ReduceKL0AL0BPingPong(Intf *self, const uint16_t& l0abFlag) {
        if (l0abFlag) {
            self->ctx.al0 = self->ctx.al0Pong;
            self->ctx.bl0 = self->ctx.bl0Pong;
        } else {
            self->ctx.al0 = self->ctx.al0Ping;
            self->ctx.bl0 = self->ctx.bl0Ping;
        }

        wait_flag(PIPE_M, PIPE_MTE1, l0abFlag);

        self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
        self->ctx.loadAL0Ins.LoadAL0();

        if constexpr (Intf::bl1bypass) {
            wait_flag(PIPE_M, PIPE_MTE2, l0abFlag);
        }
        self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
        self->ctx.loadBL0Ins.LoadBL0();
        set_flag(PIPE_MTE1, PIPE_M, l0abFlag);
        if constexpr (Intf::bl1bypass) {
            set_flag(PIPE_MTE2, PIPE_M, l0abFlag);
        }

        wait_flag(PIPE_MTE1, PIPE_M, l0abFlag);
        if constexpr (Intf::bl1bypass) {
            wait_flag(PIPE_MTE2, PIPE_M, l0abFlag);
        }

        PipeBarrier<PIPE_M>();
        self->ctx.madIns.Mad();
        set_flag(PIPE_M, PIPE_MTE1, l0abFlag);
        if constexpr (Intf::bl1bypass) {
            set_flag(PIPE_M, PIPE_MTE2, l0abFlag);
        }
    }

    static __aicore__ void inline LoadAL1Process(Intf *self, uint64_t kAL1Iter)
    {
        self->ctx.al1 = self->ctx.queueAL1.template AllocTensor<typename Intf::FmapT>();
        self->ctx.kAL1Iter = kAL1Iter;
        self->ctx.loadAl1Ins.LoadAL1();
        self->ctx.queueAL1.EnQue(self->ctx.al1);
        self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
        self->ctx.loadAL1Flag = false;  // LoopK中只有K方向可能重新载入。
        self->ctx.freeAL1TensorFlag = true;
    }

    static __aicore__ void inline LoadBL1Process(Intf *self, uint64_t kBL1Iter)
    {
        self->ctx.bl1 = self->ctx.queueBL1.template AllocTensor<typename Intf::WeightT>();
        self->ctx.kBL1Iter = kBL1Iter;
        self->ctx.loadBL1Ins.LoadBL1();
        self->ctx.queueBL1.EnQue(self->ctx.bl1);
        self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
        self->ctx.loadBL1Flag = false;  // LoopK中只有K方向可能重新载入。
        self->ctx.freeBL1TensorFlag = true;
    }

    static __aicore__ void inline LoadAL1PreloadProcess(Intf *self, uint64_t kAL1Iter)
    {
        self->ctx.al1 = self->ctx.queueAL1.template AllocTensor<typename Intf::FmapT>();
        self->ctx.kAL1Iter = kAL1Iter;
        self->ctx.loadAl1Ins.LoadAL1();
        self->ctx.queueAL1.EnQue(self->ctx.al1);
        self->ctx.loadAL1Flag = true;
        self->ctx.freeAL1TensorFlag = false;
    }

    static __aicore__ void inline LoadBL1PreloadProcess(Intf *self, uint64_t kBL1Iter)
    {
        self->ctx.bl1 = self->ctx.queueBL1.template AllocTensor<typename Intf::WeightT>();
        self->ctx.kBL1Iter = kBL1Iter;
        self->ctx.loadBL1Ins.LoadBL1();
        self->ctx.queueBL1.EnQue(self->ctx.bl1);
        self->ctx.loadBL1Flag = true;
        self->ctx.freeBL1TensorFlag = false;
    }

    // K方向迭代首次(iter==0)加载L0A, L0B
    static __aicore__ void inline ReduceKFirstIterLoadL0(Intf *self)
    {
        if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::ALL_CLOSE)) {
            // for L0PingPong::ALL_CLOSE, BL1ByPass is always ON
            self->ctx.al0 = self->ctx.al0Ping;
            self->ctx.bl0 = self->ctx.bl0Ping;
            if constexpr (Intf::bl1bypass) {
                set_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID0);
                ReduceKNoPingPongBL1ByPass(self);
            } else {
                ReduceKNoPingPongBL1NoByPass(self);
            }
        } else if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0A_OPEN)) {
            // for L0PingPong::L0A_OPEN, BL1ByPass is always ON
            self->ctx.bl0 = self->ctx.bl0Ping;
            if constexpr (Intf::bl1bypass) {
                set_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID2);
                ReduceKL0APingPongBL1ByPass(self, event_t::EVENT_ID0);
            } else {
                ReduceKL0APingPongBL1NoByPass(self, event_t::EVENT_ID0);
            }
        } else if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0B_OPEN)) {
            self->ctx.al0 = self->ctx.al0Ping;
            if constexpr (Intf::bl1bypass) {
                set_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID0);
                if (self->ctx.ddr2l1LoopD > 1) {
                    set_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID1);
                }
                ReduceKL0BPingPongBL1ByPass(self, event_t::EVENT_ID0);
            } else {
                ReduceKL0BPingPongBL1NoByPass(self, event_t::EVENT_ID0);
            }
        } else if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::ALL_OPEN)) {
            if constexpr (Intf::bl1bypass) {
                set_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID0);
                set_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID1);
                if (self->ctx.ddr2l1LoopD > 1) {
                    set_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID4);
                }
            }
            ReduceKL0AL0BPingPong(self, event_t::EVENT_ID0);
        }
    }

    // K方向迭代(iter>0)加载L0A, L0B
    static __aicore__ void inline ReduceKIterLoadL0(Intf *self, const uint16_t& isOdd)
    {
        if constexpr (Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::ALL_CLOSE)) {
            if constexpr (Intf::bl1bypass) {
                ReduceKNoPingPongBL1ByPass(self);
            } else {
                ReduceKNoPingPongBL1NoByPass(self);
            }
        } else if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0A_OPEN)) {
            if constexpr (Intf::bl1bypass) {
                ReduceKL0APingPongBL1ByPass(self, isOdd);
            } else {
                ReduceKL0APingPongBL1NoByPass(self, isOdd);
            }
        } else if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0B_OPEN)) {
            if constexpr (Intf::bl1bypass) {
                ReduceKL0BPingPongBL1ByPass(self, isOdd);
            } else {
                ReduceKL0BPingPongBL1NoByPass(self, isOdd);
            }
        } else if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::ALL_OPEN)) {
            ReduceKL0AL0BPingPong(self, isOdd);
        }
    }

    static __aicore__ void inline ReduceKIterLoadL1(Intf *self)
    {
        if (self->ctx.loadAL1Flag || (!self->ctx.kAL1fullload && self->ctx.kIter % self->ctx.multiKAL1 == 0)) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            self->ctx.freeAL1TensorFlag = false;
            LoadAL1Process(self, self->ctx.kIter / self->ctx.multiKAL1);
        }
        if constexpr (!Intf::bl1bypass) {
            if (self->ctx.loadBL1Flag || (!self->ctx.kBL1fullload && self->ctx.kIter % self->ctx.multiKBL1 == 0)) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
                self->ctx.freeBL1TensorFlag = false;
                LoadBL1Process(self, self->ctx.kIter / self->ctx.multiKBL1);
            }
        }
    }

    // K方向迭代的后处理, 当前只对bl1bypass需要添加wait_flag
    static __aicore__ void inline ReduceKPostProcessLoadL0(Intf *self)
    {
        if constexpr((Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::ALL_CLOSE)) && Intf::bl1bypass) {
            wait_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID0);
        } else if constexpr((Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0A_OPEN)) && Intf::bl1bypass) {
            wait_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID2);
        } else if constexpr((Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0B_OPEN)) && Intf::bl1bypass) {
            wait_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID0);
            if (self->ctx.ddr2l1LoopD > 1) {
                wait_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID1);
            }
        } else if constexpr((Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::ALL_OPEN)) && Intf::bl1bypass) {
            wait_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID1);
            if (self->ctx.ddr2l1LoopD > 1) {
                wait_flag(PIPE_M, PIPE_MTE2, event_t::EVENT_ID4);
            }
        }
    }

    static __aicore__ void inline ReduceK(Intf *self)
    {
        ASC_OP_LOGD("no preload in ReduceK: loadAl1Flag: %d, kAL1fullload: %d, freeAL1TensorFlag: %d\n",
                    self->ctx.loadAL1Flag, self->ctx.kAL1fullload, self->ctx.freeAL1TensorFlag);

        if (self->ctx.loadAL1Flag || !(self->ctx.kAL1fullload)) {
            if (self->ctx.freeAL1TensorFlag) {
                self->ctx.queueAL1.FreeTensor(self->ctx.al1);
                self->ctx.freeAL1TensorFlag = false;
            }
            LoadAL1Process(self, 0);
        }
        if constexpr (!Intf::bl1bypass) {
            if (self->ctx.loadBL1Flag || !(self->ctx.kBL1fullload)) {
                if (self->ctx.freeBL1TensorFlag) {
                    self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
                    self->ctx.freeBL1TensorFlag = false;
                }
                LoadBL1Process(self, 0);
            }
        }
        ReduceKFirstIterLoadL0(self);

        self->ctx.kIter = 1;
        uint16_t isOdd = 1;
        while (self->ctx.kIter < self->ctx.ddr2l0LoopK) {
            ReduceKIterLoadL1(self);
            ReduceKIterLoadL0(self, isOdd);
            self->ctx.kIter++;
            isOdd = self->ctx.kIter & 0x1;
        }

        ReduceKPostProcessLoadL0(self);
    }

    static __aicore__ void inline ReduceKPreloadDbAllLoadL1(Intf *self, const uint64_t& maxKAL1PreloadIter, const uint64_t& maxKBL1PreloadIter)
    {
        if (self->ctx.kIter == maxKAL1PreloadIter) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
        } else if (self->ctx.kIter < maxKAL1PreloadIter &&
            (self->ctx.loadAL1Flag || (!self->ctx.kAL1fullload && self->ctx.kIter % self->ctx.multiKAL1 == 0))) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            LoadAL1Process(self, (self->ctx.kIter / self->ctx.multiKAL1) + 1);
        }

        if (self->ctx.kIter == maxKBL1PreloadIter) {
            self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
        } else if (self->ctx.kIter < maxKBL1PreloadIter &&
            (self->ctx.loadBL1Flag || (!self->ctx.kBL1fullload && self->ctx.kIter % self->ctx.multiKBL1 == 0))) {
            self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            LoadBL1Process(self, (self->ctx.kIter / self->ctx.multiKBL1) + 1);
        }
    }

    static __aicore__ void inline ReduceKPreloadDbAll(Intf *self)
    {
        ASC_OP_LOGD("AL1 and BL1 db case, preload reduce k\n");

        if (self->ctx.loadAL1Flag || !(self->ctx.kAL1fullload)) {
            if (self->ctx.freeAL1TensorFlag) {
                self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            }
            LoadAL1PreloadProcess(self, 0);
        }

        if (self->ctx.loadBL1Flag || !(self->ctx.kBL1fullload)) {
            if (self->ctx.freeBL1TensorFlag) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            }
            LoadBL1PreloadProcess(self, 0);
        }

        LoadAL1Process(self, 1);
        LoadBL1Process(self, 1);
        ReduceKFirstIterLoadL0(self);

        self->ctx.kIter = 1;
        uint16_t isOdd = 1;
        uint64_t maxKAL1PreloadIter = self->ctx.ddr2l0LoopK - self->ctx.multiKAL1;
        uint64_t maxKBL1PreloadIter = self->ctx.ddr2l0LoopK - self->ctx.multiKBL1;
        while (self->ctx.kIter < self->ctx.ddr2l0LoopK) {
            ReduceKPreloadDbAllLoadL1(self, maxKAL1PreloadIter, maxKBL1PreloadIter);
            ReduceKIterLoadL0(self, isOdd);
            self->ctx.kIter++;
            isOdd = self->ctx.kIter & 0x1;
        }
    }

    static __aicore__ void inline ReduceKPreloadDbFmapLoadL1(Intf *self,  const uint64_t& maxKAL1PreloadIter)
    {
        if (self->ctx.kIter == maxKAL1PreloadIter) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
        } else if (self->ctx.kIter < maxKAL1PreloadIter && self->ctx.kIter % self->ctx.multiKAL1 == 0) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            LoadAL1Process(self, (self->ctx.kIter / self->ctx.multiKAL1) + 1);
        }

        if constexpr (!Intf::bl1bypass) {
            if (self->ctx.loadBL1Flag || (!self->ctx.kBL1fullload && self->ctx.kIter % self->ctx.multiKBL1 == 0)) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
                LoadBL1Process(self, self->ctx.kIter / self->ctx.multiKBL1);
            }
        }
    }

    static __aicore__ void inline ReduceKPreloadDbFmap(Intf *self)
    {
        ASC_OP_LOGD("AL1 db case, preload reduce k\n");

        if (self->ctx.freeAL1TensorFlag) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
        }
        LoadAL1PreloadProcess(self, 0);
        LoadAL1Process(self, 1);

        if constexpr (!Intf::bl1bypass) {
            if (self->ctx.loadBL1Flag || !(self->ctx.kBL1fullload)) {
                if (self->ctx.freeBL1TensorFlag) {
                    self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
                }
                LoadBL1Process(self, 0);
            }
        }

        ReduceKFirstIterLoadL0(self);

        self->ctx.kIter = 1;
        uint16_t isOdd = 1;
        uint64_t maxKAL1PreloadIter = self->ctx.ddr2l0LoopK - self->ctx.multiKAL1;
        while (self->ctx.kIter < self->ctx.ddr2l0LoopK) {
            ReduceKPreloadDbFmapLoadL1(self, maxKAL1PreloadIter);
            ReduceKIterLoadL0(self, isOdd);
            self->ctx.kIter++;
            isOdd = self->ctx.kIter & 0x1;
        }

        ReduceKPostProcessLoadL0(self);
    }

    static __aicore__ void inline CalcBias(Intf *self)
    {
        if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0B_OPEN)) {
            wait_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID2);
            self->ctx.loadBiasBTIns.LoadBiasL0WithBroadcast();
            set_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID0);
            set_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID2);
            wait_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID2);
            self->ctx.madIns.MadBias();
            set_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID0);
            set_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID2);
        } else {
            wait_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID0);
            self->ctx.loadBiasBTIns.LoadBiasL0WithBroadcast();
            set_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, event_t::EVENT_ID0);
            self->ctx.madIns.MadBias();
            set_flag(PIPE_M, PIPE_MTE1, event_t::EVENT_ID0);
        }
    }

    static __aicore__ void inline InitBiasWithPointWise(Intf *self, uint64_t m, uint64_t n)
    {
        if (self->ctx.enableBias) {
            self->ctx.loadBiasL1Ins.SetN(AlignB(n, BLOCK_L0_M));
            self->ctx.loadBiasBTIns.SetMN(AlignB(n, BLOCK_L0_M), AlignB(m, BLOCK_L0_N));
            if (!self->ctx.biasFullLoadFlag) {
                self->ctx.biasL1 = self->ctx.queueBiasL1.template AllocTensor<typename Intf::BiasT>();
                self->ctx.loadBiasL1Ins.LoadChannelWiseL1(self->ctx.biasL1, self->ctx.biasgm);
                self->ctx.queueBiasL1.EnQue(self->ctx.biasL1);
                self->ctx.biasL1 = self->ctx.queueBiasL1.template DeQue<typename Intf::BiasT>();
            }
            CalcBias(self);
        }
    }

    static __aicore__ void inline InitBiasWithNormal(Intf *self, uint64_t m, uint64_t n)
    {
        if (self->ctx.enableBias) {
            self->ctx.loadBiasL1Ins.SetN(n);
            self->ctx.loadBiasBTIns.SetN(AlignB(n, BLOCK_L0_N));
            if (!self->ctx.biasFullLoadFlag) {
                self->ctx.biasL1 = self->ctx.queueBiasL1.template AllocTensor<typename Intf::BiasT>();
                self->ctx.loadBiasL1Ins.LoadChannelWiseL1(self->ctx.biasL1, self->ctx.biasgm);
                self->ctx.queueBiasL1.EnQue(self->ctx.biasL1);
                self->ctx.biasL1 = self->ctx.queueBiasL1.template DeQue<typename Intf::BiasT>();
            }
            self->ctx.biasBT = self->ctx.queueBiasBT.template AllocTensor<typename Intf::L0cT>();
            self->ctx.loadBiasBTIns.LoadBiasBt();
            self->ctx.queueBiasBT.EnQue(self->ctx.biasBT);
            self->ctx.biasBT = self->ctx.queueBiasBT.template DeQue<typename Intf::L0cT>();
        }
    }

    static __aicore__ void inline IterateK(Intf *self)
    {
        // in each iterate k, cal current m,n value
        uint64_t n = CalcL0CurrentN(self);
        uint64_t m = CalcL0CurrentM(self);
        if ASCEND_IS_AIC { 
            self->ctx.cl0 = self->ctx.queueCL0.template AllocTensor<typename Intf::L0cT>();
        }

        self->ctx.loadAL0Ins.SetM(AlignB(m, BLOCK_L0_N));
        self->ctx.loadBL0Ins.SetN(AlignB(n, BLOCK_L0_M));
        if constexpr (Intf::formatType == conv::ConvFormat::NCDHW) {
            self->ctx.madIns.SetMN(AlignB(n, BLOCK_L0_M), AlignB(m, BLOCK_L0_N));
            self->ctx.copyOutIns.SetMN(n, m);
            if ASCEND_IS_AIC { 
                InitBiasWithPointWise(self, m, n);
            }
        } else {
            self->ctx.madIns.SetMN(AlignB(m, BLOCK_L0_M), AlignB(n, BLOCK_L0_N));
            self->ctx.copyOutIns.SetMN(m, AlignB(n, self->ctx.cout0));
            if ASCEND_IS_AIC { 
                InitBiasWithNormal(self, m, n);
            }
        }
        if ASCEND_IS_AIC { 
            if (self->ctx.preloadABL1DbFlag) {
                ReduceKPreloadDbAll(self);
            } else if (self->ctx.preloadAL1DbFlag) {
                ReduceKPreloadDbFmap(self);
            } else {
                ReduceK(self);
            }
            self->ctx.queueCL0.EnQue(self->ctx.cl0);
            self->ctx.cl0 = self->ctx.queueCL0.template DeQue<typename Intf::L0cT>();
            self->ctx.kIter = 0;
        }
    }

    static __aicore__ void inline UpdateL1TailLoop(Intf *self)
    {
        self->ctx.l12l0LoopM = self->ctx.mAL1Iter == self->ctx.maxMAL1Iter
                                   ? CeilDIV(self->ctx.mAL1Tail, self->ctx.conv3dTiling->mL0)
                                   : self->ctx.conv3dTiling->mAL1DivmL0;
        self->ctx.maxML0Iter = self->ctx.l12l0LoopM - 1;

        if constexpr (Intf::bl1bypass) {
            return;
        }
        self->ctx.l12l0LoopN = self->ctx.nBL1Iter == self->ctx.maxNBL1Iter
                                   ? CeilDIV(self->ctx.nBL1Tail, self->ctx.conv3dTiling->nL0)
                                   : self->ctx.conv3dTiling->nBL1DivnL0;
        self->ctx.maxNL0Iter = self->ctx.l12l0LoopN - 1;
    }

    static __aicore__ bool inline IterateImpl(Intf *self, bool enPartialSum)
    {
        if (self->ctx.isFirstIterate) {
            FirstIterateImpl(self);
        } else if (likely(self->ctx.conv3dTiling->iterateMNOrder == static_cast<int>(IterateMNOrder::ORDER_MTERFIRST))) {
            if (IterateMFirst(self) == false) {
                return false;
            }
        } else if (likely(self->ctx.conv3dTiling->iterateMNOrder == static_cast<int>(IterateMNOrder::ORDER_NTERFIRST))) {
            if (IterateNFirst(self) == false) {
                return false;
            }
        }
        UpdateL1TailLoop(self);
        IterateK(self);
        return true;
    }
};

template <class Intf, uint32_t ImplType>
struct GetTensorC {
    template <bool sync = true>
    static __aicore__ inline bool call(
        Intf *self, const GlobalTensor<typename Intf::OutputT> &output, bool enSequentialWrite = false)
    {
        if constexpr (Intf::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
            CrossCoreWaitFlag(self->ctx.V2CEvent + self->ctx.workspaceDbFlag);
            self->ctx.copyOutIns.CopyOut2Workspace(self->ctx.workspacegm);
            CrossCoreSetFlag<0x2, PIPE_FIX>(self->ctx.C2VEvent + self->ctx.workspaceDbFlag);
        } else {
            self->ctx.copyOutIns.CopyOut(output);
        }
        self->ctx.queueCL0.FreeTensor(self->ctx.cl0);
        if (self->ctx.enableBias) {
            if (!self->ctx.biasFullLoadFlag) {
                self->ctx.queueBiasL1.FreeTensor(self->ctx.biasL1);
            }
            if constexpr (Intf::formatType != conv::ConvFormat::NCDHW) {
                self->ctx.queueBiasBT.FreeTensor(self->ctx.biasBT);
            }
        }
        ASC_OP_LOGD("[GetTensorC] GetTensorC Success! \n\n");
        return false;
    }
};

template <class Intf, uint32_t ImplType>
struct VecCompute {
    static __aicore__ inline bool call(
        Intf *self, const GlobalTensor<typename Intf::OutputT> &output)
    {
        uint64_t mSize, nSize;
        uint64_t ws_startoffset = self->ctx.workspaceDbFlag * self->ctx.conv3dTiling->mL0 * self->ctx.conv3dTiling->nL0;
        self->ctx.copyOutIns.GetL0CSize(mSize, nSize);
        uint64_t n16num = nSize / BLOCK_L0_N;
        if (n16num > 1) {
            if (self->ctx.subblockIdx) {
                uint64_t halfnum = CeilDIV(n16num, 2) * BLOCK_L0_N;
                nSize = nSize - halfnum;
                self->ctx.channelOffset = halfnum;
                ws_startoffset += halfnum * mSize;
                self->ctx.outNoffset = halfnum;
            } else {
                nSize = CeilDIV(n16num, 2) * BLOCK_L0_N;
            }
        } else {
            if (self->ctx.subblockIdx) {
                if (mSize > 1) {
                    uint64_t halfnum = CeilDIV(mSize, 2);
                    mSize = mSize - halfnum;
                    ws_startoffset += BLOCK_L0_N * halfnum;
                    self->ctx.outMoffset = halfnum;
                } else {
                    CrossCoreWaitFlag(self->ctx.C2VEvent + self->ctx.workspaceDbFlag);
                    CrossCoreSetFlag<0x2, PIPE_MTE2>(self->ctx.V2CEvent + self->ctx.workspaceDbFlag);
                    return false;
                }
            } else {
                mSize = CeilDIV(mSize, 2);
            }
        }

        if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_LC0)) {
            CopyScaleAndBias<LoadChannelType::LOAD_TOTAL_LC0>(self, 0, nSize);
        }
        
        CrossCoreWaitFlag(self->ctx.C2VEvent + self->ctx.workspaceDbFlag);

        //copy from workspace
        
        uint32_t maxnUBIter = CeilDIV(nSize, self->ctx.conv3dTiling->nUB);
        uint32_t maxmUBIter = CeilDIV(mSize, self->ctx.conv3dTiling->mUB);
        uint16_t totalSrcStride = (mSize - self->ctx.conv3dTiling->mUB) * BLOCK_L0_N * self->ctx.sizeOfL0c / 32;

        for (uint32_t nIter = 0; nIter < maxnUBIter - 1; nIter++) {
            if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::NORMAL)) {
                CopyScaleAndBias<LoadChannelType::NORMAL>(self, nIter, self->ctx.conv3dTiling->nUB);
            }

            for (uint32_t mIter = 0; mIter < maxmUBIter - 1; mIter++) {
                //copy in
                uint64_t srcOffset = ws_startoffset + (nIter * self->ctx.conv3dTiling->nUB * mSize + 
                                    mIter * self->ctx.conv3dTiling->mUB * BLOCK_L0_N);
                CopyIn(self, self->ctx.totalBlockCount, self->ctx.totalBlockLen, totalSrcStride, srcOffset);
                //vec compute
                oneVecCompute(self, self->ctx.conv3dTiling->mUB, self->ctx.conv3dTiling->nUB, mIter, nIter, output);
            }
            //m tail
            uint16_t blockLen =  (mSize - (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB) * BLOCK_L0_N * self->ctx.sizeOfL0c / 32;
            uint16_t srcStride = (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB * BLOCK_L0_N * self->ctx.sizeOfL0c / 32 ;
            uint64_t srcOffset = ws_startoffset + (nIter * self->ctx.conv3dTiling->nUB * mSize + 
                                (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB * BLOCK_L0_N);
            CopyIn(self, self->ctx.totalBlockCount, blockLen, srcStride, srcOffset);
            //vec compute
            oneVecCompute(self, mSize - (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB, self->ctx.conv3dTiling->nUB, maxmUBIter - 1, nIter, output);
            if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::NORMAL)) {
                FreeScaleAndBias(self);
            }
        }
        if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::NORMAL)) {
            CopyScaleAndBias<LoadChannelType::NORMAL>(self, maxnUBIter - 1, nSize - (maxnUBIter - 1) * self->ctx.conv3dTiling->nUB);
        }
        
        // n tail
        uint16_t blockCount = (nSize - (maxnUBIter - 1) * self->ctx.conv3dTiling->nUB) / BLOCK_L0_N;
        for (uint32_t mIter = 0; mIter < maxmUBIter - 1; mIter++) {
            //copy in
            uint64_t srcOffset = ws_startoffset + (maxnUBIter-1) * self->ctx.conv3dTiling->nUB * mSize + 
                                                    mIter * self->ctx.conv3dTiling->mUB * BLOCK_L0_N ;
            CopyIn(self, blockCount, self->ctx.totalBlockLen, totalSrcStride, srcOffset);

            //vec compute
            oneVecCompute(self, self->ctx.conv3dTiling->mUB, nSize - (maxnUBIter - 1) * self->ctx.conv3dTiling->nUB, mIter, maxnUBIter - 1, output);
        }
        //m tail
        uint16_t blockLen =  (mSize - (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB) * BLOCK_L0_N * self->ctx.sizeOfL0c / 32;
        uint16_t srcStride = (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB * BLOCK_L0_N * self->ctx.sizeOfL0c / 32 ;
        uint64_t srcOffset = ws_startoffset + ((maxnUBIter-1) * self->ctx.conv3dTiling->nUB * mSize + 
                                                    (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB * BLOCK_L0_N);
        CopyIn(self, blockCount, blockLen, srcStride, srcOffset);
        //vec compute
        oneVecCompute(self, mSize - (maxmUBIter - 1) * self->ctx.conv3dTiling->mUB, nSize - (maxnUBIter - 1) * self->ctx.conv3dTiling->nUB,
                        maxmUBIter - 1, maxnUBIter - 1, output);
        if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_LC0) ||
                        Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::NORMAL)) {
            FreeScaleAndBias(self);
        }
        CrossCoreSetFlag<0x2, PIPE_MTE2>(self->ctx.V2CEvent + self->ctx.workspaceDbFlag);
        
        return false;
    }

    static __aicore__ inline void CopyIn(Intf *self, uint16_t blockCount, uint16_t blockLen, uint16_t srcStride, uint64_t srcOffset) 
    {
        self->ctx.ubin = self->ctx.queueUBin.template AllocTensor<typename Intf::L0cT>();
        DataCopyParams copyinParams;
        copyinParams.blockCount = blockCount;
        copyinParams.blockLen =  blockLen;
        copyinParams.srcStride = srcStride;
        copyinParams.dstStride = 0;
        DataCopy(self->ctx.ubin, self->ctx.workspacegm[srcOffset], copyinParams);
        self->ctx.queueUBin.EnQue(self->ctx.ubin);
    }

    template <LoadChannelType T>
    static __aicore__ inline void  CopyScaleAndBias(Intf *self, uint16_t nIter, uint16_t num)
    {
        if constexpr (T == LoadChannelType::NORMAL) {
            LocalTensor<typename Intf::BiasT> bias = self->ctx.queueUBbias.template AllocTensor<typename Intf::BiasT>();
            CopyInChannel<typename Intf::BiasT>(self, bias, self->ctx.biasgm[self->ctx.channelOffset], nIter, num);
            self->ctx.queueUBbias.EnQue(bias);
            LocalTensor<typename Intf::FP32T> scale = self->ctx.queueUBscale.template AllocTensor<typename Intf::FP32T>();
            CopyInChannel<typename Intf::FP32T>(self, scale, self->ctx.scalegm[self->ctx.channelOffset], nIter, num);
            self->ctx.queueUBscale.EnQue(scale);

            LocalTensor<typename Intf::BiasT> biasLocal = self->ctx.queueUBbias.template DeQue<typename Intf::BiasT>();

            if constexpr (IsSameType<typename Intf::BiasT, bfloat16_t>::value || IsSameType<typename Intf::BiasT, half>::value) {
                self->ctx.ubbias = self->ctx.fp32BiasBuf.template Get<typename Intf::FP32T>();
                Cast(self->ctx.ubbias, biasLocal, RoundMode::CAST_NONE, num);
                self->ctx.queueUBbias.FreeTensor(biasLocal);
                
            } else {
                self->ctx.ubbias = biasLocal;
            }
            self->ctx.ubscale = self->ctx.queueUBscale.template DeQue<typename Intf::FP32T>();
        }
        
        if constexpr (T == LoadChannelType::LOAD_TOTAL_LC0) {
            LocalTensor<typename Intf::BiasT> bias = self->ctx.queueUBbias.template AllocTensor<typename Intf::BiasT>();
            CopyInChannel<typename Intf::BiasT>(self, bias, self->ctx.biasgm[self->ctx.channelOffset], 0, num);
            self->ctx.queueUBbias.EnQue(bias);
            LocalTensor<typename Intf::BiasT> biasLocal = self->ctx.queueUBbias.template DeQue<typename Intf::BiasT>();

            if constexpr (IsSameType<typename Intf::BiasT, bfloat16_t>::value || IsSameType<typename Intf::BiasT, half>::value) {
                self->ctx.ubbias = self->ctx.fp32BiasBuf.template Get<typename Intf::FP32T>();
                Cast(self->ctx.ubbias, biasLocal, RoundMode::CAST_NONE, num);
                self->ctx.queueUBbias.FreeTensor(biasLocal);
                
            } else {
                self->ctx.ubbias = biasLocal;
            }
            
            LocalTensor<typename Intf::FP32T> scale = self->ctx.queueUBscale.template AllocTensor<typename Intf::FP32T>();
            CopyInChannel<typename Intf::FP32T>(self, scale, self->ctx.scalegm[self->ctx.channelOffset], 0, num);
            self->ctx.queueUBscale.EnQue(scale);
            self->ctx.ubscale = self->ctx.queueUBscale.template DeQue<typename Intf::FP32T>();
        }
    }

    static __aicore__ inline void  FreeScaleAndBias(Intf *self) 
    {
        if constexpr (!(IsSameType<typename Intf::BiasT, bfloat16_t>::value || IsSameType<typename Intf::BiasT, half>::value)) {
            self->ctx.queueUBbias.FreeTensor(self->ctx.ubbias);
        }
        self->ctx.queueUBscale.FreeTensor(self->ctx.ubscale);
    }

    template <typename DataTypeT>
    static __aicore__ inline void CopyInChannel(Intf *self, const LocalTensor<DataTypeT>& dst, const GlobalTensor<DataTypeT> &src, uint16_t nIter, uint16_t num) 
    {    
        DataCopyParams copyinParams;
        copyinParams.blockCount = 1;
        copyinParams.blockLen =  num * sizeof(DataTypeT) / 32;
        uint64_t Offset = self->ctx.copyOutIns.GetChannelOffset(nIter * self->ctx.conv3dTiling->nUB);
        DataCopy(dst, src[Offset], copyinParams);
    }

    static __aicore__ inline void MulScaleAddBias(uint16_t m, uint16_t n, const LocalTensor<typename Intf::FP32T>& src, 
                                                 const LocalTensor<typename Intf::FP32T>& bias, const LocalTensor<typename Intf::FP32T>& scale) 
    {
        uint64_t mask = BLOCK_L0_N;
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = BLOCK_L0_N * sizeof(typename Intf::FP32T) / 32;
        repeatParams.src0RepStride = BLOCK_L0_N * sizeof(typename Intf::FP32T) / 32;
        repeatParams.src1RepStride = 0;

        uint16_t maxNiter = n / BLOCK_L0_N;
        uint16_t maxMiter = CeilDIV(m, 255);
        for(uint16_t niter = 0; niter < maxNiter; niter++) {
            for (uint16_t miter = 0; miter < maxMiter - 1; miter++)
            {
                uint64_t offset = m * niter * BLOCK_L0_N + 255 * miter * BLOCK_L0_N;
                Mul(src[offset], src[offset], scale[niter * BLOCK_L0_N], mask, 255, repeatParams);
            }

            uint64_t offset = m * niter * BLOCK_L0_N + 255 * (maxMiter - 1) * BLOCK_L0_N;
            Mul(src[offset], src[offset], scale[niter * BLOCK_L0_N], mask, m - (maxMiter - 1) * 255, repeatParams);
        }
        
        PipeBarrier<PIPE_V>();

        for(uint16_t niter = 0; niter < maxNiter; niter++) {
            for (uint16_t miter = 0; miter < maxMiter - 1; miter++)
            {
                uint64_t offset = m * niter * BLOCK_L0_N + 255 * miter * BLOCK_L0_N;
                Add(src[offset], src[offset], bias[niter * BLOCK_L0_N], mask, 255, repeatParams);
            }
            uint64_t offset = m * niter * BLOCK_L0_N + 255 * (maxMiter - 1) * BLOCK_L0_N;
            Add(src[offset], src[offset], bias[niter * BLOCK_L0_N], mask, m - (maxMiter - 1) * 255, repeatParams);
        }
    }

    static __aicore__ inline void oneVecCompute(Intf *self, uint16_t m, uint16_t n, uint16_t mIter, uint16_t nIter, const GlobalTensor<typename Intf::OutputT> &output)
    {
        LocalTensor<typename Intf::L0cT> localUBin = self->ctx.queueUBin.template DeQue<typename Intf::L0cT>();
        LocalTensor<typename Intf::FP32T> dstLocal = localUBin.template ReinterpretCast<typename Intf::FP32T>();

        Cast(dstLocal, localUBin, AscendC::RoundMode::CAST_RINT, m * n);

        LocalTensor<typename Intf::FP32T> bias;
        LocalTensor<typename Intf::FP32T> scale;
        if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::NORMAL)) {
            bias = self->ctx.ubbias;
            scale = self->ctx.ubscale;
        }

        if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_LC0)) {
            bias = self->ctx.ubbias[nIter * self->ctx.conv3dTiling->nUB];
            scale = self->ctx.ubscale[nIter * self->ctx.conv3dTiling->nUB];
        }

        if constexpr (Intf::loadChannelType == static_cast<int8_t>(LoadChannelType::LOAD_TOTAL_CORE)) {
            uint64_t Offset = self->ctx.copyOutIns.GetChannelOffset(nIter * self->ctx.conv3dTiling->nUB) + self->ctx.channelOffset;
            bias = self->ctx.ubbias[Offset];
            scale = self->ctx.ubscale[Offset];
        }
        PipeBarrier<PIPE_V>();
        MulScaleAddBias(m, n, dstLocal, bias, scale);

        //cast to bf16
        self->ctx.ubout = self->ctx.queueUBout.template AllocTensor<typename Intf::BF16T>();
        PipeBarrier<PIPE_V>();
        Cast(self->ctx.ubout, dstLocal, AscendC::RoundMode::CAST_RINT, m * n);
        self->ctx.queueUBin.FreeTensor(localUBin);
        self->ctx.queueUBout.EnQue(self->ctx.ubout);
        LocalTensor<typename Intf::BF16T> copyubout = self->ctx.queueUBout.template DeQue<typename Intf::BF16T>();
        
        self->ctx.copyOutIns.CopyUBOut(output, mIter, nIter, m, n, copyubout);
        self->ctx.queueUBout.FreeTensor(copyubout);
    }
};

template <class Intf, uint32_t ImplType>
struct IterateAll {
    template <bool sync = true>
    static __aicore__ inline bool call(
        Intf *self, const GlobalTensor<typename Intf::OutputT> &output, bool enPartialSum = false)
    {
        self->ctx.loadBiasL1Ins.SetParams(self);
        self->ctx.loadBL1Ins.SetParams(self);
        self->ctx.loadAl1Ins.SetParams(self);
        self->ctx.loadBL0Ins.SetParams(self);
        self->ctx.madIns.SetParams(self);
        self->ctx.copyOutIns.SetParams(self);
        self->ctx.loadBiasBTIns.SetParams(self);
        if constexpr (Intf::formatType == conv::ConvFormat::NCDHW) {
            self->ctx.loadAL0Ins.SetParams(self);
        } else {
            self->ctx.loadAL0Ins.SetParams(self, &self->ctx.loadAl1Ins);
        }
        if constexpr (Intf::groupConvType) {
            IterateAllWithGroups(self, output, enPartialSum);
        } else {
            IterateAllBase(self, output, enPartialSum);
        }
        return false;
    }

    static __aicore__ void inline IterateAllBase(
        Intf *self, const GlobalTensor<typename Intf::OutputT> &output, bool enPartialSum = false) 
    {
        if ASCEND_IS_AIC {
            if (self->ctx.biasFullLoadFlag && self->ctx.enableBias) {
                self->ctx.biasL1 = self->ctx.queueBiasL1.template AllocTensor<typename Intf::BiasT>();
                self->ctx.loadBiasL1Ins.LoadChannelWiseL1(self->ctx.biasL1, self->ctx.biasgm);
                self->ctx.queueBiasL1.EnQue(self->ctx.biasL1);
                self->ctx.biasL1 = self->ctx.queueBiasL1.template DeQue<typename Intf::BiasT>();
            }
        }
        
        while (Iterate<Intf, ImplType>::call(self, enPartialSum)) {
            if ASCEND_IS_AIC {
                GetTensorC<Intf, ImplType>::call(self, output);
                if constexpr (Intf::formatType != conv::ConvFormat::NCDHW) {
                    if (self->ctx.enableBias) {
                        self->ctx.queueBiasBT.FreeAllEvent();
                    }
                }
            }
            
            if constexpr (Intf::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
                if ASCEND_IS_AIV {  
                    VecCompute<Intf, ImplType>::call(self, output);
                }
            }
            self->ctx.workspaceDbFlag = (self->ctx.workspaceDbFlag + 1) & 0x03;
        }
        if ASCEND_IS_AIC {
            if (self->ctx.biasFullLoadFlag && self->ctx.enableBias) {
                self->ctx.queueBiasL1.FreeTensor(self->ctx.biasL1);
            }
        }
        self->ctx.isFirstIterate = true;
        self->ctx.nBL0Iter = 0;
        self->ctx.nBL1Iter = 0;
    }

    static __aicore__ void inline ReCalculationKTilingWithGroups(
        Intf *self, uint64_t &updateKAL1, uint64_t &updateKBL1, uint64_t &updateKL0)
    {
        // Update kaL1/kbL1/kL0 when singleCoreCin changes.
        uint64_t curKAL1Kd = GetCurrentKD(
            self->ctx.conv3dTiling->kAL1, AlignB(self->ctx.orgCi, self->ctx.cin0), self->ctx.kernelHxkernelW);
        uint64_t curKBL1Kd = GetCurrentKD(
            self->ctx.conv3dTiling->kBL1, AlignB(self->ctx.orgCi, self->ctx.cin0), self->ctx.kernelHxkernelW);
        uint64_t curCinxKhxKw = AlignB(self->ctx.singleCoreCin, self->ctx.cin0) * self->ctx.kernelHxkernelW;
        updateKAL1 = curCinxKhxKw > self->ctx.conv3dTiling->kAL1 ? 0 : curCinxKhxKw;
        updateKBL1 = curCinxKhxKw > self->ctx.conv3dTiling->kBL1 ? 0 : curCinxKhxKw;
        if (curKAL1Kd > 1) {
            // The kAL1/kBL1 is calculated by multiplying the new cin by the kd of the previous tiling decision.
            updateKAL1 = curKAL1Kd * curCinxKhxKw;
        }
        if (updateKAL1 == 0) {
            // To ensure that kAL1/kBL1 is the factor of cin1, 1 is used as kAL1, which can be optimized in the future.
            updateKAL1 =
                curCinxKhxKw % self->ctx.conv3dTiling->kAL1 == 0 ? 0 : self->ctx.cin0 * self->ctx.kernelHxkernelW;
        }
        if (curKBL1Kd > 1) {
            // The kAL1/kBL1 is calculated by multiplying the new cin by the kd of the previous tiling decision.
            updateKBL1 = curKBL1Kd * curCinxKhxKw;
        }
        if (updateKBL1 == 0) {
            // To ensure that kAL1/kBL1 is the factor of cin1, 1 is used as kAL1, which can be optimized in the future.
            updateKBL1 =
                curCinxKhxKw % self->ctx.conv3dTiling->kBL1 == 0 ? 0 : self->ctx.cin0 * self->ctx.kernelHxkernelW;
        }
        if (updateKAL1 % self->ctx.conv3dTiling->kL0 != 0 || updateKBL1 % self->ctx.conv3dTiling->kL0 != 0) {
            // To ensure that kL0 is the factor of kAL1/kBL1, cin0 is used as kL0, which can be optimized in the future.
            updateKL0 = self->ctx.cin0;
        }
    }

    static __aicore__ void inline PreProcessGroupOptDimTail(Intf *self)
    {
        if (!self->ctx.isGroupOptDimTail) {
            return;
        }

        if (self->ctx.singleCoreCinTail != 0) {
            ASC_OP_LOGD("[IterateAllWithGroups] singleCoreCin %d update to %d \n",
                self->ctx.singleCoreCin,
                self->ctx.singleCoreCinTail);
            self->ctx.singleCoreCin = self->ctx.singleCoreCinTail;
            uint64_t updateKAL1 = 0;
            uint64_t updateKBL1 = 0;
            uint64_t updateKL0 = 0;
            ReCalculationKTilingWithGroups(self, updateKAL1, updateKBL1, updateKL0);
            InitKDirectionBaseValue<Intf>(self, updateKAL1, updateKBL1, updateKL0);
            self->ctx.preloadAL1DbFlag = false;
            self->ctx.preloadABL1DbFlag = false;
            ASC_OP_LOGD("[IterateAllWithGroups] updateKAL1 %d updateKBL1 %d updateKL0 %d \n",
                updateKAL1,
                updateKBL1,
                updateKL0);
        }
        if (self->ctx.singleCoreCoutTail != 0) {
            ASC_OP_LOGD("[IterateAllWithGroups] singleCoreCo %d update to %d \n",
                self->ctx.singleCoreCo,
                self->ctx.singleCoreCoutTail);
            self->ctx.singleCoreCo = self->ctx.singleCoreCoutTail;
            InitCoutDirectionBaseValue<Intf>(self);
        }
    }

    static __aicore__ void inline PostProcessGroupOptDimTail(Intf *self, const uint64_t &tmpSingleCoreCo,
        const uint8_t &tmpPreloadAL1DbFlag, const uint8_t &tmpPreloadABL1DbFlag)
    {
        if (!self->ctx.isGroupOptDimTail) {
            return;
        }

        if (self->ctx.singleCoreCin != self->ctx.conv3dTiling->cinOpt) {
            self->ctx.singleCoreCin = self->ctx.conv3dTiling->cinOpt;
            InitKDirectionBaseValue<Intf>(self);
            self->ctx.preloadAL1DbFlag = tmpPreloadAL1DbFlag;
            self->ctx.preloadABL1DbFlag = tmpPreloadABL1DbFlag;
        }
        if (self->ctx.singleCoreCo != tmpSingleCoreCo) {
            self->ctx.singleCoreCo = tmpSingleCoreCo;
            InitCoutDirectionBaseValue<Intf>(self);
        }
        self->ctx.isGroupOptDimTail = false;
    }

    static __aicore__ void inline IterateAllWithGroups(
        Intf *self, const GlobalTensor<typename Intf::OutputT> &output, bool enPartialSum = false)
    {
        uint64_t weightOneGroupOptSize =
            self->ctx.conv3dTiling->cinOpt * self->ctx.kernelHxkernelWxkernelD * self->ctx.conv3dTiling->coutOpt;
        while (self->ctx.groupOptIter < self->ctx.maxGroupOptIter - 1) {
            IterateAllBase(self, output, enPartialSum);
            self->SetWeight(self->ctx.bgm[weightOneGroupOptSize]);
            if (self->ctx.enableBias) {
                self->SetBias(self->ctx.biasgm[self->ctx.conv3dTiling->coutOpt]);
            }
            self->ctx.groupOptIter++;
        }
        uint64_t tmpSingleCoreCo = self->ctx.singleCoreCo;
        uint8_t tmpPreloadAL1DbFlag = self->ctx.preloadAL1DbFlag;
        uint8_t tmpPreloadABL1DbFlag = self->ctx.preloadABL1DbFlag;
        PreProcessGroupOptDimTail(self);
        IterateAllBase(self, output, enPartialSum);
        PostProcessGroupOptDimTail(self, tmpSingleCoreCo, tmpPreloadAL1DbFlag, tmpPreloadABL1DbFlag);
        self->ctx.groupOptIter = 0;
    }
};

}  // namespace Conv3dFunc

#endif
