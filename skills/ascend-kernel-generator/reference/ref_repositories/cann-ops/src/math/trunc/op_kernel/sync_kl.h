/**
 * Copyright (C) Henan KunLun Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#ifndef SYNC_KUNLUN_H
#define SYNC_KUNLUN_H
/// 跨流水同步函数。通过阻塞DST_POS流水，保证PIPE_SRC的前序读写操作完成后，PIPE_DST上的操作才被执行。
/// 支持的流水：MTE1, MTE2, MTE3, V, S, M
/// @param PIPE_SRC 源流水
/// @param PIPE_DST 目的流水
namespace kunlun{
    using AscendC::HardEvent;
    using AscendC::TQueSync;
    using AscendC::TPipe;
    __aicore__ inline void SyncVMTE3(){
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        TQueSync<PIPE_V, PIPE_MTE3> queSync;
        queSync.SetFlag(eventID);
        queSync.WaitFlag(eventID);
    }
    __aicore__ inline void SyncMTE2S(){
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        TQueSync<PIPE_MTE2, PIPE_S> queSync;
        queSync.SetFlag(eventID);
        queSync.WaitFlag(eventID);
    }
    __aicore__ inline void SyncMTE2MTE3(){
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        TQueSync<PIPE_MTE2, PIPE_MTE3> queSync;
        queSync.SetFlag(eventID);
        queSync.WaitFlag(eventID);
    }
}
#endif// SYNC_KUNLUN_H
