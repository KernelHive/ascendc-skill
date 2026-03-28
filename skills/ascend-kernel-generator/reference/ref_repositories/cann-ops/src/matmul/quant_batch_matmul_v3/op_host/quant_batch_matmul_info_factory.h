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
 * \file quant_batch_matmul_info_factory.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_INFO_FACTORY_H
#define QUANT_BATCH_MATMUL_INFO_FACTORY_H

#include <pthread.h>

#include "lock.h"
#include "quant_batch_matmul_v3_tiling.h"

namespace optiling {
class QuantBatchMatmulInfoFactory {
public:
    QuantBatchMatmulInfoFactory() = default;
    ~QuantBatchMatmulInfoFactory() = default;

    QuantBatchMatmulInfo* Get()
    {
        QuantBatchMatmulInfo *ptr = nullptr;
        auto threadId = pthread_self();
        lock_.rdlock();
        auto it = inst_.find(threadId);
        if (it == inst_.end()) {
            lock_.unlock();
            lock_.wrlock();
            ptr = &(inst_[threadId]);
        } else {
            ptr = &(it->second);
        }

        lock_.unlock();
        return ptr;
    }

private:
    std::map<pthread_t, QuantBatchMatmulInfo> inst_;
    RWLock lock_;
};

}  // namespace optiling
#endif  // QUANT_BATCH_MATMUL_INFO_FACTORY_H
