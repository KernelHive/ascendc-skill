/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file scatter_elements.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_SCATTER_ELEMENTS_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_SCATTER_ELEMENTS_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *ScatterElements(const aclTensor *data, const aclTensor *indices, const aclTensor *updates,
    int64_t axis, const std::string &reduction, aclOpExecutor *executor);
}

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_SCATTER_ELEMENTS_OP_H_