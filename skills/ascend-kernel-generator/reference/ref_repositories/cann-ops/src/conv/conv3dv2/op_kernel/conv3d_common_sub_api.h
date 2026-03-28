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
 * \file conv3d_common_sub_api.h
 * \brief
 */

#ifndef CONV3D_COMMON_SUB_API_H
#define CONV3D_COMMON_SUB_API_H

#if (__CCE_AICORE__ > 300)
    #include "conv3d_sub_api.h"
#elif (__CCE_AICORE__ > 200)
    #include "conv3d_sub_api.h"
    #include "conv3d_pointwise_sub_api.h"
    #include "conv3d_groupopt_sub_api.h"
    #include "conv3d_hw_mode_sub_api.h"
#endif
#endif // __CONV3D_COMMON_SUB_API_H__