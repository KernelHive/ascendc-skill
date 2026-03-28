/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file platform.h
 * \brief platform apator
 */
#ifndef OPS_BUILT_IN_OP_ASCENDC_PLATFORM_INFO_H_
#define OPS_BUILT_IN_OP_ASCENDC_PLATFORM_INFO_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

#ifndef KERNEL_API
#define KERNEL_API extern "C" __global__ __aicore__
#endif

namespace platform {

#define MID_THREAD_NUM 1024

__aicore__ inline constexpr bool IsDataCopyPadSupport()
{
#if __CCE_AICORE__ == 220
    return true;
#else
    return false;
#endif
}

/**
 * Get the block size of unified buffer in bytes
 */
__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

/**
 * Get the size of vector registers in bytes
 */
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

/**
 * Check whether the type is supported by atomic add for simd
 */
template<typename T>
__aicore__ inline constexpr bool IsSupportAtomicAddTypeSIMD()
{
#if __CCE_AICORE__ == 310
    return ops::IsSame<T, float>::value || ops::IsSame<T, half>::value || ops::IsSame<T, int16_t>::value ||
        ops::IsSame<T, int32_t>::value || ops::IsSame<T, int8_t>::value || ops::IsSame<T, bfloat16_t>::value;
#else
    return false;
#endif
}

} // namespace platform

namespace PlatformSocInfo {
__aicore__ inline constexpr bool IsDataCopyPadSupport()
{
    return platform::IsDataCopyPadSupport();
}

}

namespace AscendC {
namespace MicroAPI {

}
}

#endif  // OPS_BUILT_IN_OP_ASCENDC_PLATFORM_INFO_H_