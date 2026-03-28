# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME AdaptiveAvgPool3dGrad
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnnInner PRIVATE
        op_host/adaptive_avg_pool3d_grad_def.cpp
)

target_link_libraries(op_host_aclnnInner PRIVATE
        ops_utils_tiling_headers)

target_sources(opapi PRIVATE
        op_host/adaptive_avg_pool3d_backward.cpp
        op_host/aclnn_adaptive_avg_pool3d_backward.cpp
)

target_sources(optiling PRIVATE
        op_host/adaptive_avg_pool3d_grad.cpp
)
target_link_libraries(optiling PRIVATE
        ops_utils_tiling_headers)

target_sources(opsproto PRIVATE
         op_host/adaptive_avg_pool3d_grad.cpp
)
target_link_libraries(opsproto PRIVATE
        ops_utils_proto_headers
)

file(GLOB OP_KERNEL_HEADERS "op_kernel/*.h")

install(FILES op_kernel/adaptive_avg_pool3d_grad.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES ${OP_KERNEL_HEADERS}
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_host/aclnn_adaptive_avg_pool3d_backward.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)