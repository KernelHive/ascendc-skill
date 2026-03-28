add_ops_compile_options(
        OP_NAME UpsampleBicubic2dGrad
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

# 手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/upsample_bicubic2d_grad_def.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# api
target_sources(opapi PRIVATE
        op_host/upsample_bicubic2d_grad_l0.cpp
        op_host/aclnn_upsample_bicubic_2d_backward.cpp
)

# optiling
target_sources(optiling PRIVATE
        op_host/upsample_bicubic2d_grad.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# install kernel
install(FILES op_kernel/upsample_bicubic2d_grad.cpp
DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/upsample_bicubic2d_grad_base.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/upsample_bicubic2d_grad_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/upsample_bicubic2d_grad_dc.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

# install aclnn
install(FILES op_host/aclnn_upsample_bicubic_2d_backward.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)