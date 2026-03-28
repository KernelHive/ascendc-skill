add_ops_compile_options(
        OP_NAME UpsampleBicubic2dAA
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

# 手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/upsample_bicubic2d_aa_def.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# api
target_sources(opapi PRIVATE
        op_host/upsample_bicubic2d_aa_l0.cpp
        op_host/aclnn_upsample_bicubic2d_aa.cpp
)

# optiling
target_sources(optiling PRIVATE
        op_host/upsample_bicubic2d_aa.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# install kernel
install(FILES op_kernel/upsample_bicubic2d_aa.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/upsample_bicubic2d_aa.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

# install aclnn
install(FILES op_host/aclnn_upsample_bicubic2d_aa.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)