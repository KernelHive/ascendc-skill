add_ops_compile_options(
        OP_NAME UpsampleBicubic2d
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

# 手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/upsample_bicubic2d_def.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# api
target_sources(opapi PRIVATE
        op_host/upsample_bicubic2d_l0.cpp
        op_host/aclnn_upsample_bicubic2d.cpp
)

# optiling
target_sources(optiling PRIVATE
        op_host/upsample_bicubic2d.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# install kernel
install(FILES op_kernel/upsample_bicubic2d.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/upsample_bicubic2d.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/upsample_bicubic2d_310p.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

# install aclnn
install(FILES op_host/aclnn_upsample_bicubic2d.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)