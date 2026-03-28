add_ops_compile_options(
        OP_NAME MseLossGradV1
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/mse_loss_grad_v1.cpp
)

target_sources(optiling PRIVATE
        op_host/mse_loss_grad_v1.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/mse_loss_grad_v1.cpp
)

# install kernel侧源码和aclnn头文件
install(DIRECTORY op_kernel/
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic
        FILES_MATCHING PATTERN "*.cpp")

install(DIRECTORY op_kernel/
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic
        FILES_MATCHING PATTERN "*.h")