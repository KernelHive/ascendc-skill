add_ops_compile_options(
        OP_NAME PadV3GradReplicate
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(optiling PRIVATE
        op_host/pad_v3_grad_replicate.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

target_sources(opsproto PRIVATE
        op_host/pad_v3_grad_replicate_proto.cpp
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

# 手动生成的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/pad_v3_grad_replicate_def.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

target_sources(opapi PRIVATE
        ../circular_pad_grad/op_host/padv3grad.cpp
        op_host/aclnn_pad2d_backward.cpp
)

install(FILES op_kernel/pad_v3_grad_replicate.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v3_grad_replicate_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v3_grad_replicate_h_w_large.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v3_grad_replicate_h_w_mini.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v3_grad_replicate_h_w.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v3_grad_replicate_h.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v3_grad_replicate_large_h_small_w_bf16.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v3_grad_replicate_large_h_small_w.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v3_grad_replicate_small_h_large_w_bf16.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v3_grad_replicate_small_h_large_w.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v3_grad_replicate_w.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_host/aclnn_pad2d_backward.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)