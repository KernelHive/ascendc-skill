add_ops_compile_options(
        OP_NAME AdaptiveMaxPool3d
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(optiling PRIVATE
        op_host/adaptive_max_pool3d.cpp
        op_host/adaptive_max_pool3d_small_pool_tiling.cpp
        op_host/adaptive_max_pool3d_big_pool_tiling.cpp
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
)

# 手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/adaptive_max_pool3d.cpp
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
        op_host/adaptive_max_pool3d_l0.cpp
        op_host/aclnn_adaptive_max_pool3d.cpp
)

install(FILES op_host/aclnn_adaptive_max_pool3d.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)

file(GLOB OP_KERNEL_HEADERS "op_kernel/*.h")

install(FILES ${OP_KERNEL_HEADERS}
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/adaptive_max_pool3d.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)