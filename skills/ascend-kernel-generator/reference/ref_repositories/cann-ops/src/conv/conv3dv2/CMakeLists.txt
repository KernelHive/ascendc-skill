add_ops_compile_options(
        OP_NAME Conv3DV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

#optiling
target_sources(optiling PRIVATE
        op_host/conv3dv2_tiling.cpp
        op_host/conv3d_tiling.cpp
        op_host/conv3d_tiling_utils.cpp
        op_host/conv3d_tiling_util.cpp
        op_host/conv3d_tiling_base.cpp
        op_host/conv3d_tiling_algorithm.cpp
        op_host/conv3d_tiling_algorithm_hw_mode.cpp
        op_host/conv3d_tiling_algorithm_pointwise.cpp
        op_host/conv3d_base_tiling.cpp
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

#opsproto
target_sources(opsproto PRIVATE
         op_host/conv3dv2_proto.cpp
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

#手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/conv3_dv2_def.cpp
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
        ../conv2d_transpose_v2/op_host/convolution.cpp
        ../conv2d_transpose_v2/op_host/aclnn_convolution.cpp
)

# install kernel侧源码和aclnn头文件
install(DIRECTORY op_kernel/
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic
        FILES_MATCHING PATTERN "*.cpp")

install(DIRECTORY op_kernel/
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic
        FILES_MATCHING PATTERN "*.h")

install(FILES op_host/aclnn_convolution.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)