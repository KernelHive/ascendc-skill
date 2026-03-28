add_ops_compile_options(
        OP_NAME Conv3DTransposeV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

file(GLOB_RECURSE EXTRA_SRC_FILES
    "../../common/op_host/op_tiling/*.cc"
)

#optiling
target_sources(optiling PRIVATE
        op_host/conv3_d_transpose_v2_tiling.cpp
        ../conv3d_backprop_input_v2/op_host/conv3_d_backprop_input_v2.cpp
        ../conv3d_backprop_input_v2/op_host/conv3d_backprop_input.cpp
        ${EXTRA_SRC_FILES}
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${CMAKE_SOURCE_DIR}/src/conv/common
        ${CMAKE_SOURCE_DIR}/src/common/op_host
        ${CMAKE_SOURCE_DIR}/src/common/op_host/op_tiling
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn/opdev
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog/toolchain
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/mmpa
)

#opsproto
target_sources(opsproto PRIVATE
        op_host/conv3_d_transpose_v2_proto.cpp
        ${CMAKE_SOURCE_DIR}/src/conv/conv2d_transpose_v2/op_host/cube_util.cc
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_SOURCE_DIR}/src/conv/conv2d_transpose_v2/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${CMAKE_SOURCE_DIR}/src/conv/common
        ${CMAKE_SOURCE_DIR}/src/common/op_host
        ${CMAKE_SOURCE_DIR}/src/common/op_host/op_tiling
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn/opdev
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

#手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/conv3_d_transpose_v2_def.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${CMAKE_SOURCE_DIR}/src/common/op_host
        ${CMAKE_SOURCE_DIR}/src/common/op_host/op_tiling
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn/opdev
        ${ASCEND_CANN_PACKAGE_PATH}/include/graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
)

target_sources(opapi PRIVATE
        ${CMAKE_SOURCE_DIR}/src/conv/conv2d_transpose_v2/op_host/convolution.cpp
        ${CMAKE_SOURCE_DIR}/src/conv/conv2d_transpose_v2/op_host/aclnn_convolution.cpp
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${CMAKE_SOURCE_DIR}/src/conv/common
        ${CMAKE_SOURCE_DIR}/src/common/op_host
        ${CMAKE_SOURCE_DIR}/src/common/op_host/op_tiling
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn/opdev
        ${ASCEND_CANN_PACKAGE_PATH}/include/graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
)

# install kernel侧源码和aclnn头文件
install(DIRECTORY op_kernel/
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic
        FILES_MATCHING PATTERN "*.cpp")

install(DIRECTORY op_kernel/
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic
        FILES_MATCHING PATTERN "*.h")

install(FILES ../aclnn/inc/aclnn_convolution.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)