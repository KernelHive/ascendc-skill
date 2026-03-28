add_ops_compile_options(
        OP_NAME Conv3DBackpropFilterV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)


#手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/conv3_d_backprop_filter_v2_def.cpp
)

target_sources(opapi PRIVATE
        ${CMAKE_SOURCE_DIR}/src/conv/aclnn/convolutionbackward.cpp
        ${CMAKE_SOURCE_DIR}/src/conv/aclnn/aclnn_convolution_backward.cpp
)


file(GLOB_RECURSE EXTRA_SRC_FILES
    "../../common/op_host/op_tiling/*.cc"
)

# optiling
target_sources(optiling PRIVATE
        op_host/conv3_d_backprop_filter_v2.cpp
        ${EXTRA_SRC_FILES}
)


target_sources(opsproto PRIVATE
        op_host/conv3_d_backprop_filter_v2_def.cpp
)

file(GLOB KERNEL_FILES
        op_kernel/*.cpp
        op_kernel/*.h
)

install(FILES ${KERNEL_FILES}
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${CMAKE_SOURCE_DIR}/src/conv/aclnn
        ${CMAKE_SOURCE_DIR}/src/common/op_host
        ${CMAKE_SOURCE_DIR}/src/common/op_host/op_tiling
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
        ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn/opdev
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog/toolchain
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/mmpa
)


target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${CMAKE_SOURCE_DIR}/src/conv/aclnn
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

target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${CMAKE_SOURCE_DIR}/src/conv/aclnn
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


target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${CMAKE_SOURCE_DIR}/src/conv/aclnn
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
