add_ops_compile_options(
        OP_NAME Conv2DTransposeV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)


#手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/conv2_d_transpose_v2_def.cpp
)

target_sources(opapi PRIVATE
        op_host/convolution.cpp
        op_host/aclnn_convolution.cpp
)



file(GLOB_RECURSE EXTRA_SRC_FILES
    "../../common/op_host/op_tiling/*.cc"
)

# optiling
target_sources(optiling PRIVATE
        op_host/conv2_d_transpose_v2_tiling.cpp
        op_host/conv2dbp_adapt_to_conv3dbp.cc
        op_host/base/context_maker/kernel_run_context_maker.cc
        ../conv3d_backprop_input_v2/op_host/conv3_d_backprop_input_v2.cpp
        ../conv3d_backprop_input_v2/op_host/conv3d_backprop_input.cpp
        ../conv3d_transpose_v2/op_host/conv3_d_transpose_v2_tiling.cpp
        ${EXTRA_SRC_FILES}
)


target_include_directories(optiling PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/tools/msopgen/template/custom_operator_sample/TIK/PyTorch/metadef/inc/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/tools/msopgen/template/custom_operator_sample/TIK/PyTorch/metadef/inc
        ${ASCEND_CANN_PACKAGE_PATH}/tools/msopgen/template/custom_operator_sample/TIK/PyTorch/metadef/inc/graph
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${CMAKE_SOURCE_DIR}/src/conv/common
        ${CMAKE_SOURCE_DIR}/src/common/op_host
        ${CMAKE_SOURCE_DIR}/src/common/op_host/op_tiling
        ${ASCEND_CANN_PACKAGE_PATH}/include/graph/utils
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

target_sources(opsproto PRIVATE
        op_host/conv2_d_transpose_v2_proto.cpp
        op_host/cube_util.cc
)

file(GLOB KERNEL_FILES
        op_kernel/*.cpp
        op_kernel/*.h
)

install(FILES ${KERNEL_FILES}
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic
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
