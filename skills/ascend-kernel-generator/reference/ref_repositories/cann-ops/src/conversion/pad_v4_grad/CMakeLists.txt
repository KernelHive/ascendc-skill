add_ops_compile_options(
        OP_NAME PadV4Grad
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

#target_sources(op_host_aclnn PRIVATE
#op_host/pad_v4_grad.cpp
#)

#optiling
target_sources(optiling PRIVATE
        op_host/pad_v4_grad.cpp
)

target_include_directories(optiling PRIVATE
        $CMAKECURRENTSOURCEDIR/ophost
        ${CMAKE_SOURCE_DIR}/src/common/inc
        $ASCENDCANNPACKAGEPATH/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        $ASCENDCANNPACKAGEPATH/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        $ASCENDCANNPACKAGEPATH/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

#opproto
target_sources(opsproto PRIVATE
        op_host/pad_v4_grad_proto.cpp
)

target_include_directories(opsproto PRIVATE
        $CMAKECURRENTSOURCEDIR/ophost
        ${CMAKE_SOURCE_DIR}/src/common/inc
        $ASCENDCANNPACKAGEPATH/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        $ASCENDCANNPACKAGEPATH/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        $ASCENDCANNPACKAGEPATH/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

#手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/pad_v4_grad_def.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
        $CMAKECURRENTSOURCEDIR/ophost
        ${CMAKE_SOURCE_DIR}/src/common/inc
        $ASCENDCANNPACKAGEPATH/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        $ASCENDCANNPACKAGEPATH/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        $ASCENDCANNPACKAGEPATH/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

target_sources(opapi PRIVATE
        ../circular_pad_grad/op_host/padv3grad.cpp
        op_host/aclnn_reflection_pad1d_backward.cpp
)

#op_kernel
install(FILES op_kernel/pad_v4_grad.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v4_grad_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v4_grad_h_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v4_grad_h_w_bf16_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v4_grad_h_w_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v4_grad_large_h_small_w_bf16_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v4_grad_large_h_small_w_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v4_grad_mini_h_w_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v4_grad_small_h_large_w_bf16_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v4_grad_small_h_large_w_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/pad_v4_grad_w_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

#op_host
install(FILES op_host/aclnn_reflection_pad1d_backward.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
