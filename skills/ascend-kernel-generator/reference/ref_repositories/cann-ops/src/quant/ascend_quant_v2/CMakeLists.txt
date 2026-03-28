add_ops_compile_options(
        OP_NAME AscendQuantV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

# 手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/ascend_quant_v2_def.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# api
target_sources(opapi PRIVATE
        op_host/ascend_quant_v2_l0.cpp
        op_host/aclnn_ascend_quant.cpp
)

# optiling
target_sources(optiling PRIVATE
        op_host/ascend_quant_v2.cc
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# opproto
target_sources(opsproto PRIVATE
        op_host/ascend_quant_v2_proto.cc
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# install kernel
install(FILES op_kernel/ascend_quant_v2.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/ascend_quant_v2.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/ascend_quant_v2_bf16.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/ascend_quant_v2_fp16.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/ascend_quant_v2_fp32.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

# install aclnn
install(FILES op_host/aclnn_ascend_quant.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)