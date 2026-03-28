add_ops_compile_options(
        OP_NAME GroupQuant
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

# 手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/group_quant_def.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# api
target_sources(opapi PRIVATE
        op_host/group_quant_l0.cpp
        op_host/aclnn_group_quant.cpp
)

# optiling
target_sources(optiling PRIVATE
        op_host/group_quant.cc
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# opproto
target_sources(opsproto PRIVATE
        op_host/group_quant_proto.cc
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# install kernel
install(FILES op_kernel/group_quant.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/group_quant_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

# install aclnn
install(FILES op_host/aclnn_group_quant.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)