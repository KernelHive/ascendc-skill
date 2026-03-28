add_ops_compile_options(
        OP_NAME DiagV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

# 手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/diag_v2.cpp
)

target_sources(opapi PRIVATE
        op_host/diag_v2_l0.cpp
        op_host/diag_flat_l0.cpp
        op_host/aclnn_diag.cpp
)

install(FILES op_host/aclnn_diag.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)

target_sources(optiling PRIVATE
        op_host/diag_v2.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/diag_v2.cpp
)

install(FILES op_kernel/diag_v2.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES op_kernel/diag_v2_assist.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES op_kernel/diag_v2_b8.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES op_kernel/diag_v2_b16.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES op_kernel/diag_v2_b32.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES op_kernel/diag_v2_b64.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES op_kernel/diag_v2_b128.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES op_kernel/diag_v2_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES op_kernel/diag_v2_t4.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)