add_ops_compile_options(
        OP_NAME CoalesceSparse
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/coalesce_sparse.cpp
)

target_sources(optiling PRIVATE
        op_host/coalesce_sparse.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
        op_host/coalesce_sparse.cpp
)

install(FILES op_kernel/coalesce_sparse.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/coalesce_sparse.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
