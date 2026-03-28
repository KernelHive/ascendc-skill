add_ops_compile_options(
        OP_NAME LogicalOr
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
                # -sanitizer
)

target_sources(op_host_aclnn PRIVATE
op_host/logical_or.cpp
)

target_sources(optiling PRIVATE
        op_host/logical_or.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/logical_or.cpp
)

install(FILES op_kernel/logical_or.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
