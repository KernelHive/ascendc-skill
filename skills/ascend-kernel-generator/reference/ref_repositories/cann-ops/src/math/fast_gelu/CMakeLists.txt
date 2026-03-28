add_ops_compile_options(
        OP_NAME FastGelu
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/fast_gelu.cpp
)

target_sources(optiling PRIVATE
        op_host/fast_gelu.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/fast_gelu.cpp
)

install(FILES op_kernel/fast_gelu.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)