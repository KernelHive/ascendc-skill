add_ops_compile_options(
        OP_NAME Addcdiv
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/addcdiv.cpp
)

target_sources(optiling PRIVATE
        op_host/addcdiv.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/addcdiv.cpp
)

install(FILES op_kernel/addcdiv.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)