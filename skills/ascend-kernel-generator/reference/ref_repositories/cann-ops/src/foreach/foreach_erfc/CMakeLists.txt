add_ops_compile_options(
        OP_NAME ForeachErfc
        OPTIONS -I${OP_COMMON_DIR}/inc/foreach/op_kernel
                --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
op_host/foreach_erfc.cpp
)

target_include_directories(op_host_aclnn PRIVATE
        ${OP_COMMON_DIR}/inc
)

target_sources(optiling PRIVATE
        ${OP_COMMON_DIR}/src/foreach/op_tiling/foreach_tiling_func.cpp
)

target_include_directories(optiling PRIVATE
        ${OP_COMMON_DIR}/inc
)

target_sources(opsproto PRIVATE
         op_host/foreach_erfc.cpp
)

install(FILES op_kernel/foreach_erfc.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)