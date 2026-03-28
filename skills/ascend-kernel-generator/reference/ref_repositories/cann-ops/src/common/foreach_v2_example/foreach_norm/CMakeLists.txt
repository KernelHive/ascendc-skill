add_ops_compile_options(
        OP_NAME ForeachNorm
        OPTIONS -I${OP_COMMON_DIR}/inc/foreach/op_kernel_v2
                --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
    op_host/foreach_norm.cpp
)

target_sources(optiling PRIVATE
    op_host/foreach_norm_tiling.cpp
    ${OP_COMMON_DIR}/src/foreach/op_tiling/foreach_reduce_tiling_func.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${OP_COMMON_DIR}/inc
    ${OP_COMMON_DIR}/inc/foreach/op_tiling
    ${OP_COMMON_DIR}/src/foreach/op_tiling
)

target_sources(opsproto PRIVATE
         op_host/foreach_norm.cpp
)

target_include_directories(opsproto PRIVATE
        ${OP_COMMON_DIR}/inc
)

install(FILES op_kernel/foreach_norm.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)