add_ops_compile_options(
        OP_NAME SelectReduceMaxDSubExpReduceSumDRealDiv
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
op_host/select_reduce_max_d_sub_exp_reduce_sum_d_real_div.cpp
)

target_sources(optiling PRIVATE
        op_host/select_reduce_max_d_sub_exp_reduce_sum_d_real_div.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/select_reduce_max_d_sub_exp_reduce_sum_d_real_div.cpp
)

install(FILES op_kernel/select_reduce_max_d_sub_exp_reduce_sum_d_real_div.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)