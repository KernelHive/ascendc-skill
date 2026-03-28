add_ops_compile_options(
        OP_NAME AddSigmoidMulReduceSumD
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
op_host/add_sigmoid_mul_reduce_sum_d.cpp
)

target_sources(optiling PRIVATE
        op_host/add_sigmoid_mul_reduce_sum_d.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/add_sigmoid_mul_reduce_sum_d.cpp
)

install(FILES op_kernel/add_sigmoid_mul_reduce_sum_d.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)