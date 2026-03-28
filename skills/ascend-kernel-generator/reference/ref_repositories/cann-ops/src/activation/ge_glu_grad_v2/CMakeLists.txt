add_ops_compile_options(
        OP_NAME GeGluGradV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/ge_glu_grad_v2_def.cpp
)

# optiling
target_sources(optiling PRIVATE
        op_host/ge_glu_grad_v2.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/ge_glu_grad_v2_tiling.h
)


# opproto
target_sources(opsproto PRIVATE
        op_host/ge_glu_grad_v2_def.cpp
)

install(FILES op_kernel/ge_glu_grad_v2.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES op_kernel/ge_glu_grad_v2_base_310p.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_grad_v2_erf_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_grad_v2_erf_bfp16.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_grad_v2_erf_fp16.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_grad_v2_erf_fp32.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_grad_v2_fp16_310p.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_grad_v2_fp32_310p.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_grad_v2_tanh_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_grad_v2_tanh_bfp16.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_grad_v2_tanh_fp16.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_grad_v2_tanh_fp32.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        
