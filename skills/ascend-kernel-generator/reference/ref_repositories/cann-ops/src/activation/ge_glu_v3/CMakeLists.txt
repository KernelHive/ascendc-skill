add_ops_compile_options(
        OP_NAME GeGluV3
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/ge_glu_v3_def.cpp
)

# optiling
target_sources(optiling PRIVATE
        op_host/ge_glu_v3.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/ge_glu_v3_tiling.h
)


# opproto
target_sources(opsproto PRIVATE
        op_host/ge_glu_v3_def.cpp
)

install(FILES op_kernel/ge_glu_v3.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES op_kernel/ge_glu_v3_base_310p.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_v3_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_v3_fp16_align_last_axis_big_without_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_v3_fp16_stride_310p.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_v3_fp32_align_last_axis_big_without_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/ge_glu_v3_fp32_stride_310p.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/tanh/ge_glu_v3_bf16_align_last_axis_big.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/tanh/ge_glu_v3_bf16_align.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/tanh/ge_glu_v3_bf16_vreduce.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/tanh/ge_glu_v3_fp16_align_last_axis_big.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/tanh/ge_glu_v3_fp16_align.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/tanh/ge_glu_v3_fp16_vreduce.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/tanh/ge_glu_v3_fp32_align_last_axis_big.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/tanh/ge_glu_v3_fp32_align.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/tanh/ge_glu_v3_fp32_vreduce.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/erf/ge_glu_v3_bf16_align_erf.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/erf/ge_glu_v3_bf16_align_last_axis_big_erf.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/erf/ge_glu_v3_bf16_vreduce_erf.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/erf/ge_glu_v3_fp16_align_erf.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/erf/ge_glu_v3_fp16_align_last_axis_big_erf.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/erf/ge_glu_v3_fp16_vreduce_erf.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/erf/ge_glu_v3_fp32_align_erf.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/erf/ge_glu_v3_fp32_align_last_axis_big_erf.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/erf/ge_glu_v3_fp32_vreduce_erf.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
