add_ops_compile_options(
        OP_NAME WeightQuantBatchMatmulV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

file(GLOB_RECURSE EXTRA_SRC_FILES
    "../../common/op_host/op_tiling/*.cc"
)
# optiling
target_sources(optiling PRIVATE
        op_host/weight_quant_batch_matmul_v2_adaptive_split_tiling.cpp
        op_host/weight_quant_batch_matmul_v2_basic_block_tiling.cpp
        op_host/weight_quant_batch_matmul_v2_compute_matmul_tiling.cpp
        op_host/weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.cpp
        op_host/weight_quant_batch_matmul_v2_tiling_custom.cpp
        op_host/weight_quant_batch_matmul_v2_tiling_fixpipe.cpp
        op_host/weight_quant_batch_matmul_v2_tiling_msd_group.cpp
        op_host/weight_quant_batch_matmul_v2_tiling_msd.cpp
        op_host/weight_quant_batch_matmul_v2_tiling_registry.cpp
        op_host/weight_quant_batch_matmul_v2_tiling_splitk.cpp
        op_host/weight_quant_batch_matmul_v2_tiling_tool.cpp
        op_host/weight_quant_batch_matmul_v2_tiling.cpp
        op_host/weight_quant_batch_matmul_v2_weight_nz_tiling.cpp
        ${EXTRA_SRC_FILES}
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/op_host/op_tiling
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/op_host/op_tiling/aoe/op_tuning_tiling
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/op_host/op_tiling/cube/util
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
        ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn/opdev
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog/toolchain
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/mmpa
)

target_compile_options(optiling PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17)

# opproto
target_sources(opsproto PRIVATE
         op_host/weight_quant_batch_matmul_v2_proto.cpp
)

target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

# 手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/weight_quant_batch_matmul_v2_def.cpp
)

target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

target_sources(opapi PRIVATE
        op_host/weight_quant_batch_matmul_v2_l0.cpp
        op_host/aclnn_weight_quant_batch_matmul_v2.cpp
)

install(FILES op_kernel/weight_quant_batch_matmul_v2.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/weight_quant_batch_matmul_v2_common.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/weight_quant_batch_matmul_v2_constant.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/weight_quant_batch_matmul_v2_custom_mix_splitk.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/weight_quant_batch_matmul_v2_custom_nz_splitk.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/weight_quant_batch_matmul_v2_custom_weight_nz.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/weight_quant_batch_matmul_v2_custom.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/weight_quant_batch_matmul_v2_msd_group.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/weight_quant_batch_matmul_v2_msd_multicore.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/weight_quant_batch_matmul_v2_msd_split_k.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/weight_quant_batch_matmul_v2_weight_nz_performance_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/weight_quant_batch_matmul_v2_weight_nz_performance.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/tool.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_host/aclnn_weight_quant_batch_matmul_v2.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)

install(FILES op_host/aclnn_weight_quant_batch_matmul_v3.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)



