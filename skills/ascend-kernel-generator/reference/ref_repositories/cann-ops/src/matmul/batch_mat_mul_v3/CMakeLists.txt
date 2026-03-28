add_ops_compile_options(
        OP_NAME BatchMatMulV3
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

file(GLOB_RECURSE EXTRA_SRC_FILES
    "../../common/op_host/op_tiling/*.cc"
)
# optiling
target_sources(optiling PRIVATE
        op_host/batch_mat_mul_v3_tiling.cpp
        op_host/batch_mat_mul_v3_base_tiling.cpp
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

# opproto
target_sources(opsproto PRIVATE
         ../mat_mul_v3/op_host/mat_mul_v3_proto.cpp
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
        op_host/batch_mat_mul_v3_def.cpp
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
        op_host/batch_matmul_l0.cpp
        op_host/aclnn_batch_matmul.cpp
)

install(FILES op_kernel/batch_mat_mul_v3.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/batch_mat_mul_nd2nz.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/batch_mat_mul_v3_block.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/batch_mat_mul_v3_com_base_block_struct.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/batch_mat_mul_v3_com_base_block.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/batch_mat_mul_v3.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/mat_mul_base_block.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/mat_mul_base_kernel.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/mat_mul_l1_full_load.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/mat_mul_nd2nz_kernel.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/mat_mul_nd2nz_util.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/mat_mul_nd2nz.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/mat_mul_unaligned_base_kernel.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/mat_mul_v3_common.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_host/aclnn_batch_matmul.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)



