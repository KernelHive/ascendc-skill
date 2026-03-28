add_ops_compile_options(
        OP_NAME MaxPool3DWithArgmaxV2
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

# optiling
target_sources(optiling PRIVATE
        op_host/max_pool3_d_with_argmax_v2.cpp
        op_host/max_pool3d_with_argmax_v2_no_expand_indices_tiling.cpp
        op_host/max_pool3d_with_argmax_v2_tiling_base.cpp
        op_host/max_pool3d_with_argmax_v2_tiling_basesplit.cpp
        op_host/max_pool3d_with_argmax_v2_tiling_bigkernel.cpp
        op_host/max_pool3d_with_argmax_v2_tiling_hugekernel.cpp
        op_host/max_pool3d_with_argmax_v2_tiling_nosplit.cpp
        op_host/max_pool3d_with_argmax_v2_tiling_splitd.cpp
        op_host/max_pool3d_with_argmax_v2_tiling_splith.cpp
        op_host/max_pool3d_with_argmax_v2_tiling_splitw.cpp
)

target_include_directories(optiling PRIVATE
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
# opproto
target_sources(opsproto PRIVATE
)

# 手动实现的aclnn
target_sources(op_host_aclnnInner PRIVATE
        op_host/max_pool3_d_with_argmax_v2.cpp
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
        op_host/max_pool3d_with_argmax_v2_l0.cpp
        op_host/aclnn_max_pool3d_with_argmax.cpp
)

install(FILES op_host/aclnn_max_pool3d_with_argmax.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)

# kernel
install(FILES op_kernel/max_pool3d_with_argmax_v2.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_with_argmax_big_kernel.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        
install(FILES op_kernel/max_pool3d_with_argmax_v2_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_with_argmax_v2_huge_kernel.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_with_argmax_v2_no_expand_indices.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_with_argmax_v2_nosplit.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_with_argmax_v2_splitd.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_with_argmax_v2_splith.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_with_argmax_v2_splitw.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
