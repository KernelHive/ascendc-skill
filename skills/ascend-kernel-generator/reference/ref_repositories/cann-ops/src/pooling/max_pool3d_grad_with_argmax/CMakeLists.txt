add_ops_compile_options(
        OP_NAME MaxPool3DGradWithArgmax
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

# optiling
target_sources(optiling PRIVATE
        op_host/max_pool3d_grad_with_argmax.cpp
        op_host/max_pool3d_grad_with_argmax_tiling_splitw.cpp
        op_host/max_pool3d_grad_with_argmax_tiling_splith.cpp
        op_host/max_pool3d_grad_with_argmax_tiling_splitd.cpp
        op_host/max_pool3d_grad_with_argmax_tiling_nosplit.cpp
        op_host/max_pool3d_grad_with_argmax_tiling_basesplit.cpp
        op_host/max_pool3d_grad_with_argmax_tiling_base.cpp
        op_host/max_pool3d_grad_with_argmax_scatter_tiling.cpp
        op_host/max_pool3d_grad_with_argmax_normal_tiling.cpp
        op_host/max_pool3d_grad_with_argmax_cutk_tiling.cpp
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
        op_host/max_pool3_d_grad_with_argmax_def.cpp
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
        op_host/max_pool3d_grad_with_argmax_l0.cpp
        op_host/aclnn_max_pool3d_with_argmax_backward.cpp
)

install(FILES op_host/aclnn_max_pool3d_with_argmax_backward.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)

# kernel
install(FILES op_kernel/max_pool3d_grad_with_argmax.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_common.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_cutk_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_cutk_d.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_cutk_dh.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_cutk_dhw.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_normal.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_nosplit.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_scatter_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_scatter_overlap.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_scatter.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_splitd.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_splith.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/max_pool3d_grad_with_argmax_splitw.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)