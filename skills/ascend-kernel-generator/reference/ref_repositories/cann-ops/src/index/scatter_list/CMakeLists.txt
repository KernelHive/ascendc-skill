add_ops_compile_options(
        OP_NAME ScatterList
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

# 自动生成的aclnn
target_sources(op_host_aclnn PRIVATE
        op_host/scatter_list_def.cpp
)

# optiling
target_sources(optiling PRIVATE
        op_host/scatter_list.cpp
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
# target_sources(op_host_aclnnInner PRIVATE
#         op_host/scatter_list_def.cpp
# )

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

install(FILES op_kernel/scatter_list.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_base.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_neg_dim2_large.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_neg_large_e.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_neg_large.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_neg_more.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_neg_small.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_rlble_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_rlble.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_rlbse_pad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_rlbse.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_rsble.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_rsbse.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_transpose_large.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_transpose_more.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_kernel/scatter_list_transpose_small.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)