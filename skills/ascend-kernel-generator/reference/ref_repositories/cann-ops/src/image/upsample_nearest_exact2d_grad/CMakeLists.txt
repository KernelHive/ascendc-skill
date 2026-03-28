add_ops_compile_options(
        OP_NAME UpSampleNearestExact2dGrad
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

# target_sources(op_host_aclnn PRIVATE
# op_host/upsample_nearest_exact2d_grad.cpp
# )

target_sources(optiling PRIVATE
        op_host/upsample_nearest_exact2d_grad_tiling.cpp
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

target_sources(opsproto PRIVATE

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
        op_host/upsample_nearest_exact2d_grad_def.cpp
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
        op_host/upsample_nearest_exact2d_grad_l0.cpp
        op_host/aclnn_upsample_nearest_1d_backward.cpp
        op_host/aclnn_upsample_nearest_2d_backward.cpp
        op_host/aclnn_upsample_nearest_exact1d_backward.cpp
        op_host/aclnn_upsample_nearest_exact2d_backward.cpp
)

install(FILES op_kernel/upsample_nearest_exact2d_grad.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES op_kernel/upsample_nearest_exact2d_grad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/op_host/aclnn_upsample_nearest_1d_backward.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/op_host/aclnn_upsample_nearest_2d_backward.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/op_host/aclnn_upsample_nearest_exact1d_backward.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/op_host/aclnn_upsample_nearest_exact2d_backward.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
