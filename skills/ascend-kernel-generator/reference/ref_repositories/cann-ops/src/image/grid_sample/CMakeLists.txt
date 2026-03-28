add_ops_compile_options(
        OP_NAME GridSample
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

# target_sources(op_host_aclnn PRIVATE
# op_host/upsample_nearest3d.cpp
# )

target_sources(optiling PRIVATE
        op_host/grid_sample.cpp
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
         op_host/grid_sample.cpp
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
        op_host/grid_sample.cpp
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
        op_host/grid_sample_l0.cpp
        op_host/aclnn_grid_sample.cpp
)

install(FILES op_kernel/grid_sample.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
install(FILES op_kernel/grid_sample_2d_bicubic.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_2d_common.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_2d_fp16_slide_window_310b.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_2d_fp16_slide_window.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_2d_fullLoad_310p.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_2d_fullLoad.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_2d.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_2d_nearest.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_2d_slide_window_310p.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_2d_slide_window.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_3d_common.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_3d.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_3d_nearest.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_3d_portrait.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
        install(FILES op_kernel/grid_sample_common.h
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES op_host/aclnn_grid_sample.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)