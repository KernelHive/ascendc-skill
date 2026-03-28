add_ops_compile_options(
        OP_NAME ClipByValueV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
                -g
                -DASCENDC_TRACE_ON
)

target_sources(op_host_aclnn PRIVATE
op_host/clip_by_value_v2.cpp
)

target_sources(optiling PRIVATE
        op_host/clip_by_value_v2.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/clip_by_value_v2.cpp
)

install(FILES op_kernel/clip_by_value_v2.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)