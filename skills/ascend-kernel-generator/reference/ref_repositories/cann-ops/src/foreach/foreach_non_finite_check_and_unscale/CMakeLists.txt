add_ops_compile_options(
        OP_NAME ForeachNonFiniteCheckAndUnscale
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
op_host/foreach_non_finite_check_and_unscale.cpp
)

target_sources(optiling PRIVATE
        op_host/foreach_non_finite_check_and_unscale_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/foreach_non_finite_check_and_unscale.cpp
)

install(FILES op_kernel/foreach_non_finite_check_and_unscale.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)