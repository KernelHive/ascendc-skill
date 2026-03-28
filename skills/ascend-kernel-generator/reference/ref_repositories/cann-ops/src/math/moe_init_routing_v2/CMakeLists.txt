add_ops_compile_options(
        OP_NAME MoeInitRoutingV2
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
op_host/moe_init_routing_v2.cpp
)

target_sources(optiling PRIVATE
        op_host/moe_init_routing_v2.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/moe_init_routing_v2.cpp
)
  
install(FILES op_kernel/moe_init_routing_v2.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)