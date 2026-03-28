## acldvpp 返回码

调用DVPP算子API时，常见的接口返回码如下表所示。

| 状态码名称 | 状态码值 | 状态码说明 |
|------------|----------|------------|
| ACLDVPP_SUCCESS | 0 | 成功。 |
| ACLDVPP_ERR_PARAM_NULLPTR | 106001 | 参数校验错误，参数中存在非法的nullptr。 |
| ACLDVPP_ERR_PARAM_INVALID | 106002 | 参数校验错误，如输入的两个数据类型不满足输入类型推导关系。<br>详细的错误消息，可以通过aclGetRecentErrMsg接口获取（该接口在acl.h中）。 |
| ACLDVPP_ERR_UNINITIALIZE | 106101 | 执行aclDvpp算子接口前未调用aclDvppInit接口进行初始化。<br>需确保在执行各aclDvpp算子接口之前已调用过aclDvppInit接口。 |
| ACLDVPP_ERR_REPEAT_INITIALIZE | 106102 | 重复初始化。 |
| ACLDVPP_ERR_API_NOT_SUPPORT | 206001 | API接口不支持，请检查CANN版本与Driver包版本是否配套。 |
| ACLDVPP_ERR_RUNTIME_ERROR | 306001 | API内存调用npu runtime的接口异常。 |
| ACLDVPP_ERR_INNER_XXX | 506xxx | API内部发生内部异常，详细的错误消息，可以通过aclGetRecentErrMsg接口获取。 |
| ACLDVPP_ERR_INNER | 506000 | 请根据报错排查问题，或联系技术支持。<br>（您可以获取日志后单击Link联系技术支持。） |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 561101 | - |
| ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR | 561102 | - |
| ACLNN_ERR_INNER_NULLPTR | 561103 | - |
| ACL_ERROR_RT开头的返回码 | - | 请参见表2，查看返回码的详细说明。 |
