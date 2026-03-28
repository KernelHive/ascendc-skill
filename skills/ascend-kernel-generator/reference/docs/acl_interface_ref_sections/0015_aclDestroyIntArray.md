## aclDestroyIntArray

函数功能
销毁通过3.4 aclCreateIntArray接口创建的aclIntArray。

函数原型
aclnnStatus aclDestroyIntArray(const aclIntArray *array)

参数说明
参数名 输入/输出 说明

array 输入 需要销毁的aclIntArray。

返回值说明
返回0表示成功，返回其他值表示失败，返回码列表参见3.39 公共接口返回码。

约束与限制
无

调用示例
接口调用请参考3.4 aclCreateIntArray的调用示例。
