###### TransDataSpecial

支持的产品型号
Atlas 训练系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas 推理系列产品

功能说明
该函数不改变tensor数据的值，实现对tensor数据格式的转换，功能与 TransData类
似。

具体功能是将用户输入tensor的format转换为指定的dstPrimaryFormat。

函数原型
const aclTensor *TransDataSpecial(const aclTensor *x, op::Format
dstPrimaryFormat, int64_t groups, aclOpExecutor *executor)

参数说明
参数 输入/输出 说明

x 输入 待转换的tensor。数据类型支持FLOAT16、FLOAT32、
INT32、UINT32、INT8、UINT8。


参数 输入/输出 说明

dstPrimaryF 输入 输入tensor要转换的目标format。
ormat

groups 输入 分组参数，用于分组转换时传入。数据类型支持
INT64。

executor 输入 op执行器，包含了算子计算流程。

返回值说明
返回数据格式为dstPrimaryFormat的tensor。

约束说明
当输入tensor数据类型为FLOAT32、INT32、UINT32时，C0只能按照16处理。

调用示例
// 固定写法，创建OpExecutor
auto uniqueExecutor = CREATE_EXECUTOR();
// 将gradOutputReFormat的format转换为NC1HWC0
auto gradOutputTransData = l0op::TransDataSpecial(gradOutputReFormat,
op::Format::FORMAT_NC1HWC0, 0,uniqueExecutor.get());
