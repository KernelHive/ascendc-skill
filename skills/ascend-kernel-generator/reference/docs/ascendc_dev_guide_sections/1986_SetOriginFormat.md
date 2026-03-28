##### SetOriginFormat

函数功能
设置Tensor的原始Format。

该Format是指原始网络模型的Format。

函数原型
graphStatus SetOriginFormat(const ge::Format &format)

参数说明
参数名 输入/输出 描述

format 输入 需设置的原始Format值。
关于Format类型，请参见15.2.3.59 Format。

返回值
graphStatus类型：设置成功返回GRAPH_SUCCESS，否则，返回GRAPH_FAILED。

异常处理
无。

约束说明
无。
