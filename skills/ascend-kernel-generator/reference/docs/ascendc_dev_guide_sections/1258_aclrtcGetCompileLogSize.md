##### aclrtcGetCompileLogSize

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 AI Core | × |
| Atlas 推理系列产品 Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

获取编译日志的大小。用于在 `aclrtcGetCompileLog` 获取日志内容时分配对应大小的内存空间。

## 函数原型

```c
aclError aclrtcGetCompileLogSize(aclrtcProg prog, size_t *logSizeRet)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| prog | 输入 | 运行时编译程序的句柄 |
| logSizeRet | 输出 | 编译日志的长度 |

## 返回值说明

`aclError` 为 `int` 类型变量，详细说明请参考 RTC 错误码。

## 约束说明

无

## 调用示例

```c
aclrtcProg prog;
size_t logSize;
aclError result = aclrtcGetCompileLogSize(prog, &logSize);
```
