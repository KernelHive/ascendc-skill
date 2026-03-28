##### aclrtcGetBinDataSize

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |

| 产品 | 是否支持 |
|------|----------|
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

获取编译的二进制数据大小。用于在 `aclrtcGetBinData` 获取二进制数据时分配对应大小的内存空间。

## 函数原型

```c
aclError aclrtcGetBinDataSize(aclrtcProg prog, size_t *binDataSizeRet)
```

## 参数说明

**表 15-1039 接口参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| prog | 输入 | 运行时编译程序的句柄 |
| binDataSizeRet | 输出 | 编译得到的Device侧二进制数据大小 |

## 返回值说明

`aclError` 为 int 类型变量，详细说明请参考 15.1.8.1.9 RTC错误码。

## 约束说明

无

## 调用示例

```c
aclrtcProg prog;
size_t binDataSizeRet = 0;
aclError result = aclrtcGetBinDataSize(prog, &binDataSizeRet);
```
