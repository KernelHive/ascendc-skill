###### GetBufHandle

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

## 功能说明

根据 offset 取出自定义 TBufPool 中管理的内存块。

## 函数原型

```cpp
__aicore__ inline TBufHandle GetBufHandle(uint8_t offset)
```

## 参数说明

**表 参数说明**

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| offset | 输入 | 内存块的 index 值 |

## 约束说明

offset 值不可超出 `EXTERN_IMPL_BUFPOOL` 中定义的 `BUFID_SIZE` 大小，用户可根据需要添加校验。

## 返回值说明

指定的内存块，类型为 `TBufHandle`。

## 调用示例

请参考调用示例。
