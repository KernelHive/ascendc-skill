##### operator[]

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

获取距原 LocalTensor 起始地址偏移量为 offset 的新 LocalTensor，注意 offset 不能超过原有 LocalTensor 的 size 大小。

## 函数原型

```cpp
__aicore__ inline LocalTensor operator[](const uint32_t offset) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| offset | 输入 | 偏移量，单位为元素 |

## 返回值说明

返回距原 LocalTensor 起始地址偏移量为 offset 的新 LocalTensor。

## 约束说明

无

## 调用示例

参考调用示例。
