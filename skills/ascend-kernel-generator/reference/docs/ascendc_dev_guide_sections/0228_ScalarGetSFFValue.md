##### ScalarGetSFFValue

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

获取一个 `uint64_t` 类型数字的二进制表示中从最低有效位开始的第一个 0 或 1 出现的位置，如果没找到则返回 -1。

## 函数原型

```cpp
template <int countValue>
__aicore__ inline int64_t ScalarGetSFFValue(uint64_t valueIn)
```

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| countValue | 指定要查找的值，0 表示查找第一个 0 的位置，1 表示查找第一个 1 的位置，数据类型是 int，只能输入 0 或 1。 |

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| valueIn | 输入 | 输入数据，数据类型是 `uint64_t`。 |

## 返回值说明

`int64_t` 类型的数，`valueIn` 中第一个 0 或 1 出现的位置。

## 约束说明

无。

## 调用示例

```cpp
uint64_t valueIn = 28;
// 输出数据 oneCount：2
int64_t oneCount = AscendC::ScalarGetSFFValue<1>(valueIn);
```
