###### SetHcclContext

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品 AI Core | √ |
| Atlas 推理系列产品 Vector Core | √ |
| Atlas 训练系列产品 | x |

## 功能说明

设置通算融合算子每个通信域对应的 context（消息区）地址。

## 函数原型

```cpp
template <uint32_t index>
__aicore__ inline void SetHcclContext(__gm__ uint8_t* context)
```

## 参数说明

**表 参数说明**

| 参数名 | 描述 |
|--------|------|
| index | 模板参数，用来表示要设置的通信域 ID，当前只支持 2 个通信域，index 只能为 0/1。 |
| context | 对应通信域的 context（消息区）地址。 |

## 返回值说明

无

## 约束说明

当前最多只支持 2 个通信域。

## 调用示例

```cpp
// 给 GROUP_0 设置消息区地址
AscendC::SetHcclContext<0>(contextGM);
```
