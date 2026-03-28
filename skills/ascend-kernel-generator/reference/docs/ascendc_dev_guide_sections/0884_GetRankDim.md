###### GetRankDim

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

获取通信域内卡的数量。该接口默认在所有核上工作，用户也可以在调用前通过 `GetBlockIdx` 指定其在某一个核上运行。

## 函数原型

```cpp
__aicore__ inline uint32_t GetRankDim()
```

## 参数说明

无

## 返回值说明

通信域内卡的数量。

## 约束说明

无

## 调用示例

请参见 `GetWindowsInAddr` 的调用示例。
