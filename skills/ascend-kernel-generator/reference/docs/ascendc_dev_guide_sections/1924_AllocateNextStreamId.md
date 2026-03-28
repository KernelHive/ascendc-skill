##### AllocateNextStreamId

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

申请新的 Stream ID，当希望申请新逻辑流的时候，需要调用该接口。

## 函数原型

```cpp
int64_t AllocateNextStreamId()
```

## 参数说明

无

## 返回值说明

新 Stream ID。

## 约束说明

无
