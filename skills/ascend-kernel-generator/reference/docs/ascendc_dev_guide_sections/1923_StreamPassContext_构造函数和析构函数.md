##### StreamPassContext 构造函数和析构函数

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

StreamPassContext 构造函数和析构函数。

## 函数原型

```cpp
explicit StreamPassContext(int64_t current_max_stream_id)
~StreamPassContext() override = default
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| current_max_stream_id | 输入 | 当前图中最大的 Stream ID。 |

## 返回值说明

无

## 约束说明

无
