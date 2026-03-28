##### GetStreamId

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

获取节点上当前的 Stream ID，其结果表示经过内置流分配算法以后该节点被分配的 Stream ID。

## 函数原型

```cpp
int64_t GetStreamId(const GNode &node) const;
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| node | 输入 | 图上节点 |

## 返回值说明

节点所属的 Stream ID。

## 约束说明

无
