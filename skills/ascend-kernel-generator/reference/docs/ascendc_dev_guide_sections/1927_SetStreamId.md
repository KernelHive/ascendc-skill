##### SetStreamId

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

设置某节点的 Stream ID，拥有相同 Stream ID 的节点将会在同一条流上依次执行。

## 函数原型

```cpp
graphStatus SetStreamId(const GNode &node, int64_t stream_id)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| node | 输入 | 图上节点 |
| stream_id | 输入 | 待设置的 Stream ID：<br>• 若为已申请的 stream id，直接设置即可<br>• 若需要新申请 Stream ID，请先调用 `AllocateNextStreamId` 接口申请，否则若 Stream ID 超出当前图上最大的 Stream ID 接口将返回失败 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | Status | SUCCESS：设置成功<br>FAILED：设置失败 |

## 约束说明

无
