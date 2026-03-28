###### SetQuit

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

通过 `AllocMessage` 接口获取到消息空间地址后，发送退出消息，告知该消息队列对应的 AIC 无需处理该队列的消息。如下图，Queue5 对应的 AIV 发了退出消息后，Block1 将不再处理 Queue5 的任何消息。

![消息队列退出示意图](图 15-35 消息队列退出示意图)

## 函数原型

```cpp
__aicore__ inline void SetQuit(__gm__ CubeMsgType* msg)
```

## 参数说明

**表 15-454 接口参数说明**

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| msg | 输入 | 该 CubeResGroupHandle 中的消息空间地址。 |

## 返回值说明

无。

## 约束说明

无

## 调用示例

```cpp
handle.AssignQueue(queIdx);
auto msgPtr = a.AllocMessage(); // 获取消息空间指针 msgPtr
handle.SetQuit(msgPtr); // 发送退出消息
```
