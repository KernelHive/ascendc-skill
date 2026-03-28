###### PostFakeMsg

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

通过 `AllocMessage` 接口获取到消息空间地址后，AIV发送假消息，刷新消息状态 `msgState` 为 `FAKE`。

当多个AIV的消息内容一致时，AIC仅需要读取一次位置靠前的第一个消息，通过将消息结构体中自定义的参数 `skipCnt` 设置为 n，通知AIC后续 n 条消息无需处理，直接跳过。被跳过的AIV需要使用本接口发送假消息，这被称之为消息合并机制或消息合并场景。

如下图所示，假设 Queue1、2、3 的第 0 条消息与 Queue0 的第 0 条消息相同，在消息合并场景中，从 AIC 视角来看，Queue0(0)、Queue4(0) 的消息会被处理，并根据用户自定义的消息内容完成相应的 AIC 上的计算。Queue1(0)、Queue2(0)、Queue3(0) 由于发了假消息，AIC将不会读取消息内容进行计算，直接释放消息。

**图 15-34 PostFakeMessage 示意图**

## 函数原型

```cpp
__aicore__ inline uint16_t PostFakeMsg(__gm__ CubeMsgType* msg)
```

## 参数说明

**表 15-453 接口参数说明**

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| msg | 输入 | 该 CubeResGroupHandle 中某个任务的消息空间地址。 |

## 返回值说明

当前消息空间与该消息队列队首空间的地址偏移。

## 约束说明

无

## 调用示例

```cpp
hanndle.AssignQueue(queIdx);
auto msgPtr = handle.AllocMessage(); // 获取消息空间指针msgPtr
auto offset = handle.PostFakeMsg(msgPtr); // 在msgPtr指针位置，发送假消息
```
