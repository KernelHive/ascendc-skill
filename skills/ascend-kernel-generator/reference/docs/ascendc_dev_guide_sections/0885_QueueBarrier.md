###### QueueBarrier

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

阻塞服务端上指定队列的BatchWrite通信任务，直到指定范围内所有队列上的任务完成执行，从而实现指定范围内队列的同步。

## 函数原型

```cpp
template <ScopeType type = ScopeType::ALL>
__aicore__ inline void QueueBarrier(uint16_t queueID)
```

## 参数说明

### 模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| type | 输入 | 表示阻塞服务端的通信任务范围。类型为ScopeType，默认值为ScopeType::ALL。当前参数仅支持取值为ScopeType::ALL。<br><br>ScopeType的定义如下：<br>```cpp<br>enum class ScopeType: uint8_t {<br>    ALL,        // 阻塞所有队列上的通信任务<br>    QUEUE,      // 暂不支持<br>    BLOCK,      // 暂不支持<br>    INVALID_TYPE // 暂不支持<br>};<br>``` |

### 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| queueID | 输入 | 表示队列ID |

## 返回值说明

无

## 约束说明

无

## 调用示例

请参见BatchWrite的调用示例。
