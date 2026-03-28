###### CubeResGroupHandle 构造函数

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 AI Core | × |
| Atlas 推理系列产品 Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

构造 `CubeResGroupHandle` 对象，完成组内的 AIC 和消息队列分配。构造 `CubeResGroupHandle` 对象时需要传入模板参数 `CubeMsgType`，`CubeMsgType` 是由用户定义的消息结构体，请参考表 15-441。

使用此接口需要用户自主管理地址、同步事件等，因此更推荐使用 `CreateCubeResGroup` 接口快速创建 `CubeResGroupHandle` 对象。

## 函数原型

```cpp
template <typename CubeMsgType>
class CubeResGroupHandle;

__aicore__ inline CubeResGroupHandle() = default;

__aicore__ inline CubeResGroupHandle(
    GM_ADDR workspace,
    uint8_t blockStart,
    uint8_t blockSize,
    uint8_t msgQueueSize,
    uint8_t evtIDIn
);
```

## 参数说明

**表 15-446 CubeResGroupHandle 参数说明**

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| workspace | 输入 | 该 CubeResGroupHandle 的消息通讯区在 GM 上的起始地址 |
| blockStart | 输入 | 该 CubeResGroupHandle 在 AIV 视角下起始 AIC 对应的序号，即 AIC 的起始序号 × 2。例如，如果 AIC 起始序号为 0，则填入 0×2；如果为 1，则填入 1×2 |
| blockSize | 输入 | 该 CubeResGroupHandle 在 AIV 视角下分配的 Block 个数，即实际的 AIC 个数 × 2 |
| msgQueueSize | 输入 | 该 CubeResGroupHandle 分配的消息队列总数 |
| evtIDIn | 输入 | 通信框架内用于 AIV 侧消息的同步事件 |

如下图所示，`CubeResGroupHandle1` 的 `blockStart` 为 4，`blockSize` 为 4，表示起始的 AIC 序号为 2（即 `blockStart / 2`）；AIC 数量为 2（即 `blockSize / 2`）。`msgQueueSize` 为 10，表示消息队列个数为 10，每个 Block 分配的消息队列个数为 `Ceil(msgQueueSize, blockSize/2)`，Block2 和 Block3 分配到的消息队列个数均为 5。`CubeResGroupHandle2` 的 `msgQueueSize` 数量为 11，最后一个 Block 只能分配 5 个消息队列。

**图 15-32 Block 和消息队列映射示意图**

## 约束说明

- 假设芯片的 AIV 核数为 x，那么 `blockStart + blockSize <= x - 1`，`msgQueueSize <= x`
- 每个 AIC 至少被分配 1 个消息队列 `msgQueue`
- `blockStart` 和 `blockSize` 必须为偶数
- 使用该接口，UB 空间末尾的 `1600B + sizeof(CubeMsgType)` 将被占用
- 1 个 AIC 只能属于 1 个 `CubeGroupHandle`，即多个 `CubeGroupHandle` 的 `[blockStart / 2, blockStart / 2 + blockSize / 2]` 区间不能重叠
- 不能和 `REGIST_MATMUL_OBJ` 接口同时使用。使用资源管理 API 时，用户自主管理 AIC 和 AIV 的核间通信，`REGIST_MATMUL_OBJ` 内部是由框架管理 AIC 和 AIV 的核间通信，同时使用可能会导致通信消息错误等异常

## 调用示例

```cpp
uint8_t blockStart = 4;
uint8_t blockSize = 4;
uint8_t msgQueueSize = 10;
uint8_t evtIDIn = 0; // 用户自行管理事件 ID
AscendC::KfcWorkspace desc(workspace); // 用户自行管理的 workspace 指针
AscendC::CubeResGroupHandle<CubeMsgBody> handle;
handle = AscendC::CubeResGroupHandle<MatmulApiType, MyCallbackFunc, CubeMsgBody>(
    desc.GetMsgStart(), blockStart, blockSize, msgQueueSize, evtIDIn
);
```
