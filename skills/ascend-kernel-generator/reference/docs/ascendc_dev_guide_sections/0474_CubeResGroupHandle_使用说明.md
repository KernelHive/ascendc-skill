###### CubeResGroupHandle 使用说明

CubeResGroupHandle 用于在分离模式下对 AI Core 计算资源分组。分组后，开发者可以对不同的分组指定不同的计算任务。

一个 AI Core 分组可包含多个 AIV 和 AIC，AIV 和 AIC 之间采取 Client 和 Server 架构进行任务处理：

- AIV 为 Client，每一个 Cube 计算任务为一个消息，AIV 发送消息至消息队列。
- AIC 作为 Server，遍历消息队列的消息，根据消息类型及内容执行对应的计算任务。

一个 CubeResGroupHandle 中可以有一个或多个 AIC，同一个 AIC 只能属于一个 CubeResGroupHandle。AIV 无此限制，即同一个 AIV 可以属于多个 CubeResGroupHandle。

如下图所示，CubeResGroupHandle1 中有 2 个 AIC，10 个 AIV，AIC 为 Block0 和 Block1。其中 Block0 与 Queue0、Queue1、Queue2、Queue3、Queue4 进行通信，Block1 与 Queue5、Queue6、Queue7、Queue8、Queue9 进行通信。每一个消息队列对应一个 AIV，消息队列的深度固定为 4，即一次性最多可以容纳 4 个消息。CubeResGroupHandle2 的消息队列个数为 12，表明有 12 个 AIV。CubeResGroupHandle 的消息处理顺序如 CubeResGroupHandle2 中黑色箭头所示。

> 图 15-31 基于 CubeResGroupHandle 的 AI Core 计算资源分组通信示意图

## 实现步骤

基于 CubeResGroupHandle 实现 AI Core 计算资源分组步骤如下：

1. 创建 AIC 上所需要的计算对象类型。
2. 创建通信区域描述 KfcWorkspace，用于记录通信消息 Msg 的地址分配。
3. 自定义消息结构体，用于通信。
4. 自定义回调计算结构体，根据实际业务场景实现 Init 函数和 Call 函数。
5. 创建 CubeResGroupHandle。
6. 绑定 AIV 到 CubeResGroupHandle。
7. 收发消息。
8. AIV 退出消息队列。

> 下文仅提供示例代码片段，更多完整样例请参考 CubeGroup 样例。

### 步骤 1：创建 AIC 上所需要的计算对象类型

用户根据实际需求，自定义 AIC 所需要的计算对象类型，或者高阶 API 已提供的 Matmul 类型。

例如，创建 Matmul 类型如下，其中 `A_TYPE`、`B_TYPE`、`C_TYPE`、`BIAS_TYPE`、`CFG_NORM` 等含义请参考 Matmul 模板参数。

```cpp
// A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM 根据实际需求场景构造
using MatmulApiType = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, C_TYPE, CFG_NORM>;
```

### 步骤 2：创建 KfcWorkspace

使用 KfcWorkspace 管理不同 CubeResGroupHandle 的消息通信区的划分。

```cpp
// 创建 KfcWorkspace 对象前，需要对该 workspaceGM 清零
KfcWorkspace desc(workspaceGM);
```

### 步骤 3：自定义消息结构体

用户需要自行构造消息结构体 `CubeMsgBody`，用于 AIV 向 AIC 发送通信消息。构造的 `CubeMsgBody` 必须 64 字节对齐，该结构体最前面需要定义 2 字节的 `CubeGroupMsgHead`，使消息收发机制正常运行。`CubeGroupMsgHead` 结构定义请参考表 15-442。除 2 字节的 `CubeGroupMsgHead` 外，其余参数根据业务需求自行构造。

**表 15-441 CubeMsgBody 消息结构体**

| 参数名称 | 含义 |
|----------|------|
| CubeMsgBody | 用户自定义的消息结构体。结构体名称可自定义，结构体大小需要 64 字节对齐。 |

```cpp
// 这里提供 64B 对齐的结构体示例，用户实际使用时，除 CubeGroupMsgHead 外，其他参数个数及参数类型可自行构造
struct CubeMsgBody {
    CubeGroupMsgHead head; // 2B，需放在结构体最前面, 自定义的 CubeMsgBody 中，CubeGroupMsgHead 的变量名需设置为 head，否则会编译报错。
    uint8_t funcID;
    uint8_t skipCnt;
    uint32_t value;
    bool isTransA;
    bool isTransB;
    bool isAtomic;
    bool isLast;
    int32_t tailM;
    int32_t tailN;
    int32_t tailK;
    uint64_t aAddr;
    uint64_t bAddr;
    uint64_t cAddr;
    uint64_t aGap;
    uint64_t bGap;
};
```

**表 15-442 CubeGroupMsgHead 结构体参数定义**

| 参数名 | 含义 |
|--------|------|
| msgState | 表明该位置的消息状态。参数取值如下：<br>• `CubeMsgState::FREE`：表明该位置还未填写消息，可执行 `AllocMessage`。<br>• `CubeMsgState::VALID`：表明该位置已经含有 AIV 发送的消息，待 AIC 接收执行。<br>• `CubeMsgState::QUIT`：表明该位置的消息为通知 AIC 有 AIV 将退出流程。<br>• `CubeMsgState::FAKE`：表明该位置的消息为假消息。在消息合并场景，被跳过处理任务的 AIV 需要发送假消息，消息合并场景请参考 `PostFakeMsg` 中的介绍。 |
| aivID | 发送消息的 AIV 的序号。 |

### 步骤 4：自定义回调计算结构体

根据实际业务场景实现 `Init` 函数和 `Call` 函数。

```cpp
template<class MatmulApiCfg, class CubeMsgBody>
struct NormalCallbackFuncs {
    __aicore__ inline static void Call(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle){
        // 用户自行实现逻辑
    };

    __aicore__ inline static void Init(NormalCallbackFuncs<MatmulApiCfg, CubeMsgBody> &foo, MatmulApiCfg &mm, GM_ADDR tilingGM){
        // 用户自行实现逻辑
    };
};
```

**表 15-443 模板参数说明**

| 参数 | 说明 |
|------|------|
| MatmulApiCfg | 用户自定义的 AIC 上计算所需要对象的数据类型，参考步骤 1，该模板参数必须填入。 |
| CubeMsgBody | 用户自定义的消息结构体，该模板参数必须填入。 |

用户自定义回调计算结构体中需要包含固定的 `Init` 函数和 `Call` 函数，函数原型如下所示。

```cpp
// 该函数的参数和名称为固定格式，函数实现根据业务逻辑自行实现。
__aicore__ inline static void Init(MyCallbackFunc<MatmulApiCfg, CubeMsgBody> &myCallBack, MatmulApiCfg &mm, GM_ADDR tilingGM){
    // 用户自行实现内部逻辑
}
```

**表 15-444 Init 函数参数说明**

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| myCallBack | 输入 | 用户自定义的带模板参数的回调计算结构体。 |
| mm | 输入 | AIC 上计算对象，多为 Matmul 对象。 |
| tilingGM | 输入 | 用户传入的 tiling 指针。 |

```cpp
// 该函数的参数和名称为固定格式，函数实现根据业务逻辑自行实现。
__aicore__ inline static void Call(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle){
    // 用户自行实现内部逻辑
}
```

**表 15-445 Call 函数参数说明**

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| mm | 输入 | AIC 上计算对象，多为 Matmul 对象。 |
| rcvMsg | 输入 | 用户自定义的消息结构体指针。 |
| handle | 输入 | 分组管理 Handle，用户调用其接口进行收发消息，释放消息等。 |

某算子的回调计算结构体的代码示例如下：

```cpp
// 用户自定义的回调计算逻辑
template<class MatmulApiCfg, typename CubeMsgBody>
struct MyCallbackFunc
{
    template<int32_t funcId>
    __aicore__ inline static typename IsEqual<funcId, 0>::Type CubeGroupCallBack(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
    {
        GlobalTensor<int64_t> msgGlobal;
        msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *> (rcvMsg) + sizeof(int64_t));
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(msgGlobal);
        using SrcAT = typename MatmulApiCfg::AType::T;
        auto skipNum = 0;
        for (int i = 0; i < skipNum + 1; ++i)
        {
            auto tmpId = handle.FreeMessage(rcvMsg + i); // msgPtr process is complete
        }
        handle.SetSkipMsg(skipNum);
    }

    template<int32_t funcId>
    __aicore__ inline static typename IsEqual<funcId, 1>::Type CubeGroupCallBack(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
    {
        GlobalTensor<int64_t> msgGlobal;
        msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *> (rcvMsg) + sizeof(int64_t));
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(msgGlobal);
        using SrcAT = typename MatmulApiCfg::AType::T;
        LocalTensor<SrcAT> tensor_temp;
        auto skipNum = 3;
        auto tmpId = handle.FreeMessage(rcvMsg, CubeMsgState::VALID);
        for (int i = 1; i < skipNum + 1; ++i)
        {
            auto tmpId = handle.FreeMessage(rcvMsg + i, CubeMsgState::FAKE);
        }
        handle.SetSkipMsg(skipNum); // notify the cube not to process
    }

    __aicore__ inline static void Call(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
    {
        if (rcvMsg->funcId == 0)
        {
            CubeGroupCallBack<0> (mm, rcvMsg, handle);
        }
        else if(rcvMsg->funcId == 1)
        {
            CubeGroupCallBack<1> (mm, rcvMsg, handle);
        }
    }

    __aicore__ inline static void Init(MyCallbackFunc<MatmulApiCfg, CubeMsgBody> &foo, MatmulApiCfg &mm, GM_ADDR tilingGM)
    {
        auto tempTilingGM = (__gm__ uint32_t*)tilingGM;
        auto tempTiling = (uint32_t*)&(foo.tiling);
        for (int i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i, ++tempTilingGM, ++tempTiling)
        {
            *tempTiling = *tempTilingGM;
        }
        mm.SetSubBlockIdx(0);
        mm.Init(&foo.tiling, GetTPipePtr());
    }
    TCubeTiling tiling;
};
```

### 步骤 5：创建 CubeResGroupHandle

用户使用 `CreateCubeResGroup` 接口创建一个或多个 CubeResGroupHandle。

```cpp
/*
 * groupID 为用户自定义的 CreateCubeResGroup 的 groupID
 * MatmulApiType 为步骤 1 定义好的 AIC 上计算对象的类型
 * MyCallbackFunc 为步骤 4 定义好的自定义回调计算结构体
 * CubeMsgBody 为步骤 3 中的自定义消息结构体
 * desc 为步骤 2 中的用户初始化好的通信区域描述
 * groupID 为 1，blockStart 为 0，blockSize 为 12，msgQueueSize 为 48，tilingGm 为指针，存储了用户在 AIC 上所需要的 tiling 信息
 */
auto handle = AscendC::CreateCubeResGroup<groupID, MatmulApiType, MyCallbackFunc, CubeMsgBody>(desc, 0, 12, 48, tilingGM);
```

### 步骤 6：绑定 AIV 到 CubeResGroupHandle

绑定 AIV 和消息队列序号。注意：消息队列序号 `queIdx` 小于该 CubeGroupHandle 的消息队列总数，每个 AIV 需要传入不同的 `queIdx`。

```cpp
// handle 为步骤 5 中 CreateCubeResGroup 创建的 CubeResGroupHandle 对象
handle.AssignQueue(queIdx);
```

### 步骤 7：AIV 发消息

用户调用 `AllocMessage`、`PostMessage` 等接口进行消息的收发。其中，调用 `AllocMessage` 获取消息结构体指针，通过 `PostMessage` 发送消息，在消息合并场景调用 `PostFakeMessage` 发送假消息，示例如下。

```cpp
CubeGroupMsgHead head = {CubeMsgState::VALID, (uint8_t)queIdx};
CubeMsgBody aCubeMsgBody {head, 0, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0};
CubeMsgBody bCubeMsgBody {head, 1, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0};

auto offset = 0;
if (GetBlockIdx() == 0)
{
    auto msgPtr = handle.template AllocMessage(); // alloc for queue space
    offset = handle.template PostMessage(msgPtr, bCubeMsgBody); // post true msgPtr
    bool waitState = handle.template Wait<true> (offset); // wait until the msgPtr is proscessed
}
else if (GetBlockIdx() < 4)
{
    auto msgPtr = handle.AllocMessage();
    offset = handle.PostFakeMsg(msgPtr); // post fake msgPtr
    bool waitState = handle.template Wait<true> (offset); // wait until the msgPtr is proscessed
}
else
{
    auto msgPtr = handle.template AllocMessage();
    offset = handle.template PostMessage(msgPtr, aCubeMsgBody);
    bool waitState = handle.template Wait<true> (offset); // wait until the msgPtr is proscessed
}
```

### 步骤 8：AIV 退出消息队列

调用 `AllocMessage` 获取消息结构体指针后，通过 `SendQuitMsg` 发送当前消息队列退出。

```cpp
auto msgPtr = handle.AllocMessage(); // 获取消息空间指针 msgPtr
handle.SetQuit(msgPtr); // 发送退出消息
```
