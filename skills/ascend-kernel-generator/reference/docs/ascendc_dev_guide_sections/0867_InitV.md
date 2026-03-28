###### InitV

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

Hccl客户端初始化接口。该接口默认在所有核上工作，用户也可以在调用前通过 `GetBlockIdx` 指定其在某一个核上运行。

## 函数原型

```cpp
__aicore__ inline void InitV2(GM_ADDR context, const void *initTiling)
```

## 参数说明

**表 15-873 接口参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| context | 输入 | 通信上下文，包含rankDim，rankID等相关信息。通过框架提供的获取通信上下文的接口 `GetHcclContext` 获取 context。 |
| initTiling | 输入 | 通信域初始化 `Mc2InitTiling` 的地址。`Mc2InitTiling` 在 Host 侧计算得出，具体请参考表1 `Mc2InitTiling` 参数说明，由框架传递到 Kernel 函数中使用。 |

## 返回值说明

无

## 约束说明

- 本接口必须与 `SetCcTilingV2` 接口配合使用。
- 调用本接口时，必须使用标准 C++ 语法定义 `TilingData` 结构体的开发方式，具体请参考《Ascend C算子开发指南》中的"算子实现 > 工程化算子开发 > Host 侧 tiling 实现 > 使用标准 C++ 语法定义 Tiling 结构体"。
- 调用本接口传入的 `initTiling` 参数，不能使用 Global Memory 地址，建议通过 `GET_TILING_DATA_WITH_STRUCT` 接口获取 `TilingData` 的栈地址。
- 本接口不支持使用相同的 context 初始化多个 Hccl 对象。

## 调用示例

用户自定义 `TilingData` 结构体：

```cpp
class UserCustomTilingData {
    AscendC::tiling::Mc2InitTiling initTiling;
    AscendC::tiling::Mc2CcTiling tiling;
    CustomTiling param;
};
```

在所有核上创建 Hccl 对象，并调用 `InitV2` 接口初始化：

```cpp
extern "C" __global__ __aicore__ void userKernel(GM_ADDR aGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
    REGISTER_TILING_DEFAULT(UserCustomTilingData);
    GET_TILING_DATA_WITH_STRUCT(UserCustomTilingData,tilingData,tilingGM);

    GM_ADDR contextGM = AscendC::GetHcclContext<0>();
    Hccl hccl;
    hccl.InitV2(contextGM, &tilingData);
    hccl.SetCcTilingV2(offsetof(UserCustomTilingData, tiling));

    // 调用Hccl的Prepare、Commit、Wait、Finalize接口
}
```
