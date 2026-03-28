###### SetLocalWorkspace

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 AI Core | √ |
| Atlas 推理系列产品 Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

对于某些场景，Matmul 内部需要额外占用 VECCALC 空间。如果用户希望在算子中复用这个额外占用的 VECCALC 空间，则该空间需要用户预留，并申请好 LocalTensor，将其起始物理地址传入给 Matmul。

具体需要申请的 VECCALC 临时空间大小由 tiling 接口 `MatmulGetTmpBufSize` 给出。满足以下几个条件之一就需要使用该接口传入 UB 临时空间：

- C 矩阵 Position 为 `TPosition::GM`
- C 矩阵 CubeFormat 为 `CubeFormat::ND`
- A 矩阵或者 B 矩阵 CubeFormat 为 `CubeFormat::ND`
- 存在 Bias 且 Bias 的 Position 不是 VECCALC

请在 `Iterate` 或者 `IterateAll` 之前调用该接口。

获取到的 UB 临时空间大小以字节为单位。

## 函数原型

```cpp
__aicore__ inline void SetLocalWorkspace(const LocalTensor<uint8_t>& tmpBuffer)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| tmpBuffer | 输入 | 临时空间，由用户申请并管理，TPosition 为 VECCALC |

## 返回值说明

无

## 约束说明

当使能 MixDualMaster（双主模式）场景时，即模板参数 `enableMixDualMaster` 设置为 `true`，不支持使用该接口。

## 调用示例

```cpp
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetLocalWorkspace(mmFormatUb); // 设置临时 VECCALC 空间
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
