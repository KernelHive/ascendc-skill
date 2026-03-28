###### ValueDepend

## 功能说明

标识该输入是否为“数据依赖输入”。数据依赖输入是指在 Tiling/InferShape 等函数实现时依赖该输入的具体数据。该输入数据为 host 侧数据，开发者在 Tiling 函数/InferShape 函数中可以通过以下方式获取这个输入数据：

- TilingContext 类的 `GetInputTensor`（参考 15.2.2.35.3）
- InferShapeContext 类的 `GetInputTensor`（参考 15.2.2.14.3）

## 函数原型

```cpp
OpParamDef &ValueDepend(Option value_depend)
OpParamDef &ValueDepend(Option value_depend, DependScope scope)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| value_depend | 输入 | value_depend 有以下两种取值：<br>● **REQUIRED**：表示算子的输入必须是 Const 类型。在调用算子的 SetCheckSupport 时，会校验算子的输入是否是 Const 类型。若校验通过，则将此输入的值下发到算子；否则报错。<br>● **OPTIONAL**：表示算子的输入可以是 Const 类型，也可以不是 Const 类型。如果输入是 Const 类型，则将输入的值下发到算子，否则不下发。 |
| scope | 输入 | scope 类型为枚举类型 DependScope，支持的取值为：<br>● **ALL**：指在 Tiling/InferShape 等函数实现时都依赖该输入的具体数据，行为与调用单参数 ValueDepend 重载接口一致。<br>● **TILING**：指仅在 Tiling 时依赖 Tensor 的值，可以支持 Tiling 下沉。 |

## 返回值说明

返回 OpParamDef 算子定义（OpParamDef 请参考 15.1.6.1.4）。

## 约束说明

- 仅支持对算子输入配置
- 仅支持输入的参数数据类型配置为：
  - DT_INT64
  - DT_FLOAT
  - DT_BOOL
- 对应生成的输出类型分别为：
  - aclIntArray
  - aclFloatArray
  - aclBoolArray
（均为 aclnn 数据类型）
