###### DataTypeForBinQuery

## 功能说明

设置 Input/Output 用于运行时算子二进制查找的数据类型，与 DataType/DataTypeList 的数量一致，且一一对应。

算子编译过程中，会根据数据类型生成多个 .o 文件，并通过这些数据类型在运行时索引算子二进制。某些算子支持多种数据类型，且对数据类型不敏感，这时可以使用该接口，将多种数据类型映射到同一个算子二进制，多个数据类型可以复用一个 .o 文件，从而减少二进制文件的生成。

例如，如果一个算子的输入支持多种数据类型（`ge::DT_INT16` 和 `ge::DT_INT32`），并且使用 `ge::DT_INT16` 输入时可以复用 `ge::DT_INT32` 的二进制文件而不影响最终结果，那么可以采用如下配置：

```cpp
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_INT16, ge::DT_INT32})
    .DataTypeForBinQuery({ge::DT_INT32, ge::DT_INT32})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
```

这样，只需生成一个目标文件（.o），就能实现对多种数据类型的支持。

## 函数原型

```cpp
OpParamDef &DataTypeForBinQuery(std::vector<ge::DataType> types)
```

## 参数说明

| 参数   | 输入/输出 | 说明                                                         |
| ------ | --------- | ------------------------------------------------------------ |
| `types` | 输入      | 算子参数数据类型，`ge::DataType` 请参考 15.2.3.58 DataType。 |

## 返回值说明

`OpParamDef` 算子定义，`OpParamDef` 请参考 15.1.6.1.4 OpParamDef。

## 约束说明

- `DataTypeForBinQuery` 的参数个数需要和当前算子参数的 `DataType` 或者 `DataTypeList` 的参数个数保持一致。
- 不支持与 `To`（指定数据类型）的接口同时使用。
- 需要保证使用 `DataTypeForBinQuery` 后，产生新的算子参数属性集合（使用 `DataTypeForBinQuery` 替换原本 `DataType` 序列）存在于原本支持的参数属性集合中。

参数属性集合的定义为：算子所支持的所有参数的属性的集合，相当于一列参数的集合。

例如示例一中，算子支持四种原集合，没有重复：

1. `x : DT_FLOAT16, FORMAT_ND` `y : DT_INT16, FORMAT_ND`
2. `x : DT_FLOAT, FORMAT_ND` `y : DT_INT16, FORMAT_ND`
3. `x : DT_INT16, FORMAT_ND` `y : DT_INT16, FORMAT_ND`
4. `x : DT_INT32, FORMAT_ND` `y : DT_INT16, FORMAT_NC`

使用 `DataTypeForBinQuery` 替换原本 `DataType` 序列后，新集合为：

1. `x : DT_INT16, FORMAT_ND` `y : DT_INT16, FORMAT_ND`
2. `x : DT_FLOAT16, FORMAT_ND` `y : DT_INT16, FORMAT_ND`
3. `x : DT_FLOAT16, FORMAT_ND` `y : DT_INT16, FORMAT_ND`
4. `x : DT_INT16, FORMAT_ND` `y : DT_INT16, FORMAT_NC`

此时发现，新集合 1 与原集合 3 一致，新集合 2、新集合 3 与原集合 1 一致，设置生效。新集合 4 不属于原集合，设置失效，此时会按照原本的集合 4 进行编译。

## 调用示例

### 示例一

```cpp
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT16, ge::DT_INT32})
    .DataTypeForBinQuery({ge::DT_INT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, ge::DT_INT16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NC});
```

如下图所示，没有设置 `DataTypeForBinQuery` 之前，会生成 4 个二进制。通过上述代码设置 `DataTypeForBinQuery` 后：

- 替换后第 1 列使用原来第 3 列的二进制，第 2 列和第 3 列使用原来第 1 列的二进制。第 4 列仍使用第 4 列的二进制。
- 替换后，第 2 列和第 3 列完全一致，达成二进制复用的效果，算子总二进制会由原来的四个（bin1，bin2，bin3，bin4）缩减至现在的三个（bin1，bin3、bin4）。

### 示例二

```cpp
// 简单用例，此时会有两对复用，1、2列->1列，3、4列->4列。总共生成1、4两个二进制。所有支持的DataType会传入这两个二进制运行。
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT16, ge::DT_INT32})
    .DataTypeForBinQuery({ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
```

### 示例三

```cpp
// 复杂用例，可以多个Input/Output同时使用DataTypeForBinQuery，此时也会产生两对复用。1、2列->2列，3、4列->1列。总共生成1、2两个二进制。所有支持的DataType会传入这两个二进制运行。
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT16, ge::DT_INT32})
    .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

this->Input("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16})
    .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

this->Output("z")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT16})
    .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
```
