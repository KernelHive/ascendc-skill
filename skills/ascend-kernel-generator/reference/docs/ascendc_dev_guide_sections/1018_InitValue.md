###### InitValue

## 功能说明

通过该 host 侧接口设置算子输出的初始值，设置后会在算子执行前对算子输出的 GM 空间进行清零操作或者插入 memset 类算子进行初始值设置。

InitValue 和 SetNeedAtomic 接口配合使用，SetNeedAtomic 接口需要配置为 true。

## 函数原型

- **OpParamDef &InitValue(uint64_t value)**  
  在算子执行前，对输出参数对应 GM 空间进行清零。

- **OpParamDef &InitValue(const ScalarVar &value)**  
  指定输出参数初值的类型和值，输出参数调用该接口，会在算子执行前，对输出参数对应 GM 空间插入对应类型和值的 memset 类算子。

- **OpParamDef &InitValue(const std::vector<ScalarVar> &value)**  
  指定输出参数初值类型和值的列表，依次对应输出参数的数据类型和数据格式组合，会在算子执行前，对输出参数对应 GM 空间插入对应类型和值的 memset 类算子。

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| value | 输入 | - **uint64_t 类型参数**：仅支持输入 0，输出参数调用该接口，会在算子执行前，对输出参数对应 GM 空间进行清零。<br>- **ScalarVar 类型参数**：ScalarVar 用于指定输出参数初值的类型 ScalarType 和值 ScalarNum，具体定义如下：<br>  ```cpp<br>  enum class ScalarType : uint32_t {<br>    UINT64 = 0,<br>    INT64 = 1,<br>    UINT32 = 2,<br>    INT32 = 3,<br>    UINT16 = 4,<br>    INT16 = 5,<br>    UINT8 = 6,<br>    INT8 = 7,<br>    FLOAT32 = 8,<br>    FLOAT16 = 9,<br>    INVALID_DTYPE = static_cast<uint32_t>(-1),<br>  };<br><br>  union ScalarNum {<br>    uint64_t value_u64;<br>    int64_t value_i64;<br>    float value_f32;<br>    ScalarNum() : value_u64(0) {}<br>    explicit ScalarNum(uint64_t value) : value_u64(value) {}<br>    explicit ScalarNum(int64_t value) : value_i64(value) {}<br>    explicit ScalarNum(float value) : value_f32(value) {}<br>  };<br><br>  struct ScalarVar {<br>    ScalarType scalar_type;<br>    ScalarNum scalar_num;<br>    ScalarVar();<br>    ScalarVar(ScalarType type, uint64_t num);<br>    ScalarVar(ScalarType type, int64_t num);<br>    ScalarVar(ScalarType type, int num);<br>    ScalarVar(ScalarType type, unsigned int num);<br>    ScalarVar(ScalarType type, float num);<br>    ScalarVar(ScalarType type, double num);<br>    bool operator==(const ScalarVar& other) const;<br>  };<br>  ```<br>  - ScalarType 当前仅支持：UINT64/INT64/UINT32/INT32/UINT16/INT16/UINT8/INT8/FLOAT32/FLOAT16<br>  - ScalarNum 支持 uint64_t/int64_t/float 类型<br>  - 为方便使用，ScalarVar 也支持立即数初始化，示例：<br>    ```cpp<br>    InitValue({ScalarType::INT16, 1});<br>    ```<br>- **const std::vector<ScalarVar> &value 类型**：指定输出参数初值类型和值的列表，依次对应输出参数的数据类型和数据格式组合。 |

## 返回值说明

返回 OpParamDef 算子定义，OpParamDef 请参考 15.1.6.1.4 OpParamDef。

## 约束说明

- InitValue 和 SetNeedAtomic 接口配合使用，否则会出现初始化不生效的情况。
- 针对如下产品型号：
  - Atlas A3 训练系列产品 / Atlas A3 推理系列产品
  - Atlas A2 训练系列产品 / Atlas A2 推理系列产品
  - Atlas 200I/500 A2 推理产品
  - Atlas 推理系列产品
  - Atlas 训练系列产品

  插入 memset 类算子时，仅在入图场景下支持初始值设置任意值，单算子 API 执行的场景下仅支持清零。

- 针对 `OpParamDef &InitValue(uint64_t value)` 接口，算子输出参数的数据类型支持范围如下：UINT64/INT64/UINT32/INT32/UINT16/INT16/UINT8/INT8/FLOAT32/FLOAT16，超出该范围为未定义行为。

- 针对 `OpParamDef &InitValue(const std::vector<ScalarVar> &value)` 接口，输入 value 的 size 需要与输出参数配置的 DataType 或 DataTypeList 接口参数的 size 一致。同时，相同数据类型需保证设置的类型和值相同，否则将会报错。

- 对于同一个输出参数仅支持调用一种接口设置初值，调用多种 InitValue 接口为未定义行为；多次调用同一种接口以最后一次调用设置的初值为准。

- 基于旧版本 CANN 包（不支持 InitValue 特性）生成的自定义算子工程，无法兼容 InitValue 接口。在使用非当前版本 CANN 包生成的自定义算子工程时，需特别注意兼容性问题。您可以通过查看自定义算子工程下 `cmake/util/ascendc_impl_build.py` 中有无 `output_init_value` 字段来确认当前工程是否支持该特性，如果未找到该字段，则需要重新生成自定义算子工程以启用 InitValue 特性。

## 调用示例

```cpp
// OpParamDef &InitValue(uint64_t value)示例
this->Output("z")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
    .FormatList({ge::FORMAT_ND})
    .InitValue(0);
```

```cpp
// OpParamDef &InitValue(const ScalarVar &value)示例
this->Output("z")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
    .FormatList({ge::FORMAT_ND})
    .InitValue({ScalarType::INT16, 1});
```

```cpp
// OpParamDef &InitValue(const std::vector<ScalarVar> &value)示例
this->Output("z")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
    .FormatList({ge::FORMAT_ND})
    .InitValue({{ScalarType::INT16, 1}, {ScalarType::FLOAT32, 3.2}, {ScalarType::INT64, 7}});

this->Output("z")
    .ParamType(REQUIRED)
    .DataType({ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT32})  // 第一个和第三个 DataType 相同
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NHWC})
    .InitValue({{ScalarType::INT16, 1}, {ScalarType::FLOAT32, 3.2}, {ScalarType::INT16, 1}});  // InitValue 对应的数据类型和数值也需相同
```
