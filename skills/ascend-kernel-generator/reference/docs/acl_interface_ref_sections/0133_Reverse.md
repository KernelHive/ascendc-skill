### Reverse

## 输入

### x

- **是否必填**: 必填
- **数据类型**: float16、float32、int8、uint8、int16、uint16、int32、uint32、int64、uint64
- **参数解释**: 输入Tensor
- **规格限制**: 无

## 属性

### axis

- **是否必填**: 必填
- **数据类型**: int
- **参数解释**: 是一个向量，表示反向的轴，取值范围在 [-rank(x), rank(x))
- **规格限制**: [-rank(x), rank(x))

## 输出

### y

- **是否必填**: 必填
- **数据类型**: float16、float32、int8、uint8、int16、uint16、int32、uint32、int64、uint64
- **参数解释**: 输出Tensor，与输入Tensor的数据类型和Shape相同
- **规格限制**: 无
