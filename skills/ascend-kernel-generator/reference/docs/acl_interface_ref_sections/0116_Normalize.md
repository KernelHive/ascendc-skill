### Normalize

## 输入

### x1
- **是否必填**: 必填
- **数据类型**: float32、float16
- **参数解释**: 需要normalized的输入Tensor
- **规格限制**: 无

### x2
- **是否必填**: 非必填
- **数据类型**: float32、float16
- **参数解释**: normalized的scale输入Tensor，是一个N维的向量。如果channel_shared==True，则N=1；否则channel_shared==False，N=channel
- **规格限制**: 无

## 属性

### across_spatial
- **是否必填**: 非必填
- **数据类型**: bool
- **参数解释**: True表示在CHW维度做normalize，False表示只在C维度上做normalize，默认为True
- **规格限制**: True或者False

### channel_shared
- **是否必填**: 非必填
- **数据类型**: bool
- **参数解释**: 用于控制x2的参数，默认为True
- **规格限制**: True或者False

### eps
- **是否必填**: 非必填
- **数据类型**: 无
- **参数解释**: 为了防止除0，默认值1e-10
- **规格限制**: 无

## 输出

### y
- **是否必填**: 必填
- **数据类型**: float32、float16
- **参数解释**: 输出Tensor
- **规格限制**: 无
