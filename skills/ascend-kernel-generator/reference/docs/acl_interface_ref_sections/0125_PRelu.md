### PRelu

## 输入

### x

- **是否必填**: 必填
- **数据类型**: int8、uint8、float16、float32
- **参数解释**: 激活函数，y = weight × min(x, 0) + max(x, 0)
- **规格限制**: float16

### weight

- **是否必填**: 必填
- **数据类型**: int8、uint8、float16、float32
- **参数解释**: 只根据 weight 的维度区分是否为 channel_shared 场景：
  - 如果 weight 只有一个数，则为 channel_shared = True
  - 如果 weight 是 channel 维度，则为 channel_shared = False
- **规格限制**: 支持 float16，是 channel 维度的向量或者标量

## 输出

### y

- **是否必填**: 必填
- **数据类型**: int8、uint8、float16、float32
- **参数解释**: 无
- **规格限制**: float16
