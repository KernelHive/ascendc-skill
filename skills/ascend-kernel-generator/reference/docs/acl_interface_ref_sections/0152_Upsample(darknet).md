### Upsample(darknet)

## 输入

### x

- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: `y[N,C,i,j] = scale * x[N,C,i/stride_h,j/stride_w]`，其中  
  `0 <= i <= stride_h * x.h - 1`，`0 <= j <= stride_w * x.w - 1`
- **规格限制**: float16

## 属性

### stride

- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 如果同时有 stride、stride_h 和 stride_w，以 stride 的值为准
- **规格限制**: 无

### stride_h

- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 默认值为 2，h 方向放大的比例
- **规格限制**: 大于 1

### stride_w

- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 默认值为 2，w 方向放大的比例
- **规格限制**: 大于 1

### scale

- **是否必填**: 非必填
- **数据类型**: float
- **参数解释**: 默认值为 1，输出值的比例系数
- **规格限制**: 无

## 输出

### y

- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: `y[N,C,i,j] = scale * x[N,C,i/stride_h,j/stride_w]`，其中  
  `0 <= i <= stride_h * x.h - 1`，`0 <= j <= stride_w * x.w - 1`
- **规格限制**: float16
