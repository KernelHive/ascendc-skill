### ArgMax

## 输入

### x

- **是否必填**：必填
- **数据类型**：float32、float16
- **参数解释**：输入的Tensor
- **规格限制**：无

## 属性

### axis

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：指定在输入Tensor做削减的轴，如果不提供此参数，则每个batch求top_k
- **规格限制**：无

### out_max_val

- **是否必填**：非必填
- **数据类型**：bool
- **参数解释**：是否需要输出最大值
- **规格限制**：无

#### 行为说明

- 如果 `out_max_val` 为 `True`，且有2个top：
  - 如果有轴 `axis`，只输出每个轴的最大值
  - 如果没有轴 `axis`，则输出最大值的索引和最大值

```protobuf
layer {
  name: "argmax"
  type: "ArgMax"
  bottom: "data"
  top: "indices"
  top: "values"
  argmax_param {
    out_max_val: True
    top_k: 1
  }
}
```

- 如果 `out_max_val` 为 `True`，且有1个top，则输出最大值：

```protobuf
layer {
  name: "argmax"
  type: "ArgMax"
  bottom: "data"
  top: "values"
  argmax_param {
    out_max_val: True
    top_k: 1
    axis: 1
  }
}
```

- 如果 `out_max_val` 为 `False`，则输出每个轴的最大索引：

```protobuf
layer {
  name: "argmax"
  type: "ArgMax"
  bottom: "data"
  top: "indices"
  argmax_param {
    out_max_val: False
    top_k: 1
    axis: 1
  }
}
```

### top_k

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：默认是1，表示每个axis轴中前top_k个数（取值大于等于1），其取值范围必须在 `[1, x.shape(axis)]`，对应于caffe中的top_k
- **规格限制**：当前只支持1

## 输出

### indices

- **是否必填**：非必填
- **数据类型**：int32
- **参数解释**：输出的最大值的索引
- **规格限制**：无

### values

- **是否必填**：非必填
- **数据类型**：float32、float16
- **参数解释**：输出的Tensor，包含最大值索引或最大值
- **规格限制**：无
