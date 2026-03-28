### Concat

## 输入

### x

- **是否必填**: 必填
- **数据类型**: float16、float32、int8、int16、int32、int64、uint8、uint16、uint32、uint64
- **参数解释**: 将多个Tensor按照指定轴合并为一个Tensor。其中x为变长的tensor list
- **规格限制**: bottom num最大为16

## 属性

### axis

- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 默认为1，表示哪个维度concat，可以为负数
- **规格限制**: 支持配置，需要在维度范围之内

### concat_dim

- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 默认为1，跟axis含义相同
- **规格限制**: 跟axis含义相同，不支持负值

## 输出

### y

- **是否必填**: 必填
- **数据类型**: float16、float32、int8、int16、int32、int64、uint8、uint16、uint32、uint64
- **参数解释**: 将多个Tensor按照指定轴合并为一个Tensor。其中x为变长的tensor list
- **规格限制**: 无

---

## 输入

### x

- **是否必填**: 必填
- **数据类型**: float16
- **参数解释**: 对输入x做卷积
- **规格限制**: 无

### filter

- **是否必填**: 必填
- **数据类型**: float16
- **参数解释**: 卷积核，DepthwiseConv和ConvolutionDepthwise都通过转换为Convolution算子来使用，例如ConvolutionDepthwise算子：

```protobuf
layer {
  name: "resx1_conv2"
  type: "ConvolutionDepthwise"
  bottom: "resx1_conv1"
  top: "resx1_conv2"
  convolution_param {
    num_output: 54
    kernel_size: 3
    stride: 2
    pad: 1
    bias_term: false
    weight_filler {
      type: "msra"
    }
  }
}
```

需要修改为：

```protobuf
layer {
  name: "resx1_conv2"
  type: "Convolution"
  bottom: "resx1_conv1"
  top: "resx1_conv2"
  convolution_param {
    num_output: 54
    group: 54
    kernel_size: 3
    stride: 2
    pad: 1
    bias_term: false
    weight_filler {
      type: "msra"
    }
  }
}
```

如果DepthwiseConv和ConvolutionDepthwise中已携带group属性，则只修改type为Convolution即可；如果没有携带group属性，则需添加group属性：`group == num_output`（group、output_channel和input_channel必须相同）

**说明**：修改后的模型文件（.prototxt）和权重文件（.caffemodel）的op name、op type必须保持名称一致（包括大小写）。

- **规格限制**:
  - Filter_W和Filter_h配置范围在1~255

### bias

- **是否必填**: 非必填
- **数据类型**: float16
- **参数解释**: bias，可以为None（表示不需要bias）
- **规格限制**: 无

## 属性

### pad

- **是否必填**: 非必填
- **数据类型**: ListInt
- **参数解释**: 高和宽轴开始和结束的Padding，pad和pad_h/pad_w不能同时提供，默认值为0；pad的List长度最大为2
- **规格限制**: pad < filter.size

### pad_h

- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 高轴开始和结束的Padding，pad和pad_h/pad_w不能同时提供，默认值为0；pad的List长度最大为2
- **规格限制**: pad < filter.size

### pad_w

- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 宽轴开始和结束的Padding，pad和pad_h/pad_w不能同时提供，默认值为0；pad的List长度最大为2
- **规格限制**: pad < filter.size

### stride

- **是否必填**: 非必填
- **数据类型**: ListInt
- **参数解释**: 高和宽轴的stride。stride和stride_h/stride_w不能同时提供，默认值为1；stride的List长度最大为2
- **规格限制**: 1 <= stride <= 63

### stride_h

- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 高轴的stride。stride和stride_h/stride_w不能同时提供，默认值为1；stride的List长度最大为2
- **规格限制**: 1 <= stride <= 63

### stride_w

- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 宽轴的stride。stride和stride_h/stride_w不能同时提供，默认值为1；stride的List长度最大为2
- **规格限制**: 1 <= stride <= 63

### dilation

- **是否必填**: 非必填
- **数据类型**: ListInt
- **参数解释**: Filter的高和宽轴的放大系数，List长度最大为2，默认值为1
- **规格限制**: 支持1~255，配置后 `(kernel-1)*dilation+1 < 256`

### group

- **是否必填**: 非必填，默认值为1
- **数据类型**: int
- **参数解释**: 无
- **规格限制**: group能被channel整除，且feature map的input channel = group * weight的input channel

## 输出

### y

- **是否必填**: 必填
- **数据类型**: float16
- **参数解释**: 无
- **规格限制**: 无
