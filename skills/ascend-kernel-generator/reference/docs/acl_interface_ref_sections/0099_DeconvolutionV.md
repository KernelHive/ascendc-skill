### DeconvolutionV

## 输入

### input_data
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：目标检测领域的特征图
- **规格限制**：
  - 输入：1个输入
  - 约束如下：
    - Kw = weight[2]
    - Kh = weight[3]
    - stride = strides[0]
    - pad = pads[0]
    - dilation = dilations[0] = 1
    - dilation仅支持1
    - K - stride - 2 * pad = 0
    - 0 <= (Kw - 1 - pad) // stride <= 255
    - 0 <= (Kh - 1 - pad) // stride <= 255
    - Kh - pad - 1 >= 0
    - Kw - pad - 1 >= 0

#### group = 1
- Kw * Kh * 16 * 3 * unitsize + stride * ceil(input_w, 16.0) * 16 * 2 <= 245760
- ceil(input_w / 8.0) * 8 * 16 * unitsize <= 245760
- ceil(input_w, 16) * 16 * ks2 * unitsize <= 65536
- 16 * ks2 * 16 * unitsize <= 65536
- stride * ceil(input_w, 16.0) * 16 * unitsize <= 262144
- Kw * Kh * 16 * 16 + 2 * ks * input_w * 16 <= 1048576
- Kw = Kh，Kw * Kh % 16 = 0，Kw % stride = 0

#### group = channel
- ceil(input_w, 16) * 16 * ks2 * unitsize <= 65536
- 16 * ks2 * 16 * unitsize <= 65536
- stride * ceil(input_w, 16.0) * 16 * unitsize <= 262144
- Kw * Kh * 256 * unitsize + 2 * ks * w * 16 * unitsize <= 1048576
- Kw * Kh * 16 * 3 * unitsize + stride * ceil(input_w, 16.0) * 16 * 2 + 543 <= 245760
- Kw * Kh * 16 * 3 * unitsize + 16 * Kw * Kh * 16 * unitsize + 543 <= 245760
- Kw = Kh，Kw * Kh % 16 = 0，Kw % stride = 0

### 说明
- unitsize为2代表fp16占2个字节。
- ks为kernelsize与stride的比值。
- ks2为ks的平方。
- KB为千字节。

### weight
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：反卷积权重
- **规格限制**：请参见input_data的规格限制

### bias
- **是否必填**：非必填
- **数据类型**：float16
- **参数解释**：偏置
- **规格限制**：请参见input_data的规格限制

## 属性

### strides
- **是否必填**：必填
- **数据类型**：listInt
- **参数解释**：反卷积步长
- **规格限制**：请参见input_data的规格限制

### pads
- **是否必填**：必填
- **数据类型**：listInt
- **参数解释**：特征图pad值
- **规格限制**：请参见input_data的规格限制

### dilations
- **是否必填**：非必填
- **数据类型**：listInt
- **参数解释**：膨胀系数
- **规格限制**：请参见input_data的规格限制

### groups
- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：分组反卷积的组数
- **规格限制**：请参见input_data的规格限制

### data_format
- **是否必填**：非必填
- **数据类型**：str
- **参数解释**：数据排布格式
- **规格限制**：请参见input_data的规格限制

### offset_x
- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：特征图偏移
- **规格限制**：请参见input_data的规格限制

## 输出

### featuremap
- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：两个输入的featuremap的相似性
- **规格限制**：NCHW
