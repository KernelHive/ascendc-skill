### Pooling

## 输入

### x

- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：对输入x做Pooling，输出y
- **规格限制**：无

## 属性

### pool

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：pooling的类型，包括两种：
  - MAX：取值为0，默认为MAX
  - AVE：取值为1
- **规格限制**：支持配置0或1

### kernel_size

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：kernel大小
- **规格限制**：
  当global_pooling==False, pool==1，x_w(Padding后)>kernel_w时，有如下约束：
  - Filter_h*filter_w*32+feature_w*(31*stride_h+filter_w)<=32768
  - 满足kernel_size(kernel_h,kernel_w)配置范围在1~255

### kernel_h

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：kernel高。（在global_pooling==False时，kernel_size、kernel_h/kernel_w是二选一的，不能两者都提供）
- **规格限制**：
  当global_pooling==False, pool==1，x_w(Padding后)>kernel_w时，有如下约束：
  - Filter_h*filter_w*32+feature_w*(31*stride_h+filter_w)<=32768
  - 满足kernel_size（kernel_h,kernel_w）配置范围在1~255

### kernel_w

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：kernel宽
- **规格限制**：
  当global_pooling==False, pool==1，x_w(Padding后)>kernel_w时，有如下约束：
  - Filter_h*filter_w*32+feature_w*(31*stride_h+filter_w)<=32768
  - 满足kernel_size（kernel_h,kernel_w）配置范围在1~255

### stride

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：stride (在高和宽方向)（default=1）如果提供了Stride_h，则优先取值Stride_h和Stride_w
- **规格限制**：支持配置范围1~63，且stride<=2*kernel_size

### stride_h

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：stride高，默认值为1
- **规格限制**：支持配置范围1~63，且stride<=2*kernel_size

### stride_w

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：stride宽，默认值为1
- **规格限制**：支持配置范围1~63，且stride<=2*kernel_size

### pad

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：padding大小(在高和宽方向)（default=0），如果提供了Pad_h，则以Pad_h为准
- **规格限制**：pad<kernel

### pad_h

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：padding高(默认值 0)
- **规格限制**：pad<kernel

### pad_w

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：padding宽(默认值 0)
- **规格限制**：pad<kernel

### global_pooling

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：是否全平面做，默认为false
- **规格限制**：支持配置True或False

### round_mode

- **是否必填**：非必填
- **数据类型**：int
- **参数解释**：pooling ceil的模式
  - 0: DOMI_POOLING_CEIL，默认为0
  - 1: DOMI_POOLING_FLOOR
- **规格限制**：支持配置0或1

## 输出

### y

- **是否必填**：必填
- **数据类型**：float16
- **参数解释**：对输入x做Pooling，输出y
- **规格限制**：无
