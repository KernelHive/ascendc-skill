/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file main.cpp
 */
#include <acl/acl.h>
#include "atb/atb_infer.h"
#include "aclnn_eye_operation.h"
#include "securec.h"

#include <fstream>
#include <random>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
namespace common{
    struct InputData{
        void* data;
        uint64_t size;
    };

    aclError CheckAcl(aclError ret) {
        if (ret != ACL_ERROR_NONE) {
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << ret << std::endl;
        }
        return ret;
    }

    void* ReadBinFile(const std::string filename, size_t& size) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return nullptr;
        }
        // 获取文件大小
        size = file.tellg();
        file.seekg(0, std::ios::beg);
        // 分配内存
        void* buffer;
        int ret = aclrtMallocHost(&buffer,size);
        if (!buffer) {
            std::cerr << "内存分配失败" << std::endl;
            file.close();
            return nullptr;
        }
        // 读取文件内容到内存
        file.read(static_cast<char*>(buffer), size);
        if (!file) {
            std::cerr << "读取文件失败" << std::endl;
            delete[] static_cast<char*>(buffer);
            file.close();
            return nullptr;
        }
        file.close();
        return buffer;
    }

    static void SetCurrentDevice()
    {
        const int deviceId = 0;
        std::cout << "[INFO]: aclrtSetDevice " << deviceId << std::endl;
        int ret = aclrtSetDevice(deviceId);
        if (ret != 0) {
            std::cout << "[ERROR]: aclrtSetDevice fail, error:" << ret << std::endl;
            return;
        }
        std::cout << "[INFO]: aclrtSetDevice success" << std::endl;
    }

    static void FreeTensor(atb::Tensor &tensor)
    {
        if (tensor.deviceData) {
            int ret = aclrtFree(tensor.deviceData);
            if (ret != 0) {
                std::cout << "[ERROR]: aclrtFree fail" << std::endl;
            }
            tensor.deviceData = nullptr;
            tensor.dataSize = 0;
        }
        if (tensor.hostData) {
            int ret = aclrtFreeHost(tensor.hostData);
            if (ret != 0) {
                std::cout << "[ERROR]: aclrtFreeHost fail, ret = " << ret << std::endl;
            }
            tensor.hostData = nullptr;
            tensor.dataSize = 0;
        }
    }

    static void FreeTensors(atb::SVector<atb::Tensor> &inTensors)
    {
        for (size_t i = 0; i < inTensors.size(); ++i) {
            FreeTensor(inTensors.at(i));
        }
    }
    static bool SaveMemoryToBinFile(void* memoryAddress, size_t memorySize, size_t i) {
        // 生成文件名
        std::string filename = "script/output/output_" + std::to_string(i) + ".bin";
        // 打开文件以二进制写入模式
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        // 写入数据
        file.write(static_cast<const char*>(memoryAddress), memorySize);
        if (!file) {
            std::cerr << "写入文件时出错: " << filename << std::endl;
            file.close();
            return false;
        }
        // 关闭文件
        file.close();
        std::cout << "数据已成功保存到: " << filename << std::endl;
        return true;
    }    
}
using namespace common;
int main(int argc, const char *argv[])
{
    // 初始化
    const int deviceId = 0;
    std::cout << "[INFO]: aclrtSetDevice " << deviceId << std::endl;
    int ret = aclrtSetDevice(deviceId);
    if (ret != 0) {
        std::cout << "[ERROR]: aclrtSetDevice fail, error:" << ret << std::endl;
        return 1;
    }
    std::cout << "[INFO]: aclrtSetDevice success" << std::endl;
    atb::Context *context = nullptr;
    ret = atb::CreateContext(&context);
    void *stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != 0) {
        std::cout << "[ERROR]: AsdRtStreamCreate fail, ret:" << ret << std::endl;
        return 1;
    }
    context->SetExecuteStream(stream);
    // 读取输入文件
    std::vector<InputData> input;
    std::string xPath = "./script/input/input0.bin";
    InputData inputX;
    inputX.data = ReadBinFile(xPath,inputX.size);
    input.push_back(inputX);
    // 创建算子
    EyeAttrParam eyeAttrParam;
    eyeAttrParam.num_rows = 133; // 设置行数为133 
    eyeAttrParam.num_columns = 4095; // 设置列数为4095
    std::vector<int64_t> batchShape = {3,4};
    eyeAttrParam.batch_shape = aclCreateIntArray(batchShape.data(),batchShape.size());
    eyeAttrParam.dtype = 0;
    EyeOperation *op = new EyeOperation("Eye",eyeAttrParam);
    std::cout << "[INFO]: complete CreateOp!" << std::endl;
    if(input.size() != op->GetInputNum()){
        std::cout << "[ERROR]: Operation actual input num is not equal to GetInputNum()";
    } 
    // 创建输入描述
    atb::SVector<atb::TensorDesc> intensorDescs;
    atb::SVector<atb::TensorDesc> outtensorDescs;
    intensorDescs.resize(op->GetInputNum());
    outtensorDescs.resize(op->GetOutputNum());
    atb::TensorDesc xDesc;
    xDesc.dtype = ACL_FLOAT;
    xDesc.format = ACL_FORMAT_ND;
    constexpr int INPUT_FIRST_DIM_NUM = 4; // 第一个输入是四维tensor
    constexpr int INPUT_FIRST_DIM_FIRST = 3; // 第一个输入的第一个维度为3 
    constexpr int INPUT_FIRST_DIM_SECOND = 4; // 第一个输入的第二个维度为4
    constexpr int INPUT_FIRST_DIM_THIRD = 133; // 第一个输入的第三个维度为133
    constexpr int INPUT_FIRST_DIM_FOURTH = 4095; // 第一个输入的第四个维度为4095
    constexpr int DIMENSION_INDEX_FIRST = 0; // 第一个输入的第一个维度
    constexpr int DIMENSION_INDEX_SECOND = 1; // 第一个输入的第二个维度 
    constexpr int DIMENSION_INDEX_THIRD = 2; // 第一个输入的第三个维度 
    constexpr int DIMENSION_INDEX_FOURTH = 3; // 第一个输入的第四个维度 
    xDesc.shape.dimNum = INPUT_FIRST_DIM_NUM;  
    xDesc.shape.dims[DIMENSION_INDEX_FIRST] = INPUT_FIRST_DIM_FIRST; 
    xDesc.shape.dims[DIMENSION_INDEX_SECOND] = INPUT_FIRST_DIM_SECOND; 
    xDesc.shape.dims[DIMENSION_INDEX_THIRD] = INPUT_FIRST_DIM_THIRD; // 第一个输入的第三个维度为133 
    xDesc.shape.dims[DIMENSION_INDEX_FOURTH] = INPUT_FIRST_DIM_FOURTH; // 第一个输入的第四个维度为4095 
    intensorDescs.at(0) = xDesc;
    // 算子InferShape推导
    atb::Status st = op->InferShape(intensorDescs,outtensorDescs);
    if (st != 0) {
        std::cout << "[ERROR]: Operation InferShape fail" << std::endl;
        return -1;
    }
    std::cout << "[INFO]: Operation InferShape success" << std::endl;
    atb::VariantPack variantPack;
    variantPack.inTensors.resize(op->GetInputNum());
    for(size_t i=0;i<op->GetInputNum();i++){
        variantPack.inTensors.at(i).desc = intensorDescs.at(i);
        variantPack.inTensors.at(i).hostData = input[i].data;
        variantPack.inTensors.at(i).dataSize = input[i].size;
        CheckAcl(aclrtMalloc(&variantPack.inTensors.at(i).deviceData, input[i].size, ACL_MEM_MALLOC_HUGE_FIRST));
        CheckAcl(aclrtMemcpy(variantPack.inTensors.at(i).deviceData, input[i].size, input[i].data, input[i].size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    std::cout << "[INFO]: Operation Input prepare sucess" << std::endl;
    uint64_t workspaceSize = 0;
    st = op->Setup(variantPack, workspaceSize, context);
    if (st != 0) {
        std::cout << "[ERROR]: Operation setup fail" << std::endl;
        return -1;
    }
    std::cout << "[INFO]: Operation setup success" << std::endl;
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    std::cout << "[INFO]: Operation execute start" << std::endl;
    st = op->Execute(variantPack, (uint8_t*)workspace, workspaceSize, context);
    if (st != 0) {
        std::cout << "[ERROR]: Operation execute fail" << std::endl;
        return -1;
    }
    std::cout << "[INFO]: Operation execute success" << std::endl;
    ret = aclrtSynchronizeStream(stream);
    CheckAcl(aclrtMemcpy(variantPack.inTensors.at(0).hostData, variantPack.inTensors.at(0).dataSize, variantPack.inTensors.at(0).deviceData,
    variantPack.inTensors.at(0).dataSize, ACL_MEMCPY_DEVICE_TO_HOST));
    SaveMemoryToBinFile(variantPack.inTensors.at(0).hostData,variantPack.inTensors.at(0).dataSize,0);
    FreeTensors(variantPack.inTensors);
    st = atb::DestroyContext(context);
    CheckAcl(aclrtDestroyStream(stream));
    CheckAcl(aclrtResetDevice(0));
    CheckAcl(aclFinalize());
    return atb::ErrorType::NO_ERROR;
}
