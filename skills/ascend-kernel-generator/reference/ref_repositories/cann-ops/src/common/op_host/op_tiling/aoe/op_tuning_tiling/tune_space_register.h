/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TUNE_SPACE_REGISTER_H
#define TUNE_SPACE_REGISTER_H

#include <iostream>
#include <memory>
#include <map>
#include <list>
#include <algorithm>
#include "tune_space.h"
#include "common_utils.h"

namespace OpTuneSpace {
using CreateFunction = std::shared_ptr<TuneSpace>(*)();
// define tune space manager class
class TuneSpaceManager {
public:
    // singleton factory
    static TuneSpaceManager* GetInstance()
    {
        static TuneSpaceManager instance;
        return &instance;
    }

    bool Regist(std::string typeName, CreateFunction func)
    {
        if (!func) {
            return false;
        }
        std::map<std::string, CreateFunction>::const_iterator iter = createMap.find(typeName);
        if (iter == createMap.end()) {
            createMap.insert(make_pair(typeName, func));
        }
        return true;
    }

    std::shared_ptr<TuneSpace> CreateObject(const std::string& typeName)
    {
        if (typeName.empty()){
            return nullptr;
        }
        std::map<std::string, CreateFunction>::const_iterator iter = createMap.find(typeName);
        if (iter == createMap.end()) {
            return nullptr;
        }
        return iter->second();
    }

private:
    std::map<std::string, CreateFunction> createMap;
    TuneSpaceManager() {}
    ~TuneSpaceManager() {}
    TuneSpaceManager(const TuneSpaceManager&) = delete;
    TuneSpaceManager& operator=(const TuneSpaceManager&) = delete;
}; // TuneSpaceManager

// Register Tool
class RegisterClassAction {
public:
    RegisterClassAction(const std::string& className, CreateFunction func)
    {
        TuneSpaceManager::GetInstance()->Regist(className, func);
    }
    ~RegisterClassAction() {}
};

#define TUNE_SPACE_REGISTER(type, clazz) \
    static std::shared_ptr<TuneSpace> Creator_##type##_Class() \
    { \
        std::shared_ptr<clazz> ptr = std::make_shared<clazz>(); \
        return std::shared_ptr<TuneSpace>(ptr); \
    }   \
    RegisterClassAction g_##type##_Class_Creator(#type, Creator_##type##_Class)
}   // namespace OpTuneSpace
#endif