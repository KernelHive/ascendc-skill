/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_framework_util.h
 * \brief 
 */

#ifndef CONV2D_FRAMEWORK_H
#define CONV2D_FRAMEWORK_H

#include "kernel_utils.h"
using namespace AscendC;

#ifdef ASC_OP_DEBUG_TEST
#define ASC_OP_LOGD(fmt, ...)       \
    do {                            \
        printf(fmt, ##__VA_ARGS__); \
    } while (0)
#else
#define ASC_OP_LOGD(fmt, ...)
#endif

#define CONV_DECLARE_REG_IMPL(MEMBER, args...) namespace __AuxCheckImpl {                                             \
    template<typename T, typename U>                                                                                  \
    struct _has_impl_##MEMBER {                                                                                       \
        template<typename C>                                                                                          \
        static auto check(int) -> decltype(std::declval<typename C::template MEMBER<U, 0, ##args>>(), TrueType());    \
        template<typename C>                                                                                          \
        static FalseType check(...);                                                                                  \
        enum { value = IsSameType<decltype(check<T>(0)), TrueType>::value };                                          \
    };                                                                                                                \
}

#define CONV_REG_IMPL(Config, Current, MEMBER)                                                                        \
    template<bool Default, class T>                                                                                   \
    struct __##MEMBER { using Type = typename Current::MEMBER<Intf, ImplType>; };                         \
    template<class T>                                                                                                 \
    struct __##MEMBER<true, T> { using Type = typename T::template MEMBER<Intf, ImplType>; };             \
    using MEMBER = typename __##MEMBER<Current::__AuxCheckImpl::_has_impl_##MEMBER<Config, Intf>::value, Config>::Type

// 接口调用校验
#define CONV_DECLARE_CHECK_FUN(T, NAMESPACE)                                                                          \
    template<class... Ts>                                                                                             \
    static __aicore__ inline NAMESPACE::TypeFalse call(Intf* self, Ts... args) { return (NAMESPACE::TypeFalse){0}; }

#define CONV_CHECK_FUN(T, NAMESPACE, ... )                                                                            \
    (!IsSameType<decltype(T::call(__VA_ARGS__)), NAMESPACE::TypeFalse>::value)

#define CONV_CHECK_FUN_TEMPLATE(T, NAMESPACE, ARG, ...)                                                               \
    (!IsSameType<decltype(T::template call<ARG>(__VA_ARGS__)), NAMESPACE::TypeFalse>::value)

// 检查是否定义成员变量， 或者函数
#define CONV_DECLARE_CHECK_MEMBER(MEMBER) namespace __AuxCheck {                                                      \
    template <typename T>                                                                                             \
    struct _has_member_##MEMBER {                                                                                     \
        template<typename U>                                                                                          \
        static void check(decltype(&U::MEMBER));                                                                      \
        template<typename U>                                                                                          \
        static int check(...);                                                                                        \
        enum { value = IsSameType<decltype(check<T>(0)), void>::value };                                              \
    };                                                                                                                \
}

#define CONV_CHECK_MEMBER(OBJ, MEMBER) (__AuxCheck::_has_member_##MEMBER<typename OBJ>::value)

#define CONV_DECLARE_DEFINE_MEMBER(OBJ, MEMBER, TYPE, DEFAULT)                                                        \
    template<typename T>                                                                                              \
    constexpr TYPE _get_##MEMBER##_value(TYPE value)                                                                  \
    {                                                                                                                 \
        if constexpr (T::MEMBER > DEFAULT) {                                                                          \
            return T::MEMBER;                                                                                         \
        } else {                                                                                                      \
            return value;                                                                                             \
        }                                                                                                             \
    }                                                                                                                 \

#define CONV_DEFINE_MEMBER(OBJ, MEMBER, DEFAULT, TYPE)                                                                \
    constexpr static TYPE MEMBER = _get_##MEMBER##_value<OBJ>(DEFAULT)                                                \

// 数据常量化定义
#define CONV_DECLARE_DEFINE_STRUCT(T, M, U) namespace __AuxTiling {                                                   \
    template <typename T>                                                                                             \
    struct T##_##M {                                                                                                  \
        U M;                                                                                                          \
        constexpr static bool __CONST_TYPE_##M = false;                                                               \
    };                                                                                                                \
    template<typename T>                                                                                              \
    struct T##_CT_##M {                                                                                               \
        constexpr static U M = T::M;                                                                                  \
        constexpr static bool __CONST_TYPE_##M = true;                                                                \
    };                                                                                                                \
    template <typename T>                                                                                             \
    static constexpr bool _is_const_##T##_##M() {                                                                     \
        if constexpr (T::M > 0) {return true;} else {return false;}                                                   \
    };                                                                                                                \
    template <typename T>                                                                                             \
    typename Conditional<_is_const_##T##_##M<T>(), T##_CT_##M<T>, T##_##M<T>>::type T##_##M##_checkdefine();          \
}

#define CONV_DEFINE_STUCT(T, M) public decltype(__AuxTiling::T##_##M##_checkdefine<T>())

// 数据自定义
#define CONV_DEFINE_STUCT_FIELD(T, FIELD)                                                                             \
  T FIELD;                                                                                                            \
  constexpr static bool __CONST_TYPE_##FIELD = false

// 数据常量化校验
#define CONV_CHECK_CONST(T,M) (T::__CONST_TYPE_##M)

#endif

