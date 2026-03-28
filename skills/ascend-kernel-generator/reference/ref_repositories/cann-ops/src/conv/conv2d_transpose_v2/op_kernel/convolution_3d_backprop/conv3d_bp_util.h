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
 * \file conv3d_bp_util.h
 * \brief
 */

#ifndef CONV3D_BP_UTIL_H
#define CONV3D_BP_UTIL_H

#include "kernel_utils.h"

static __aicore__ inline uint64_t ShiftCeilBlockCube(uint64_t a)
{
    return (a + 15) >> 4; // 4: bit; 15: 2**4 - 1
}

static __aicore__ inline uint64_t CalcDiv(uint64_t a, uint32_t b)
{
    if (b > a) {
        return 0;
    } else if (b == a) {
        return 1;
    } else if (b == 1) {
        return a;
    } else {
        return a / b;
    }
}

static __aicore__ inline uint64_t CalcFloorAlign(uint64_t a, uint32_t b)
{
    if (b >= a) {
        return b;
    } else if (b == 1) {
        return a;
    } else {
        return a / b * b;
    }
}

static __aicore__ inline uint64_t CalcRemainder(uint64_t a, uint32_t b)
{
    if (b > a) {
        return a;
    } else if (b == 1 || b == a) {
        return 0;
    } else {
        return a % b;
    }
}

// API类中定义call函数的默认重载函数，支持任意类型任意数量的参数
#define DECLARE_DEFAULT_OVERLOADING_FUN(T, NAMESPACE)                       \
    template <class... Ts>                                                  \
    static __aicore__ inline NAMESPACE::TypeFalse call(T *self, Ts... args) \
    {                                                                       \
        return (NAMESPACE::TypeFalse){0};                                   \
    }

// 检查类T中是否有call(...)成员函数
#define CHECK_FUN(T, NAMESPACE, ...) (!IsSameType<decltype(T::call(__VA_ARGS__)), NAMESPACE::TypeFalse>::value)

/*
定义一个校验性的模板类，用于判断类型T是否具有模板成员函数MEMBER<U>
和宏DECLARE_IMPL配套使用，调用方式_has_impl_MEMBER<T, U>::value
49行：decltype获取表达式的类型，declval是模板函数，获取模板参数T的右值引用，如果T没有MEMBER<U>，会报错，否则返回TrueType
*/
#define DECLARE_CHECK_IMPL(MEMBER, args...)                                                                     \
    namespace __AuxCheckImpl {                                                                                  \
    template <typename T, typename U>                                                                           \
    struct _has_impl_##MEMBER {                                                                                 \
        template <typename C>                                                                                   \
        static auto check(int) -> decltype(std::declval<typename C::template MEMBER<U, ##args>>(), TrueType()); \
        template <typename C>                                                                                   \
        static FalseType check(...);                                                                            \
        static constexpr bool value = IsSameType<decltype(check<T>(0)), TrueType>::value;                       \
    };                                                                                                          \
    }

// 定义一个模板类，用于判断类型T是否具有模板成员函数MEMBER<typename U, bool sync>
#define DECLARE_CHECK_SYNC_IMPL(MEMBER, args...)                                                                      \
    namespace __AuxCheckImpl {                                                                                        \
    template <typename T, typename U, bool sync>                                                                      \
    struct _has_impl_##MEMBER {                                                                                       \
        template <typename C>                                                                                         \
        static auto check(int) -> decltype(std::declval<typename C::template MEMBER<U, sync, ##args>>(), TrueType()); \
        template <typename C>                                                                                         \
        static FalseType check(...);                                                                                  \
        static constexpr bool value = IsSameType<decltype(check<T>(0)), TrueType>::value;                             \
    };                                                                                                                \
    }

// 定义成员函数MEMBER<U>, 如果Config中存在MEMBER成员，MEMBER函数指向Config成员，否者指向Current::Init
#define DECLARE_IMPL(Config, Current, MEMBER, U)     \
    template <bool Default, class T>                 \
    struct __##MEMBER {                              \
        using Type = typename Current::MEMBER<U>;    \
    };                                               \
    template <class T>                               \
    struct __##MEMBER<true, T> {                     \
        using Type = typename T::template MEMBER<U>; \
    };                                               \
    using MEMBER = typename __##MEMBER<__AuxCheckImpl::_has_impl_##MEMBER<Config, U>::value, Config>::Type

// 定义成员函数MEMBER<U, sync>, 如果Config中存在MEMBER成员，MEMBER函数指向Config成员，否者指向Current::Init
#define DECLARE_SYNC_IMPL(Config, Current, MEMBER, U)      \
    template <bool Default, class T, bool sync>            \
    struct __##MEMBER {                                    \
        using Type = typename Current::MEMBER<U, sync>;    \
    };                                                     \
    template <class T, bool sync>                          \
    struct __##MEMBER<true, T, sync> {                     \
        using Type = typename T::template MEMBER<U, sync>; \
    };                                                     \
    template <bool sync>                                   \
    using MEMBER = typename __##MEMBER<__AuxCheckImpl::_has_impl_##MEMBER<Config, U, sync>::value, Config, sync>::Type

/*
定义一个模板类，用于判断类型T是否具有成员MEMBER
和宏CHECK_MEMBER配套使用，调用方式_has_member_MEMBER<T>::value
*/
#define DECLARE_CHECK_MEMBER(MEMBER)                                                        \
    namespace __AuxCheck {                                                                  \
    template <typename T>                                                                   \
    struct _has_member_##MEMBER {                                                           \
        template <typename U>                                                               \
        static void check(decltype(&U::MEMBER));                                            \
        template <typename U>                                                               \
        static int check(...);                                                              \
        static constexpr bool value = IsSameType<decltype(check<T>(nullptr)), void>::value; \
    };                                                                                      \
    }

// 检查类OBJ中是否有成员变量MEMBER
#define CHECK_MEMBER(OBJ, MEMBER) (__AuxCheck::_has_member_##MEMBER<typename OBJ>::value)

/*
定义两个辅助模板类，一个成员M是变量，一个成员M是常量；
同时定义一个校验性的模板函数，函数根据模板参数T是否有常量且值>0的成员M，返回对应的模板类
和宏DEFINE_STUCT配套使用，
*/
#define DECLARE_DEFINE_STRUCT(T, M, U)                                                                               \
    namespace __AuxTiling {                                                                                          \
    template <typename TT>                                                                                           \
    struct T##_##M {                                                                                                 \
        U M;                                                                                                         \
        constexpr static bool __CONST_TYPE_##M = false;                                                              \
    };                                                                                                               \
    template <typename TT>                                                                                           \
    struct T##_CT_##M {                                                                                              \
        constexpr static U M = TT::M;                                                                                \
        constexpr static bool __CONST_TYPE_##M = true;                                                               \
    };                                                                                                               \
    template <typename TT>                                                                                           \
    constexpr bool _is_const_##T##_##M()                                                                             \
    {                                                                                                                \
        return TT::M > 0;                                                                                            \
    };                                                                                                               \
    template <typename TT>                                                                                           \
    typename std::conditional<_is_const_##T##_##M<TT>(), T##_CT_##M<TT>, T##_##M<TT>>::type T##_##M##_checkdefine(); \
    }

// 供类继承使用，返回一个供继承的父类类型
#define DEFINE_STUCT(T, M) \
public                     \
    decltype(__AuxTiling::T##_##M##_checkdefine<T>())

#define DEFINE_STUCT_FIELD(T, FIELD) \
    T FIELD;                         \
    constexpr static bool __CONST_TYPE_##FIELD = false

#define CHECK_CONST(T, M) (T::__CONST_TYPE_##M)

#define DEFINE_STUCT_TEMPLATE_FIELD(T, FIELD, C, ...) \
    T<C, ##__VA_ARGS__> FIELD;                        \
    constexpr static bool __CONST_TYPE_##FIELD = false

#define DEFINE_STUCT_ARRAY_FIELD(T, FIELD, NUM) \
    T FIELD[NUM];                               \
    constexpr static bool __CONST_TYPE_##FIELD = false

#endif