# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CAMKE_SKIP_RPATH TRUE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")
add_compile_options(
    -fstack-protector-all
    -D_FORTIFY_SOURCE=2 -O2
    )
# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_subdirectory(src)
endif()