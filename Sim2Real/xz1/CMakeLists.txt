cmake_minimum_required(VERSION 3.16)
project(nazarite_xz1_deploy LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(xz1_deploy
  imu/dm_imu.cpp
  motor_control/im6014.cpp
  motor_control/async_motor_system.cpp
  remote_controler/evdev_gamepad.cpp
  him/xz1_him.cpp
)
target_include_directories(xz1_deploy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(xz1_deploy PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(xz1_deploy PUBLIC Threads::Threads)

add_executable(xz1_deploy_test tests/test_main.cpp)
target_link_libraries(xz1_deploy_test PRIVATE xz1_deploy)
target_compile_options(xz1_deploy_test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(xz1_him_config him/config_view.cpp)
target_link_libraries(xz1_him_config PRIVATE xz1_deploy)

add_executable(xz1_imu_reader imu/imu_reader.cpp)
target_link_libraries(xz1_imu_reader PRIVATE xz1_deploy)

add_executable(xz1_motor_test motor_control/motor_test.cpp)
target_link_libraries(xz1_motor_test PRIVATE xz1_deploy)

add_executable(xz1_motor_system_probe motor_control/motor_system_probe.cpp)
target_link_libraries(xz1_motor_system_probe PRIVATE xz1_deploy Threads::Threads)

add_executable(xz1_motor_sequence motor_control/motor_sequence.cpp)
target_link_libraries(xz1_motor_sequence PRIVATE xz1_deploy)
target_compile_options(xz1_motor_sequence PRIVATE -Wall -Wextra -Wpedantic)

add_executable(xz1_gamepad_test remote_controler/gamepad_test.cpp)
target_link_libraries(xz1_gamepad_test PRIVATE xz1_deploy)

get_filename_component(SIM2REAL_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_default_onnxruntime_root
  "${SIM2REAL_ROOT}/sdk/onnxruntime-linux-x64-1.31.1")
if(NOT EXISTS "${_default_onnxruntime_root}/include/onnxruntime_cxx_api.h")
  set(_default_onnxruntime_root "")
endif()
set(ONNXRUNTIME_ROOT "${_default_onnxruntime_root}" CACHE PATH
  "ONNX Runtime C++ SDK root directory")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}" ENV ONNXRUNTIME_ROOT PATH_SUFFIXES include)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}" ENV ONNXRUNTIME_ROOT PATH_SUFFIXES lib lib64)
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  target_sources(xz1_deploy PRIVATE him/onnx_policy.cpp)
  target_include_directories(xz1_deploy PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(xz1_deploy PUBLIC ${ONNXRUNTIME_LIBRARY})
  target_compile_definitions(xz1_deploy PUBLIC XZ1_HAS_ONNXRUNTIME=1)
  add_executable(xz1_onnx_test him/onnx_test.cpp)
  target_link_libraries(xz1_onnx_test PRIVATE xz1_deploy)
  get_filename_component(_onnxruntime_lib_dir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  set_target_properties(xz1_onnx_test PROPERTIES
    BUILD_RPATH "${_onnxruntime_lib_dir}")
else()
  message(STATUS "ONNX Runtime C++ SDK not found; hardware interfaces/tests remain available")
endif()

enable_testing()
add_test(NAME xz1_deploy_test COMMAND xz1_deploy_test)

# Optional host-side UART emulation uses only the Python standard library.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_executable(xz1_async_uart_test_client tests/async_uart_client.cpp)
  target_link_libraries(xz1_async_uart_test_client PRIVATE xz1_deploy)
  add_test(NAME xz1_async_uart_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_async_uart.py
      $<TARGET_FILE:xz1_motor_sequence> $<TARGET_FILE:xz1_async_uart_test_client>)
  set_tests_properties(xz1_async_uart_test PROPERTIES TIMEOUT 60)
endif()
