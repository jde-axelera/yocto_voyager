cmake_minimum_required(VERSION 3.16)
project(yolo_voyager CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sysroot of the Axelera runtime (extracted from the aarch64 axelera-runtime/axelera-runtime2
# wheels). See scripts/04_build.sh for how to populate it.
set(AX_SYSROOT "${CMAKE_SOURCE_DIR}/../sysroot/axelera" CACHE PATH "Axelera runtime sysroot")

if(NOT EXISTS "${AX_SYSROOT}/include/axruntime/axruntime.h")
    message(FATAL_ERROR
        "AX_SYSROOT not populated. Run scripts/04_build.sh, or pass "
        "-DAX_SYSROOT=/path/to/sysroot/axelera to cmake.")
endif()

add_executable(yolo_demo_multi yolo_demo_multi.cpp)
target_include_directories(yolo_demo_multi PRIVATE "${AX_SYSROOT}/include")
target_link_directories(yolo_demo_multi PRIVATE "${AX_SYSROOT}/lib")
target_link_libraries(yolo_demo_multi PRIVATE axruntime pthread m)
target_compile_options(yolo_demo_multi PRIVATE -Wall -Wextra -O3 -pthread)
target_link_options(yolo_demo_multi PRIVATE
    -Wl,--unresolved-symbols=ignore-in-shared-libs
    -static-libstdc++ -static-libgcc
)
set_target_properties(yolo_demo_multi PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib:$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)
