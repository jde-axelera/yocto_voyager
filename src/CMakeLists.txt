cmake_minimum_required(VERSION 3.16)
project(yolo_voyager CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Where the Axelera runtime aarch64 wheels were extracted (libaxruntime.so + headers).
# Populated by scripts/04_build.sh; override with -DAX_SYSROOT=... if you stage it elsewhere.
set(AX_SYSROOT "${CMAKE_SOURCE_DIR}/../sysroot/axelera" CACHE PATH "Axelera runtime sysroot")

if(NOT EXISTS "${AX_SYSROOT}/include/axruntime/axruntime.h")
    message(FATAL_ERROR
        "AX_SYSROOT not populated. Run scripts/04_build.sh, or pass "
        "-DAX_SYSROOT=/path/to/sysroot/axelera to cmake.")
endif()

# Modularized library — each translation unit owns one piece of the pipeline.
add_library(yolo_voyager_core STATIC
    drawing.cpp
    font.cpp
    yolo_preproc.cpp
    yolo_postproc.cpp
    dma_heap.cpp
    subprocess.cpp
    py_aipu_client.cpp
    task_factory.cpp
    tasks/detection.cpp
    tasks/classify.cpp
)
target_include_directories(yolo_voyager_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${AX_SYSROOT}/include"
)
target_compile_options(yolo_voyager_core PRIVATE -Wall -Wextra -O3 -pthread)

# Thin orchestrator: parses argv, builds threads, wires the modules together.
add_executable(yolo_demo_multi yolo_demo_multi.cpp)
target_link_directories(yolo_demo_multi PRIVATE "${AX_SYSROOT}/lib")
target_link_libraries(yolo_demo_multi PRIVATE yolo_voyager_core axruntime pthread m)
target_compile_options(yolo_demo_multi PRIVATE -Wall -Wextra -O3 -pthread)
target_link_options(yolo_demo_multi PRIVATE
    -Wl,--unresolved-symbols=ignore-in-shared-libs    # ignore transitive libz reference from bundled libzip
    -static-libstdc++ -static-libgcc                  # Voyager Linux libstdc++ predates GLIBCXX_3.4.30
)
set_target_properties(yolo_demo_multi PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib:$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)
