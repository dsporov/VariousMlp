# VariousMlp - C++ ports of Karpathy's "Neural Networks: Zero to Hero", and a
# few other implementations of the GPT from the last of them.
#
# Targets:
#   VariousMlp   the walkthroughs, as the Visual Studio project builds them
#   gpt_blocked  the reference GPT with a cache-blocked, vectorised multiply
#   gpt_aten     the GPT on ATen tensors; needs libtorch (CMAKE_PREFIX_PATH)
#   gpt_cuda     the GPT as CUDA kernels; needs a CUDA 12.x toolkit
#   gpt_split    CUDA and the blocked multiply sharing each batch; needs CUDA too
#   gpt_scaled   the video's scale-up, trained in the same way; needs CUDA
#                too. A separate project in VariousMlp/GptScaled, with its
#                own copies of every source it compiles
#
# The last four are skipped, with a message, if what they need is missing.
# CMakePresets.json has a Windows/MSVC and a Linux/clang configuration.

cmake_minimum_required(VERSION 3.24)
project(VariousMlp LANGUAGES CXX)

# C++20, because recent libtorch headers need it, and one standard for all
# targets is simpler than two. The older sources were written for C++14 and
# 17, and none of the changes in C++20 (aggregates, rewritten comparisons, u8
# literals) affect them. The vcxproj uses the same standard.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Without this, single-configuration generators (Ninja, Makefiles) build
# without any optimisation, and these programs are meant to run in Release.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VARIOUSMLP_NATIVE "Tune for this CPU: -march=native, or /arch:AVX2 on MSVC" ON)
option(VARIOUSMLP_FAST_MATH "Let the compiler reorder floating point arithmetic" ON)
option(VARIOUSMLP_VEC_REPORT "Have the compiler report which loops it vectorised" OFF)
option(VARIOUSMLP_ATEN "Build gpt_aten if libtorch can be found" ON)
option(VARIOUSMLP_CUDA "Build gpt_cuda, gpt_split and gpt_scaled if a CUDA compiler can be found" ON)

set(MAKEMORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VariousMlp/makemore)
set(SCALED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VariousMlp/GptScaled)

# The vcxproj uses /openmp:llvm. CMake 3.30 and later can request the same
# runtime. Older versions ignore this setting and use plain /openmp, which
# gpt_model.h also supports.
set(OpenMP_RUNTIME_MSVC llvm)
find_package(OpenMP)

# Flags that all targets share. They are wrapped in COMPILE_LANGUAGE:CXX, so
# that none of them reach nvcc, which rejects most of them.
function(variousmlp_setup target)
    # The programs look for input.txt and names.txt relative to a few working
    # directories that a Visual Studio run might have. A build tree is not one
    # of them, so the path of the data directory is compiled in.
    target_compile_definitions(${target} PRIVATE VARIOUSMLP_DATA_DIR="${MAKEMORE_DIR}")
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W3>
            $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
        if(VARIOUSMLP_NATIVE)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        endif()
        if(VARIOUSMLP_FAST_MATH)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>)
        endif()
        if(VARIOUSMLP_VEC_REPORT)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Qvec-report:2>)
        endif()
    else()
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
        if(VARIOUSMLP_NATIVE)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
        endif()
        if(VARIOUSMLP_FAST_MATH)
            # -ffast-math is /fp:fast plus a promise that no value is ever
            # infinite or NaN. But the masked softmax in gpt.cpp uses
            # -infinity. So the reordering is kept, and the promise is removed.
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
                $<$<COMPILE_LANGUAGE:CXX>:-fno-finite-math-only>)
        endif()
        if(VARIOUSMLP_VEC_REPORT)
            if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                target_compile_options(${target} PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-Rpass=loop-vectorize>
                    $<$<COMPILE_LANGUAGE:CXX>:-Rpass-missed=loop-vectorize>)
            else()
                target_compile_options(${target} PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-fopt-info-vec-optimized>
                    $<$<COMPILE_LANGUAGE:CXX>:-fopt-info-vec-missed>)
            endif()
        endif()
    endif()
endfunction()

function(variousmlp_openmp target)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

if(NOT OpenMP_CXX_FOUND)
    message(STATUS "OpenMP not found: everything builds, and runs on one thread "
        "(on Ubuntu with clang: apt install libomp-dev)")
endif()

set(MODEL_HEADERS
    ${MAKEMORE_DIR}/check.h
    ${MAKEMORE_DIR}/gpt_model.h
    ${MAKEMORE_DIR}/gpt_gemm.h)

# The walkthroughs. The same sources as VariousMlp.vcxproj.
add_executable(VariousMlp
    VariousMlp/VariousMlp.cpp
    ${MAKEMORE_DIR}/micrograd.cpp
    ${MAKEMORE_DIR}/makemore_bigram.cpp
    ${MAKEMORE_DIR}/makemore_mlp.cpp
    ${MAKEMORE_DIR}/makemore_batchnorm.cpp
    ${MAKEMORE_DIR}/gpt.cpp
    ${MAKEMORE_DIR}/gpt_chat.cpp
    ${MODEL_HEADERS})
variousmlp_setup(VariousMlp)
variousmlp_openmp(VariousMlp)

add_executable(gpt_blocked
    ${MAKEMORE_DIR}/gpt_blocked.cpp
    ${MAKEMORE_DIR}/gpt_bench.h
    ${MODEL_HEADERS})
variousmlp_setup(gpt_blocked)
variousmlp_openmp(gpt_blocked)

if(VARIOUSMLP_ATEN)
    find_package(Torch QUIET)
    if(Torch_FOUND)
        add_executable(gpt_aten
            ${MAKEMORE_DIR}/gpt_aten.cpp
            ${MAKEMORE_DIR}/gpt_bench.h
            ${MODEL_HEADERS})
        variousmlp_setup(gpt_aten)
        # No OpenMP, on purpose: libtorch brings its own runtime, and two
        # runtimes in one process conflict (see gpt_aten.cpp). The compiler
        # then does not know the pragmas in gpt_model.h. That is expected, so
        # the warning is turned off.
        if(MSVC)
            target_compile_options(gpt_aten PRIVATE /wd4068)
        else()
            target_compile_options(gpt_aten PRIVATE -Wno-unknown-pragmas)
        endif()
        target_link_libraries(gpt_aten PRIVATE ${TORCH_LIBRARIES})
        # Windows looks for DLLs next to the exe, so libtorch's DLLs are
        # copied there.
        if(MSVC)
            file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
            add_custom_command(TARGET gpt_aten POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
                    $<TARGET_FILE_DIR:gpt_aten>)
        endif()
    else()
        message(STATUS "gpt_aten skipped: libtorch not found "
            "(pass -DCMAKE_PREFIX_PATH=<unpacked libtorch>)")
    endif()
endif()

if(VARIOUSMLP_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # "native" asks the driver at configure time which GPU is installed.
        # On the desktop, that is a GTX 1050 Ti (compute 6.1). To configure on
        # a machine without that GPU, pass -DCMAKE_CUDA_ARCHITECTURES=61.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        add_executable(gpt_cuda
            ${MAKEMORE_DIR}/gpt_cuda_main.cpp
            ${MAKEMORE_DIR}/gpt_cuda.cu
            ${MAKEMORE_DIR}/gpt_cuda.h
            ${MAKEMORE_DIR}/gpt_bench.h
            ${MODEL_HEADERS})
        variousmlp_setup(gpt_cuda)
        variousmlp_openmp(gpt_cuda)

        # gpt_cuda.cu is compiled a second time, instead of being shared
        # through a library, so that the definition of gpt_cuda above stays as
        # it was. The GPU half runs on a std::async thread, and older versions
        # of glibc need -pthread for that.
        find_package(Threads)
        add_executable(gpt_split
            ${MAKEMORE_DIR}/gpt_split.cpp
            ${MAKEMORE_DIR}/gpt_cuda.cu
            ${MAKEMORE_DIR}/gpt_cuda.h
            ${MAKEMORE_DIR}/gpt_bench.h
            ${MODEL_HEADERS})
        variousmlp_setup(gpt_split)
        variousmlp_openmp(gpt_split)
        if(TARGET Threads::Threads)
            target_link_libraries(gpt_split PRIVATE Threads::Threads)
        endif()

        # Nothing from MAKEMORE_DIR is compiled here: GptScaled has its own
        # copies, so a change to the originals does not affect it. It still
        # gets VARIOUSMLP_DATA_DIR from variousmlp_setup. That is how it finds
        # input.txt, and where it saves its checkpoint.
        add_executable(gpt_scaled
            ${SCALED_DIR}/gpt_scaled.cpp
            ${SCALED_DIR}/gpt_cuda.cu
            ${SCALED_DIR}/gpt_cuda.h
            ${SCALED_DIR}/check.h
            ${SCALED_DIR}/gpt_model.h
            ${SCALED_DIR}/gpt_gemm.h)
        variousmlp_setup(gpt_scaled)
        variousmlp_openmp(gpt_scaled)
        if(TARGET Threads::Threads)
            target_link_libraries(gpt_scaled PRIVATE Threads::Threads)
        endif()
    else()
        message(STATUS "gpt_cuda, gpt_split and gpt_scaled skipped: no CUDA compiler found")
    endif()
endif()
