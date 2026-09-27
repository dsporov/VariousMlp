# VariousMlp - C++ ports of Karpathy's "Neural Networks: Zero to Hero", and a
# few other implementations of the GPT from the last of them.
#
# Targets:
#   VariousMlp   the walkthroughs, as the Visual Studio project builds them
#   gpt_blocked  the reference GPT with a cache-blocked, vectorised multiply
#   gpt_aten     the GPT on ATen tensors; needs libtorch (CMAKE_PREFIX_PATH)
#   gpt_cuda     the GPT as CUDA kernels; needs a CUDA 12.x toolkit
#
# The last two are skipped, with a message, when what they need is missing.
# CMakePresets.json has a Windows/MSVC and a Linux/clang configuration.

cmake_minimum_required(VERSION 3.24)
project(VariousMlp LANGUAGES CXX)

# The sources were written against C++14 - the vcxproj never set a standard -
# and nothing in them means anything different under 17. libtorch needs 17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-configuration generators (Ninja, Makefiles) otherwise default to no
# optimisation at all, and these programs are meant to be run in Release.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VARIOUSMLP_NATIVE "Tune for this CPU: -march=native, or /arch:AVX2 on MSVC" ON)
option(VARIOUSMLP_FAST_MATH "Let the compiler reorder floating point arithmetic" ON)
option(VARIOUSMLP_VEC_REPORT "Have the compiler report which loops it vectorised" OFF)
option(VARIOUSMLP_ATEN "Build gpt_aten if libtorch can be found" ON)
option(VARIOUSMLP_CUDA "Build gpt_cuda if a CUDA compiler can be found" ON)

set(KARPATHY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/VariousMlp/Karpathy)

# The vcxproj uses /openmp:llvm; CMake 3.30 and later can ask for the same
# runtime, and older versions ignore this and fall back to plain /openmp,
# which gpt_model.h is written to work with too.
set(OpenMP_RUNTIME_MSVC llvm)
find_package(OpenMP)

# Flags every target shares. Wrapped in COMPILE_LANGUAGE:CXX so that none of
# them reach nvcc, which rejects most of them.
function(variousmlp_setup target)
    # The programs find input.txt and names.txt relative to a few working
    # directories a Visual Studio run might have; a build tree is none of
    # them, so the data directory is compiled in.
    target_compile_definitions(${target} PRIVATE VARIOUSMLP_DATA_DIR="${KARPATHY_DIR}")
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
            # infinite or NaN, and gpt.cpp's masked softmax is built out of
            # -infinity. So the reordering is kept and the promise is not.
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
    ${KARPATHY_DIR}/check.h
    ${KARPATHY_DIR}/gpt_model.h
    ${KARPATHY_DIR}/gpt_gemm.h)

# The walkthroughs. The same sources as VariousMlp.vcxproj.
add_executable(VariousMlp
    VariousMlp/VariousMlp.cpp
    ${KARPATHY_DIR}/micrograd.cpp
    ${KARPATHY_DIR}/makemore_bigram.cpp
    ${KARPATHY_DIR}/makemore_mlp.cpp
    ${KARPATHY_DIR}/makemore_batchnorm.cpp
    ${KARPATHY_DIR}/gpt.cpp
    ${KARPATHY_DIR}/gpt_chat.cpp
    ${MODEL_HEADERS})
variousmlp_setup(VariousMlp)
variousmlp_openmp(VariousMlp)

add_executable(gpt_blocked
    ${KARPATHY_DIR}/gpt_blocked.cpp
    ${KARPATHY_DIR}/gpt_bench.h
    ${MODEL_HEADERS})
variousmlp_setup(gpt_blocked)
variousmlp_openmp(gpt_blocked)

if(VARIOUSMLP_ATEN)
    find_package(Torch QUIET)
    if(Torch_FOUND)
        add_executable(gpt_aten
            ${KARPATHY_DIR}/gpt_aten.cpp
            ${KARPATHY_DIR}/gpt_bench.h
            ${MODEL_HEADERS})
        variousmlp_setup(gpt_aten)
        # Deliberately no OpenMP: libtorch brings its own runtime, and two in
        # one process fight (see gpt_aten.cpp). gpt_model.h's pragmas are then
        # unknown to the compiler, which is expected and not worth a warning.
        if(MSVC)
            target_compile_options(gpt_aten PRIVATE /wd4068)
        else()
            target_compile_options(gpt_aten PRIVATE -Wno-unknown-pragmas)
        endif()
        target_link_libraries(gpt_aten PRIVATE ${TORCH_LIBRARIES})
        # Windows finds DLLs next to the exe, so libtorch's are copied there.
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
        # "native" asks the driver which GPU is installed when configuring,
        # which is a GTX 1050 Ti (compute 6.1) on the desktop. Pass
        # -DCMAKE_CUDA_ARCHITECTURES=61 to configure on a machine without it.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        add_executable(gpt_cuda
            ${KARPATHY_DIR}/gpt_cuda_main.cpp
            ${KARPATHY_DIR}/gpt_cuda.cu
            ${KARPATHY_DIR}/gpt_cuda.h
            ${KARPATHY_DIR}/gpt_bench.h
            ${MODEL_HEADERS})
        variousmlp_setup(gpt_cuda)
        variousmlp_openmp(gpt_cuda)
    else()
        message(STATUS "gpt_cuda skipped: no CUDA compiler found")
    endif()
endif()
