# VariousMlp

C++20 ports of Andrej Karpathy's "Neural Networks: Zero to Hero" videos, plus
alternative implementations of the GPT from the last video (blocked CPU
multiply, ATen, CUDA), made to try out new tools. Performance is a side
interest here, not the goal. Built with CMake: Visual Studio 2022 (MSVC, x64)
on Windows and clang on Ubuntu. The original Visual Studio project is still
there for the walkthrough executable.

## Layout

- `CMakeLists.txt`, `CMakePresets.json` - the build, at the repo root. Presets:
  `windows-msvc-release`/`-debug`, `linux-clang-release`/`-debug`, building
  into `out/build/<preset>/` (gitignored).
- `VariousMlp/VariousMlp.cpp` - `main` for the walkthroughs. The calls in it
  pick which one runs; `VariousMlp chat` starts the prompt in `gpt_chat.cpp`.
- `VariousMlp/Karpathy/` - one `.cpp` per video, plus the GPT's supporting
  files:
  - `gpt_model.h` - the reference GPT. Used by `gpt.cpp`, `gpt_chat.cpp` and
    every bench.
  - `gpt_gemm.h` - the cache-blocked, vectorised multiply (AVX2/FMA micro-kernel,
    portable fallback). Switched on with `gpt::active_kernels() =
    gpt::Kernels::kBlocked`. The walkthrough stays on `kScalar`, because its
    reproduction checks are to four decimals.
  - `gpt_bench.h` - the harness that checks another implementation against the
    reference: gradients, side-by-side training, throughput. Its header comment
    lists the members a backend has to provide.
  - `gpt_blocked.cpp`, `gpt_aten.cpp`, `gpt_cuda_main.cpp` with `gpt_cuda.cu`
    and `gpt_cuda.h` - one bench executable per implementation.
  - `gpt_model_plain_cpu.h` - the pre-OpenMP double version. Included by
    nothing; kept for reference.
  - `input.txt`, `names.txt` - the data. Checkpoints (`*.ckpt`, gitignored)
    are written next to `input.txt`.

## Targets

| target | sources | needs |
|---|---|---|
| `VariousMlp` | `VariousMlp.cpp` and every walkthrough | OpenMP if found |
| `gpt_blocked` | `gpt_blocked.cpp` | OpenMP if found |
| `gpt_aten` | `gpt_aten.cpp` | libtorch via `-DCMAKE_PREFIX_PATH`; skipped without it |
| `gpt_cuda` | `gpt_cuda_main.cpp`, `gpt_cuda.cu` | CUDA 12.x (the GTX 1050 Ti is Pascal, sm_61, which CUDA 13 dropped); skipped without nvcc |

- `gpt_aten` is built without OpenMP on purpose: libtorch brings its own
  runtime, and two runtimes in one process conflict.
- Everything is C++20, because recent libtorch headers need it: CMake sets it
  for C++ and CUDA, and the vcxproj sets `<LanguageStandard>stdcpp20`. Keep
  the two in step.
- `gpt_cuda.h` is plain C++ so that nvcc never compiles `gpt_model.h`. Keep it
  that way: CUDA types stay in `gpt_cuda.cu`.
- The CMake build defines `VARIOUSMLP_DATA_DIR` (absolute path of
  `Karpathy/`), so the data is found from any working directory. The vcxproj
  build does not define it and falls back to relative paths. Keep both
  working.

## Building and running

- **Do not build, and do not run anything.** No cmake configure or build, no
  MSBuild, no `cl`, `clang++` or `nvcc`, not even a syntax-only compile, and no
  executables. I build and run myself.
- This means you cannot confirm that code compiles or that results are right.
  Report what you changed and say plainly that it is unverified. Never phrase
  an unbuilt change as if it had been checked.
- A new source or header file goes into `CMakeLists.txt`. If it is part of the
  `VariousMlp` executable, it also has to be added to
  `VariousMlp/VariousMlp.vcxproj` and `VariousMlp/VariousMlp.vcxproj.filters`
  to show up in the solution. The bench executables each have their own
  `main`, so never add their `.cpp`/`.cu` files to the vcxproj.
- Flags live in `variousmlp_setup()` in `CMakeLists.txt`. Anything that must
  not reach nvcc is wrapped in `$<$<COMPILE_LANGUAGE:CXX>:...>`.

## Formatting

These apply to code you write or modify, including `.cu` files and CMake
files where they make sense. Do not reformat untouched code as a side effect
of an unrelated change.

- Opening brace on the same line as the statement it belongs to; the closing
  brace always gets its own line. Never collapse a block onto one line.
- Always brace the body, including single-statement bodies.

```cpp
// yes
for (int i = 0; i < n; ++i) {
    sum += x[i];
}

// no
for (int i = 0; i < n; ++i) sum += x[i];
for (int i = 0; i < n; ++i) { sum += x[i]; }
```

- 4-space indent, no tabs.
- ASCII only, comments included.
- Comments explain *why*, in prose sentences, at the density of the surrounding
  file. No narration of lines that speak for themselves.
- Named `constexpr` constants (`kVocab`, `kInputs`) rather than bare literals.

## Sanity checks

`Karpathy/check.h` holds the two check flavours these files use:

- `CHECK(cond, msg)` for structural invariants. It throws, and stays on in
  Release on purpose. Keep it out of inner arithmetic loops. CUDA API failures
  go through it too (`CUDA_CHECK` in `gpt_cuda.cu`). libtorch defines its own
  one-argument `CHECK`, so `gpt_aten.cpp` undefines it before including
  `check.h`.
- `sanity::Expect` for numbers the videos predict. It reports and carries on,
  since those values only hold for the video's dataset. `gpt_bench.h` uses it
  for agreement between implementations as well.
