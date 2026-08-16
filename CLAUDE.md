# VariousMlp

C++17 ports of Andrej Karpathy's "Neural Networks: Zero to Hero" videos, built
with Visual Studio 2022 (toolset v143, x64). One project, one exe; the call in
`VariousMlp/VariousMlp.cpp` picks which walkthrough runs.

## Building and running

- **Do not build, and do not run the exe.** No MSBuild, no `cl`, no
  `x64\Release\VariousMlp.exe`. I build and run in Visual Studio myself.
- This means you cannot confirm that code compiles or that results are right.
  Report what you changed and say plainly that it is unverified — never phrase
  an unbuilt change as if it had been checked.
- A new source or header file has to be added to `VariousMlp/VariousMlp.vcxproj`
  and `VariousMlp/VariousMlp.vcxproj.filters` to show up in the solution.

## Formatting

These apply to code you write or modify. Do not reformat untouched code as a
side effect of an unrelated change.

- Opening brace on the same line as the statement it belongs to; the closing
  brace always gets its own line — never collapse a block onto one line.
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

- `CHECK(cond, msg)` for structural invariants — throws, and stays on in Release
  on purpose. Keep it out of inner arithmetic loops.
- `sanity::Expect` for numbers the videos predict — reports and carries on,
  since those values only hold for the video's dataset.
