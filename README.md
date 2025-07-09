# Typst rewrite of my masters thesis

This repo contains a rewritten version of my masters thesis from Latex to Typst. 
Structure of the thesis is based on the template TODO.

This thesis includes the following typesetting objects (subjectively chosen):

- Multi-page figures with several subfigures
- Different captions in the content and outlines
- Algorithms with multi-line formulaes
- Tables with footnotes and automatic numbers rounding/formatting/alignment

## Compilation
To compile the thesis [install Typst](https://github.com/typst/typst#installation) and run:

```bash
typst c thesis.typ
```

which should produce a `thesis.pdf` file.

## Compilation benchmarks
| Command | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `typst c thesis.typ` | 1.214 ± 0.024 | 1.186 | 1.262 | 1.00 |
| `latexmk -pdflatex="lualatex --shell-escape --interaction=nonstopmode %O %S" -pdf main.tex` | 29.938 ± 0.145 | 29.760 | 30.202 | 24.65 ± 0.50 |
