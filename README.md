# Heuristic hyperparameter optimization for neural networks

This repo contains a rewritten version of my masters thesis from Latex to Typst. 
Structure of the thesis is based on the [wut-thesis](https://typst.app/universe/package/wut-thesis) template.
[Rendered version](./build/rendered.pdf)

This thesis includes the following interesting typesetting objects (subjectively chosen):

- Multi-page figures with several subfigures
- Different captions in the content and outlines
- Algorithms with multi-line formulaes
- Tables with footnotes and automatic numbers rounding/formatting/alignment

## Compilation
To compile the thesis [install Typst](https://github.com/typst/typst#installation), make sure you have `Adagio_Slab` font installed as described in the template instructions and run:

```bash
typst c thesis.typ
```

which should produce a `thesis.pdf` file.

## Compilation benchmarks
| Command | Mean [s] | Min [s] | Max [s] | Relative |
|:---|---:|---:|---:|---:|
| `typst c thesis.typ` | 1.278 ± 0.021 | 1.237 | 1.305 | 1.00 |
| `latexmk -pdflatex="lualatex --shell-escape --interaction=nonstopmode %O %S" -pdf main.tex` | 29.407 ± 0.233 | 29.195 | 29.995 | 23.02 ± 0.42 |
