#import "../utils.typ": table-with-notes

= Tables

#let hline(columns, ..args) = {
  return (
    table.cell(colspan: columns, inset: 1pt)[],
    table.hline(..args),
    table.cell(colspan: columns, inset: 1pt)[],
  )
}

#figure(
  kind: table,
  // @typstyle off
  table-with-notes(
    columns: (1fr, 1fr, 1fr, 1fr),
    align: center + horizon,
    inset: 3pt,
    notes: [
      #super[1] Provided for reference, not part of the optimization process.
    ],
    table.header(
      table.cell(rowspan: 2, [Heuristic algorithm]),
      table.cell(rowspan: 2, [Run number]),
      table.cell(colspan: 2, [Receiver Operating Characteristics]),
      table.hline(stroke: 0pt),
      table.hline(start: 1, end: 4, stroke: 0.3pt),
      [Area Under Curve],
      [Equal Error Rate],
      table.hline(stroke: 1pt),
    ),
    table.cell(rowspan: 10, [DES]), [1], [0.674], [0.366],
    [2], [0.678], [0.360],
    [3], [0.678], [0.352],
    [4], [0.673], [0.365],
    [5], [0.665], [0.379],
    [6], [0.651], [0.393],
    [7], [0.649], [0.381],
    [8], [0.676], [0.371],
    [9], [0.666], [0.383],
    [10], [0.679], [0.359],
    ..hline(4, stroke: 0.3pt),
    table.cell(rowspan: 10, [CMA-ES]), [1], [0.683], [0.356],
    [2], [0.676], [0.357],
    [3], [0.674], [0.362],
    [4], [0.655], [0.392],
    [5], [0.672], [0.367],
    [6], [0.668], [0.372],
    [7], [0.648], [0.396],
    [8], [0.681], [0.346],
    [9], [0.657], [0.379],
    [10], [0.646], [0.388],
    ..hline(4, stroke: 0.3pt),
    table.cell(rowspan: 10, [jSO]), [1], [0.679], [0.353],
    [2], [0.680], [0.362],
    [3], [0.666], [0.365],
    [4], [0.659], [0.376],
    [5], [0.671], [0.371],
    [6], [0.665], [0.376],
    [7], [0.677], [0.346],
    [8], [0.652], [0.372],
    [9], [0.653], [0.383],
    [10], [0.675], [0.365],
    ..hline(4, stroke: 0.3pt),
    [Logistic Regression#super[1]], [], [0.654], [0.392],
  ),
  caption: [RAUC and EER values for the best individuals found during optimization on the Aspartus dataset.],
)<tab:aspartus_rocs>

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr),
    align: center + horizon,
    inset: 3pt,
    table.header(
      table.cell(rowspan: 2, [Heuristic algorithm]),
      table.cell(rowspan: 2, [Run number]),
      table.cell(colspan: 2, [Receiver Operating Characteristics]),
      table.hline(stroke: 0pt),
      table.hline(start: 1, end: 4, stroke: 0.3pt),
      [Area Under Curve],
      [Equal Error Rate],
      table.hline(stroke: 1pt),
    ),
    table.cell(rowspan: 10, [DES]), [1], [0.868], [0.185],
    [2], [0.879], [0.179],
    [3], [0.864], [0.160],
    [4], [0.873], [0.179],
    [5], [0.870], [0.179],
    [6], [0.867], [0.185],
    [7], [0.872], [0.204],
    [8], [0.878], [0.191],
    [9], [0.870], [0.185],
    [10], [0.876], [0.185],
    ..hline(4, stroke: 0.3pt),
    table.cell(rowspan: 10, [CMA-ES]), [1], [0.873], [0.179],
    [2], [0.868], [0.191],
    [3], [0.872], [0.185],
    [4], [0.872], [0.173],
    [5], [0.876], [0.160],
    [6], [0.867], [0.179],
    [7], [0.877], [0.179],
    [8], [0.869], [0.185],
    [9], [0.871], [0.185],
    [10], [0.874], [0.185],
    ..hline(4, stroke: 0.3pt),
    table.cell(rowspan: 10, [jSO]), [1], [0.873], [0.179],
    [2], [0.859], [0.210],
    [3], [0.871], [0.160],
    [4], [0.878], [0.185],
    [5], [0.878], [0.185],
    [6], [0.877], [0.185],
    [7], [0.881], [0.179],
    [8], [0.878], [0.179],
    [9], [0.878], [0.179],
    [10], [0.872], [0.185],
  ),
  caption: [RAUC and EER values for the best individuals found during optimization on the Titanic dataset.],
)<tab:titanic_rocs>
