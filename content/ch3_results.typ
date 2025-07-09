#import "../utils.typ": todo, silentheading, flex-caption, table-with-notes, multipage-subfigures
#import "@preview/subpar:0.2.2"
#import "@preview/zero:0.4.0"

= Results and conclusions
<ch::results>
== Results of re-implementations' verification
<results-of-re-implementations-verification>
Results of statistical test for each function, along with the combined 'meta' p-values
calculated using the Fisher's method, can be found in @tab:implementation_verification.
Null hypothesis for the Wilcoxon rank-sum test states that two sets of measurements are
drawn from the same distribution. Assuming $alpha = 0.001$, 'meta' p-values indicate
that this hypothesis can be rejected for the jSO re-implementation ($p approx 10^(-
112)$) and can not be rejected for DES re-implementation, with $p approx 0.48$. These
results are consistent with ECDF curves calculated for selected benchmark functions,
which are provided in @fig:impl_verification. Characteristics of 'DES' and 'DESpy'
curves for all problems and all dimensionalities indicate that re-implementation has
been successful and does not significantly differ from the original. On the other hand,
curves for 'jSO' and 'jSOpy' do not appear to be similar. More specifically, Python's
based version of the algorithm tends to perform better than the original on lower
dimensionalities. While I was not able to find and correct the source of the difference,
it is plausible that it originates from differing implementations of random
distributions --- authors of the original jSO algorithms used built-in C++ randomness
source to implement these distributions, while Python-based version uses Python's
`random` module for this task. I decided to use the re-implementation, as it generally
outperforms the original on low-dimensional problems and optimization problems in this
thesis are defined as low-dimensional due to the hardware and time limitations.

#[
  #show table: zero.format-table(none, (digits: 4), (digits: 4), (digits: 4), (digits: 4))
  #figure(
    kind: table,
    table-with-notes(
      columns: (1fr, 1fr, 1fr, 1fr, 1fr),
      align: center + horizon,
      inset: 3pt,
      notes: [
        #super[1] Two-sided.\
        #super[2] Aggregated p-value from Fisher's method on one-sided p-values from presented tests.
      ],
      table.header(
        table.cell(rowspan: 2, [Function number\ in CEC’2017 set]), table.cell(colspan: 2,
        [DES]), table.cell(colspan: 2, [jSO]), table.hline(stroke: 0pt),
        table.hline(start: 1, stroke: 0.3pt), [p-value#super[1]], [Statistic],
        [p-value#super[1]], [Statistic],
        table.hline(stroke: 1pt)
      ),
      [1], [0.45552434024319266], [-0.7462368685261888], [1.0], [0.0],
      [2], [1.0], [0.0], [1.0], [0.0],
      [3], [1.0], [0.0], [1.0], [0.0],
      [4], [1.0], [0.0], [1.0], [0.0],
      [5], [0.7103059221927315], [-0.37144525742783385], [6.035487568844058e-08], [5.417746592573541],
      [6], [0.5270857869927841], [-0.6324608437284739], [1.0], [0.0],
      [7], [0.6322740412104184], [0.4785285748845067], [6.503977252321083e-08], [5.404361177891456],
      [8], [0.10246466604382465], [1.6330205912142606], [3.2811864117889187e-07], [5.106535701215085],
      [9], [1.0], [0.0], [1.0], [0.0],
      [10], [0.9014616869792204], [0.12381508580927794], [2.6108085446085493e-12], [6.997225525059465],
      [11], [0.7103059221927315], [-0.37144525742783385], [7.449729398523342e-07], [4.9492570787005965],
      [12], [0.9439760323236952], [0.07027342708094153], [0.0595653655064729], [1.8839971165033373],
      [13], [0.294911121390416], [1.047408698873081], [7.214850578275691e-09], [5.785845496330853],
      [14], [0.8225972515819387], [-0.22420569592490872], [4.233593653685743e-11], [6.595663084596941],
      [15], [0.8018322585394155], [-0.25097652528907693], [0.014707935415592826], [2.4394918258098275],
      [16], [0.9439760323236952], [-0.07027342708094153], [4.7471001841380774e-07], [5.036262274134144],
      [17], [0.40470822372478277], [0.8332420639597353], [1.1675551540879605e-15], [8.007824333556814],
      [18], [0.680631530912969], [-0.41160150147408614], [0.9386504858414535], [-0.07696613442198359],
      [19], [0.13822467304401687], [-1.4824346760408142], [1.7157362377626246e-10], [6.3848428033541165],
      [20], [0.3334959395857826], [-0.9670962107805764], [0.0018366120289253305], [3.115455267255075],
      [21], [0.796662193258135], [-0.257669232630119], [0.010983328560786158], [2.5432287895959793],
      [22], [0.7812063064399718], [0.2777473546532451], [7.616270101872297e-13], [7.167889562256037],
      [23], [0.10821904373109499], [1.6062497618500924], [1.537826704331319e-10], [6.401574571706722],
      [24], [0.0030612163029477978], [2.9615229984111076], [0.9067630763276631], [0.1171223784682359],
      [25], [0.3972027926912226], [0.8466274786418194], [0.6466288513925331], [0.45845045286138053],
      [26], [0.6708462126762573], [-0.42498691615617024], [0.7328566152664366], [0.3413280743931446],
      [27], [0.13822467304401687], [-1.4824346760408142], [1.6241145478155159e-06], [-4.79532480985663],
      [28], [0.603980549861837], [-0.518684818930759], [0.7028439051398435], [-0.3814843184393969],
      [29], [0.5671685517514178], [-0.5722264776590954], [3.096915878542318e-17], [8.442850310724548],
      [30], [0.8592281814900367], [-0.17735674453761435], [1.1765764716365598e-11], [6.783058890146119],
      table.hline(stroke: 0.3pt),
      [Fisher's p-value#super[2]], table.cell(colspan: 2, [0.47964320313813663]), table.cell(colspan: 2, [6.793993449424372e-112])

    ),
    caption: [Results of two-sided Wilcoxon rank-sum tests between different implementations of DES and jSO algorithms.],
  )<tab:implementation_verification>
]

#figure(image("../images/ecdfs.svg"),
  caption: [
    ECDF curves for various implementations of selected heuristic
    algorithms on specific CEC'2017 functions. Re-implementations in
    Python language have suffix 'py'.
  ]
)
<fig:impl_verification>

== Results for Aspartus dataset
<results-for-aspartus-dataset>

ECDF curves are illustrated in @fig:aspartus_ecdfs and numeric results can be found in
@tab:aspartus_ecdfs. While DES and CMA-ES have similar results, with the former slightly
outperforming the latter, there is a noticeable difference between them and the jSO
algorithm. All three optimizers locate similar maxima, however it takes jSO
significantly much more time to arrive at such solution. Early results are comparable
for all algorithms, with DES having a better starting position, probably due to its
initialization strategy (uniform distribution between 0.1 and 0.9 for each
hyperparameter). Worse performance of jSO can have its source in the suboptimal
parameters set for this algorithm -- default values are used.

#let auc-table(dataset: none, ..args) = {
  figure(
    table(
      columns: (1fr, 1fr, 1fr),
      align: center,
      inset: 5pt,
      table.header(
        [Heuristic algorithm], [EAUC], [NEAUC]
      ),
      ..args
    ),
    caption: [ECDF AUC values resulting from optimization on the #dataset dataset. Each curve has been created based on ten independent runs of the experiment.],
  )
}

#auc-table(
  dataset: [Aspartus],
  [DES], [662.44], [0.95],
  [CMA-ES], [661.46], [0.94],
  [jSO], [654.58], [0.94],
)<tab:aspartus_ecdfs>

#figure(image("../images/aspartus_3_runs/ecdfs.svg"),
  caption: [
    ECDF curves resulting from optimization on the Aspartus dataset.
  ]
)
<fig:aspartus_ecdfs>



To better understand hyperparameter values selected by different heuristics their
distributions are plotted in @fig:aspartus_distributions. Note that these distributions
vary between runs and algorithms, as there does not necessarily need to be a single
optimal value for hyperparameter, but rather different values are acceptable and it is
their combination that directly influences classifier's results. This is the case with
hyperparameters such as dropout, learning rate's decay or hidden layer size. On the
other hand, optimizers converge on the similar values of learning rate and Nesterov's
momentum, which indicates that these hyperparameters might be the biggest factors in
proper classifier training. Because regularization can effectively be modeled by either
activity or kernel (weight) regularization it should be the case that only one of these
ratios is set high, while the other remains low, or both have moderate rates, assuming
that there is a need for the regularization in the first place. Observing high rates of
both types of regularization is highly unlikely, due to the nature of the study --- such
regularization would substantially slow down learning process and thus the results
achieved after training process runs out of budget would be worse, even if progress was
successfully being made. Such balance can be seen in this experiment for all heuristics.
jSO and DES go with more polarizing strategies, while some of CMA-ES runs yield more
balanced rates. All optimizers opt for high learning rates, strongly indicating that in
the future optimizations on this architecture and dataset, range for learning rate could
be limited, possibly resulting in better initial performance of the optimization
process.

#multipage-subfigures(
  ("aspartus_3_runs/des_distributions.svg", [DES]),
  ("aspartus_3_runs/cmaes_distributions.svg", [CMA-ES]),
  ("aspartus_3_runs/jso_distributions.svg", [jSO]),
  caption: [
    Hyperparameters' distribution estimation resulting from optimization on the Aspartus
    dataset. Plots created based on the last 5 generations. Numbers in the legend
    correspond to the number of the experiment run.
  ],
  label: <fig:aspartus_distributions>
)

Visualizations of the best individuals throughout optimization processes can be found in
@fig:aspartus_3_runs_best. All algorithms achieved comparable results in terms of the
best individuals throughout entire optimization process, being placed between 0.72 and
0.73 RAUC. DES shows tendency to steadily improve its elite over time, whereas jSO
allows for significant oscillations of the elite, indicating stronger mutations of the
individuals between generations. Latter technique is also feasible, as indicated by
the end results. CMA-ES rapidly improves its elite in the first 20 epochs, after which
the progress slows down.

#multipage-subfigures(
  ("aspartus_3_runs/des_best.svg", [DES]),
  ("aspartus_3_runs/cmaes_best.svg", [CMA-ES]),
  ("aspartus_3_runs/jso_best.svg", [jSO]),
  caption: [
    Best fitness in population (dashed lines) and 'running' best fitness (solid lines)
    found during optimization on the Aspartus dataset. Color associated with the number
    of the run is depicted in legends.
  ],
  label: <fig:aspartus_3_runs_best>
)

Plots of the logarithmic loss are presented in @fig:aspartus_3_runs_losses. Plots of
mean values have been created by selecting the best individual in each run of the
experiment for each optimizer and calculating a 'running' mean for each optimizer.
'Best' plots are created for selecting the best individual across all runs of the
experiment for each optimizer. There are several interesting conclusions that can be
made based on these backlog plots, as listed below.

- It seems that selected number of epochs (60) is larger than needed for the classifier
  to achieve an optimal solution. Specifically, the best individuals for both DES and
  jSO are stopped before reaching maximal number of epochs.

- While all optimizers do fairly well in terms of managing overfit amongst their elite
  (as can be seen in the means plot), the best solutions do vary in this regard. DES is
  marginally better than jSO, however CMA-ES yields individuals which do not control
  overfit well.

- Overall training histories of elites show that all optimizers find similar solutions
  (DES and jSO having the closest characteristics), which could indicate an optimum
  being reached.

#multipage-subfigures(
  ("aspartus_3_runs/mean_loss.svg", [Mean]),
  ("aspartus_3_runs/best_loss.svg", [Best]),
  caption: [
    Loss histories for selected individuals found during optimization on the Aspartus
    dataset. Dotted vertical lines in the means plot denote early training termination
    of a classifier.
  ],
  label: <fig:aspartus_3_runs_losses>
)

RAUC history plots over selected individuals' training are depicted in
@fig:aspartus_raucs. Overfit can be seen more easily in these plots, especially for
CMA-ES. While CMA-ES does not cope well with the overfit problem, its models have
slightly smoother history curves, possibly indicating stronger regularization.
Additionally, it is easier to observe jumps in quality of the classifier as opposed to
the loss graphs.

Differences between RAUC achieved on the validation and test subsets can stem from
multiple reasons:

- Due to the way objective function is set up optimization process will create
  individuals, which are overfitted on the validation subset. When these individuals are
  trained on a bigger input data and tested against new subset their performance could
  be worse.

- Even though test, train and validation subsets are created in a stratified fashion,
  quality of the data might vary and so it might happen, that test subset contains more
  samples which are hard to classify properly. This could explain the observed
  difference, especially as the Aspartus dataset has been built upon real-life data.

#multipage-subfigures(
  ("aspartus_3_runs/mean_rauc.svg", [Mean]),
  ("aspartus_3_runs/best_rauc.svg", [Best]),
  caption: [
    RAUC histories for selected individuals resulting from optimization on the Aspartus
    dataset. Dotted vertical lines in the means plot denote early training termination
    of a classifier.
  ],
  label: <fig:aspartus_raucs>
)

Finally, results of the selected models are visualized in @fig:aspartus_raucs_eers and
@fig:aspartus_rocs. Exact results of all experiments can be found in the Appendix,
@tab:aspartus_rocs. Additionally to these models, a Logistic Regression with l2 penalty
classifier is trained on the merged training and validations subsets and evaluated on
the test subset. It has been selected due to the best performance achieved amongst
several models in the original study on the given dataset and its results are presented
as a reference. jSO, which performed worse throughout most of the optimization process
(shown in @fig:aspartus_ecdfs), has comparable results. Most individuals are equally
good or better than reference model (logistic regression). All heuristics outperform
reference classifier, with best individuals yielding AUC improvements of $0.03$.
Comparing EER values it can be noted that both DES and jSO perform best with mean EER
value of 0.364, closely followed by CMA-ES with 0.374. The lowest achieved EER for an
individual is 0.346, yielded by both CMA-ES and jSO. There is a notable difference
between Logistic Regression model and optimized individuals, namely 0.02 and 0.03
respectively for CMA-ES and both jSO and DES. $0.04$ difference is noted for the best
individuals.

#multipage-subfigures(
  ("aspartus_3_runs/rocs_values.svg", [ROC AUC]),
  ("aspartus_3_runs/eers_values.svg", [Equal Error Rate]),
  caption: [
    Box plots of RAUC and EER values for each run of the experiment on the Aspartus
    dataset. Each dot denotes a single experiment run. Purple, dashed line marks result
    yielded by the reference classifier.
  ],
  label: <fig:aspartus_raucs_eers>
)

#figure(image("../images/aspartus_3_runs/rocs.svg"),
  caption: [
    Mean ROC curves for the best solutions found during optimization on the Aspartus
    dataset. Curve for Logistic Regression model is provided as a reference.
  ]
)<fig:aspartus_rocs>

== Results for the Icebergs dataset
<results-for-the-icebergs-dataset>
Due to the time and hardware constraints this experiment has not been repeated multiple
times, and as such its results can not be used to directly compare different optimizers.
However, they can be used to test the possibility of optimization --- if the results are
bad then it can be argued that run experiment was an outlier, which could be detected if
multiple repetitions had been evaluated. On the other hand, if the results are good then
it is reasonable to assume that such heuristic algorithm is able to correctly tune
hyperparameters on the Icebergs dataset. Notably, this is also the primary way in which
user would be tuning the hyperparameters -- with a one-shot approach.

ECDF curves for optimizers are illustrated in @fig:icebergs_ecdfs and numeric results
are written in @tab:icebergs_ecdfs. jSO starts in the significantly worst place, but
quickly improves its performance in the first 100 function evaluations. Going further,
the progress is slower, but the process does not plateau. DES shows a flat start and for
the first 40 function evaluations it is unable to improve. Its performance then slowly
ramps up and in the end almost catches up with CMA-ES. CMA-ES has the best starting
position and it significantly outperforms other algorithms in the first 350 function
evaluations. However, further progress is much slower.

#auc-table(
  dataset: [Icebergs],
  [DES], [567.14], [0.81],
  [CMA-ES], [610.54], [0.87],
  [jSO], [535.50], [0.77],
)<tab:icebergs_ecdfs>

#figure(image("../images/icebergs/ecdfs.svg"),
  caption: [
    ECDF curves resulting from optimization on the Icebergs dataset.
  ]
)<fig:icebergs_ecdfs>


Distribution plots for different hyperparameters are depicted in
@fig:icebergs_distributions. All optimizers favour learning rate around $10^(-3)$, which
is vastly different than general learning rate chosen for both Aspartus and Titanic
datasets. This may indicate that lower learning rate allows for much smoother training
phase. This could also be a result of a differently defined objective function -- for
this dataset tuning algorithms optimize log loss on the validation set, as opposed to
RAUC, which is the metric used for optimization on other datasets.

DES and jSO tend to set the second dropout value lower than the third one, while CMA-ES
keeps them at equal level (0.2), which is also a generally suggested lowest effective
dropout rate. This illustrates the trade-off between dropout that is too high
(effectively slowing down training process) and too low (might result in strong
overfit).

Regularization ratios vary significantly between optimizers, with DES opting for the
strongest regularization and jSO choosing the lowest ratios, particularly for the weight
regularization. One of the possible explanations for this behavior is that classifiers
do not have significant overfit problems, especially since multiple dropout layers are
introduced.

#figure(image("../images/icebergs/distributions.svg"),
  caption: [
    ECDF AUC values resulting from optimization on the Icebergs dataset. Each curve has
    been created based on ten independent runs of the experiment.
  ]
)<fig:icebergs_distributions>

Plots of the 'running' best and best-in-generation individuals are depicted in
@fig:icebergs_best. Differences in generation numbers arise from different population
sizes of each algorithm. There is a substantial difference between 'running' best and
best-in-generation for jSO in the last 15 generations. Plots for DES suggest that it
transits smoothly from exploration to exploitation phase. CMA-ES, on the other hand,
shows greater oscillations of its elite, but still manages to consecutively find better
individuals throughout the optimization process.

Mean score per generation, along with the respective standard deviation is plotted in
@fig:icebergs_means. Out of all algorithms, CMA-ES is the most likely to significantly
worsen its average performance, possibly caused by step size expansion, combined with
boundary crossing. To an extent, similar behavior is observable for DES, although with
smaller deviation from the reasonable solution. This can be explained by the fact that
DES has bigger population size, and so only part of the population can be put out of
boundaries, or into the worse place in the search space. DES has a pretty smoothly
descending mean. jSO's characteristic does not contain such drastic peaks, but rather
has a fairly constant standard deviation. Its mean, however, can worsen from time to
time, and does so for several generations.

#figure(image("../images/icebergs/best.svg"),
  caption: [
    Best fitness in population and 'running' best fitness found during optimization on
    the Icebergs dataset.
  ]
)<fig:icebergs_best>


#multipage-subfigures(
  ("icebergs/means.svg", [Limited for CMA-ES]),
  ("icebergs/means_limited.svg", [Limited for DES and jSO]),
  caption: [
    Mean fitness in generation and standard deviation resulting from optimization on the
    Icebergs dataset.
  ],
  label: <fig:icebergs_means>
)

Logarithmic loss history for the best individuals is shown in @fig:icebergs_losses.
Histories for all individuals indicate that there is no problem of overfitting -- either
due to the nature of the dataset, or because it has been taken care of through proper
hyperparameters. A continuous improvement over the course of the training process is
observed, without reaching any plateau. This means that further improvement should be
observed if these individuals were given bigger training budget. Interestingly, all
optimizers decided upon a fairly low starting learning rate (~$10^(-3)$). In theory its
increase could lead to faster convergence, but could also cause problems with stability,
which might be an explanation for the chosen values.

#figure(image("../images/icebergs/loss.svg"),
  caption: [
    Loss histories for selected individuals found during optimization on the Icebergs
    dataset.
  ]
) <fig:icebergs_losses>

ROC curves for the best individuals are illustrated in @fig:icebergs_rocs and numeric
results can be found in @tab:icebergs_rocs. DES yields a significantly better individual
(over $0.04$ difference in log loss) than other two algorithms. This, however, only
means that all algorithms should be capable of reaching results presented, but might
have worse runs (there are no repeated runs of experiment for Icebergs dataset).
Notably, this difference is greater than the ones observed on other datasets. Despite
this fact both CMA-ES and jSO come up with individuals that can still be used to achieve
reasonable classification results (with RAUCs of roughly $0.91$).

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr),
    align: center + horizon,
    inset: 5pt,
    table.header(
      table.cell(rowspan: 2, [Heuristic algorithm]), table.cell(colspan: 2, [Receiver
      Operating Characteristics]), table.cell(rowspan: 2, [Logarithmic loss]),
      table.hline(stroke: 0pt), table.hline(start: 1, end: 3, stroke: 0.3pt), [Area
      Under Curve], [Equal Error Rate], table.hline(stroke: 1pt)
    ),
    [DES], [0.951], [0.094], [0.307],
    [CMA-ES], [0.919], [0.176], [0.351],
    [jSO], [0.912], [0.176], [0.347],
  ),
  caption: [Results of optimization on the Iceberg dataset.],
)<tab:icebergs_rocs>

#figure(image("../images/icebergs/rocs.svg"),
  caption: [ROC curves for the best solutions found on the Iceberg dataset.]
)<fig:icebergs_rocs>


== Results for the Titanic dataset
<results-for-the-titanic-dataset>
Results for the Titanic dataset are presented for 10 separate repetitions of
optimization process for each algorithm. ECDF curves are depicted in @fig:titanic_ecdfs.
It can be observed that the quality of initial populations vary greatly for this
dataset. CMA-ES shows a strong start and steady improvement in the first 70 generations.
Both DES and jSO despite notably worse initial populations rapidly reach a 0.83
threshold in a similar manner. At this stage the two diverge, with DES eventually
reaching the level of CMA-ES. jSO, on the other hand, improves slowly, but ultimately
scores worse than the other two algorithms. Early results favor CMA-ES, but after
several hundred function evaluations DES is also a viable algorithm in terms of the
population quality. Difference between the two in terms of NEAUC are minor (0.01), with
jSO obtaining worse results (0.04 worse NEAUC than DES).

#auc-table(
  dataset: [Titanic],
  [DES], [633.84], [0.91],
  [CMA-ES], [645.17], [0.92],
  [jSO], [605.66], [0.87],
)<tab:titanic_ecdfs>

#figure(image("../images/titanic/ecdfs.svg"),
  caption: [ECDF curves resulting from optimization on the Titanic dataset.]
) <fig:titanic_ecdfs>


Distribution plots for each algorithm are presented in @fig:titanic_distributions. In
terms of architecture, it seems that hidden layer size distributions vary mildly between
algorithms and experiment runs. Around 300 neurons mark a soft border, which is crossed
by only 2 experiment runs. In contrast to the Aspartus dataset, regularization rates are
in general lower, indicating less overfit tendencies during classifier's training. This
is further validated by, on average, lower dropout, which is predominantly centered
around 0.2 for DES and jSO. CMA-ES also provides two runs which yield dropouts over 0.4,
with the rest of experiments being placed around 0.2. Similar pattern can be observed in
Nesterov's momentum distribution, where all heuristics are almost uniformly centered
around 0.75, with CMA-ES again being placed roughly 0.05 to the right. Learning rate
values are high (between $10^(-2)$ and $10^(-1)$). This trend is even stronger than
the one seen for the Aspartus dataset, and it further strengthens the theory that
shorter range for this hyperparameter should've been chosen, under assumed conditions
(budget, architecture etc.). Learning rate decay is uniformly distributed over different
runs of experiments, highlighting the fact that such values are too low to have a real
impact on the training process and that wider range should be used in the optimization
process.

#multipage-subfigures(
  ("titanic/des_distributions.svg", [DES]),
  ("titanic/cmaes_distributions.svg", [CMA-ES]),
  ("titanic/jso_distributions.svg", [jSO]),
  caption: [
    Hyperparameters' distribution estimation resulting from optimization on the Titanic
    dataset. Plots for the first and last five generations. Numbers in the legend
    correspond to the number of the experiment run.
  ],
  label: <fig:titanic_distributions>
)


Plots of fitness for the 'running' best individual and best individual in generation are
presented in @fig:titanic_best. Interestingly, despite having a strong early lead in
terms of ECDF score, first 10 generations of CMA-ES seem to greatly vary in the
performance of best individuals. This is partially due to the fact that CMA-ES has the
smallest population size (9) and in parts caused by the nature of the algorithm itself.
However, around the 15th generation it starts converging and shows a stable, but small
improvement over time, with a limited variation between best individuals in each
generation. On the other hand, early on DES improves in a slower and less aggressive
fashion, but seems to be carrying the momentum and steadily improving over time. jSO's
has the least stable learning characteristic, with significant variations of the best
individual's fitness in each generation. Nonetheless, this behavior is consistent with
algorithm's design, as it is the only tested algorithm with changing population size. It
starts with a large population and this is visible in the performance of best
individuals in early generations, and it shrinks the population size significantly over
time, thus resulting in greater variance in the best individual, as there is smaller
number of individuals to choose the best from. Results show that when using provided
parameters on the given problem, jSO prefers exploration over exploitation.

#multipage-subfigures(
  ("titanic/des_best.svg", [DES]),
  ("titanic/cmaes_best.svg", [CMA-ES]),
  ("titanic/jso_best.svg", [jSO]),
  caption: [
    Best fitness in population (dashed lines) and 'running' best fitness
    (solid lines) found during optimization on the Titanic dataset.
    Color associated with the number of the run is depicted in legends.
  ],
  label: <fig:titanic_best>
)

Overfit is significant for the Titanic dataset, based on the loss history graphs for
best individuals, shown in @fig:titanic_losses. In both mean and best plots there is a
gap of roughly 0.1 loss difference between training and validation subsets. No algorithm
is able to effectively bridge this gap. Moreover, loss on validation subset is
oscillating, which might indicate too high learning rate or too small learning rate
decay. Another interesting observation is the number of epochs reached. All models were
given 150 epochs for the training process, however best solutions never reach more than
60 epochs --- their training is stopped when they do not show any improvement on the
validation set over the span of 20 epochs.

#multipage-subfigures(
  ("titanic/mean_loss.svg", [Mean]),
  ("titanic/best_loss.svg", [Best]),
  caption: [
    Loss histories for selected individuals found during optimization on the Titanic
    dataset. Dotted vertical lines in the means plot denote early training termination
    of a classifier.
  ],
  label: <fig:titanic_losses>
)

Analogously to loss histories, RAUC histories show great similarities between
individuals from different optimizers. These plots can be seen in @fig:titanic_raucs.
Overfit is also clearly visible, and no optimizer seems to be dealing well with it.
Curvature and oscillation of the plots highlights one of the potential problems with the
way individuals are assessed, namely choosing RAUC value at the last epoch, as it does
not take into the consideration the shape of the history plot (characteristic of the
training). Rather the optimizer tries to set such hyperparameters that will ensure
terminating training process after the right (highest RAUC yielding) epoch.

#multipage-subfigures(
  ("titanic/mean_rauc.svg", [Mean]),
  ("titanic/best_rauc.svg", [Best]),
  caption: [
    RAUC histories for selected individuals resulting from optimization
    on the Titanic dataset. Dotted vertical lines in the means plot
    denote early training termination of a classifier.
  ],
  label: <fig:titanic_raucs>
)


Results on the test subset can be seen in @fig:titanic_raucs_eers and @fig:titanic_rocs.
Full results of all experiments can be found in the Appendix, @tab:titanic_rocs. Both
DES and CMA-ES yield very similar RAUC results (medians of respectively 0.871 and
0.872), with the former having slightly greater variation. jSO seems to perform better
(median of 0.878), with a crucial exception of one outlier run, which scores noticeably
worse (0.859 RAUC). Outliers are also present for EER results, this time in both
directions, and for all optimizers. CMA-ES and jSO have identical medians (0.182),
while the median for DES is slightly worse -- 0.185.

#figure(image("../images/titanic/rocs.svg"),
  caption: [
    Mean ROC curves for the best solutions found during optimization on the Titanic
    dataset. Curve for Logistic Regression model is provided as a reference.
  ]
)<fig:titanic_rocs>

#multipage-subfigures(
  ("titanic/rocs_values.svg", [ROC AUC]),
  ("titanic/eers_values.svg", [Equal Error Rate]),
  caption: [
    Box plots of RAUC and EER values for each run of the experiment on
    the Titanic dataset. Each dot denotes a single run of the experiment
  ],
  label: <fig:titanic_raucs_eers>
)
