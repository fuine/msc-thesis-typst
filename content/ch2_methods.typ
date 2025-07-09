#import "../utils.typ": todo, ub, table-with-notes, algorithm, larrow, comment

= Methods, datasets and tools
<ch::methods>
== Heuristic algorithms
<heuristic-algorithms>
This section contains descriptions of three different heuristic optimizers used to tune
hyperparameters. The following symbols are used by convention in all algorithms'
listings and their descriptions:

- $lambda$ -- population size;
- $N$ -- dimensionality of the problem;
- $t$ -- generation number.

=== CMA-ES
<cma-es>
Covariance Matrix Adaptation Evolution Strategy (CMA-ES)@Hansen:2001:CDS:1108839.1108843
is a state-of-the-art heuristic optimization algorithm. It samples new candidates from a
multivariate normal distribution over real numbers. CMA-ES represents this distribution
internally by a covariance matrix. Optimization process consists of modifications of the
covariance matrix such that sampling of better individuals is achieved. Amongst
advantages of CMA-ES are fast convergence and relatively low number of required
parameters -- user should provide population size (authors suggest usage of $lambda = 4
+ 3 * log N$), initial individual and standard deviation of the starting population
$sigma$.

Pseudo-code for the CMA-ES is provided in~@alg:cmaes. Main loop of this heuristic
revolves around adaptation of three parameters to achieve a normal distribution, which
yields better individuals. Modified parameters are: the reference point
$ub(m)^t$, the covariance matrix $ub(C)^t$ and the step size
$sigma^t$. CMA-ES operates on two sets of points, namely $ub(d)_i^t$, called
#emph[base points] and $ub(x)_i^t$ -- #emph[individuals];. Base points are
sampled from the Gaussian distribution, with mean $0$ and described by covariance matrix
$ub(C)^t$. Individuals are created as a result of linear transformation of
base points. $D^t$ denotes population of $lambda$ base points, while $D_mu^t$ contains
$mu$ base points used to create the best individuals in the $t$th generation. To modify
covariance matrix, two different operations are combined, namely rank-1 and rank-$mu$
updates. After covariance matrix has been updated, step size is adapted. There are 5
constants used throughout optimization process:

- $c_s$ and $c_c$ -- cumulation constants, respectively for step size and covariance matrix;
- $c_1$ and $c_mu$ -- learning rates for rank-1 and rank-$mu$ updates;
- $d_sigma$ -- damping for step size.

These constants are calculated during initialization of the algorithm, using formulas
described in the original paper~@Hansen:2001:CDS:1108839.1108843.

#algorithm(caption: [Covariance Matrix Adaptation Evolution Strategy])[
- *Require:* $ub(m)^1$ initial solution
- *Require:* $sigma^1$ initial step size multiplier
+ $ub(p)^1 <- 0$
+ $ub(s)^1 <- 0$
+ $ub(C)^1 <- ub(I)$ #comment[Covariance matrix]
+ $t <- 1$
+ *while* !stop *do*
  + *for* $i <- 1$ *to* $lambda$ *do*
    + $ub(d)^t_i ~ N(0, ub(C)^t)$
    + $ub(x)^t_i <- ub(m)^t + sigma^t ub(d)^t_i$
  + #smallcaps[evaluate]$(X^t)$
  + sort $X^t$ according to their fitness
  + $ub(m)^(t+1) <- ub(m)^t + sigma^t angle.l D^t_mu angle.r$
  + $ub(s)^(t+1) <- (1 - c_s) ub(s)^t + sqrt(mu c_s (2 - c_s)) dot.op (ub(C)^t)^(-1/2) angle.l D_mu^t angle.r$
  + $ub(p)^(t+1) <- (1 - c_p) ub(p)^t + sqrt(mu c_p (2 - c_p)) dot.op angle.l D_mu^t angle.r$
  + $ub(C)^t_1 <- (ub(p)^t)(ub(p)^t)^T$ #comment[rank-1 update]
  + $ub(C)^t_mu <- 1/mu  sum^mu_(i=1) (ub(d)^t_i)(ub(d)^t_i)^T$ #comment[rank-$mu$ update]
  + $ub(C)^(t+1) <- (1 - c_1 - c_mu) ub(C)^t + c_1 ub(C)_1^t + c_mu ub(C)_mu^t$ #comment[update covariance matrix]
  + $sigma^(t+1) <- sigma^t exp(c_s/d_sigma ((norm(ub(s)^(t + 1)))/(E norm(N(0, ub(I)))) - 1))$ #comment[update step size]
  + $t <- t + 1$
]<alg:cmaes>

=== DES
<des>
Differential Evolution Strategy (DES)@7969529 is a hybrid between CMA-ES and
Differential Evolution (DE) class of algorithms. While it does not directly store nor
operate on the covariance matrix it uses a number of techniques to generate new
individuals resembling these of CMA-ES. DES strives to achieve much better performance
on problems with high dimensionality, as it avoids exponential complexity of specific
matrix calculations, which CMA-ES incurs. Additionally DES can be a parameter-free
method according to its authors, which is a desired feature from this research point of
view. Outline of the Differential Evolution Strategy is provided in~@alg:des. As opposed
to CMA-ES, DES does not have a notion of step size. Optimization starts with
initialization of the first population $X^1$, using a uniform distribution over
restricted range for each optimized component between set boundaries. Next, main loop of
the algorithm is entered, which starts with the evaluation of the population $X^t$ and
its midpoint $angle.l X^t angle.r$. Following the evaluation, a difference vector
$delta^t$ is created by calculating the difference between the midpoint of the whole
population $angle.l X^t angle.r$ and the midpoint of the best $mu$ points $angle.l
X_mu^t angle.r$. This vector is then saved in the $ub(p)^t$ vector. Adaptation of the
population is achieved through the repeated mutation of the elite's midpoint $X_mu^t$.
First random values $h_1$ and $h_2$ are sampled uniformly over the set ${ 1, dots.h, H
}$, with $H$ being a user-selected constant. The mutation vector $ub(d)^(t + 1)$
consists of four different components:

- the difference between two randomly selected individuals from the historical
  population $X^(t - h_1)$;
- random vector along the direction $delta^(t - h_1)$;
- random vector along the direction $ub(p)^(t - h_2)$;
- sample from the standard Gaussian distribution, scaled by $epsilon$.

DES has 4 constants: $epsilon$, $c_c$, $c_p$ and $H$, which are calculated during the
initialization phase, according to the formulas provided by authors in their original
work~@7969529.

#[
  #set par(leading: 0.3em) // change line spacing for the sublines of line 11
  #algorithm(caption: [Differential Evolution Strategy])[
  + $t <- 1$
  + $ub(p)^1 <- 0$
  + #smallcaps[initialize]$(X^1)$ #comment[Initialize first population]
  + *while* !stop *do*
    + #smallcaps[evaluate]$(X^t, angle.l X^t angle.r)$
    + $delta^t <- angle.l X^t_mu angle.r - angle.l X^t angle.r$
    + $ub(p)^t <- (1 - c_p) ub(p)^(t-1) + sqrt(mu c_p (2 - c_p)) delta^t$
    + *for* $k <- 1$ *to* $lambda$ *do*
      + pick at random $h_1, h_2 in {1, ..., H}$
      + $j, k ~ cal(U)(1, ..., mu)$
      + $ub(d)^(t+1)_i <- & sqrt(c_c / 2) (ub(x)^(t-h_1)_j - ub(x)^(t-h_1)_k) \
                          & + sqrt(c_c) delta^(t-h_1) dot.op N(0, 1) \
                          & + sqrt(1 - c_c) ub(p)^(t-h_2) dot.op N(0, 1) \
                          & + epsilon dot.op N(ub(0), ub(I))$
      + $ub(x)^(t+1)_i <- angle.l X^t_mu angle.r + ub(d)^(t+1)_i$
    + $t <- t + 1$
  ]<alg:des>
]

=== jSO
<jso>
jSO@7969456 is an algorithm for single-objective continuous optimization. It is an
improved version of the iL-SHADE algorithm, both classified as Differential Evolution
(DE) algorithms. It has been chosen as a representative of the DE class due to its win
in the IEEE Congress on Evolutionary Computation (CEC) competition from
2017~@MSU-CSE-06-2. Notably, jSO has more parameters, which could imply that it is not
as generic and easy to use in terms of hyperparameters optimization. It is also the only
algorithm with varying population sizes, as both CMA-ES and DES have fixed population
size. Initial starting population should contain $sqrt(N) * log N$ individuals
according to authors.

@alg:jso provides an outline of the jSO method. As it is an example of a DE class of
algorithms its core mechanism of optimization can be divided into three main parts.

#strong[Mutation] -- used to modify individuals in the population. Yields a mutant
vector $v_i^t$ according to a specific mutation strategy, called
DE/current-to-$cal(p)$Best-w/1:
$
  ub(v)_i^t = ub(x)_i^t + F_w (ub(x)_(p"Best")^t - ub(x)_i^t) + F (ub(x)_(r_1)^t - ub(x)_(r_2)^t)
$<eq:mutation_strategy>

where $r_1, r_2$ are random,
mutually different integers from the set ${1, dots.h, lambda}$,
$ub(x)_(p"Best")^t$ is a randomly chosen individual from the $cal(p)$ best
individuals in population, and $F_w$ is described as:
$ F_w = cases(
  delim: "{",
  0.7 * F\, & quad "if " italic("nfes") < 0.2 italic("max_nfes"),
  0.8 * F\, & quad "if " italic("nfes") < 0.4 italic("max_nfes"),
  1.2 * F\, & quad "otherwise"
) $ where $F$ is a control parameter, $italic("nfes")$ is number of evaluated objective
functions and $italic("max_nfes")$ is the budget (maximal number of allowed objective
function evaluations).

#strong[Crossover] -- the result of mutation (vector $ub(v)_i^t$) is used to
perform a crossover operation, which allows for swapping components between the original
individual and its mutated version in the following fashion:
$
  forall i in {1, ..., lambda}. forall j in 1, ..., D.ub(u)_(i, j)^t =
  cases(
    delim: "{",
    ub(v)_(i, j)^t\, & quad "if" r a n d (0\, 1) <= C R " or " j = j_(r a n d),
    ub(x)_(i, j)^t\, & quad "otherwise"
  )
$<eq:crossover>
where $C R in [0, 1]$ is a crossover parameter and $j_(r a n d) in {1, ..., N}$ is a
randomly chosen index, which guarantees that at least one component is taken from the
mutated version of the individual.

#strong[Selection] -- trial vector $ub(u)_i^t$, created by crossover, is then evaluated
using objective function $f$, and its fitness is compared to the fitness of the original
individual $ub(x)_i^t$. Better individual outlives the other, as it is included in the
future population:
$
  ub(x)_i^(t + 1) = cases(
    delim: "{",
    ub(u)_i^t\, & quad "if" f(ub(u)_i^t) <= f (ub(x)_i^t),
    ub(x)_i^t\, & quad "otherwise"
  )
$<eq:selection>

After these three stages archive is potentially shrunk, memories for $C R$ and $F$
parameters are updated, Linear Population Size Reduction (described in~@6900380) method
is applied and $cal(p)$ parameter is adapted.

jSO is characterized by its distinctive self-adaptation techniques for $C R$ and $F$
control parameters, which are described in lines 9-22, as well as shrinking of its
population size.

#algorithm(caption: [jSO algorithm])[
- *Require:* $d_sigma$ damping for step size
- *Require:* $p_(i n i t)$ initial $p$ rate
+ $t <- 1, p <- p_(i n i t)$
+ $ub(A) <- emptyset$ #comment[archive]
+ $M^i_F <- 0.5, thin i in {1, ..., H}$ #comment[initialize scaling factor memory]
+ $M^i_(C R) <- 0.8, thin i in {1, ..., H}$ #comment[initialize crossover control parameter memory]
+ *while* !stop *do*
  + $S_(C R) <- emptyset$
  + $S_F <- emptyset$
  + *for* $i <- 1$ *to* $lambda$ *do*
    + pick at random $r in {1, ..., H}$
    + *if* $r = H$ *then*
      + $M^r_F <- 0.9$
      + $M^r_(C R) <- 0.9$
    + *if* $M^r_(C R) < 0$ *then*
      + $C R^t_i <- 0$
    + *else*
      + $C R^t_i ~ cal(N)_i (M^r_(C R), 0.1)$
    + *if* $t < 0.25T_(M A X)$ *then*
      + $C R^t_i <- max(C R^t_i, 0.7)$
    + *else if* $t < 0.5T_(M A X)$ *then*
      + $C R^t_i <- max(C R^t_i, 0.6)$
    + $F^t_i ~ cal(C) (M^r_F, 0.1)$
    + *if* $t < 0.6T_(M A X)$ *and* $F^t_i > 0.7$ *then*
      + $F^t_i <- 0.7$
    + $ub(u)^t_i <-$ #smallcaps[current-to-pBest-w/1/bin] #comment[mutation and crossover, using @eq:mutation_strategy and @eq:crossover]
  + *for* $i <- 1$ *to* $lambda$ *do*
    + *if* $f(ub(u)^t_i) <= f(ub(x)^t_i)$ *then* #comment[selection, using @eq:selection]
      + $ub(x)^(t+1)_i <- ub(u)^t_i$
    + *else*
      + $ub(x)^(t+1)_i <- ub(x)^t_i$
    + *if* $f(ub(u)^t_i) < f(ub(x)^t_i)$ *then*
      + $ub(x)^t_i -> ub(A)$
      + $C R^t_i -> S_(C R)$
      + $F^t_i -> S_F$
    + Shrink $ub(A)$ if necessary
    + Update $M_(C R)$ and $M_F$
    + Apply Linear Population Size Reduction, as described in @6900380
    + $p <- p_(i n i t) (1 - (n f e s) / (2 m a x\_ n f e s))$ #comment[update $p$]
    + $t <- t + 1$
]<alg:jso>

=== Default parameters
<ssec:default_heuristic_parameters>
The default optimization algorithms' parameter values used in this research are depicted
in @tab:heuristic_params. If parameter is not specified in @tab:heuristic_params and its
value is not described in the experiment, a default value has been used based on the
reference paper. All heuristics optimize hyperparameters based on the provided allowed
values ranges. Non-lamarckian approach is assumed, with the harshest possible penalty
for boundary crossing, effectively killing individuals, which leave provided boundaries.

#figure(
  table-with-notes(
    columns: 3,
    align: center + horizon,
    inset: (y: 4pt, x: 10pt),
    notes: [
      #super[1] Number of objective function (classifier training) evaluations.
    ],
    table.header([Algorithm], [Parameter name], [Parameter value]),
    [DES], [$lambda$], [28],
    [CMA-ES], [$lambda$], [9],
    [CMA-ES], [$sigma$], [0.2],
    [CMA-ES], [initial parameter value], [0.5],
    [jSO], [archive rate], [1.0],
    [jSO], [memory size], [5],
    [jSO], [$cal(p)$Best rate], [0.25],
    table.hline(stroke: 0.3pt),
    [All], [budget#super[1]], [700],
    [All], [lower bound], [0.0],
    [All], [upper bound], [1.0],
  ),
  kind: table,
  caption: [Paramerers of heuristic algorithms used in the study.],
)<tab:heuristic_params>

== Datasets
<datasets>
Three different real-life datasets are used to compare heuristic hyperparameter
optimization. Due to their nature, achieving correct classification on them can be
challenging, with problems such as:

- high target class imbalance;
- poor quality of attributes;
- mislabeled samples;
- small number of samples in the dataset.

Optimizers should be able to correctly tune hyperparameters in order to account for
these factors, as mentioned obstacles are often found in real-life datasets. These
characteristics motivated the choice of datasets for the thesis.

#let dataset-class-table(dataset-name: none, ..args) = {
  figure(
    table(
      columns: 3,
      align: center,
      table.header(
        [Class], [number of records], [% in the dataset],
      ),
      ..args
    ),
    caption: [Proportions of the target class in the #dataset-name dataset.],
  )
}

=== Aspartus
<ssec:aspartus>
After a road accident takes place, the victim can call an insurance company of the
perpetrator for insurance claims. As a part of that process, insurance companies in
Poland are obliged to ask the victim whether or not their vehicle needs to be repaired,
and if so, it must propose a replacement vehicle, or cash compensation. The task of the
model trained on this dataset is to indicate which option will be taken by the victim.
Supplied dataset contains 12072 examples. Target class has 2 different values:

- *CASH* cash compensation chosen;
- *CAR* replacement car chosen.

Class proportions are presented in @tab:aspartus_classes_ratio.

#dataset-class-table(
  dataset-name: [Aspartus],
  [CASH], [1227], [10.16],
  [CAR], [10845], [89.84],
)<tab:aspartus_classes_ratio>

There are 110 attributes in the dataset, which provide information, amongst other
things, about victim's and perpetrator's vehicles, their homes' locations, date of the
accident, weather during the accident and forecast, upcoming holidays etc. All
attributes are described in-depth in~@aspartus.

For the purpose of this thesis only a selected group of attributes is used (33), all of
which are continuous, and it is assumed that the missing values have been filled by mean
of the attribute where applicable. This dataset is used for Multilayer Perceptron's
hyperparameter tuning, as outlined in @ssec:mlp.


=== Titanic
<sec:titanic_dataset>
A sample in the Titanic dataset describes a single passenger of the
fatal last cruise. It contains information about sex and age of the
person, their family aboard the ship, ticket class and price, as well as
the port of embarkation. Four of these attributes are continuous (age,
ticket price, number of siblings/spouses and parents/children aboard the
boat), while the others are discrete. One-hot encoding was used to
encode discrete attributes. Target class is binary and indicates
survival of the person. This is a small (1309 samples), benchmark
dataset, with mild target class imbalance. This dataset has been shared
by the Vanderbilt University's Department of
Biostatistics~@titanic_dataset.

#dataset-class-table(
  dataset-name: [Titanic],
  [SURVIVED], [500], [38.20],
  [NO_SURVIVAL], [809], [61.80],
)<tab:titanic_classes_ratio>

Hyperparameters for Multilayer Perceptron are tuned on this dataset (described in depth
in @ssec:mlp.


=== Icebergs
<sec:icebergs>
This dataset has been provided as a part of the 'Statoil/C-CORE Iceberg Classifier
Challenge' competition~@icebergs_kaggle hosted on the Kaggle platform. This dataset
contains satellite radar images of icebergs and ships. Each sample is described by two
vectors of floating point numbers, each of length 5625. These vectors represent radar
images from two bands, each being single channel with dimensions 75 x 75 in pixels. Each
pixel on the image represents value in decibels. Both channels are signals resulting
from radar backscatter with different polarizations - respectively HH (transmit/receive
horizontally) and HV (transmit horizontally and receive vertically).

Classifier's goal is to label given image, deciding if the sample is a ship or an
iceberg. Target class proportions are described in~@tab:icebergs_classes_ratio. Based on
this dataset, convolutional neural network classifier is optimized, as described in @ssec:cnn.

#dataset-class-table(
  dataset-name: [Icebergs],
  [ICEBERG], [753], [46.95],
  [SHIP], [851], [53.05],
)<tab:icebergs_classes_ratio>


== Classifiers
<classifiers>
Architectures of two different classifiers are described in this section. Moreover, each
subsection describes specific hyperparameters, that will be optimized.

To simplify optimization process, as well as comparisons of distributions of different
hyperparameters all individuals are represented as vectors of floating point numbers
ranging between 0 and 1. In order to convert such representation to the desired
hyperparameters' values a _transformation_ function is used (e.g. for a value $0.42$
obtained from an individual and a transformation function defined as $10^(-1 - 5*x)$ the
real hyperparameter value used for the neural network roughly equals to $7.94 *
10^(-4)$). These functions are presented for described architectures and
hyperparameters.

Weights of all evaluated models are randomly initialized (with identically seeded random
number generator) using Glorot's uniform initialization rule~@Glorot10understandingthe.


#let hyperparams-table(caption: none, ..args) = {
  figure(
    table-with-notes(
      columns: 4,
      align: center + horizon,
      inset: (y: 4pt, x: 10pt),
      notes: [
        #super[1] L2 regularization applied on `dense_1` and `dense_2` layers.
      ],
      table.header(table.cell(rowspan: 2, [Hyperparameter]), table.cell(colspan: 2,
      [Range]), table.cell(rowspan: 2, [Transformation]),
      table.hline(stroke: 0pt),
      table.hline(start: 1, end: 3, stroke: .3pt), [Lower], [Upper],
      table.hline(stroke: 1pt)),
      ..args
    ),
    kind: table,
    caption: caption,
  )
}

=== Multilayer Perceptron
<ssec:mlp>
In this study a Multilayer Perceptron (MLP) with one hidden layer is used. This is an
almost classical MLP design, which allows for non-linear classification, while being the
simplest to train and reason about. The architecture is not fixed completely, as one of
the tuned hyperparameters is the number of neurons in the hidden layer. The only
difference between used architecture and classic design is the fact that a single
dropout layer~@dropoutcit is added, between hidden and output layers. Due to its nature,
it does not change the way that predictions are made, but it changes training procedure,
preventing possible overfitting tendencies of the model. As the dropout ratio is tuned,
it is possible that this layer becomes effectively disabled, if that hyperparameter is
tuned to the values close to $0$. As such, heuristic optimizer can modify the
architecture of the model in two different ways.


Stochastic Gradient Descent~@kiefer1952 (SGD) with Nesterov's
Momentum~@Sutskever:2013:IIM:3042817.3043064 is used as an optimizer for the neural
network's training process.

This architecture has been chosen for tuning due to its popularity, especially in
low-dimensional datasets' classification, and simple architecture, which guarantees fast
training process.

All tuned hyperparameters for the Multilayer Perceptron classifier are described
in @tab:mlp_hyperparameters.

#hyperparams-table(
  caption: [Hyperparameters tuned for the MLP architecture.],
  [Hidden layer size], [1], [1001], [$x * 10^3 + 1$],
  [Weight regularization ratio#super[1]], [$10^(-6)$], [$10^(-2)$], [$10^(-2 - 4*x)$],
  [Activity regularization ratio#super[1]], [$10^(-6)$], [$10^(-2)$], [$10^(-2 - 4*x)$],
  [Dropout], [0.0], [0.9], [$x * 0.9$],
  [Learning rate], [$10^(-6)$], [$10^(-1)$], [$10^(-1 - 5*x)$],
  [Learning rate decay], [$10^(-8)$], [$10^(-3)$], [$10^(-3 - 5*x)$],
  [Nesterov's momentum], [0.0], [2.0], [$x * 2.0$],
)<tab:mlp_hyperparameters>

=== Convolutional Neural Network
<ssec:cnn>
To test optimizers' performance for more complex models I use a relatively simple
convolutional network, architecture of which is depicted on~@fig:cnn_architecture. It
is simple enough to allow for relatively fast training process, while lending itself to
effective small images classification (~100 x 100 pixels). All layers use ReLU
activation function, with the exception of the output layer, which uses sigmoid
activation function. Optimizer used for the CNN architecture is
ADAM~#cite(label("DBLP:journals/corr/KingmaB14"));. To prevent overfitting, dropout
layers and both weight and activity L2 regularization~@Ng:2004:FSL:1015330.1015435 are
used. Hyperparameters tuned for this architecture are depicted
in~@tab:cnn_hyperparameters.

#hyperparams-table(
  caption: [Hyperparameters tuned for the MLP architecture.],
  [`dropout_2` value], [0.0], [0.9], [$x * 0.9$],
  [`dropout_3` value], [0.0], [0.9], [$x * 0.9$],
  [Weight regularization ratio#super[1]], [$10^(-6)$], [$10^(-2)$], [$10^(-2 - 4*x)$],
  [Activity regularization ratio#super[1]], [$10^(-6)$], [$10^(-2)$], [$10^(-2 - 4*x)$],
  [Learning rate], [$10^(-5)$], [$10^(-1)$], [$10^(-1 - 4*x)$],
)<tab:cnn_hyperparameters>

#figure(image("../images/iceberg_architecture.svg", width: 67.0%),
  caption: [
    Convolutional Neural Network architecture used in the study. Numbers correspond to
    the input and output dimensions of data tensor, with _None_ being the placeholder
    for the number of samples.
  ]
)
<fig:cnn_architecture>

== Default experiment description
<default-experiment-description>
Unless explicitly stated otherwise all experiments are run in the fashion described
below.

#{
  set enum(numbering: "1.a)")
[
  + Initialize chosen heuristic algorithm. Its parameters are described
    in~@ssec:default_heuristic_parameters, or in the experiment description, if default
    parameters have not been used.

  + Create specific subsets, based on the dataset used in the experiment, in the following
    manner:

    + Split dataset into two parts using 80:20 proportions. Use stratification, as it
      preserves target class' proportions in all subsets. The smaller subset (i.e. 20% of
      the dataset) will be further referred to as _test_ subset.

    + Split the remaining 80% of the dataset using the same proportions (80:20) and
      stratification. The bigger of two subsets created that way will be referred to as
      _training_ subset, while the smaller one as _validation_ subset.

  + Define objective function as follows:

    + Given an individual from heuristic optimizer, transform it to obtain proper
      hyperparameters' values. Transformations are described separately for each
      classifier, and can be found in~@tab:mlp_hyperparameters and @tab:cnn_hyperparameters.

    + Create a classifier using provided hyperparameters.

    + Train the classifier on the training subset of the dataset.

    + Evaluate trained classifier on the validation subset.

  + Run heuristic optimization process until it runs out of budget (function evaluations).

  + Select the best individual from the history of the optimization.

  + Create a classifier using selected individual.

  + Train created classifier using combined train and validation subsets.

  + Evaluate trained model on the test subset.
]}

Each classifier starts with the fixed, known state of the pseudo-random number
generator, so that results are repeatable and starting weights are identical for all
individuals. There are 3 separate stopping conditions for the neural network's training
process:

- each model is given fixed amount of epochs, defined per dataset/experiment;

- each model has maximal CPU/GPU time budget, after which training is stopped regardless
  of the state of the training;

- training is stopped if the process plateaus, i.e. no significant change in logarithmic
  loss, calculated for the validation subset, is observed over the period of 20 epochs.

== Evaluation methods
<evaluation-methods>
Evaluation of experimental results should consider various characteristic of the
hyperparameter tuning process. More specifically, this process can be studied by
considering optimization itself, focusing on the quality of population, its elite,
convergence rate, etc. Furthermore, one could examine classifiers which are created and
tuned based on different individuals found by heuristics, exploring such things as
history plots of loss function and target metric, or classification quality of the
trained model. Evaluation methods for each approach are described in this section.

=== Heuristics evaluation methods
<heuristics-evaluation-methods>
Two different tools were used to visualize and compare heuristic optimization results:
Empirical Cumulative Distribution Function (ECDF)@6557689 and distribution plots for
each hyperparameter.

#heading(level: 4, numbering: none)[ECDF]
<sssec:ecdf_description>
ECDF is a function of the proportion of executed objective function evaluations when the
algorithm successfully reaches certain step. For all of the experiments described in
this thesis the following budget steps are assumed:
${0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}$,
where each number denotes percentage of the budget reached (e.g.~assuming budget of 100
objective function evaluations and step 0.3 this step would be reached after 30
objective function evaluations). These have been inspired by budget steps used by the
CEC competition. For each algorithm and for each run of the experiment, a 'running best'
individual history is created (i.e.~for each generation the best solution found so far
is selected). Thresholds span between worst individual in all histories in the first
iteration and the best individual in the last iteration. ECDF curves allow for
comparison of different heuristics, averaging through repeated runs of the experiment.
In order to quantify these curves an Area Under the Curve is used, referred to as EAUC
(ECDF AUC). Moreover, to further simplify comparisons I propose Normalized ECDF AUC
(NEAUC), such that its value is in the range of $[0, 1]$.

#heading(level: 4, numbering: none)[Hyperparameters distribution plots]
<hyperparameters-distribution-plots>
To visualize how heuristics optimize hyperparameters, distribution plots are created for
each hyperparameter, each illustrating last 5 generations. Kernel Density Estimation
with Gaussian kernel is used to estimate these distributions. To calculate bandwidth
Scott's method~@scott1992multivariate is applied:

$ N^(-1 / (D + 4)) $ where $N$ is the number of data points and $D$ is the number of
dimensions. Those distributions are then plotted using 256-point mesh.

=== Best and mean fitness plots
<best-and-mean-fitness-plots>
To track the behavior of the elite plots for the best individual in each generation, as
well as 'running' best individual (i.e. best individual found so far) are provided.
These plots can be used to further clarify the exploration vs. exploitation strategy
used by the heuristic and indicate trends in the optimization process.

In order to observe these characteristics with respect to the whole population mean and
standard deviation plots are presented for selected experiments. This technique has a
limitation, namely the plot loses on readability as the number of plotted experiment
runs increases, which is the reason why it is not shown for all experiments.

=== Classifier evaluation methods
<classifier-evaluation-methods>
Classifiers' performance is assessed using selection of tools: ROC
curves, logarithmic loss, as well as training history plots.

#heading(level: 4, numbering: none)[ROC AUC]
<roc-auc>
To evaluate neural network's performance on a binary dataset (i.e.~one with 2 target
classes) an Area Under the Curve (AUC) will be measured for the Receiver Operator
Characteristic (ROC) curve. ROC is a standard tool for graphical assessment of binary
classifiers. It is created by plotting True Positive Rate (TPR, also known as
sensitivity, recall) against False Positive Rate (FPR, also fall-out, $1 - s p e c i f i
t y$) at various thresholds. To generate these rates predicted probability of each
sample is mapped onto predicted class by thresholding it, using specified cut-off
probability. Afterwards, TPR and FPR can be calculated as:
$ T P R = (T P) / (T P + F N) quad quad F P R = (F P) / (F P + T N) $
where $T P$, $F P$, $F N$, $T N$ are numbers of True Positive, False Positive, False
Negative and True Negative samples.

After plotting ROC, its AUC can be measured, with random classifier and perfect
classifier yielding values $0.5$ and $1.0$ respectively.

In this thesis ROC's AUC will be referred to as RAUC in order to prevent confusion with
ECDF's AUC (EAUC).

#heading(level: 4, numbering: none)[Log loss]
<log-loss>
For non-binary datasets, models will be evaluated by the multinomial
cross-entropy loss (also known as logarithmic loss or log loss in
short):
$ cal(L)(theta) = -1 / n sum_(i = 1)^n sum_(j = 1)^m y_(i j) log(p_(i j)) $
where $n$ is the number of samples, $m$ number of target classes, $y_i$
is a one-hot encoded label of the sample (a binary vector with dimension
$m$, in which 1 denotes the target class, and 0 otherwise), $p_i$ is a
vector of probabilities for each class for the given sample such that
$ p_(i j) in (0, 1) : forall i sum_(j = 1)^m p_(i j) = 1 $

#heading(level: 4, numbering: none)[Training history plots]
<training-history-plots>
Another way to visualize final results of optimization is the plot of loss function
throughout classifier's training. In order to do so, each classifier stores a backlog of
selected metrics, as well as logarithmic loss for both training and validation subset,
calculated after each epoch of the training process. This backlog can then be plotted to
better understand how does the classifier learn, check for possible overfit, etc. Most
notably, these plots are the backbone of manual hyperparameter tuning, and so plotting
these backlogs for selected individuals might help to better understand characteristics
of the heuristic optimization process. In this thesis two different methods of plotting
these backlogs are used:

- Simple plots --- given a set of backlogs plot selected metrics for each of them, with
  distinct plot lines for train and validation subsets;

- Aggregations --- given a set of sets of backlogs (e.g. few selected individuals for
  each optimizer) first calculate a 'running' mean and standard deviation for each set
  of backlogs, and then use this information with the first described technique. Here I
  call the statistic 'running' if it is calculated at each step for all available
  backlogs. The reason behind this approach is that there are 3 distinct stopping
  conditions for the classifier and so backlogs can differ in length (number of epochs)
  recorded. In other words, a 'running' mean of a metric is a 1-D vector calculated for
  each epoch in all backlogs, where at each epoch all available values of a metric are
  averaged.

== Method for verification of the heuristics re-implementations
<method-for-verification-of-the-heuristics-re-implementations>
In order to easily integrate neural network related code with heuristic algorithms, an
implementation of DES and jSO algorithms in the Python programming language have been
provided. They were rewritten from the original projects, which used C++ (jSO) and R
language (DES). To test these re-implementations I compared them against Congress on
Evolutionary Computation (CEC) benchmark set from 2017~@MSU-CSE-06-2. Numeric results
are presented for all functions with the dimensionality of 10, representing different
modalities, as documented in~@tab:cec_functions_description. To account for the
randomness an optimization process is run 51 times for each of the functions, as
required by the benchmark rules, and results are aggregated.

#figure(
  table(
    columns: 2,
    align: center + horizon,
    inset: 5pt,
    table.header([Function number in CEC'2017 set], [Function type]),
    [1-3], [Unimodal],
    [4-10], [Simple Multimodal],
    [11-20], [Hybrid],
    [21-30], [Composition],
  ),
  caption: [Characteristics of CEC’2017 functions used for implementation validation.],
)<tab:cec_functions_description>


Two methods were used to compare these results: visual inspection of Empirical
Cumulative Distribution Functions (described in~@heuristics-evaluation-methods) and statistical
approach using a non parametric test. The second method uses best individuals found
throughout the optimization process. Specifically, for each heuristic and for each
function, an optimization process is independently run 51 times, yielding a 1-D vector
with 51 fitness values for each of the best individual in each run. Next, for each
benchmark function a two-sample Wilcoxon rank-sum test~#cite(label("10.2307/3001968"))
is calculated, using two vectors (one from each implementation) as input samples. These
p-values can then be aggregated by the means of a meta-analysis technique called
Fisher's method~#cite(label("10.2307/2681650"));@fishers_elston. It can be divided into
two phases. First, a $X^2$ test statistic is calculated: 
$ X_(2 k)^2 ~ - 2 sum_(i = 1)^k ln(p_i) $
where $p_i$ is the p-value for the $i$th test and $k$ is the number of tests. Next,
assuming that all tests have been independent (as is the case in the described
procedure), a final 'meta' p-value can be calculated, based on the fact that $X^2$
follows a chi-squared distribution with $2 k$ degrees of freedom.

To account for directionality of calculated rank-sum tests, their
p-values are changed into one-sided versions, while taking the
directionality under account. This is achieved using the following
formula:

$
p_"one-sided" = cases(delim: "{",
    p / 2\, & quad "if" Z > 0,
    1 - p / 2\, & quad "otherwise")
$

where $p$ is the two-tailed p-value and $Z$ is Wilcoxon rank-sum test's statistic.

== Language, tooling and hardware used
<language-tooling-and-hardware-used>
All experiments have been implemented using the Python programming
language, version 3.4, using the following libraries:

- `numpy` --- linear algebra library, on top of which re-implementations of DES and jSO
  have been built;

- `matplotlib` --- used to create all plots presented in the thesis;

- `deap` (Distributed Evolutionary Algorithms in Python) --- provides implementation of
  the CMA-ES algorithm, which was slightly modified to better suit the needs of this
  research;

- `keras` --- high-level library for neural network classification, aids the process of
  creating a classifier, abstracting over different computational backends;

- `theano` --- library, which provides efficient composition and optimization of
  complex, multi-array mathematical operations. Used as one of the backends for `keras`
  library;

- `tensorflow` --- machine learning framework, an alternative to `theano`, also used as
  a backend for `keras`;

- `scipy` --- framework for scientific computing, provides variety of statistical tests,
  i.a. implementation of Wilcoxon rank-sum test;

- `scikit-learn` --- machine learning toolkit library, implements various metrics (e.g.
  ROC AUC, confusion matrices generation), as well as Logistic Regression classifier
  used as a reference classifier.

Experiments for the Icebergs dataset have been run using GPU with `theano` backend for
the `keras` library. The rest of experiments have been run on CPU with 6 cores per an
experiment, using `tensorflow` as the backend. Evaluation platform is described
in~@tab:hal_hardware.

#figure(
  table(
    columns: 3,
    align: center + horizon,
    stroke: none,
    inset: 5pt,
    table.hline(),
    table.cell(rowspan: 2, [System]), [Kernel], "4.9.0-6-amd64 x86_64 (64 bit)",
    table.hline(start: 2, stroke: 0.3pt),
    [Distribution], [Debian GNU/Linux 9 (stretch)],
    table.hline(start: 2, stroke: 0.3pt),
    table.cell(colspan: 2, [CPUs]), [4x 12-Core Intel Xeon E7-4830 v3, L2 cache: 120 MB],
    table.hline(start: 2, stroke: 0.3pt),
    table.cell(colspan: 2, [GPU]), [NVIDIA Tesla K20m],
    table.hline(start: 2, stroke: 0.3pt),
    table.cell(colspan: 2, [Memory]), [252 GB],
    table.hline()
  ),
  caption: [Software and hardware used as a platform for experiments in the thesis.],
)<tab:hal_hardware>
