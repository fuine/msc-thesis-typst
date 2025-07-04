= Introduction
<introduction>
In the past few years neural networks became a very popular classification method in the
field of machine learning. They have been successfully used in a variety of different
settings, such as medicine~@Chi2017#cite(label("DBLP:journals/corr/abs-1709-00753"));,
physics~@deOliveira:2017pjk and machine translation~@wu2016google. One of the crucial
parts of neural network's training is the process of _hyperparameter tuning_, during
which user tweaks hyperparameters of a model with fixed architecture (either with
pre-trained weights or trained from scratch) to achieve the best performance possible.
This process can be both mundane and lead to sub-par quality of the final classifier, as
humans do not always have the necessary knowledge to effectively tune the
hyperparameters, or lack the ability to do it if the model is complex enough, which can
result in high dimensionality of the objective function used in the optimization
process.

On the other hand, there is a branch of algorithms that specialize in mathematical
optimization, particularly in cases which are computationally challenging, either due to
extensive parameterization or to the nature of the optimized function. They are called
algorithms, as usually the solution provided as a result of optimization is
sub-optimal, while being acceptable for the user.

Combining these two ideas was described in an earlier study~@2016arXiv160407269L, which
heavily influenced this thesis. Its authors compare one heuristic algorithm (Covariance
Matrix Adaptation Evaluation Strategy) with more traditional approaches (described
in~#link(<sec:related>)[1.2];). I focus on heuristic algorithms exclusively and test
these methods on a selection of various, real-life datasets.

== Main hypothesis
<main-hypothesis>

#block(spacing: 2em)[
*Can Differential Evolution Strategy and jSO algorithms perform on par with Covariance
Matrix Adaptation Evaluation Strategy when tuning neural network's hyperparameters?*
]

Detailed hypotheses are described below.

- Do optimization processes differ in their characteristics between tested heuristic
  algorithms?

- Can optimal hyperparameters' subranges be found, assuming non-Lamarckian approach with
  fixed, high penalty for boundary crossing?

- How much does complexity of the tuned network influence heuristic algorithms' ability
  to tune?

- Can heuristic algorithms be used to find values of hyperparameters, which are
  infeasible for the given classification task?

- Is 'one-shot' optimization viable, as opposed to repeated optimization?

== Related work
<sec:related>
Autonomous hyperparameter tuning has been explored for at least 20 years, with several
different approaches being developed.

#heading(level: 4, numbering: none)[Grid search]
<grid-search>
Grid search is one of the simplest techniques used for hyperparameter tuning. After user
provides sets of values for each hyperparameter, all possible combinations are tested.
This approach does not scale well and it requires user to pick specific values for each
hyperparameter. Moreover, if user provides value that results in bad performance, all
combinations using it will be tested, even if all such combinations are infeasible.

#heading(level: 4, numbering: none)[Random search]
<random-search>
A slightly different technique, called random search, can be used to achieve much better
results. With this approach, user can either provide the set of values, or a
distribution for tuned hyperparameters. Furthermore, a budget (i.e. number of model
evaluations) should be defined. Random search creates individuals in the following
manner: for each hyperparameter, if distribution was provided then sample the
distribution, otherwise uniformly sample provided set of values. Next, evaluate the
model and save results. This process is repeated until the number of evaluated models
reaches set budget. In the end, the best found individual is chosen. While simple, this
method shows surprisingly good performance, as described
in~@Bergstra:2012:RSH:2188385.2188395.

#heading(level: 4, numbering: none)[Bayesian optimization]
<bayesian-optimization>
This approach uses Bayesian method based on Gaussian processes to tune hyperparameters.
The basic idea behind this technique is to use all available information from previous
function evaluation to inform the process of choosing new hyperparameters. By their
nature these techniques require, that a prior over objective function is chosen, as well
as an acquisition function is defined, in order to determine next individual for
evaluation. These methods are known to perform well and have been extensively studied.
Examples of this approach are TPE~@NIPS2011_4443,
SMAC~#cite(label("10.1007/978-3-642-25566-3_40"));, or the works by Snoek et
al.~@pmlr-v37-snoek15@NIPS2012_4522.

#heading(level: 4, numbering: none)[Evolutionary optimization]
<evolutionary-optimization>
Solution based on evolutionary optimizations often combine hyperparameter tuning with
structure evolution. This approach is called neuroevolution, and has been studied quite
extensively over the years. Examples of these strategies
are~#cite(label("10.1007/3-540-58484-6_288"));@ROSTAMI201750#cite(label("DBLP:journals/corr/abs-1711-09846"));.
Usually these solutions are based on modified genetic algorithms, which are tailored to
evolve the architecture of the classifier during the course of optimization. However,
standard state-of-the-art heuristic optimizers are not widely used for the task of
hyperparameter optimization using fixed structure of the neural network, as described
in~@2016arXiv160407269L. This research focuses on Evolution Strategies and Differential
Evolution to better understand their applicability for hyperparameter tuning.

== Thesis layout
<thesis-layout>
This thesis is divided into 4 chapters.

+ Introduction --- outline of the problem, main hypothesis and related work.

+ Methods, datasets and tools --- descriptions of used algorithms (heuristics and
  classifiers), datasets for classification, as well as various statistical tools used
  to evaluate heuristics and classifiers.

+ Results --- verification of algorithms' re-implementations and results of the carried
  experiments.

+ Summary --- conclusions and ideas for future research.
