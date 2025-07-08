= Summary
<ch::summary>
== Final conclusions
<final-conclusions>
The main goal of the study was to compare CMA-ES, DES and jSO algorithms and evaluate
their performance in the neural network's hyperparameter tuning problem. Based on the
carried out experiments I conclude that all three algorithms are suitable for
hyperparameter optimization task for MLP and shallow CNN architectures. Notably, there
is a lack of substantial differences in performance amongst best found individuals by
different optimizers. Moreover, results are comparable amongst multiple runs of an
experiment using the same optimizer, which indicates that these algorithms can be used
in a 'one-shot tuning' fashion, without the need to repeat the optimization process.

While all algorithms would eventually yield similar results their optimization
characteristics differ. Usually DES and CMA-ES show greater similarity in the
optimization characteristics, with jSO slightly underperforming, throughout most of the
optimization. Therefore it is advised to use either CMA-ES or DES if hardware does not
allow for the prolonged tuning process or optimization time is a matter.

Another way to speed up the tuning process is reducing the number of optimized
hyperparameters. Depending on the classification problems, a dropout layer could not be
needed, or regularization could be turned off. On tested datasets learning rate decay
did not influence the results, and as such it could possibly be excluded from tuning.

Even though non-Lamarckian approach was used with severe penalty for crossing the
boundaries (effectively killing the individual), optimizers were able to choose
hyperparameter values near boundaries (e.g. learning rates in
@fig:aspartus_distributions and @fig:titanic_distributions).

As described in~@2016arXiv160407269L, CMA-ES can be used as an alternative for the
Gaussian optimization approaches -- after 5 minutes of training the model on the MNIST
dataset it has managed to find an individual, which achieved roughly $0.55 %$ validation
error, being only surpassed by the Gaussian TPE. However, after 30 minutes of training
it became the leading method, with validation error of $0.27 %$. Results obtained in
this thesis indicate that DES jSO algorithms can compete with CMA-ES optimizer on tested
datasets -- the difference between best individuals was minimal for 2 datasets
(difference of $0.001$ RAUC). The one-shot experiment showed better performance of DES
over comparable results of CMA-ES and jSO (respectively $0.951, 0.912$ and $0.912$
RAUC), although that difference could be at lest in part motivated by randomness. I
therefore conclude, that DES and jSO could be a valid alternatives for CMA-ES algorithm
for hyperparameter tuning problems, especially on problems with high dimensionality.

== Future research
<future-research>
#heading(level: 4, numbering: none)[Scalability with respect to
architecture and dataset used]
<scalability-with-respect-to-architecture-and-dataset-used>
While all tested algorithms managed to find satisfying solutions, datasets and
classifiers used were fairly minimal. It could be beneficial to run similar experiments
on bigger datasets and more complex architectures to see how does optimization process
scale with mentioned factors.

#heading(level: 4, numbering: none)[Scalability with respect to number
of tuned hyperparameters]
<scalability-with-respect-to-number-of-tuned-hyperparameters>
In theory, DES should scale better performance-wise with the number of optimized
hyperparameters. Seeing as deep neural networks have potentially dozens, if not hundreds
of hyperparameters, this could be exploited to achieve even better results by optimizing
higher number of these hyperparameters. This approach requires a significant
computational power, but is a way to naturally scale the optimization process
horizontally with more available hardware.

#heading(level: 4, numbering: none)[Different initialization of
population strategy]
<different-initialization-of-population-strategy>
Initialization can often have a significant impact on the quality of the early
solutions. Combined with the fact that often users have a good guess for decent
hyperparameters, it could be beneficial to change initialization strategies for
different algorithms. Obviously, such strategies touch on the trade-off between
exploration and exploitation, but I theorize that a good initialization strategy can
further boost optimization process. Proposed initialization strategies are:

- changing the distribution from which initial population is sampled for DES and jSO.
  Currently these algorithms use uniform distribution between the 5th and 95th
  percentile of optimized attribute's value. Given a starting point these approaches
  could use normal, beta or student's `t`-distribution;

- providing starting point directly to CMA-ES (currently the starting point chosen is
  the middle of the value ranges for all attributes);

- if multiple starting points are proposed, then these could be directly used as part of
  the initial population, while the rest of the population is sampled in a canonical
  way. This strategy could be easily used in DES and jSO optimizers.

#heading(level: 4, numbering: none)[jSO's parameters tuning]
<jsos-parameters-tuning>
Usually when a tuning algorithm is used to optimize hyperparameters, user does not tune
parameters of the optimizer itself. Parameters for jSO have not been tuned as a part of
this thesis and as such it is possible that different set of parameters could yield
overall better results. By researching the impact of jSO's parameters on its performance
in the hyperparameter optimization setting, an optimal set of parameters could be found,
for which satisfying results are reached on variety of datasets and architectures.

#heading(level: 4, numbering: none)[Improved objective function]
<improved-objective-function>
Objective function could be more sophisticated, i.e. taking under consideration the
slice of training history, rather than looking at the last training epoch. This could
lead to smoother learning curve of classifiers and could help with robustness of the
found individuals.

#heading(level: 4, numbering: none)[Difference between jSO
implementations]
<difference-between-jso-implementations>
Re-implementation of the jSO algorithm written in the Python language for this thesis
has in general been performing better, than the original implementation. The source of
this difference is unknown, as I have tried to keep the former identical to the latter.
One possible explanation could be the usage of vastly different pseudo random number
generators. Further researching differences between these two implementations could
possibly improve the jSO algorithm's performance on low-dimensional problems
(performance on high-dimensional problems were not tested).
