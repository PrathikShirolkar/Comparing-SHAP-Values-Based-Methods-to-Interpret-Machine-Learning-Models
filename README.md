# Abstract
As a part of EECS 545 (University of Michigan, Ann Arbor), in this project 
we explore explanation methods based on SHAP values which
allow to explain contributions of individual features to the overall prediction
on a machine learning method. We implement three methods: SHAP values,
Kernel SHAP, and Deep SHAP. We apply these methods to explain machine
learning models such as linear regression, XGBoost, and neural network for tabular
and image datasets. We verify first two methods implementation against ’shap’
Python’s library. Qualitatively, verification shows excellent agreement between
outcomes. Quantitatively, there is a discrepancy with the SHAP values method
results, possibly, caused by a difference in background treatment. The Deep SHAP
is verified by satisfying theoretical properties of the results. For validation, we
apply Deep SHAP to explain the neural network prediction of image classification.
The results are excellent and greatly agree with human intuition. The time efficiency
test confirms our expectations: the Kernel SHAP is faster than SHAP values. The
Deep SHAP is the most time efficient for neural networks, therefore it is applied to
image classification. The key learning points during implementations for us are:
importance of background choice and model averaging strategies. Also, now we
fully appreciate how powerful the SHAP values can be for revealing the individual
feature impacts to the model prediction.

# Motivation
Machine learning methods have demonstrated remarkable performances in many real-world applications such as 
computer vision, natural language processing, traffic prediction, etc. In many
tasks, there is a need to use complex machine learning (ML) models with high performance such as
ensemble and deep neural networks. For example, deep neural networks can significantly improve
image classification compared to other ML models (e.g. AlexNet [4]). However, unlike simple
models such as linear regression or decision trees, these ’black-boxes’ lack providing the association
between cause and effect. We know little about the underlying mechanism, i.e., how and why the
model makes a certain decision. This becomes crucial when such methods are used to make decisions
for high-stakes applications affecting individuals and social groups in real-life problems, e.g. in
ML-driven medical decisions, autonomous driving, and social-policy changes. The effect can be
negative and easily overlooked due to limitations in understanding the mechanisms of ML outcomes.
Thus, there is a pressing need to explain (or interpret) ML models.
In recent years, there were significant efforts in ML research community to develop methods to
explain ML models [3]. In artificial intelligence theory, one would distinguish interpretability and
explanability. Roughly, an algorithm is interpretable if its decision can be explained to a non-expert,
without necessarily knowing ’why’. Explanability reveals details trying to answer why a particular
decision was chosen. As mentioned in [6], correctly interpreting the predictions of a model can also
provide insight into how a model may be improved. This also motivates the research in explanability
and interpretability.

To address the need for universal interpreting methods, in 2017 Lundberg and Lee proposed the
unified approach for interpreting model predictions, called SHapley Additive exPlanations (SHAP) [6].
The novelty of the paper is in defining the class of additive feature attribution methods that unifies
explanation methods that were established previously: LIME [7], DeepLIFT [9; 10], layer-wise
relevance propagation [1], and classic Shapley value estimation methods [5; 11; 2]. The paper
proposed SHAP values as well as Kernel SHAP, Linear SHAP, Low-order SHAP, Max SHAP, and
Deep SHAP for approximating SHAP values. This leads to improvements of the existing methods
because the SHAP values provide the unique solution for feature contributions that satisfy the
desirable properties of local accuracy, missingness, and consistency. Noteworthy, the Kernel SHAP is
an approximation of SHAP values and reduces the computational complexity. Deep SHAP, designed
specifically for deep networks, has even lower computational complexity than Kernel SHAP.




