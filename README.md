From scratch implementation for SHAPLEY VALUES, KERNEL SHAP and DEEP SHAP, following the ["A Unified Approach to Interpreting Model Predictions"](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) reserach paper.
# Challenges
For all 3 algorithms, we found incporporating missingness particularly challenging.
Since model architecture needs to be changes for dropping a variable, the agorithms emulates missingness, by passing a bunch of background values for features that need to be dropped, and taking the expectation of the final predictions.

NOTE: For Shapley values, even after following the formulae mentioned in the paper, we couldnt get the the scale of the shapley values right (as can be seen in the plots y axis). But the relative ratio of each shapley values are consistent with that of expected.

# Results
Comparing implementation values with that of python's shap library

## SHAPLEY Values
<img src="https://user-images.githubusercontent.com/16356237/147423000-b4dcde52-7559-43e1-b677-de1d22eb91a0.PNG" alt="first">
SHAP values results against ’shap’ library: features contribution to the housing price for a modelwith 5 features (left) and 15 (features). Green bar - positive, red - negative contributions, shown relatively to the previous feature.

## Kernel SHAP
Linear Regression 
<img src="https://user-images.githubusercontent.com/16356237/147423186-3ee565aa-3267-44e8-ac44-b21d6039acf1.PNG" alt="second">

XGBoost
<img src="https://user-images.githubusercontent.com/16356237/147423198-5c000073-3164-41c7-ad07-0fe14221cdfa.PNG" alt="third">

## Deep SHAP

<img src="https://user-images.githubusercontent.com/16356237/147423204-6e2bcd4d-8b8e-4969-b2f1-413758fae03d.PNG" alt="fourth">


<img src="https://user-images.githubusercontent.com/16356237/147423212-d16e4d28-ee0d-4fda-8600-a81c0312ce9e.PNG" alt="fifth">

