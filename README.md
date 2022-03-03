# TwoTestCharacteristics
Code for taking results of two diagnostic tests and using that to infer the diagnostic characteristics of the first test. Also possible to infer other characteristics as well. Interval inputs can be used to calculate with incomplete information, and outputs are functions for a consistent multi-dimensional possibility structure, or alpha cuts can be taken to return marginal intervals of confidence sets.

Import TwoTestCombination and use the Combine_Results class to create a combination. You need to pass it two 1-dimensional iterables:
	Combo = tt.Combine_Results(ResultsA, ResultsB)
This initialises the structure. You can assess the possibility of a point, consisting of the following characteristics in this order:

	Sens_A: Sensitivity of test A
	Spec_A: Specificity of test A
	Sens_B: Sensitivity of test B
	Spec_B: Specificity of test B
	Prev:   Prevalence
	
Optionally you may also specify:

	Corr:   Correlation coefficient between test results
You can assess the possibility of a point as:

	Combo.possibility(Sens_A, Spec_A, Sens_B, Spec_B, Prev)
or

	Combo.possibility(Sens_A, Spec_A, Sens_B, Spec_B, Prev, Corr)
You can replace any of these inputs with intervals that are subsets of the unit interval \[0,1\].

You can take an alpha cut to get marginal intervals for Sens_A and Spec_A, the cartesian product of which bounds the actual set created by the alpha cut.

	Combo.cut(alpha=0.95)
You can specify known values for Sens_B, Spec_B, Prev and Corr when taking the alpha cut

	Combo.cut(0.95, 0.8, 0.8, 0.2, Interval(0,1))
