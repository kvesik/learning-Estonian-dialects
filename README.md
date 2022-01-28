# Laying a foundation for bidialectalism: necessary biases for algorithmic learning of two dialects of Estonian (supplementary materials)

This script implements the Gradual Learning Algorithm [(Boersma & Hayes, 2001)](#boersmahayes2001) in such a way that any combination of three particular biases can be applied simultaneously. All other behaviour mirrors the default GLA settings in OTSoft [(Hayes et al., 2013)](#otsoft). The biases are:
  1. Markedness over faithfulness (via setting different initial values for M vs F constraints).
  2. Specific over general faithfulness (via *a priori* ranking of one constraint value a specified amount over another).
  3. [Magri (2012)](#magri2012) update rule

Input as well as tracking history output are both via tab-separated plain text files, in the same format as for OTSoft.

Note that due to the project for which this implementation was used, the script is not sufficiently general to use for other applications without (possibly significant) revision. Examples of restrictions include, but are not limited to:
  - Only constraints beginning with "Id" are recognized as faithfulness constraints.
  - Only one *a priori* ranking relationship.


## References

<a id="boersmahayes2001">Boersma, P., & Hayes, B. (2001).</a>
Empirical Tests of the Gradual Learning Algorithm. 
*Linguistic Inquiry*, 32, 45-86.

<a id="otsoft">Hayes, B., Tesar, B., & Zuraw, K. (2013).</a>
*"OTSoft" software package.* Retrieved April 11, 2021, from http://www.linguistics.ucla.edu/people/hayes/otsoft/ (Version 2.6 beta)

<a id="magri2012">Magri, G. (2012).</a>
Convergence of error-driven ranking algorithms. 
*Phonology, 29*(2), 213–269.
