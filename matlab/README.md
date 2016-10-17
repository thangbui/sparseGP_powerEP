## Matlab implementation

We provide two Matlab implementations of pseudo-point based sparse Gaussian processes using Power-Expectation Propagation for **regression**. 

1. A barebone implementation [sgp.m](./pep/sgp.m) based on Carl E. Rasmussen's implementation of VFE and FITC. See [test\_sgp.m](./tests/test_sgp.m) for usage. This code is highly optimised and much faster compared to the second option.

2. A GPML-compatible implementation of PEP and VFE. See cov\*.m and inf\*.m and gp\_new.m in this [folder](./matlab/pep). Note that a FITC implementation is available in the GPML code. See [test\_sgp\_gpml.m](./tests/test_sgp_gpml.m) for usage. 

More details can be found in our [paper](https://arxiv.org/abs/1605.07066).
