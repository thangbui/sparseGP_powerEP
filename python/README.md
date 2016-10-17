## Python implementation

We provide a GPy-compatible implementation of the collapsed Power-EP case for **regression**, i.e. when the optimal form for the posterior and the Power-EP energy can be found analytically -- please see [PEP\_reg.py](./sgp/pep/PEP_reg.py) for the code, and [test\_PEP\_GPy.py](./sgp/tests/test_PEP_GPy.py) for usage and an example comparing PEP with VFE, FITC and exact inference.

'''
python -m sgp.tests.test_PEP_GPY
'''

We also provide a barebone implementation of Power-EP for both **regression** and **classification**, that deals with the uncollapsed case, i.e. when the optimal posterior cannot be found. For regression tasks, running this should converge to the solution of the collapsed case, as presented in the code above. See [SGP\_PEP.py](./sgp/pep/SGP_PEP.py) for the code and [test\_SGP\_PEP\_xor.py](./sgp/tests/test_SGP_PEP_xor.py) for an example run on the xor classification problem.

'''
python -m sgp.tests.test_SGP_PEP_xor
'''

More details can be found in our [paper](https://arxiv.org/abs/1605.07066).
