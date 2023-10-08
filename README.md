# Project Information

- Team: QemSpin
- Last updated: 8th October 2023.

## Requirements

We optimized the conventional Givens gate by eliminating the minimal contributions from both single and double excitation states. And we implement it on the SpinQit platform._**[If you are not interested in the process of circuit optimization, you may disregard this requirement.]**_
> Python
```python
python = 3.9.13
```
> Pypi(Linux x86_64) :
```python
   pip install spinqkit-0.0.2-cp39-cp39-linux_x86_64.whl
```

## Getting Started

Our code is divided into three parts, which are **Circuit_Optimization_main.ipynb** in Circuit_Optimization, **Error_mitigation_main.ipynb** and **main.ipynb** in GitHub_QemSpin. The **Circuit_Optimization_main.ipynb** is used for optimizing general Givens gates and obtaining the quantum circuit we want. On the other hand, **Error_mitigation_main.ipynb** is used for calculating the ground state energy of different scales of quantum circuits, preparing data processing for subsequent **main.ipynb**.


### Running the Code

Follow these steps to run the code:

1. Step 1: Run the **Circuit_Optimization_main.ipynb**  to obtain sequences of optimized single excitation gates and double excitation gates, and modify the _singles_list and doubles_list_ in all _Adapt_Givens_Ansatz()_ functions within the Ansatz.py.  _**[If you are not interested in the process of circuit optimization, you can choose not to run this code.]**_
2. Step 2: Run the **Error_mitigation_main.ipynb** to obtain the expected values of different quantum circuit scales under the same noise model. These expected values will be saved in _GitHub_QemSpin_Documents/Error_mitigation_data_.
3. Step 3: Run the **main.ipynb** to obtain the performance of the quantum circuit without noise mitigation (mainly the final optimized loss value). Also, read the data from **Error_mitigation_main.ipynb** and perform exponential fitting, then make a comparison.
