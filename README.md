# CSDS 440 Fall 2020 Final Project - Privacy Preserving Linear Regression
Author: Thomas Patton (<tjp94@case.edu>)

Case Western Reserve University

Deparment of Computer Science

## Description
The goal of this project was to implement differentially private linear regression as described in *Yue Wang et al.* This form of privacy is dependent on a peturbation of the objective function, usually by injecting Laplacian noise. This project implements this idea with a 1-layer network in TensorFlow with a custom objective function. Additionally, this project expands on that idea by allowing users to use Gaussian noise. The comparison of these two noise types is the subject of the corresponding paper for this project. See ``PrivLinReg_Writeup.pdf`` for the full writeup.

## Usage
``python tf_main.py <data_dir> <epsilon> <noise_type>``

### Arguments:
* ``<data_dir>`` - The location of the ``.data`` and ``.names`` file (ex ``data/cancer``)
* ``<epsilon>`` - The level of e-differential privacy to hold. Lower numbers indicate more noise added to the system
* ``<noise_type>`` - One of either 'none', 'laplace', or 'gauss'. Indicates whether to use none, Laplace, or Gaussian noise respectively

## Requirements
This project was built with TensorFlow v2.3.1 and Python v3.7.6. Other configurations may work but are currently untested. This project also makes use of the appropriate configurations of NumPy, Pandas, and SciPy.

