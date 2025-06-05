# Convolutional Neural Network in RISC-V

## Overview
A implementation of Convolutional Neural Networks (CNNs) on the RISC-V architecture. This project demonstrates how each layers work on a low level and making use of the standard Risc-V ISA and RVV.

[Video Link](https://youtu.be/qFiORxSO10g)

## Table of Contents
- [Convolutional Neural Network in RISC-V](#convolutional-neural-network-in-risc-v)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Acknowledgments](#acknowledgments)

## Installation
The following tools will be needed to simulate Risc-V assembly code:
- RISC-V toolchain setup - [link](https://github.com/riscv-collab/riscv-gnu-toolchain)
- Veer-ISS - [link](https://github.com/chipsalliance/VeeR-ISS)


## Usage
1. Run `./build.sh <filename.s>` to compile the assembly file

2. Open `build/log.txt`

3. Search for `sw       t0, 0x0(a0)` in the log file

4. The value to its left will be the number predicted by the model. It will be 5 as that is the number currently loaded in

## Project Structure
- The `cpp/` directory contains C++ implementations of some layers that we used as reference when coding in assembly

- The `python/` directory contains helper scripts that were used to convert data to the correct format, obtain weights and biases, and get images to pass through the CNN.

- The `src/` contains the non-vectorized and vectorized variants of the code. `cnn.s` contains the complete CNN.

***The vectorized code is not fully complete as dense layer and softmax are still non-RVV***


## Acknowledgments
This was a group Project and could not have been done without the help of:
- [Abdullah Faraz](https://github.com/Hypxr-7)
- [Sharique Baig](https://github.com/ShariqueBaig)
- [Hamza Ahsan](https://github.com/mzhmza2)
- [Faraz Ansari](https://github.com/farazz1)
