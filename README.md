# Phase 1 - Probability based edge detection

This repository contains the scripts required for the homework 0 submission for the CMSC733 (Classical and Deep Learning Approaches for Geometric Computer Vision) course. Made by Vinay Lanka (120417665) at the University of Maryland. 

This contains a python script for the PbLite boundary detection algorithm developed as an alternative to the classical approaches of the Canny and Sobel edge detection algorithms.
The program will create a set of filter banks and will then find Texton, Brightness and Color gradient maps. It will then perform boundary detection by taking those and the Canny and Sobel baselines.

To get this to work you would need the BSDS500 folder that is distributed as a part of the starter code. Make sure that the folder is present in the Phase 1 directory and then run

```bash
$ python3 /Phase1/Wrapper.py
```