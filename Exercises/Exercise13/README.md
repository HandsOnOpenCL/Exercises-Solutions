Exercise 13 - Porting CUDA to OpenCL
====================================

Goal
----
* To port the CUDA/serial C program to OpenCL

Procedure
---------
* Examine the CUDA kernel and identify which parts need changing
    * Change them to the OpenCL equivalents
* Examine the Host code and part the commands to the OpenCL equivalents

Expected output
---------------
* The OpenCL and CUDA programs should produce the same output - check this!

Examples
--------
Some example input is provided in the Examples/ directory.
The `.dat` files list the co-ordinates of the grid with a live cell, followed by a 1 (to signify alive).
The `input.params` file lists the size of the grid (X then Y) and the number of iterations.

Notes
-----

See the Exercises/Exercise13/Examples directory for some sample input .dat, input.params files
along with the expected final_state.dat for four different Game of Life patterns.
