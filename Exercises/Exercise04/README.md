Exercise 4 - Chaining vector add kernels
========================================

Goal
----
* To verify that you understand manipulating kernel invocations and buffers in OpenCL.

Procedure
---------
* Start with your VADD program.
* Add additional buffer objects and assign them to vectors defined on the host (see the solutions to Exercise 3).
* Chain vadds ... e.g. C=A+B; D=C+E; F=D+G.
* Read back the final result and verify that this is correct.
* Have a think about how this would differ in the C, C++ or Python.

Expected output
---------------
* A message to standard output verifying that the chain of vector additions produced the correct result.
