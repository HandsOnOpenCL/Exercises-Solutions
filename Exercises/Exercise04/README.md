Exercise 4 - Chaining vector add kernels (C++/Python)
=====================================================

Goal
----
* To verify that you understand manipulating kernel invocations and buffers in OpenCL.

Procedure
---------
* Start with your VADD program in C++ or Python.
* Add additional buffer objects and assign them to vectors defined on the host
  (see the provided vadd programs for examples of how to do this).
* Chain vadds ... e.g. C=A+B; D=C+E; F=D+G.
* Read back the final result and verify that this is correct.
* Compare the complexity of your host code to C.

Expected output
---------------
* A message to standard output verifying that the chain of vector additions produced the correct result.

Note
----

Sample solution is for C = A + B; D = C + E; F = D + G; return F
