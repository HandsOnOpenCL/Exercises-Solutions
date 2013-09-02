Exercise 8 - using local memory
===============================

Goal
----
* Use local memory to minimize memory movement costs and optimize performance of your matrix multiplication program.

Procedure
---------
* Start with your matrix multiplication program that already uses private memory from Exercise 7.
* Modify the kernel so that each work-group collaboratively copies its own column of B into local memory.
* Optimize step by step, saving the intermediate versions and tracking performance improvements.

Expected output
---------------
* A message to standard output verifying that the matrix multiplication program is generating the correct results.
* Report the runtime and the MFLOPS.

Extra
-----
* Look at the fast, blocked implementation from the NVIDIA OpenCL SDK example.
  Try running it and compare to yours.

