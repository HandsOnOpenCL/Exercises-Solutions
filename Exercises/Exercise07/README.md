Exercise 7 - using private memory
=================================

Goal
----
* Use private memory to minimize memory movement costs and optimize performance of your matrix multiplication program.

Procedure
---------
* Start with your matrix multiplication program.
* Modify the kernel so that each work-item copies its own row of A into private memory.
* Optimize step by step, saving the intermediate versions and tracking performance improvements.

Expected output
---------------
* A message to standard output verifying that the matrix multiplication program is generating the correct results.
* Report the runtime and the MFLOPS.
