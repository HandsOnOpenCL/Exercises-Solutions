Exercise 9 - The Pi program
===========================

Goal
----
* To understand synchronization between work-items in the OpenCL C kernel programming language.

Procedure
---------
* Start with the provided serial program to estimate Pi through numerical integration.
* Write a kernel and host program to compute the numerical integral using OpenCL.
* Note: you will need to implement a reduction.

Expected output
---------------
* Output result plus an estimate of the error in the result.
* Report the runtime.

Hint
----
You will want each work-item to do many iterations of the loop, i.e. don't create one work-item per loop iteration.
To do so would make the reduction so costly that performance would be terrible.
