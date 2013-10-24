//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multipliplication driver
//
//  PURPOSE: This is a driver program to test various ways of computing
//           the product:
//
//                C  = A * B
//
//           A and B are set to constant matrices so we
//           can make a quick test of the multiplication.
//
//  USAGE:   The matrices are constant matrices, square and the order is
//           set as a constant, ORDER (see mult.h).
//
//  HISTORY: Written by Tim Mattson, August 2010 
//           Modified by Simon McIntosh-Smith, September 2011
//           Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
//           Updated to C++ Wrapper v1.2.6 by Tom Deakin, August 2013
//
//------------------------------------------------------------------------------

#include "matmul.hpp"
#include "matrix_lib.hpp"

int main(void)
{

    int Mdim, Ndim, Pdim;   // A[N][P], B[P][M], C[N][M]
    int szA, szB, szC;      // Number of elements in each matrix


    double start_time;      // Starting time
    double run_time;        // Timing data
    util::Timer timer;      // Timer


    Ndim = ORDER;
    Pdim = ORDER;
    Mdim = ORDER;

    szA = Ndim * Pdim;
    szB = Pdim * Mdim;
    szC = Ndim * Mdim;

    std::vector<float> h_A(szA); // Host memory for Matrix A
    std::vector<float> h_B(szB); // Host memory for Matrix B
    std::vector<float> h_C(szC); // Host memory for Matrix C

    cl::Buffer d_a, d_b, d_c;   // Matrices in device memory

    initmat(Mdim, Ndim, Pdim, h_A, h_B, h_C);

    printf("\n===== Sequential, matrix mult (dot prod), order %d on host CPU ======\n",ORDER);
    for(int i = 0; i < COUNT; i++)
    {
        zero_mat(Ndim, Mdim, h_C);
        start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

        seq_mat_mul_sdot(Mdim, Ndim, Pdim, h_A, h_B, h_C);

        run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;
        results(Mdim, Ndim, Pdim, h_C, run_time);
    }

    try
    {

//--------------------------------------------------------------------------------
// Create a context and queue for DEVICE
//--------------------------------------------------------------------------------

        cl::Context context(DEVICE);
        cl::CommandQueue queue(context);

//--------------------------------------------------------------------------------
// Setup the buffers, initialize matrices, and write them into global memory
//--------------------------------------------------------------------------------

        //  Reset A, B and C matrices (just to play it safe)
        initmat(Mdim, Ndim, Pdim, h_A, h_B, h_C);

        d_a = cl::Buffer(context, begin(h_A), end(h_A), true);

        d_b = cl::Buffer(context, begin(h_B), end(h_B), true);

        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * szC);

//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... Naive
//--------------------------------------------------------------------------------

        // Create the compute program from the source buffer
        cl::Program program(context, util::loadProgram("../C_elem.cl"), true);

        // Create the compute kernel from the program
        auto naive_mmul = cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer>(program, "mmul");

        printf("\n===== OpenCL, matrix mult, C(i,j) per work item, order %d ======\n",Ndim);

        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            zero_mat(Ndim, Mdim, h_C);

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

            // Execute the kernel over the entire range of C matrix elements ... computing
            // a dot product for each element of the product matrix.  The local work
            // group size is set to NULL ... so I'm telling the OpenCL runtime to
            // figure out a local work group size for me.
            cl::NDRange global(Ndim, Mdim);
            naive_mmul(cl::EnqueueArgs(queue, global),
                    Mdim, Ndim, Pdim, d_a, d_b, d_c);

            queue.finish();

            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;

            cl::copy(queue, d_c, begin(h_C), end(h_C));

            results(Mdim, Ndim, Pdim, h_C, run_time);

        } // end for loop

//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... C row per work item
//--------------------------------------------------------------------------------

        // Create the compute program from the source buffer
        program = cl::Program(context, util::loadProgram("../C_row.cl"), true);

        // Create the compute kernel from the program
        auto crow_mmul = cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer>(program, "mmul");

         printf("\n===== OpenCL, matrix mult, C row per work item, order %d ======\n",Ndim);

        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            zero_mat(Ndim, Mdim, h_C);

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

            cl::NDRange global(Ndim);
            crow_mmul(cl::EnqueueArgs(queue, global),
                    Mdim, Ndim, Pdim, d_a, d_b, d_c);

            queue.finish();

            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;

            cl::copy(queue, d_c, begin(h_C), end(h_C));

            results(Mdim, Ndim, Pdim, h_C, run_time);

        } // end for loop

//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... C row per work item, A row in pivate memory
//--------------------------------------------------------------------------------

        // Create the compute program from the source buffer
        program = cl::Program(context, util::loadProgram("../C_row_priv.cl"), true);

        // Create the compute kernel from the program
        auto arowpriv_mmul = cl::make_kernel<int, int, int, cl::Buffer, cl::Buffer, cl::Buffer>(program, "mmul");

        printf("\n===== OpenCL, matrix mult, C row, A row in priv mem, order %d ======\n",Ndim);

        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            zero_mat(Ndim, Mdim, h_C);

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;


            cl::NDRange global(Ndim);
            cl::NDRange local(ORDER / 16);
            arowpriv_mmul(cl::EnqueueArgs(queue, global, local),
                    Mdim, Ndim, Pdim, d_a, d_b, d_c);

            queue.finish();

            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;

            cl::copy(queue, d_c, begin(h_C), end(h_C));

            results(Mdim, Ndim, Pdim, h_C, run_time);

        } // end for loop

    } catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
