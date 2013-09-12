/*
 * Display Device Information
 *
 * Script to print out some information about the OpenCL devices
 * and platforms available on your system
 *
 * History: C++ version written by Tom Deakin, 2012
 *          Ported to C by Tom Deakin, July 2013
*/

#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

char* err_code (cl_int);

int main(void)
{
    cl_int err;
    // Find the number of OpenCL platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms < 0)
    {
        printf("Error: could not find a platform\n%s\n", err_code(err));
        return EXIT_FAILURE;
    }
    // Create a list of platform IDs
    cl_platform_id platform[num_platforms];
    err = clGetPlatformIDs(num_platforms, platform, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: could not get platforms\n%s\n", err_code(err));
    }

    printf("\nNumber of OpenCL platforms: %d\n", num_platforms);
    printf("\n-------------------------\n");

    // Investigate each platform
    for (int i = 0; i < num_platforms; i++)
    {
        cl_char string[10240] = {0};
        // Print out the platform name
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, sizeof(string), &string, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: could not get platform information\n%s\n", err_code(err));
            return EXIT_FAILURE;
        }
        printf("Platform: %s\n", string);

        // Print out the platform vendor
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, sizeof(string), &string, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: could not get platform information\n%s\n", err_code(err));
            return EXIT_FAILURE;
        }
        printf("Vendor: %s\n", string);

        // Print out the platform OpenCL version
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, sizeof(string), &string, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: could not get platform information\n%s\n", err_code(err));
            return EXIT_FAILURE;
        }
        printf("Version: %s\n", string);

        // Count the number of devices in the platform
        cl_uint num_devices;
        err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (err != CL_SUCCESS)
        {
            printf("Error: could not get devices for platform\n%s\n", err_code(err));
            return EXIT_FAILURE;
        }
        // Get the device IDs
        cl_device_id device[num_devices];
        err = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, num_devices, device, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Error: could not get devices for platform\n%s\n", err_code(err));
            return EXIT_FAILURE;
        }
        printf("Number of devices: %d\n", num_devices);

        // Investigate each device
        for (int j = 0; j < num_devices; j++)
        {
            printf("\t-------------------------\n");

            // Get device name
            err = clGetDeviceInfo(device[j], CL_DEVICE_NAME, sizeof(string), &string, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tName: %s\n", string);

            // Get device OpenCL version
            err = clGetDeviceInfo(device[j], CL_DEVICE_OPENCL_C_VERSION, sizeof(string), &string, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tVersion: %s\n", string);

            // Get Max. Compute units
            cl_uint num;
            err = clGetDeviceInfo(device[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tMax. Compute Units: %d\n", num);

            // Get local memory size
            cl_ulong mem_size;
            err = clGetDeviceInfo(device[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tLocal Memory Size: %ld KB\n", mem_size/1024);

            // Get global memory size
            err = clGetDeviceInfo(device[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tGlobal Memory Size: %ld MB\n", mem_size/(1024*1024));

            // Get maximum buffer alloc. size
            err = clGetDeviceInfo(device[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_size, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tMax Alloc Size: %ld MB\n", mem_size/(1024*1024));

            // Get work-group size information
            size_t size;
            err = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tMax Work-group Size: %ld\n", size);

            // Find the maximum dimensions of the work-groups
            err = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &num, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            // Get the max. dimensions of the work-groups
            size_t dims[num];
            err = clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), &dims, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: could not get device information\n%s\n", err_code(err));
                return EXIT_FAILURE;
            }
            printf("\t\tMax Work-item Dims: ( ");
            for (size_t k = 0; k < num; k++)
            {
                printf("%ld ", dims[k]);
            }
            printf(")\n");

            printf("\t-------------------------\n");
        }

        printf("\n-------------------------\n");
    }

    return EXIT_SUCCESS;
}
