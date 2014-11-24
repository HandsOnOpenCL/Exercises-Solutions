#!/usr/bin/env python3

# Usage: ./genErrCode.py /path/to/cl.h > err_code.h

from __future__ import print_function
import sys

if len(sys.argv) != 2:
    print("Usage: python genErrCode.py /path/to/cl.h", file = sys.stderr)
    sys.exit(1)

hfile = open(sys.argv[1], "r")

# Find the start of the error code list
for l in hfile:
    if l == "/* Error Codes */\n":
        # Found the error code comment
        break

errors = []
# Loop through the errors and construct the list of errors
for l in hfile:
    # Skip if a blank line
    if l == "\n":
        continue

    tokens = l.split()
    # We expect the line to be of the form:
    # #define CL_... int
    # OpenCL error numbers are 0 or negative
    if len(tokens) != 3 or int(tokens[2]) > 0:
        # We are done or some error
        break
    else:
        errors.append(tokens[1])

# Print out the C file
print('''
#pragma once
/*----------------------------------------------------------------------------
 *
 * Name:     err_code()
 *
 * Purpose:  Function to output descriptions of errors for an input error code
 *           and quit a program on an error with a user message
 *
 *
 * RETURN:   echoes the input error code / echos user message and exits
 *
 * HISTORY:  Written by Tim Mattson, June 2010
 *           This version automatically produced by genErrCode.py
 *           script written by Tom Deakin, August 2013
 *           Modified by Bruce Merry, March 2014
 *           Updated by Tom Deakin, October 2014
 *               Included the checkError function written by
 *               James Price and Simon McIntosh-Smith
 *
 *----------------------------------------------------------------------------
 */
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
 #include <cstdio>
#endif

const char *err_code (cl_int err_in)
{
    switch (err_in) {''')
for err in errors:
    print('        case ' + err + ':')
    print('            return (char*)"' + err.strip() + '";')

print('''
        default:
            return (char*)"UNKNOWN ERROR";
    }
}
''')

# Check error funtion
print('''
void check_error(cl_int err, const char *operation, char *filename, int line)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error during operation '%s', ", operation);
        fprintf(stderr, "in '%s' on line %d\\n", filename, line);
        fprintf(stderr, "Error code was \\"%s\\" (%d)\\n", err_code(err), err);
        exit(EXIT_FAILURE);
    }
}
''')

# Macro version of checkError without need for file and line
print('''
#define checkError(E, S) check_error(E,S,__FILE__,__LINE__)
''')

