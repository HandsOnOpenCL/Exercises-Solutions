
import sys

if len(sys.argv) != 2:
    print "Usage: python genErrCode.py /path/to/cl.h"
    sys.exit(-1)

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
print '''
//------------------------------------------------------------------------------
//
// Name:     err_code()    
//
// Purpose:  Function to output descriptions of errors for an input error code
//
//
// RETURN:   echoes the input error code
//
// HISTORY:  Written by Tim Mattson, June 2010
//           This version automatically produced by genErrCode.py
//           script written by Tom Deakin, August 2013
//
//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef APPLE
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

char *err_code (cl_int err_in)
{
    switch (err_in) {
'''
for err in errors:
    print '        case', err, ':'
    print '            return (char*)"', err, '";'

print '        default:'
print '            return (char*)"UNKNOWN ERROR";'
print '''
    }
}
'''
