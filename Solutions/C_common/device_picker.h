/*------------------------------------------------------------------------------
 *
 * Name:       device_picker.h
 *
 * Purpose:    Provide a simple CLI to specify an OpenCL device at runtime
 *
 * Note:       Must be included AFTER the relevant OpenCL header
 *             See one of the Matrix Multiply exercises for usage
 *
 * HISTORY:    Method written by James Price, October 2014
 *             Extracted to a common header by Tom Deakin, November 2014
 */

#pragma once

#include <string.h>
#include <err_code.h>

#define MAX_PLATFORMS     8
#define MAX_DEVICES      16
#define MAX_INFO_STRING 256


unsigned getDeviceList(cl_device_id devices[MAX_DEVICES])
{
  cl_int err;

  // Get list of platforms
  cl_uint numPlatforms = 0;
  cl_platform_id platforms[MAX_PLATFORMS];
  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &numPlatforms);
  checkError(err, "getting platforms");

  // Enumerate devices
  unsigned numDevices = 0;
  for (int i = 0; i < numPlatforms; i++)
  {
    cl_uint num = 0;
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-numDevices, devices+numDevices, &num);
    checkError(err, "getting deviceS");
    numDevices += num;
  }

  return numDevices;
}

void getDeviceName(cl_device_id device, char name[MAX_INFO_STRING])
{
  cl_device_info info = CL_DEVICE_NAME;

  // Special case for AMD
#ifdef CL_DEVICE_BOARD_NAME_AMD
  clGetDeviceInfo(device, CL_DEVICE_VENDOR, MAX_INFO_STRING, name, NULL);
  if (strstr(name, "Advanced Micro Devices"))
    info = CL_DEVICE_BOARD_NAME_AMD;
#endif

  clGetDeviceInfo(device, info, MAX_INFO_STRING, name, NULL);
}


int parseUInt(const char *str, cl_uint *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[], cl_uint *deviceIndex)
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--list"))
    {
      // Get list of devices
      cl_device_id devices[MAX_DEVICES];
      unsigned numDevices = getDeviceList(devices);

      // Print device names
      if (numDevices == 0)
      {
        printf("No devices found.\n");
      }
      else
      {
        printf("\n");
        printf("Devices:\n");
        for (int i = 0; i < numDevices; i++)
        {
          char name[MAX_INFO_STRING];
          getDeviceName(devices[i], name);
          printf("%2d: %s\n", i, name);
        }
        printf("\n");
      }
      exit(0);
    }
    else if (!strcmp(argv[i], "--device"))
    {
      if (++i >= argc || !parseUInt(argv[i], deviceIndex))
      {
        printf("Invalid device index\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./program [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h  --help               Print the message\n");
      printf("      --list               List available devices\n");
      printf("      --device     INDEX   Select device at INDEX\n");
      printf("\n");
      exit(0);
    }
  }
}

