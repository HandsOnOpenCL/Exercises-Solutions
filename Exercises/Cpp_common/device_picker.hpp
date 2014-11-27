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

#include <vector>
#include <err_code.h>
#include <iostream>

#define MAX_INFO_STRING 256


unsigned getDeviceList(std::vector<cl::Device>& devices)
{
  cl_int err;

  // Get list of platforms
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  // Enumerate devices
  for (int i = 0; i < platforms.size(); i++)
  {
    cl_uint num = 0;
    std::vector<cl::Device> plat_devices;
    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &plat_devices);
    devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
  }

  return devices.size();
}

void getDeviceName(cl::Device& device, std::string& name)
{
  cl_device_info info = CL_DEVICE_NAME;

  // Special case for AMD
#ifdef CL_DEVICE_BOARD_NAME_AMD
  device.getInfo(CL_DEVICE_VENDOR, &name);
  if (strstr(name.c_str(), "Advanced Micro Devices"))
    info = CL_DEVICE_BOARD_NAME_AMD;
#endif

  device.getInfo(info, &name);
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
      std::vector<cl::Device> devices;
      unsigned numDevices = getDeviceList(devices);

      // Print device names
      if (numDevices == 0)
      {
        std::cout << "No devices found.\n";
      }
      else
      {
        std::cout << "\nDevices:\n";
        for (int i = 0; i < numDevices; i++)
        {
          std::string name;
          getDeviceName(devices[i], name);
          std::cout << i << ": " << name << "\n";
        }
        std::cout << "\n";
      }
      exit(0);
    }
    else if (!strcmp(argv[i], "--device"))
    {
      if (++i >= argc || !parseUInt(argv[i], deviceIndex))
      {
        std::cout << "Invalid device index\n";
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      std::cout << "\n";
      std::cout << "Usage: ./program [OPTIONS]\n\n";
      std::cout << "Options:\n";
      std::cout << "  -h  --help               Print the message\n";
      std::cout << "      --list               List available devices\n";
      std::cout << "      --device     INDEX   Select device at INDEX\n";
      std::cout << "\n";
      exit(0);
    }
  }
}

