#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <fstream>
#include <iostream>
#include <CL/cl.h>

const char *kernel_code =
"	__kernel void histogram(                                           \n"
"		__global unsigned int *image_data,                             \n"
"		__global unsigned int *ref_histogram_results,                  \n"
"		unsigned int input_size)                                       \n"
"	{                                                                  \n"
"		int i = get_global_id(0) * 3;                                  \n"
"		int j;                                                         \n"
"		if(i < input_size)                                             \n"
"			for (j = 0; j < 3; j++) {                                  \n"
"				unsigned int index = image_data[i + j];                \n"
"				atomic_inc(&ref_histogram_results[index + j * 256]);   \n"
"			}                                                          \n"
"	};                                                                 \n";

int main(int argc, char const *argv[])
{
	unsigned int i=0, a, input_size;
	std::fstream inFile("input", std::ios_base::in);
	std::ofstream outFile("0116230.out", std::ios_base::out);

	unsigned int *histogram_results = new unsigned int[256 * 3];
	memset(histogram_results, 0, sizeof(unsigned) * 256 * 3);

	inFile >> input_size;
	unsigned int *image = new unsigned int[input_size];
	while( inFile >> a ) {
		image[i++] = a;
	}
	
	cl_int err;
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue commandqueue;
	cl_program program;
	cl_kernel kernel;
	cl_mem input, output;
	size_t local_work_size;
	size_t global_work_size = input_size/3;

	// get first platfrom
	err = clGetPlatformIDs(1, &platform, NULL);
	if(err != CL_SUCCESS) { 
		std::cerr << "Unable to get platforms\n";
		return 0;
	}
	// get first GPU device on platform
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if(err != CL_SUCCESS) { 
		std::cerr << "Unable to get devices\n";
		return 0;
	}
	// create context for selected device
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "Can't create OpenCL context\n";
		return 0;
	}
	// create commadqueue
	commandqueue = clCreateCommandQueue(context, device, NULL, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "Can't create command queue\n";
		clReleaseContext(context);
		return 0;
	}
	// create buffer on device
	input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned)*input_size, NULL, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "Can't create OpenCL buffer\n";
		clReleaseMemObject(input);
		clReleaseCommandQueue(commandqueue);
		clReleaseContext(context);
		return 0;
	}
	output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned)*256*3, NULL, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "Can't create OpenCL buffer\n";
		clReleaseMemObject(input);
		clReleaseMemObject(output);
		clReleaseCommandQueue(commandqueue);
		clReleaseContext(context);
		return 0;
	}
	// write input data to device
	err = clEnqueueWriteBuffer(commandqueue, input, CL_TRUE, 0, sizeof(unsigned)*input_size, image, NULL, NULL, NULL);
	if(err != CL_SUCCESS) {
		std::cerr << "Can't write data to OpenCL buffer\n";
		clReleaseMemObject(input);
		clReleaseMemObject(output);
		clReleaseCommandQueue(commandqueue);
		clReleaseContext(context);
		return 0;
	}
	err = clEnqueueWriteBuffer(commandqueue, output, CL_TRUE, 0, sizeof(unsigned)*256*3, histogram_results, NULL, NULL, NULL);
	if(err != CL_SUCCESS) { 
		std::cerr << "Can't write data to OpenCL buffer\n";
		clReleaseMemObject(input);
		clReleaseMemObject(output);
		clReleaseCommandQueue(commandqueue);
		clReleaseContext(context);
		return 0;
	}
	// create program
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_code, NULL, &err);
	if(err != CL_SUCCESS) {
		std::cerr << "Can't create program\n";
		clReleaseMemObject(input);
		clReleaseMemObject(output);
		clReleaseCommandQueue(commandqueue);
		clReleaseContext(context);
		return 0;
	}
	// build program
	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if(err != CL_SUCCESS) { 
		std::cerr << "Can't build program\n" << err << std::endl;
		if (err == CL_BUILD_PROGRAM_FAILURE) {
			size_t log_size;
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			char *log = (char *) malloc(log_size);
			clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			printf("%s\n", log);
		}
		clReleaseMemObject(input);
		clReleaseMemObject(output);
		clReleaseCommandQueue(commandqueue);
		clReleaseContext(context);
		return 0;
	}
	// use func "histogram" as the kernel
	kernel = clCreateKernel(program, "histogram" , &err);
	if(err != CL_SUCCESS) {
		std::cerr << "Can't load kernel\n";
		clReleaseProgram(program);
		clReleaseMemObject(input);
		clReleaseMemObject(output);
		clReleaseCommandQueue(commandqueue);
		clReleaseContext(context);
		return 0;
	}
	// set arguments
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output);
	err = clSetKernelArg(kernel, 2, sizeof(unsigned), (void *)&input_size);

	// err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, size(work_group_size), &work_group_size, NULL);	
	// execute kernel, without specifying a work-group size
	err = clEnqueueNDRangeKernel(commandqueue , kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

	// read back data from device
	if(err == CL_SUCCESS) {
		err = clEnqueueReadBuffer(commandqueue, output, CL_TRUE, 0, sizeof(unsigned)*256*3, histogram_results, NULL, NULL, NULL);
		if(err == CL_SUCCESS) {
			// print result
			for(unsigned int i = 0; i < 256 * 3; ++i) {
				if (i % 256 == 0 && i != 0)
					outFile << std::endl;
				outFile << histogram_results[i]<< ' ';
			}
		} else {
			std::cerr << "Can't read back data\n";
		}
	} else {
		std::cerr << "Kernel execution failed\n";
	}
	
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseCommandQueue(commandqueue);
	clReleaseContext(context);

	inFile.close();
	outFile.close();

	return 0;
}
