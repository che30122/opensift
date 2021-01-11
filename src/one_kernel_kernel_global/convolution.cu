//#include <opencv2/gpu/gpu.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#if CV_VERSION_EPOCH == 2
#define OPENCV2
#include <opencv2/gpu/gpu.hpp>
namespace GPU = cv::gpu;
#elif CV_VERSION_MAJOR == 4 
#define  OPENCV4
#include <opencv2/core/cuda.hpp>
namespace GPU = cv::cuda;
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/imgproc.hpp>
#include <iostream>

#define THREAD_X 32
#define THREAD_Y 32
#define WRAP_SIZE 32
#define MAX_WRAP_NUM 32

//using namespace cv;
//using namespace cv;
int KERNEL_SIZE;
__global__ void conv(int* dev){
        int pixel_i=blockDim.x*blockIdx.x+threadIdx.x;
	int pixel_j=blockDim.y*blockIdx.y+threadIdx.y;
		dev[pixel_i]=1;
		//        printf("idx%d %d,%d\n",pixel_i,pixel_j,dev[pixel_i]);

}
__global__ void convolution(GPU::PtrStepSz<float> src,GPU::PtrStepSz<double> guass_kernel,GPU::PtrStepSz<float> dst,int kernel_size,int kernel_radius,int orign_width,int orign_height){
	__shared__ int  share_mem[WRAP_SIZE][MAX_WRAP_NUM];
	int pixel_i=blockDim.x*blockIdx.x+threadIdx.x;
	int pixel_j=blockDim.y*blockIdx.y+threadIdx.y;
	//need to do bound check
	//printf("pixel %d %d block dim %d %d\n",pixel_i,pixel_j,blockDim.x,blockDim.y);
	/*int thread_block_index=pixel_i+pixel_j*;
	int share_i=thread_block_index%WRAP_NUM;
	int share_j=thread_block_index/WRAP_NUM;*/
	double sum=0;
	//share_mem[share_i][share_j]=src(pixel_i,pixel_j);
	//share_mem[threadIdx.x][threadIdx.y]=src(pixel_i,pixel_j).x;
	//__syncthreads();
	 //printf("%d %d %d\n",pixel_i,pixel_j,share_mem[pixel_i][pixel_j]);
	if(!(pixel_i<kernel_radius || pixel_j<kernel_radius || pixel_i>=orign_width+kernel_radius  || pixel_j>=orign_height+kernel_radius)){
		int start_i=pixel_i-kernel_radius,start_j=pixel_j-kernel_radius;
		for(int i=0;i<kernel_size;i++){
			for(int j=0;j<kernel_size;j++){
				int index_i=start_i+i,index_j=start_j+j;
				//sum+=share_mem[][index_j]*guass_kernel(i,j).x;
				sum+=src(index_j,index_i)*(float)guass_kernel(i,j);
			}
		}

		dst(pixel_j-kernel_radius,pixel_i-kernel_radius)=sum;//sum;
	}
	return ;
}

void guassain_conv(const Mat *src,Mat *dst,double sigma){
//	int depth = CV_MAT_DEPTH(src.type());
	KERNEL_SIZE = cvRound(sigma* 4 * 2 + 1)|1;
	int kernel_radius=KERNEL_SIZE/2;
	int orign_width=src->cols,orign_height=src->rows;
	Mat padding_image;
	GPU::GpuMat device_image,g_kernel,result;

	if(GPU::getCudaEnabledDeviceCount()==0){
		std::cout<<"not use GPU module"<<std::endl;
		return ;
	}
	Mat gauss_x=getGaussianKernel(KERNEL_SIZE,sigma),gauss_y=getGaussianKernel(KERNEL_SIZE,sigma); //3*3 filter
	Mat gauss_kernel=gauss_x*gauss_y.t();
	//allocate
/*	double* gs_kernel,*dev_kernel;
	cudaHostAlloc(&gs_kernel,sizeof(double)*KERNEL_SIZE*KERNEL_SIZE,cudaHostAllocDefault);
	for(int i=0;i<KERNEL_SIZE;i++){
		double* row=gauss_kernel.ptr<double>(i);
		for(int j=0;j<KERNEL_SIZE;j++){
			gs_kernel[i*KERNEL_SIZE+j]=row[j];
		}
	}
	cudaMalloc(&dev_kernel,sizeof(double)*KERNEL_SIZE*KERNEL_SIZE);*/
	//allocate 
	copyMakeBorder(*src,padding_image,kernel_radius,kernel_radius,kernel_radius,kernel_radius,BORDER_CONSTANT, 0);
	int grid_num_x=(padding_image.cols+THREAD_X-1)/THREAD_X,grid_num_y=(padding_image.rows+THREAD_Y-1)/THREAD_Y;
	result.upload(*dst);
	g_kernel.upload(gauss_kernel);
	device_image.upload(padding_image);
	
	//cudaMemcpy(dev_kernel,gs_kernel,sizeof(double)*KERNEL_SIZE*KERNEL_SIZE,cudaMemcpyHostToDevice);
	
	dim3 thread_block(THREAD_X,THREAD_Y);
	dim3 grid(grid_num_x,grid_num_y);
	convolution<<<grid,thread_block>>>(device_image,g_kernel,result,KERNEL_SIZE,kernel_radius,orign_width,orign_height);

	result.download(*dst);
	return ;
}


