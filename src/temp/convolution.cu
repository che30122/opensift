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

__constant__ double guass_kernel[2048];
__constant__ double guass_kernel_x[128*2];
__constant__ double guass_kernel_y[128];
int KERNEL_SIZE;

__global__ void convolution(GPU::PtrStepSz<float> src,/*const double* __restrict__ guass_kernel,*/GPU::PtrStepSz<float> dst,int kernel_size,int kernel_radius,int orign_width,int orign_height){
	__shared__ int  share_mem[WRAP_SIZE][MAX_WRAP_NUM];
	int pixel_i=blockDim.x*blockIdx.x+threadIdx.x;
	int pixel_j=blockDim.y*blockIdx.y+threadIdx.y;
	//need to do bound check
	//printf("pixel %d %d block dim %d %d\n",pixel_i,pixel_j,blockDim.x,blockDim.y);
	int thread_block_index=threadIdx.x+threadIdx.y*blockDim.x;
	/*int share_i=thread_block_index%WRAP_NUM;
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
				sum+=src(index_j,index_i)*(float)guass_kernel[j];
			}
		}

		dst(pixel_j-kernel_radius,pixel_i-kernel_radius)=sum;//sum;
	}
	return ;
}
//not need to padding
__global__ void conv_x(GPU::PtrStepSz<float> src,/*const double* __restrict__ guass_kernel,*/GPU::PtrStepSz<float> dst,int kernel_size,int kernel_radius,int orign_width,int orign_height){
	__shared__ float  share_mem[THREAD_Y+50][THREAD_X+50];
	int pixel_i=blockDim.x*blockIdx.x+threadIdx.x;
	int pixel_j=blockDim.y*blockIdx.y+threadIdx.y;

	int shared_i=threadIdx.x+kernel_size/2;
	int shared_j=threadIdx.y;
	float sum=0;
	if(!(pixel_i>=orign_width || pixel_j>=orign_height)){
		share_mem[shared_j][shared_i]=src(pixel_j,pixel_i);
	__syncthreads();
		int start_i=shared_i-kernel_radius,start_j=shared_j;
		for(int i=0;i<kernel_size;i++){
			sum+=share_mem[start_j][start_i+i]*(float)guass_kernel_x[i];
		}
		dst(pixel_j,pixel_i)=sum;//src(pixel_j,pixel_i);//sum;//sum;
	}
	return ;
}
__global__ void conv_y(GPU::PtrStepSz<float> src,/*const double* __restrict__ guass_kernel,*/GPU::PtrStepSz<float> dst,int kernel_size,int kernel_radius,int orign_width,int orign_height){
	__shared__ float  share_mem[THREAD_X+50][THREAD_Y+50];
	int pixel_i=blockDim.x*blockIdx.x+threadIdx.x;
	int pixel_j=blockDim.y*blockIdx.y+threadIdx.y;
        int shared_i=threadIdx.x;
	int shared_j=threadIdx.y+kernel_size/2;
	float sum=0;
//	for(int i=0;i<THREAD_X+20;i++)
//		for(int j=0;j<THREAD_X+20;j++)
//			share_mem[i][j]=0;
	if(!(pixel_i>=orign_width || pixel_j>=orign_height)){
		share_mem[shared_j][shared_i]=src(pixel_j,pixel_i);
	
	__syncthreads();
		int start_i=shared_i,    start_j=shared_j-kernel_radius;
		for(int i=0;i<kernel_size;i++){
			sum+=share_mem[start_j+i][start_i]*(float)guass_kernel_x[i];
			//sum+=share_mem[start_j+i][start_i];
		}
		dst(pixel_j,pixel_i)=sum;//share_mem[shared_j][shared_i];//sum;
	}
	return ;
}
void guassain_conv(const Mat *src,Mat *dst,double sigma){
//	int depth = CV_MAT_DEPTH(src.type());
	KERNEL_SIZE = cvRound(sigma* 4 * 2 + 1)|1;
	int kernel_radius=KERNEL_SIZE/2;
	int orign_width=src->cols,orign_height=src->rows;
	Mat padding_image;
	GPU::GpuMat device_image,g_kernel,result, dev_image,resul;

	if(GPU::getCudaEnabledDeviceCount()==0){
		std::cout<<"not use GPU module"<<std::endl;
		return ;
	}
	Mat gauss_x=getGaussianKernel(KERNEL_SIZE,sigma),gauss_y=getGaussianKernel(KERNEL_SIZE,sigma); //3*3 filter
	Mat gauss_kernel=gauss_x*gauss_y.t();
	//allocate
	double* gs_kernel,*dev_kernel;
	cudaHostAlloc(&gs_kernel,sizeof(double)*KERNEL_SIZE*KERNEL_SIZE,cudaHostAllocDefault);
	for(int i=0;i<KERNEL_SIZE;i++){
		double* row=gauss_kernel.ptr<double>(i);
		for(int j=0;j<KERNEL_SIZE;j++){
			gs_kernel[i*KERNEL_SIZE+j]=row[j];
		}
	}
	cudaMalloc(&dev_kernel,sizeof(double)*KERNEL_SIZE*KERNEL_SIZE);
	//allocate 
	//allocate
	double* x,*y;
	cudaHostAlloc(&x,sizeof(double)*KERNEL_SIZE*2,cudaHostAllocDefault);
	double *row_x=gauss_x.ptr<double>(0),*row_y=gauss_y.ptr<double>(0);
	for(int i=0;i<KERNEL_SIZE*2;i++){
		if(i<KERNEL_SIZE){
			x[i]=row_x[i];
			//std::cout<<x[i]<<std::endl;
		}
		else
			x[i]=row_y[i-KERNEL_SIZE];
	}
	//cudaHostAlloc(&y,sizeof(double)*KERNEL_SIZE,cudaHostAllocDefault);
	//allocate
	copyMakeBorder(*src,padding_image,kernel_radius,kernel_radius,kernel_radius,kernel_radius,BORDER_CONSTANT, 0);
	int grid_num_x=(src->cols+THREAD_X-1)/THREAD_X,grid_num_y=(src->rows+THREAD_Y-1)/THREAD_Y;
	//int grid_num_x=(src->cols+THREAD_X-1)/THREAD_X,grid_num_y=(src->rows+THREAD_Y-1)/THREAD_Y;
	result.upload(*dst);
	//g_kernel.upload(gauss_kernel);

	//use seperate do no padding
	//device_image.upload(padding_image);

	//device_image.upload(padding_image);
	device_image.upload(*src);
	cudaMemcpyToSymbol(guass_kernel_x,x,sizeof(double)*2*KERNEL_SIZE);
	cudaMemcpyToSymbol(guass_kernel,gs_kernel,sizeof(double)*KERNEL_SIZE*KERNEL_SIZE);
	dim3 thread_block(THREAD_X,THREAD_Y);
	dim3 grid(grid_num_x,grid_num_y);
	//convolution<<<grid,thread_block>>>(device_image,result,KERNEL_SIZE,kernel_radius,orign_width,orign_height);
	conv_x<<<grid,thread_block>>>(device_image,result,KERNEL_SIZE,kernel_radius,orign_width,orign_height);
	cudaDeviceSynchronize();
	Mat re;
	result.download(re);
	resul.upload(re);
	//device_image.upload(*dst);
	conv_y<<<grid,thread_block>>>(resul,device_image,KERNEL_SIZE,kernel_radius,orign_width,orign_height);
	device_image.download(*dst);
	return ;
}


