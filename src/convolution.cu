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
#define WRAP_NUM 32
#define MAX_WRAP_NUM 32

//using namespace cv;
//using namespace cv;

__constant__ double guass_kernel_x[128*2];
__constant__ double guass_kernel_y[128];
static int KERNEL_SIZE;

//not need to padding
__global__ void conv_x(GPU::PtrStepSz<float> src,/*const double* __restrict__ guass_kernel,*/GPU::PtrStepSz<float> dst,int kernel_size,int kernel_radius,int orign_width,int orign_height){
	__shared__ float  share_mem[100][100];
	int left_limit=kernel_radius,right_limit=blockDim.x-kernel_radius;
	int pixel_i=blockDim.x*blockIdx.x+threadIdx.x-2*blockIdx.x*kernel_radius;
	int pixel_j=blockDim.y*blockIdx.y+threadIdx.y;
	int thread_block_index=threadIdx.x+threadIdx.y*blockDim.x;
	share_mem[thread_block_index%32][thread_block_index/32]=0;
	__syncthreads();
	float sum=0,sum1=0,sum2=0;
	if(!(pixel_i<kernel_radius || pixel_j<kernel_radius || pixel_i>=orign_width+kernel_radius  || pixel_j>=orign_height+kernel_radius)){//real image size
		share_mem[thread_block_index%32][thread_block_index/32]=src(pixel_j,pixel_i);
		__syncthreads();
		if(threadIdx.x>= left_limit && threadIdx.x<right_limit){ //non padding size
			int x=threadIdx.x-kernel_radius,y=threadIdx.y;

			for(int i=0;i<kernel_size;i++){
				
				thread_block_index=(x+i)+y*blockDim.x;
				sum+=share_mem[thread_block_index%32][thread_block_index/32]*(float)guass_kernel_x[i];
			}
			dst(pixel_j-kernel_radius,pixel_i-kernel_radius)=sum;//src(pixel_j,pixel_i);
		}
	}
	return ;
}
__global__ void conv_y(GPU::PtrStepSz<float> src,/*const double* __restrict__ guass_kernel,*/GPU::PtrStepSz<float> dst,int kernel_size,int kernel_radius,int orign_width,int orign_height){
	__shared__ float  share_mem[100][100];
	int top_limit=kernel_radius,down_limit=blockDim.y-kernel_radius;
	int pixel_i=blockDim.x*blockIdx.x+threadIdx.x;
	int pixel_j=blockDim.y*blockIdx.y+threadIdx.y-2*blockIdx.y*kernel_radius;
	int thread_block_index=threadIdx.x+threadIdx.y*blockDim.x;
	share_mem[thread_block_index%32][thread_block_index/32]=0;
	__syncthreads();
	float sum=0.0,sum1=0,sum2=0;

	if(!(pixel_i<kernel_radius || pixel_j<kernel_radius || pixel_i>=orign_width+kernel_radius  || pixel_j>=orign_height+kernel_radius)){
		share_mem[thread_block_index%32][thread_block_index/32]=src(pixel_j,pixel_i);
		__syncthreads();
		if(threadIdx.y>= top_limit && threadIdx.y<down_limit){
			int x=threadIdx.x,y=threadIdx.y-kernel_radius;
			for(int i=0;i<kernel_size;i++){
				thread_block_index=x+(y+i)*blockDim.x;
				sum+=share_mem[thread_block_index%32][thread_block_index/32]*(float)guass_kernel_x[i];
			}
		
		dst(pixel_j-kernel_radius,pixel_i-kernel_radius)=sum;//src(pixel_j,pixel_i);//sum;
		}
	}
	return ;
}

void guassain_conv(const Mat *src,Mat *dst,double sigma,int thread_x,int thread_y,int thread_x1,int thread_y1){
	
	KERNEL_SIZE = cvRound(sigma* 4 * 2 + 1)|1;
	//std::cout<<KERNEL_SIZE<<std::endl;
	int kernel_radius=KERNEL_SIZE/2;
	int orign_width=src->cols,orign_height=src->rows;
	Mat padding_image;
	GPU::GpuMat device_image,g_kernel,result, dev_image,resul;

	if(GPU::getCudaEnabledDeviceCount()==0){
		std::cout<<"not use GPU module"<<std::endl;
		return ;
	}
	Mat gauss_x=getGaussianKernel(KERNEL_SIZE,sigma);//,gauss_y=getGaussianKernel(KERNEL_SIZE,sigma); //3*3 filter
	//allocate 
	//allocate
	double* x,*y;
	cudaHostAlloc(&x,sizeof(double)*KERNEL_SIZE,cudaHostAllocDefault);
	double *row_x=gauss_x.ptr<double>(0);//,*row_y=gauss_y.ptr<double>(0);
	for(int i=0;i<KERNEL_SIZE;i++){
			x[i]=row_x[i];
	}
	//allocate
	copyMakeBorder(*src,padding_image,kernel_radius,kernel_radius,kernel_radius,kernel_radius,BORDER_CONSTANT, 0);
	int t_x=thread_x-2*kernel_radius,t_y=thread_y;
	int grid_num_x=(padding_image.cols+t_x-1)/t_x,grid_num_y=(padding_image.rows+t_y-1)/t_y;

	result.upload(*dst);
	device_image.upload(padding_image);
	cudaMemcpyToSymbol(guass_kernel_x,x,sizeof(double)*KERNEL_SIZE);

	dim3 thread_block(thread_x,thread_y);
	dim3 grid(grid_num_x,grid_num_y);
	conv_x<<<grid,thread_block>>>(device_image,result,KERNEL_SIZE,kernel_radius,orign_width,orign_height);
	cudaDeviceSynchronize();

	Mat re;
	result.download(re);
	copyMakeBorder(re,padding_image,kernel_radius,kernel_radius,kernel_radius,kernel_radius,BORDER_CONSTANT, 0);
	device_image.upload(padding_image);

	
	t_x=thread_x1;
	t_y=thread_y1-2*kernel_radius;
	grid_num_x=(padding_image.cols+t_x-1)/t_x,grid_num_y=(padding_image.rows+t_y-1)/t_y;
	dim3 thread_block1(thread_x1,thread_y1);
	dim3 grid1(grid_num_x,grid_num_y);

	conv_y<<<grid1,thread_block1>>>(device_image,result,KERNEL_SIZE,kernel_radius,orign_width,orign_height);
	result.download(*dst);
	return ;
}


