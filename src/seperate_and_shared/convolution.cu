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
int KERNEL_SIZE;

//not need to padding
__global__ void conv_x(GPU::PtrStepSz<float> src,/*const double* __restrict__ guass_kernel,*/GPU::PtrStepSz<float> dst,int kernel_size,int kernel_radius,int orign_width,int orign_height){
	__shared__ float  share_mem[100][100];
	/*int pixel_i=blockDim.x*blockIdx.x+threadIdx.x;
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
	}*/
	int left_limit=kernel_radius,right_limit=blockDim.x-kernel_radius;
	int pixel_i=blockDim.x*blockIdx.x+threadIdx.x-2*blockIdx.x*kernel_radius;
	int pixel_j=blockDim.y*blockIdx.y+threadIdx.y;
	int thread_block_index=threadIdx.x+threadIdx.y*blockDim.x;
	//share_mem[threadIdx.y][threadIdx.x]=0;
	share_mem[thread_block_index%32][thread_block_index/32]=0;
	//share_mem[10]=src(pixel_j,pixel_i);
	__syncthreads();
	float sum=0,sum1=0,sum2=0;
	if(!(pixel_i<kernel_radius || pixel_j<kernel_radius || pixel_i>=orign_width+kernel_radius  || pixel_j>=orign_height+kernel_radius)){//real image size
		share_mem[thread_block_index%32][thread_block_index/32]=src(pixel_j,pixel_i);
		__syncthreads();
		if(threadIdx.x>= left_limit && threadIdx.x<right_limit){ //non padding size
			int x=threadIdx.x-kernel_radius,y=threadIdx.y;

			for(int i=0;i<kernel_size;i++){
				
				thread_block_index=(x+i)+y*blockDim.x;
	//			if(thread_block_index>=2048 || thread_block_index<0)
	//				                                       printf("%d\n",thread_block_index/WRAP_NUM);
	//			if(x+i<100 && x+i>=0)
	//			sum+=share_mem[y][x+i]*(float)guass_kernel_x[i];
				//sum+=src(pixel_j,pixel_i-kernel_radius+i)*(float)guass_kernel_x[i];
				sum+=share_mem[thread_block_index%32][thread_block_index/32]*(float)guass_kernel_x[i];
				//sum1+=src(pixel_j,pixel_i-kernel_radius+i).y*(float)guass_kernel_x[i];
				//sum2+=src(pixel_j,pixel_i-kernel_radius+i).z*(float)guass_kernel_x[i];
			}
			dst(pixel_j-kernel_radius,pixel_i-kernel_radius)=sum;//src(pixel_j,pixel_i);
			//dst(pixel_j-kernel_radius,pixel_i-kernel_radius).y=sum1;
			//dst(pixel_j-kernel_radius,pixel_i-kernel_radius).z=sum2;
		}
		//dst(pixel_j,pixel_i)=sum;	
	}
	//dst(pixel_j,pixel_i)=sum;
	return ;
}
__global__ void conv_y(GPU::PtrStepSz<float> src,/*const double* __restrict__ guass_kernel,*/GPU::PtrStepSz<float> dst,int kernel_size,int kernel_radius,int orign_width,int orign_height){
	__shared__ float  share_mem[100][100];
	/*int pixel_i=blockDim.x*blockIdx.x+threadIdx.x;
	int pixel_j=blockDim.y*blockIdx.y+threadIdx.y;
        int shared_i=threadIdx.x;
	int shared_j=threadIdx.y+kernel_size/2;
	float sum=0;
	if(!(pixel_i>=orign_width || pixel_j>=orign_height)){
		share_mem[shared_j][shared_i]=src(pixel_j,pixel_i);
	
	__syncthreads();
		int start_i=shared_i,    start_j=shared_j-kernel_radius;
		for(int i=0;i<kernel_size;i++){
			sum+=share_mem[start_j+i][start_i]*(float)guass_kernel_x[i];
			//sum+=share_mem[start_j+i][start_i];
		}
		dst(pixel_j,pixel_i)=sum;//share_mem[shared_j][shared_i];//sum;
	}*/
	int top_limit=kernel_radius,down_limit=blockDim.y-kernel_radius;
	int pixel_i=blockDim.x*blockIdx.x+threadIdx.x;
	int pixel_j=blockDim.y*blockIdx.y+threadIdx.y-2*blockIdx.y*kernel_radius;
	int thread_block_index=threadIdx.x+threadIdx.y*blockDim.x;
//	share_mem[threadIdx.y][threadIdx.x]=0;
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
				//sum+=src(pixel_j-kernel_radius+i,pixel_i)*(float)guass_kernel_x[i];
//				if(y+i<100 && y>=0)
//				sum+=share_mem[y+i][x]*(float)guass_kernel_x[i];
				//sum1+=/*share_mem[thread_block_index%WRAP_NUM][thread_block_index/WRAP_NUM]src(pixel_j-kernel_radius+i,pixel_i).y*(float)guass_kernel_x[i];
				  //                              sum2+=src(pixel_j-kernel_radius+i,pixel_i).z*(float)guass_kernel_x[i];
			}
		
		dst(pixel_j-kernel_radius,pixel_i-kernel_radius)=sum;//src(pixel_j,pixel_i);//sum;
//		dst(pixel_j-kernel_radius,pixel_i-kernel_radius).y=sum1;
//		                        dst(pixel_j-kernel_radius,pixel_i-kernel_radius).z=sum2;
		}
	}
	//dst(pixel_j,pixel_i)=sum;
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
//	Mat gauss_kernel=gauss_x*gauss_y.t();
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
	//allocate
	double* x,*y;
	cudaHostAlloc(&x,sizeof(double)*KERNEL_SIZE,cudaHostAllocDefault);
	double *row_x=gauss_x.ptr<double>(0);//,*row_y=gauss_y.ptr<double>(0);
	for(int i=0;i<KERNEL_SIZE;i++){
		//if(i<KERNEL_SIZE){
			x[i]=row_x[i];
			//std::cout<<x[i]<<std::endl;
		//}
		//else
		//	x[i]=row_y[i-KERNEL_SIZE];
	}
	//cudaHostAlloc(&y,sizeof(double)*KERNEL_SIZE,cudaHostAllocDefault);
	//allocate
	copyMakeBorder(*src,padding_image,kernel_radius,kernel_radius,kernel_radius,kernel_radius,BORDER_CONSTANT, 0);
//	int orign_grid_num_x=(src->cols+thread_x-1)/thread_x,orign_grid_num_y=(src->rows+thread_y-1)/thread_y;
//	int grid_num_x=orign_grid_num_x+(2*kernel_radius*orign_grid_num_x+thread_x-1)/thread_x,grid_num_y=orign_grid_num_y+(2*kernel_radius*orign_grid_num_y+thread_y-1)/thread_y;
	int t_x=thread_x-2*kernel_radius,t_y=thread_y;
	        int grid_num_x=(padding_image.cols+t_x-1)/t_x,grid_num_y=(padding_image.rows+t_y-1)/t_y;
	//int grid_num_x=(src->cols+THREAD_X-1)/THREAD_X,grid_num_y=(src->rows+THREAD_Y-1)/THREAD_Y;
	result.upload(*dst);
	//g_kernel.upload(gauss_kernel);

	//use seperate do no padding
	//device_image.upload(padding_image);

	device_image.upload(padding_image);
	//device_image.upload(*src);
	cudaMemcpyToSymbol(guass_kernel_x,x,sizeof(double)*KERNEL_SIZE);
	//cudaMemcpyToSymbol(guass_kernel,gs_kernel,sizeof(double)*KERNEL_SIZE*KERNEL_SIZE);
	dim3 thread_block(thread_x,thread_y);
	dim3 grid(grid_num_x,grid_num_y);
	//convolution<<<grid,thread_block>>>(device_image,result,KERNEL_SIZE,kernel_radius,orign_width,orign_height);
	conv_x<<<grid,thread_block>>>(device_image,result,KERNEL_SIZE,kernel_radius,orign_width,orign_height);
	cudaDeviceSynchronize();
	Mat re;
	result.download(re);
	copyMakeBorder(re,padding_image,kernel_radius,kernel_radius,kernel_radius,kernel_radius,BORDER_CONSTANT, 0);
	//resul.upload(re);
	device_image.upload(padding_image);
	//result.upload(*dst);	
	//orign_grid_num_x=(src->cols+thread_x1-1)/thread_x1,orign_grid_num_y=(src->rows+thread_y1-1)/thread_y1;
	//grid_num_x=orign_grid_num_x+(2*kernel_radius*orign_grid_num_x+thread_x1-1)/thread_x1,grid_num_y=orign_grid_num_y+(2*kernel_radius*orign_grid_num_y+thread_y1-1)/thread_y1;

	
	t_x=thread_x1;
	t_y=thread_y1-2*kernel_radius;
	grid_num_x=(padding_image.cols+t_x-1)/t_x,grid_num_y=(padding_image.rows+t_y-1)/t_y;
	dim3 thread_block1(thread_x1,thread_y1);
	dim3 grid1(grid_num_x,grid_num_y);

	conv_y<<<grid1,thread_block1>>>(device_image,result,KERNEL_SIZE,kernel_radius,orign_width,orign_height);
	result.download(*dst);
	return ;
}


