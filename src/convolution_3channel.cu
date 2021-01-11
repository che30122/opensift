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
#define THREAD_X1 32
#define THREAD_Y1 32
#define WRAP_NUM 32
#define MAX_WRAP_NUM 32

//using namespace cv;
//using namespace cv;

__constant__ double guass_kernel[2048];
__constant__ double guass_kernel_x[128*2];
__constant__ double guass_kernel_y[128];
static int KERNEL_SIZE;

//not need to padding
__global__ void conv_3x(GPU::PtrStepSz<uchar3> src,/*const double* __restrict__ guass_kernel,*/GPU::PtrStepSz<uchar3> dst,int kernel_size,int kernel_radius,int orign_width,int orign_height){
	__shared__ float  share_mem[100][100];
	int left_limit=kernel_radius,right_limit=blockDim.x-kernel_radius;
	int pixel_i=blockDim.x*blockIdx.x+threadIdx.x-2*blockIdx.x*kernel_radius;
	int pixel_j=blockDim.y*blockIdx.y+threadIdx.y;
	int thread_block_index=threadIdx.x+threadIdx.y*blockDim.x;
//	share_mem[thread_block_index%32][thread_block_index/32]=src(pixel_j,pixel_i);
	//share_mem[10]=src(pixel_j,pixel_i);
	__syncthreads();
	float sum=0,sum1=0,sum2=0;
	if(!(pixel_i<kernel_radius || pixel_j<kernel_radius || pixel_i>=orign_width+kernel_radius  || pixel_j>=orign_height+kernel_radius)){//real image size
		if(threadIdx.x>= left_limit && threadIdx.x<right_limit){ //non padding size
			int x=threadIdx.x-kernel_radius,y=threadIdx.y;

			for(int i=0;i<kernel_size;i++){
				
				thread_block_index=(x+i)+y*blockDim.x;
	//			if(thread_block_index>=2048 || thread_block_index<0)
	//				                                        printf("%d\n",thread_block_index/WRAP_NUM);
				sum+=src(pixel_j,pixel_i-kernel_radius+i).x*(float)guass_kernel_x[i];
	//			sum+=share_mem[thread_block_index%32][thread_block_index/32]*(float)guass_kernel_x[i];
				sum1+=src(pixel_j,pixel_i-kernel_radius+i).y*(float)guass_kernel_x[i];
				sum2+=src(pixel_j,pixel_i-kernel_radius+i).z*(float)guass_kernel_x[i];
			}
			dst(pixel_j-kernel_radius,pixel_i-kernel_radius).x=sum;//sum;//src(pixel_j,pixel_i);
			dst(pixel_j-kernel_radius,pixel_i-kernel_radius).y=sum1;//sum1;
			dst(pixel_j-kernel_radius,pixel_i-kernel_radius).z=sum2;//sum2;
		}
		//dst(pixel_j,pixel_i)=sum;	
	}
	//dst(pixel_j,pixel_i)=sum;
	return ;
}
__global__ void conv_3y(GPU::PtrStepSz<uchar3> src,/*const double* __restrict__ guass_kernel,*/GPU::PtrStepSz<uchar3> dst,int kernel_size,int kernel_radius,int orign_width,int orign_height){
	__shared__ float  share_mem[100][100];
	int top_limit=kernel_radius,down_limit=blockDim.y-kernel_radius;
	int pixel_i=blockDim.x*blockIdx.x+threadIdx.x;
	int pixel_j=blockDim.y*blockIdx.y+threadIdx.y-2*blockIdx.y*kernel_radius;
	int thread_block_index=threadIdx.x+threadIdx.y*blockDim.x;
//	share_mem[thread_block_index%32][thread_block_index/32]=src(pixel_j,pixel_i);
	__syncthreads();
	float sum=0.0,sum1=0,sum2=0;

	if(!(pixel_i<kernel_radius || pixel_j<kernel_radius || pixel_i>=orign_width+kernel_radius  || pixel_j>=orign_height+kernel_radius)){
		if(threadIdx.y>= top_limit && threadIdx.y<down_limit){
			int x=threadIdx.x,y=threadIdx.y-kernel_radius;
			for(int i=0;i<kernel_size;i++){
				thread_block_index=x+(y+i)*blockDim.x;
//				sum+=share_mem[thread_block_index%32][thread_block_index/32]*(float)guass_kernel_x[i];
				sum+=src(pixel_j-kernel_radius+i,pixel_i).x*(float)guass_kernel_x[i];
				
				sum1+=/*share_mem[thread_block_index%WRAP_NUM][thread_block_index/WRAP_NUM]*/src(pixel_j-kernel_radius+i,pixel_i).y*(float)guass_kernel_x[i];
				                                sum2+=src(pixel_j-kernel_radius+i,pixel_i).z*(float)guass_kernel_x[i];
			}
		
		dst(pixel_j-kernel_radius,pixel_i-kernel_radius).x=sum;//sum;//src(pixel_j,pixel_i);//sum;
		dst(pixel_j-kernel_radius,pixel_i-kernel_radius).y=sum1;//sum1;
		                        dst(pixel_j-kernel_radius,pixel_i-kernel_radius).z=sum2;//sum2;
		}
	}
	//dst(pixel_j,pixel_i)=sum;
	return ;
}

void guassain_3conv(const Mat *src,Mat *dst,double sigma){
//	int depth = CV_MAT_DEPTH(src.type());
	KERNEL_SIZE = cvRound(sigma* 4 * 2 + 1)|1;
	std::cout<<KERNEL_SIZE<<std::endl;
	int kernel_radius=KERNEL_SIZE/2;
	int orign_width=src->cols,orign_height=src->rows;
	Mat padding_image;
	GPU::GpuMat device_image,g_kernel,result, dev_image,resul;

	if(GPU::getCudaEnabledDeviceCount()==0){
		std::cout<<"not use GPU module"<<std::endl;
		return ;
	}
	Mat gauss_x=getGaussianKernel(KERNEL_SIZE,sigma),gauss_y=getGaussianKernel(KERNEL_SIZE,sigma); //3*3 filter
	//Mat gauss_kernel=gauss_x*gauss_y.t();
	//allocate
	/*double* gs_kernel,*dev_kernel;
	cudaHostAlloc(&gs_kernel,sizeof(double)*KERNEL_SIZE*KERNEL_SIZE,cudaHostAllocDefault);
	for(int i=0;i<KERNEL_SIZE;i++){
		double* row=gauss_kernel.ptr<double>(i);
		for(int j=0;j<KERNEL_SIZE;j++){
			gs_kernel[i*KERNEL_SIZE+j]=row[j];
		}
	}*/
	//cudaMalloc(&dev_kernel,sizeof(double)*KERNEL_SIZE*KERNEL_SIZE);
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
	//int orign_grid_num_x=(padding_image.cols+THREAD_X-1)/THREAD_X,orign_grid_num_y=(padding_image.rows+THREAD_Y-1)/THREAD_Y;
	int t_x=THREAD_X-2*kernel_radius,t_y=THREAD_Y;
	int grid_num_x=(padding_image.cols+t_x-1)/t_x,grid_num_y=(padding_image.rows+t_y-1)/t_y;
	//int grid_num_x=orign_grid_num_x+(2*kernel_radius*orign_grid_num_x+THREAD_X-1)/THREAD_X,grid_num_y=orign_grid_num_y+(2*kernel_radius*orign_grid_num_y+THREAD_Y-1)/THREAD_Y;
	//int grid_num_x=(src->cols+THREAD_X-1)/THREAD_X,grid_num_y=(src->rows+THREAD_Y-1)/THREAD_Y;
	result.upload(*dst);
	//g_kernel.upload(gauss_kernel);

	//use seperate do no padding
	//device_image.upload(padding_image);

	device_image.upload(padding_image);
	//device_image.upload(*src);
	cudaMemcpyToSymbol(guass_kernel_x,x,sizeof(double)*2*KERNEL_SIZE);
	//cudaMemcpyToSymbol(guass_kernel,gs_kernel,sizeof(double)*KERNEL_SIZE*KERNEL_SIZE);
	dim3 thread_block(THREAD_X,THREAD_Y);
	dim3 grid(grid_num_x,grid_num_y);
	//convolution<<<grid,thread_block>>>(device_image,result,KERNEL_SIZE,kernel_radius,orign_width,orign_height);
	conv_3x<<<grid,thread_block>>>(device_image,result,KERNEL_SIZE,kernel_radius,orign_width,orign_height);
	cudaDeviceSynchronize();
	Mat re;
	result.download(re);
	copyMakeBorder(re,padding_image,kernel_radius,kernel_radius,kernel_radius,kernel_radius,BORDER_CONSTANT, 0);
	//resul.upload(re);
	device_image.upload(padding_image);  
	

	
	//orign_grid_num_x=(padding_image.cols+THREAD_X1-1)/THREAD_X1,orign_grid_num_y=(padding_image.rows+THREAD_Y1-1)/THREAD_Y1;
	t_x=THREAD_X1;
	t_y=THREAD_Y1-2*kernel_radius;
        //grid_num_x=orign_grid_num_x+(2*kernel_radius*orign_grid_num_x+THREAD_X1-1)/THREAD_X1,grid_num_y=orign_grid_num_y+(2*kernel_radius*orign_grid_num_y+THREAD_Y1-1)/THREAD_Y1;
	grid_num_x=(padding_image.cols+t_x-1)/t_x,grid_num_y=(padding_image.rows+t_y-1)/t_y;
	dim3 thread_block1(THREAD_X1,THREAD_Y1);
        dim3 grid1(grid_num_x,grid_num_y);
	conv_3y<<<grid1,thread_block1>>>(device_image,result,KERNEL_SIZE,kernel_radius,orign_width,orign_height);
	result.download(*dst);
	return ;
}


