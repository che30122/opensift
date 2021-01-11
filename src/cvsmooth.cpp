#include<opencv2/opencv.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include<opencv2/core/core.hpp>
//#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
using namespace cv;

//extern void guassain_conv(const Mat*,Mat*,double,int,int,int,int);
extern void guassain_conv(const Mat*,Mat*,double);
int main(int argc,char** argv){
	Mat img=imread(argv[1]);
	IplImage* img1,*img2;
	Mat result=imread(argv[1]);
	img1 = cvCreateImage(cvSize(img.cols,img.rows),8,3);
IplImage ipltemp2=cvIplImage(img);
cvCopy(&ipltemp2,img1);
img2 = cvCreateImage(cvSize(result.cols,result.rows),8,3);
IplImage ipltemp1=cvIplImage(result);
cvCopy(&ipltemp1,img2);
	//guassain_conv(&img,&result,atof(argv[3]),atoi(argv[4]),atoi(argv[5]),atoi(argv[6]),atoi(argv[7]));
	//guassain_conv(&img,&result,atof(argv[3]));
	cvSmooth(img1,img2,CV_GAUSSIAN,0,0,atof(argv[3]),atof(argv[3]));
	//GaussianBlur(img,result,0,atof(argv[3]),atof(argv[3]));
	
	imwrite(argv[2],cvarrToMat(img2));
	return 0;
}
