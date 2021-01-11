#include<opencv2/opencv.hpp>

using namespace cv;

extern void guassain_3conv(const Mat*,Mat*,double/*,int,int,int,int*/);

int main(int argc,char** argv){
	Mat img=imread(argv[1]);
	Mat result=imread(argv[1]);
//	guassain_3conv(&img,&result,atof(argv[3]),atoi(argv[4]),atoi(argv[5]),atoi(argv[6]),atoi(argv[7]));
	guassain_3conv(&img,&result,atof(argv[3]));
	imwrite(argv[2],result);
	return 0;
}
