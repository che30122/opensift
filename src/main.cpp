#include<opencv2/opencv.hpp>

using namespace cv;

extern void guassain_conv(const Mat*,Mat*,double);
int main(int argc,char** argv){
	Mat img=imread(argv[1]);
	Mat result=imread(argv[1]);
	guassain_conv(&img,&result,2);
	imwrite(argv[2],result);
	return 0;
}
