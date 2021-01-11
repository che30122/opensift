CC	= g++
NVCC    = nvcc
CFLAGS	+= -std=c++11
BIN_DIR = ../bin
INC_DIR	= ../include
LIB_DIR	= ../lib
INCL	= -I$(INC_DIR) `pkg-config --cflags opencv gtk+-2.0`
LIBS	= -L$(LIB_DIR) -lopensift -lm `pkg-config --libs opencv gtk+-2.0`
OBJ	= imgfeatures.o utils.o sift.o kdtree.o minpq.o xform.o convolution.o match.o convolution_3channel.o
BIN     = siftfeat match dspfeat match_num
INC_OPENCV_DIR=/usr/local/lib
INC_OPENCV=-I$(INC_OPENCV_DIR) -I/usr/local/include/opencv4
OPENCV_INC=`pkg-config --cflags opencv gtk+-2.0`
OPENCV_LIBS= `pkg-config --libs opencv gtk+-2.0`

all: sift conv cvsmooth

cvsmooth: cvsmooth.o
	$(CC) $(CFLAGS)  cvsmooth.cpp $(INC_OPENCV) $(OPENCV_INC) $(OPENCV_LIBS) -o cvsmooth
sift: $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(OBJ)  $(INC_OPENCV) $(OPENCV_LIBS) -G -o sift
conv: main.o convolution_3channel.o
	$(NVCC) $(NVCCFLAGS) main.o convolution_3channel.o  $(INC_OPENCV) $(OPENCV_LIBS) -o conv
cvsmooth.o: cvsmooth.cpp
	$(CC) $(CFLAGS) -c cvsmooth.cpp $(INC_OPENCV) $(OPENCV_INC) $(OPENCV_LIBS)
main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp $(INC_OPENCV) $(OPENCV_INC) $(OPENCV_LIBS)
match.o: match.cpp
	$(CC) $(CFLAGS) $(INCL) -c match.cpp -o $@
imgfeatures.o: imgfeatures.cpp $(INC_DIR)/imgfeatures.h
	$(CC) $(CFLAGS) $(INCL) -c imgfeatures.cpp -o $@

utils.o: utils.cpp $(INC_DIR)/utils.h
	$(CC) $(CFLAGS) $(INCL) -c utils.cpp -o $@

sift.o: sift.cpp $(INC_DIR)/sift.h
	$(CC) $(CFLAGS) $(INCL) -c sift.cpp -o $@

kdtree.o: kdtree.cpp $(INC_DIR)/kdtree.h
	$(CC) $(CFLAGS) $(INCL) -c kdtree.cpp -o $@

minpq.o: minpq.cpp $(INC_DIR)/minpq.h
	$(CC) $(CFLAGS) $(INCL) -c minpq.cpp -o $@

xform.o: xform.cpp $(INC_DIR)/xform.h
	$(CC) $(CFLAGS) $(INCL) -c xform.cpp -o $@
convolution.o: convolution.cu
	$(NVCC) $(NVCC_FLAGS) -c convolution.cu $(INC_OPENCV)  $(OPENCV_LIBS) -lopencv_gpu
convolution_3channel.o: convolution_3channel.cu
	$(NVCC) $(NVCC_FLAGS) -c convolution_3channel.cu $(INC_OPENCV)  $(OPENCV_LIBS) -lopencv_gpu
clean:
	rm -f *~ *.o sift conv cvsmooth

.PHONY: clean
