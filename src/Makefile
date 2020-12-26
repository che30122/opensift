CC	= g++
CFLAGS	+= -std=c++11
BIN_DIR = ../bin
INC_DIR	= ../include
LIB_DIR	= ../lib
INCL	= -I$(INC_DIR) `pkg-config --cflags opencv gtk+-2.0`
LIBS	= -L$(LIB_DIR) -lopensift -lm `pkg-config --libs opencv gtk+-2.0`
OBJ	= imgfeatures.o utils.o sift.o kdtree.o minpq.o xform.o
BIN     = siftfeat match dspfeat match_num

all: $(BIN) libopensift.a

libopensift.a: $(OBJ)
	ar rc $(LIB_DIR)/$@ $(OBJ)
	ranlib  $(LIB_DIR)/$@

siftfeat: libopensift.a siftfeat.cpp
	$(CC) $(CFLAGS) $(INCL) siftfeat.cpp -o $(BIN_DIR)/$@ $(LIBS)

match: libopensift.a match.cpp
	$(CC) $(CFLAGS) $(INCL) match.cpp -o $(BIN_DIR)/$@ $(LIBS)

match_num: libopensift.a match.cpp
	$(CC) $(CFLAGS) $(INCL) match_num.cpp -o $(BIN_DIR)/$@ $(LIBS)

dspfeat: libopensift.a dspfeat.cpp
	$(CC) $(CFLAGS) $(INCL) dspfeat.cpp -o $(BIN_DIR)/$@ $(LIBS)

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

clean:
	rm -f *~ *.o core

.PHONY: clean
