##

SRCS = device.cu umatrix.cc model.cc main.cc 
OBJS = $(SRCS:.cc=.o) 

CC = nvcc
CXX_FLAGS = -arch=sm_35 -rdc=true -lcudadevrt 

.SUFFIXES: .cc .o
.cc.o:
	$(CC) $(CXX_FLAGS) -c -o $@ $<

default: main
all: main

main: $(OBJS) 
	$(CC) $(CXX_FLAGS) -o $@ $^

clean:
	/bin/rm -f *.o
	/bin/rm -f main
