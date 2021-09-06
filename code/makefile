CC = g++
INCLUDE_PATHS = -I/usr/local/include/
CC_FLAGS = -O4 -Ofast -w
LIB_PATHS = -L/usr/local/lib/ -L/home/szaman5/FS-Laser-ML/fftw-3.3.9/lib

all: compile 
	$(CC) $(LIB_PATHS) LGspacelin_5pcr.o -lgsl -lgslcblas -lm -lfftw3 -fopenmp
compile:
	$(CC) $(CC_FLAGS) $(INCLUDE_PATHS) -c LGspacelin_5pcr.cpp -lfftw3 -fopenmp

.PHONY : clean

clean:
	rm *.o a.out