#--------------------------------------------------------------------------
#usage: sh make.sh [emu=1]
#--------------------------------------------------------------------------

gcc cpu.c -o cpu
nvcc gpu.cu -o gpu
nvcc zero.cu -o zero
nvcc writeC.cu -o writeC
nvcc two.cu -o two
./cpu
./gpu
./zero
./writeC
./two

