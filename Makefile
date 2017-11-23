gpu: sudoku_gpu.cu
	nvcc -g -o sudokusolver sudoku_gpu.cu
gpu_debug: sudoku_gpu.cu
	nvcc -DDEBUG -g -o sudokusolver sudoku_gpu.cu
cpu_ga: sudoku_ga_cpu.c
	gcc -g -o sudokusolver sudoku_ga_cpu.c
