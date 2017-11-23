cpu_bt:	sudoku_bt_cpu.c
	gcc -g -o sudoku_bt_cpu sudoku_bt_cpu.c
cpu_bt_iter: sudoku_bt_iter_cpu.c
	gcc -g -o sudoku_bt_iter_cpu sudoku_bt_iter_cpu.c
cpu_ga: sudoku_ga_cpu.c
	gcc -g -o sudoku_ga_cpu sudoku_ga_cpu.c
gpu: sudoku_gpu.cu
	nvcc -g -o sudoku_gpu sudoku_gpu.cu
gpu_debug: sudoku_gpu.cu
	nvcc -DDEBUG -g -o sudoku_gpu sudoku_gpu.cu
