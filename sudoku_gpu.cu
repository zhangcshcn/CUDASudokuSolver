/*  Author: Chen Zhang, NYU Courant
 *  
 *  This is a sukodu solver using stochastic methods.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


#define puzzlePb 32
#define NBLOCK 9
#define index(x, y) (9 * (x) + (y))
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void printBoard(int *board){
  for (int i = 0; i<9; i++){
    for (int j=0; j< 9; j++){
      printf("%d ", board[index(i, j)]);
    }
    printf("\n");
  }
}


void printBoardReadable(int* su){
  for (int ii = 0; ii < 3; ii ++){
    for (int jj = 0; jj < 3; jj ++){
      for (int i = 0; i < 3; i ++){
        for (int j = 0; j < 3; j ++){
          printf("%d ",su[index(3*ii+i,3*jj+j)]);
        }
      }
      printf("\n");
    }
  }
}


// Fill the blocks such that each block contains a permutation of [1..9]
void initBoard(int* su, int* mask, int* mutableIdx, int* mutableCnt){
  int rec[9]; 
  for (int k = 0; k < 9; k++){
    for (int i = 0; i < 9; i++)
      rec[i] = 0;
    int j = 0;
    for (int i = 0; i < 9; i++){
      if (mask[index(k,i)]){
        // Use a bitmap to mark the existing numbers.
        rec[su[index(k,i)]-1] = 1;
      } else {
        mutableIdx[index(k, j)] = i;
        j ++;
      }
    }
    
    for (int i = 0, kk = 0; i < 9; i++){
      if (!rec[i]){
        rec[kk] = i+1;
        kk++;
      }
    }
    mutableCnt[k] = j;

    for (int i = 0; i < 9; i++){
      if (!mask[index(k,i)]){
        int idx = rand() % j;
        int tmp = rec[idx];
        rec[idx] = rec[j-1];
        rec[j-1] = tmp;
        j--;
        su[index(k,i)] = tmp;
      }
    }
  }
}


int _assertInit(int* su){
  for (int k = 0; k< 9; k++){
    int nums[9];
    for (int i = 0; i < 9; i++)
      nums[i] = 0;
    for (int i = 0; i < 9; i++){
      if (nums[su[index(k,i)]-1]) {
        printf("Duplication in a block detected at %d, %d\n", k, i);
        exit(1);
      } else  nums[su[index(k,i)]-1] = 1;
    }
  }
  return 1;
}


__global__ void initRandKernel(curandState_t *state, unsigned int seed) {
  int idx= blockIdx.x*puzzlePb +threadIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}


/* Load the given board.
 * A1 A2 A3 B1 B2 B3 C1 C2 C3
 * A4 A5 A6 B4 B5 B6 C4 C5 C6
 * A7 A8 A9 B7 B8 B9 C7 C8 C9
 * D1 D2 D3 E1 E2 E3 F1 F2 F3
 * D4 D5 D6 E4 E5 E6 F4 F5 F6
 * D7 D8 D9 E7 E8 E9 F7 F8 F9
 * G1 G2 G3 H1 H2 H3 I1 I2 I3
 * G4 G5 G6 H4 H5 H6 I4 I5 I6
 * G7 G8 G9 H7 H8 H9 I7 I8 I9 
 * 
 * Store it as rows of sub-blocks
 * A1 A2 A3 A4 A5 A6 A7 A8 A9
 * B1 B2 B3 B4 B5 B6 B7 B8 B9
 * C1 C2 C3 C4 C5 C6 C7 C8 C9
 * D1 D2 D3 D4 D5 D6 D7 D8 D9
 * E1 E2 E3 E4 E5 E6 E7 E8 E9
 * F1 F2 F3 F4 F5 F6 F7 F8 F9
 * G1 G2 G3 G4 G5 G6 G7 G8 G9
 * H1 H2 H3 H4 H5 H6 H7 H8 H9
 * I1 I2 I3 I4 I5 I6 I7 I8 I9 
 * 
 * Initialize the board so that each sub-block contains a permutation of [1..9]
 */
void init(char* fname, int* su, int* mutableIdx, int* mutableCnt, curandState* state){
  int mask[81]; // The bitmap-ish mask of given elements.
  // Read the puzzle.
  char buf[10];
  FILE *fp = fopen(fname, "r");
  // Serialized the board.
  for (int k = 0; k < 3; k++){
    for (int kk=0; kk < 3; kk++){
      fscanf(fp, "%s\n", buf);
      for (int j = 0; j < 3; j++){
        for (int i = 0; i < 3; i++){
          su[index(3*k+j, 3*kk+i)] = buf[i+3*j] - '0';
          mask[index(3*k+j, 3*kk+i)] = (buf[i+3*j] - '0')? 1:0;
        }
      }
    }
  }
  fclose(fp);
  printf("Board loaded.\n");
  // Initialize randon seed.
  time_t t;
  srand((unsigned) time(&t));
  initRandKernel<<<9, 32>>>(state, (unsigned) t);
  gpuErrchk(cudaDeviceSynchronize());

  printf("Original board:\n");
  printBoard(su);
  
  memset(mutableIdx, 0, 81*sizeof(int));
  memset(mutableCnt, 0, 9*sizeof(int));

  printf("Initializing the board...\n");
  initBoard(su, mask, mutableIdx, mutableCnt);
  printf("\tDone.\n");
  _assertInit(su);
  printf("\tChecked.\n");
  
  printBoard(su);
  #ifdef DEBUG
  printf("\n");
  printBoard(mutableIdx);
  #endif
}


// Thread layout 32 * 9
__global__ void solveSukoduKernel(int* su, int* mutableIdx, int* mutableCnt, 
                                  int resolution, int mutation_rate, int accept_rate,
                                  curandState* state, int* boards_best, int* scores_best){
  __shared__ int mirror[81];
  __shared__ int boards[puzzlePb*81];
  __shared__ int scores_arch[puzzlePb];
  __shared__ int scores[puzzlePb];
  __shared__ int argmax[puzzlePb];

  // The index of thread in the block.
  int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
  // The global index of puzzles.
  int block_index = blockIdx.x * puzzlePb + threadIdx.x;
  
  // Copy to shared memory.
  if (thread_index < 81){
    mirror[thread_index] = su[thread_index];
  }
  __syncthreads();
  
  // TODO: Optimize memory access.
  // Further copy.
  for (int i = 0; i < 9; i++){
    boards[81*threadIdx.x+index(threadIdx.y, i)] = mirror[index(threadIdx.y, i)];
  }
  if (thread_index<32){
      scores_arch[thread_index] = scores_best[blockIdx.x];
  }
  __syncthreads();
  int k, x, y;
  int mut;
  
  for (int it = 0; it < resolution; it++){
    // The first warp do the mutation (or not).

    if (thread_index<32){
      scores[thread_index] = 0;
      argmax[thread_index] = thread_index;
      k = curand(state+block_index) % 9;
      mut = curand(state+block_index) % 100 <mutation_rate ? 1 : 0;
      
      if (mut){
        
        x = curand(state+block_index) % mutableCnt[k];
        y = curand(state+block_index) % mutableCnt[k];    
        
        if (x == y){
          y = (y+1) % mutableCnt[k];
        }
        x = mutableIdx[index(k,x)];
        y = mutableIdx[index(k,y)];
        int tmp = boards[81*threadIdx.x + index(k, x)];
        boards[81*threadIdx.x+index(k, x)] = boards[81*threadIdx.x+index(k, y)];
        boards[81*threadIdx.x+index(k, y)] = tmp;
        #ifdef DEBUG
        printf("It %d Thread %d.%d swaped %d, %d (%d) with %d (%d)\n", 
              it, blockIdx.x, thread_index, k, 
              x, boards[81*threadIdx.x+index(k, y)], y, boards[81*threadIdx.x+index(k, x)]);
        #endif
      }
    }
    // TODO: make use of all threads.
    // Compute scores
    //  column
    if (threadIdx.y == 0){
      int subblock_x;
      int subblock_y;
      int sum = 0;
      int loc[9] = {0,0,0,0,0,0,0,0,0};
      for (int i = 0; i < 9; i++){
        subblock_x = i/3;
        subblock_y = i%3;
        for (int i = 0; i < 9; i+=3){
          for (int j = 0; j < 9; j+=3){
            loc[boards[81*threadIdx.x + index(i + subblock_x, j + subblock_y)]-1] = 1;
          }
        }
        sum = 0;
        for (int ii = 0; ii < 9; ii++){
          if (loc[ii])  sum++;
          loc[ii] = 0;
        }
        scores[threadIdx.x] += sum;
      }
      //  row
      for (int i = 0; i < 9; i++){
        subblock_x = (i/3) * 3;
        subblock_y = (i%3) * 3;
        for (int i = 0; i < 3; i++){
          for (int j = 0; j < 3; j++){
            loc[boards[81*threadIdx.x + index(i + subblock_x, j + subblock_y)]-1] = 1;
          }
        }
        sum = 0;
        for (int ii = 0; ii < 9; ii++){
          if (loc[ii])  sum++;
          loc[ii] = 0;
        }
      
        scores[threadIdx.x] += sum;
      }
      if (threadIdx.y == 0){
        if (scores[threadIdx.x] > scores_arch[threadIdx.x] || 
          scores_arch[threadIdx.x] != 162 && curand(state+block_index)%100 < accept_rate){
          scores_arch[threadIdx.x] = scores[threadIdx.x];
        }else{
          // Undo the swap if necessary.
          if (mut){
            int tmp = boards[81*threadIdx.x + index(k, x)];
            boards[81*threadIdx.x+index(k, x)] = boards[81*threadIdx.x+index(k, y)];
            boards[81*threadIdx.x+index(k, y)] = tmp;
          }
        }
      }
    }
  }
  __syncthreads();
  // Reduce
  if (threadIdx.y == 0){
    #ifdef DEBUG
    printf("Thread %d.%d have sum %d\n", blockIdx.x, threadIdx.x, scores_arch[threadIdx.x]);
    #endif
    for (int stride = 16; stride > 0 && threadIdx.x < stride; stride /= 2){
      if (scores_arch[threadIdx.x] < scores_arch[threadIdx.x+ stride]) {
        scores_arch[threadIdx.x] = scores_arch[threadIdx.x+ stride];
        argmax[threadIdx.x] = argmax[threadIdx.x+stride];
      }
    }
    scores_best[blockIdx.x] = scores_arch[0];
  }
  __syncthreads();
  
  // Write back

  if (thread_index < 81){
    boards_best[81*blockIdx.x + thread_index] = boards[81*argmax[0] + thread_index];
  }
  __syncthreads();

  #ifdef DEBUG
  for (int ii = 0 ; ii < NBLOCK; ii++){
    if (blockIdx.x == ii){
      if(thread_index == 0){
        printf("Best score: %d at %d. Block %d\n", scores_arch[0], argmax[0], blockIdx.x);
        for (int i = 0; i<9; i++){
          for (int j=0; j< 9; j++){
            printf("%d ", boards_best[81*blockIdx.x + index(i, j)]);
          }
          printf("\n");
        }
      }
    }__syncthreads();
  }
  #endif
}

__global__ void updateBoardKernel(int* su, int* newSu, int opt){
  su[threadIdx.x] = newSu[threadIdx.x + 81*opt];
  __syncthreads();  
}

void solveSukodu(int* su, int* su_kernel, int* mutableIdx_kernel, int* mutableCnt_kernel, 
                 int iterations, int resolution, int mutation_rate, int accept_rate, curandState *state){
  int score = 0;
  int *boards_best;
  int *scores_best;
  cudaMalloc((void**)&boards_best, NBLOCK*81*sizeof(int));
  cudaMallocManaged((void**)&scores_best, NBLOCK*sizeof(int));
  gpuErrchk(cudaDeviceSynchronize());
  dim3 dimBlock(32, 9); 
  for (int it = resolution; it <= iterations; it+=resolution){
    solveSukoduKernel<<<NBLOCK, dimBlock>>>(
        su_kernel, mutableIdx_kernel, mutableCnt_kernel,
        resolution, mutation_rate, accept_rate, state, boards_best, scores_best);
    gpuErrchk(cudaDeviceSynchronize());

    int opt = 0;
    int s = scores_best[0];
    for (int i = 1; i < NBLOCK; i++){
      if (s < scores_best[i]){
        s = scores_best[i];
        opt = i;
      }
    }
    
    if (s > score || rand()%100 < 100*pow(1-accept_rate/100., NBLOCK)){
      updateBoardKernel<<<1,81>>>(su_kernel, boards_best, opt);
      gpuErrchk(cudaDeviceSynchronize());
      score = s;
      if (score == 162){
        cudaMemcpy(su, su_kernel, 81*sizeof(int), cudaMemcpyDeviceToHost);
        gpuErrchk(cudaDeviceSynchronize());
        printf("Solution found!");
        printf("\n%d/%d\tscore: %d\n", it, iterations, score);
        printBoardReadable(su);
        
        
        return;
      } 
    }
    cudaMemcpy(su, su_kernel, 81*sizeof(int), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaDeviceSynchronize());
    printf("\n%d/%d\tscore: %d at %d\n", it, iterations, score, opt);
    printBoardReadable(su);
  }
  cudaFree(boards_best);
  cudaFree(scores_best);
}

int main(int argc, char** argv){
  int su[81]; // The board.
  // Store the index and the total number
  // of mutable elements of each block
  int mutableIdx[81]; 
  int mutableCnt[9];

  curandState_t *state;
  cudaMalloc((void**) &state, NBLOCK*32 * sizeof(curandState_t));

  printf("Initializing...\n");

  init(argv[1], su,  mutableIdx, mutableCnt, state);

  printf("initialized.\n");
  int mutation_rate = 30;
  int accept_rate = 5;
  int iterations = 1000000;
  int resolution = 1000;

  int *su_kernel;
  int *mutableIdx_kernel;
  int *mutableCnt_kernel;
  
  cudaMalloc((void**)&su_kernel, 81*sizeof(int));
  cudaMalloc((void**)&mutableIdx_kernel, 81*sizeof(int));
  cudaMalloc((void**)&mutableCnt_kernel, 9*sizeof(int));
  

  cudaMemcpy(su_kernel, su, 81*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(mutableIdx_kernel, mutableIdx, 81*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(mutableCnt_kernel, mutableCnt, 9*sizeof(int), cudaMemcpyHostToDevice);

  char fname[100];
  int name_len = strlen(argv[1]);
  for (int i = 0; i < name_len-2; i++)
    fname[i] = argv[1][i];
  strcpy(fname+name_len-2, "out");

  printf("Start solving...\n");
  solveSukodu(su, su_kernel, mutableIdx_kernel, mutableCnt_kernel, 
              iterations, resolution, mutation_rate, accept_rate, state);
  
  FILE *fp = fopen(fname, "w+");
  for (int ii = 0; ii < 3; ii ++){
    for (int jj = 0; jj < 3; jj ++){
      for (int i = 0; i < 3; i ++){
        for (int j = 0; j < 3; j ++){
          fprintf(fp, "%d",su[index(3*ii+i,3*jj+j)]);
        }
      }
      fprintf(fp, "\n");
    }
  }
        fclose(fp);
  cudaFree(su_kernel);
  cudaFree(mutableIdx_kernel);
  cudaFree(mutableCnt_kernel);
  return 0;
}