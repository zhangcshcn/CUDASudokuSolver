/*  Author: Chen Zhang, NYU Courant
 *  
 *  This is a sukodu solver using stochastic methods.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define index(x, y) (9 * (x) + (y))

int su[81]; // The board.
int mask[81]; // The bitmap-ish mask of given elements.
// Store the index and the total number
// of mutable elements of each block
int mutableIdx[81]; 
int mutableCnt[9];


void printBoard(int *board){
  for (int i = 0; i<9; i++){
    for (int j=0; j< 9; j++)
      printf("%d", board[index(i, j)]);
    printf("\n");
  }
}

void printBoardReadable(){
  for (int ii = 0; ii < 3; ii ++){
    for (int jj = 0; jj < 3; jj ++){
      for (int i = 0; i < 3; i ++){
        for (int j = 0; j < 3; j ++){
          printf("%d",su[index(3*ii+i,3*jj+j)]);
        }
      }
      printf("\n");
    }
  }
}

int _judge(int x, int y){
  int gx = x/3 * 3;
  int gy = y/3 * 3;
  int it = su[index(x, y)];

  // In the small cell. 
  for (int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      if (su[index(gx+i, gy+j)] == it)
        if (gx+i != x && gy+j != y)  return 0;
  // The row.
  for(int i = 0; i < 9; i++)
    if (su[index(i, y)] == it)
      if (i != x) return 0;
  // The column.
  for(int j = 0; j < 9; j++)
    if (su[index(x, j)] == it)
      if (j != y) return 0;
  
  return 1;
}

// Fill the blocks such that each block contains a permutation of [1..9]
void initBoard(){
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
    
    for (int i = 0, k = 0; i < 9; i++){
      if (!rec[i]){
        rec[k] = i+1;
        k++;
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

int _assertInit(){
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
void init(char* fname){
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
  // Initialize randon seed.
  time_t t;
  srand((unsigned) time(&t));
  memset(mutableIdx, 0, 81*sizeof(int));
  memset(mutableCnt, 0, 9*sizeof(int));
  printf("Original board:\n");
  printBoard(su);

  //printf("\nReadable:\n");
  //printBoardReadable();
  
  initBoard();
  _assertInit();
  printf("\n");
  printBoard(mutableIdx);
  for (int i = 0; i < 9; i++)
  printf("%d ", mutableCnt[i]);
  printf("\n");
  printf("\nInitialized:\n");
  printBoard(su);
}

int scoreSudoku(){
  int score = 0;
  // Score the rows
  for (int ii = 0; ii < 3; ii++){
    for (int jj = 0; jj < 3; jj++){
      int nums[9];
      memset(nums, 0, 9*sizeof(int));
      for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
          nums[su[index(3*ii+i, 3*jj+j)]-1] ++;
        }
      }
      for (int k = 0; k < 9; k++){
        if (nums[k])  score += 1;
      }
    }
  }
  // Score the columns
  for (int ii = 0; ii < 3; ii++){
    for (int jj = 0; jj < 3; jj++){
      int nums[9];
      memset(nums, 0, 9*sizeof(int));
      for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j ++){
          nums[su[index(ii+3*i,jj+3*j)]-1] ++;
        }
      }
      for (int k = 0; k < 9; k++){
        if (nums[k])  score += 1;
      }
    }
  }
  return score;
}

int _assertSudoku(){
  if (scoreSudoku() == 162) return 1;
  else return 0;
}

// Randomly swap two elements in the k-th block.
void swapIdx(int k, int *x, int *y){
  if (mutableCnt[k] <= 1) return;
  int j = mutableCnt[k];
  int idx_x = rand()%j;
  *x = mutableIdx[index(k,idx_x)];
  mutableIdx[index(k,idx_x)] = mutableIdx[index(k,j-1)];
  mutableIdx[index(k,j-1)] = *x;
  *y = mutableIdx[index(k,rand()%(j-1))];
}

void swap(int k, int x, int y){
  int tmp = su[index(k, x)];
  su[index(k, x)] = su[index(k, y)];
  su[index(k, y)] = tmp;
}

void undoSwap(int *idxs){
  for (int k = 0; k < 9; k++){
    if (idxs[2*k] != -1){
      swap(k, idxs[2*k], idxs[2*k+1]);
    }
  }
}

int main(int argc, char** argv){
 
  init(argv[1]);
  int score = 0;
  int idxs[18];
  int mutation_rate = 30;
  int accept_rate = 1;
  int iterations = 10000000;
  for (int it = 1; it <= iterations; it ++){
    for (int k = 0; k < 18; k ++) idxs[k] = -1;
      if (rand() % 100 <= mutation_rate) {
        int k = rand()%9;
        int x, y;
        swapIdx(k, &x, &y);
        idxs[2*k] = x, idxs[2*k+1] = y;
        swap(k, x, y);
      }
    int new_score = scoreSudoku();
    if (new_score > score || rand() % 100 <= accept_rate){
      score = new_score;
    } else {
      undoSwap(idxs);
    }
    if (_assertSudoku())  break;
    if ((it % 1000000 )== 0){
      printf("%d/%d, score: %d\n", it, iterations, score);
      printBoard(su);
    }
  }
  printf("\n%d\n", scoreSudoku());
  printBoardReadable(su);

  return 0;
}