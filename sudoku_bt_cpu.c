/*  Author: Chen Zhang, NYU Courant
 *  
 *  This is a sukodu solver using randomized backtracking.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define index(x, y) (9 * (x) + y)

int ATTEMPT = 0;  // Track the number of attempts made. 


int _judge(int *su, int x, int y){
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

int solveSukodu(int *su, int *mask, int x, int y){
  do {
    // Find the next undecided position.
    // If there is none, the sukodu has been solved. Return 1.
    if (y == 8){
      if (x < 8) x++, y=0;
      else return 1;
    } else y++;
  } while (mask[index(x, y)]);
  // Try every possible value at (x, y).
  // If a value is potentially legal, recursively solve the function.
  // If all poosible values at (x, y) under the current setting doesn't
  // lead to a solution, then, we have a dead end. Return 0.
  int c = rand() % 9;
  for (int r = 1; r <= 9; r ++){
    ATTEMPT++;
    c = c%9+1;
    su[index(x, y)] = c;
    // su[index(x, y)] = r;
    if (_judge(su, x, y))
      if (solveSukodu(su, mask, x, y))
        return 1;
  }
  su[index(x, y)] = 0;
  return 0;
}

void init(int *su, int *mask, char* fname){
  // Read the puzzle.
  char buf[20];
  FILE *fp = fopen(fname, "r");
  for (int i = 0; i < 9; i ++){
    fscanf(fp, "%s\n", buf);
    for (int j = 0; j < 9; j++){
      su[index(i, j)] = buf[j]-'0';
      if(buf[j]-'0')  mask[index(i, j)] = 1;
      else  mask[index(i, j)] = 0;
    }
  }
  fclose(fp);
  // Initialize randon seed.
  time_t t;
  srand((unsigned) time(&t));
}

void print(int *su){
  for (int i = 0; i<9; i++){
    for (int j=0; j< 9; j++)
      printf("%d", su[index(i, j)]);
    printf("\n");
  }
}

int assert(int *su){
  int sum = 0;
  for (int i = 0; i < 9; i++){
    sum = 0;
    for (int j = 0; j < 9; j++){
      sum += su[index(i, j)];
    }
    if (sum != 45)  return 0;
  }

  for (int j = 0; j < 9; j++){
    sum = 0;
    for (int i = 0; i < 9; i++){
      sum += su[index(i, j)];
    }
    if (sum != 45)  return 0;
  }

  for (int ii = 0; ii < 9; ii += 3){
    for (int jj = 0; jj < 9; jj += 3){
      sum = 0;
      for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
          sum += su[index(i, j)];
        }
      }
      if (sum != 45)  return 0;
    }  
  }

  return 1;
}

int main(int argc, char** argv){
  
  int su[81];
  int mask[81];
  
  init(su, mask, argv[1]);
  
  printf("Original puzzle:\n");
  print(su);

  int solved = solveSukodu(su, mask, 0, -1);

  if (solved){
    printf("Solution Found in %d attempts!\n", ATTEMPT);
    print(su);
    if (assert(su)) 
      printf("Solution is correct!\n");
    else 
      printf("Solution wrong...\n");
  } else {
    printf("No solution round after %d attempts...\n", ATTEMPT);
  }

  return 0;
}