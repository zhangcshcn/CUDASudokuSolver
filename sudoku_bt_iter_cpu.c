/*  Author: Chen Zhang, NYU Courant
 *  
 *  This is a sukodu solver using randomized backtracking.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define index(x, y) (9 * (x) + (y))

int ATTEMPT = 0;  // Track the number of attempts made. 
int su[81];
int mask[81];

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

struct Node{
  int x,y,c,r;
  struct Node *prev, *next;
};

struct Node *head = NULL, *end = NULL;

void pushNode(struct Node *new_node){
  ATTEMPT ++;
  if (!head){
    head = new_node;
  } 
  new_node->next = NULL;
  new_node->prev = end;
  end = new_node;
  su[index(new_node->x, new_node->y)] = new_node->c;
}

struct Node* popNode(){
  if (!head){
    return NULL;
  } else {
    struct Node* ret = end;
    end = end->prev;
    if (end)  end->next = NULL;
    ret->prev = NULL;
    su[index(ret->x, ret->y)] = 0;
    return ret;
  }
}

int getNext(int *x, int *y){
  do {
    // Find the next undecided position.
    // If there is none, the sukodu has been solved. Return 1.
    if (*y == 8){
      if (*x < 8) (*x)++, (*y)=0;
      else return 0;
    } else (*y)++;
  } while (mask[index(*x, *y)]);
  return 1;
}

int solveSukoduIter(){
  // Find the next undecided position.
  // If there is none, the sukodu has been solved. Return 1.
  int x = 0, y = -1;
  getNext(&x, &y);
  
  struct Node* prb = (struct Node*)malloc(sizeof(struct Node));
  prb->x = x, prb->y = y, prb->r = 1, prb->c = rand()%9+1;
  pushNode(prb);

  while (head){
    if (_judge(end->x, end->y)){
      // If the last attempt is potentially correct,
      // try the next location.
      if (getNext(&x, &y)){
        prb = (struct Node*)malloc(sizeof(struct Node));
        prb->x = x, prb->y = y, prb->r = 1, prb->c = rand()%9+1;
        pushNode(prb);
      } else return 1;
    } else {
      // If the last attempt was wrong, 
      // undo the last attempt.
      prb = popNode();
      if (prb->r < 9){
        // If there are still changes at the last location, try again.
        prb->r ++, prb->c = prb->c%9+1;
        x = prb->x, y = prb->y;
        pushNode(prb);
      } else {
        // All attempts at the location fails.
        // The environment must have been wrong.
        // Retry previous steps.
        do {
          free(prb);
          prb = popNode();
        } while (prb && prb->r>=9);
        if (prb){
           prb->r ++, prb->c = prb->c%9+1;
           x = prb->x, y = prb->y;
           pushNode(prb);
        } else {
          return 0;
        }
      }
    }
  }
  return 0;
}
void init(char* fname){
  // Read the puzzle.
  char buf[100];
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

void printSu(){
  for (int i = 0; i<9; i++){
    for (int j=0; j< 9; j++)
      printf("%d", su[index(i, j)]);
    printf("\n");
  }
}

int assert(){
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
 
  init(argv[1]);
  
  printf("Original puzzle:\n");
  printSu();

  int solved = solveSukoduIter();

  if (solved){
    printf("Solution Found in %d attempts!\n", ATTEMPT);
    printSu();
    if (assert()) 
      printf("Solution is correct!\n");
    else 
      printf("Solution wrong...\n");
  } else {
    printf("No solution round after %d attempts...\n", ATTEMPT);
  }

  return 0;
}