## Sudoku solver in CUDA

Author: Chen Zhang (cz1389)

### Stochastic approach to Sudoku

There are some literatures about solving Sudoku in a parallelized 
fashion. The fancy names of the approachs include Genetic Algorithms, 
Particle Swarm Optimization, Simulated Annealing, and so on. 
Generally, these algorithms are developed with the same philosophy 
underneath. Below is an algorithm with the idea is a stochastic process.

1. Initialize the board such that each 3*3 grid contains a permutation 
   of 1..9.
1. In each step, with a probability, randomly choose two ungiven points 
   in a 3*3 grid and swap them.
  2. Compute the score by summing up the number of unique numbers in each 
     row and column. (A correct solution would have 162.)
  2. If the new score is higher than the old score, accept the new board.
  2. If the new score is lower, accept the change with a propability, so 
     that the search won't be stuck in a local optimum.
1. Repeat until a solution is found.

### Parallelization

Now that we are implementing on CUDA, we need to explore the part that can be parallelized. 

#### Launching a number of copies of Sudoku boards
  
  For simplicity, 32 copies are assigned into each block. Only the first wrap of 32 threads 
  are responsible for making decisions on the change and whether to accept the change. Due
  to the power of randomization and parallelization combined, the GPU version gets a solution
  in less iterations than CPU on average, though, it is not guaranteed that the algorithm 
  finally converges. 

  After a few iterations, the blocks return the best result to the host, can a tournament
  takes place to elect the initial board for the next round. 

#### Load and store

  Loading and storing is a classic parallelizable operation. For each of the 32 boards in 
  a block, we can use multiple threads to help with data transfer.

#### Computing scores

  The score is defined as the summation of the unique numbers in each row and column.
  this is very parallelzable. (But hard to coalesce.) We can have multiple threads computing 
  the score for each row and columns, and sum them up to get the total score.

### A bit details

  The block layout is 32*9. We have each thread is resposible of the threadIdx.x-th board.
  only the threadIdx.y == 0 are responsible for the decision making. 
  The blockDim.y is chosen to be 9 such that the 81 elements in a board can be evently distributed.

### How to run

  > make
  > ./sudokusolver filename.in

  It generates filename.sol

### Reference
1. Wang, Zhiwen, Toshiyuki Yasuda, and Kazuhiro Ohkura. "An evolutionary approach to sudoku puzzles with filtered mutations." In Evolutionary Computation (CEC), 2015 IEEE Congress on, pp. 1732-1737. IEEE, 2015.
1. Sato, Yuji, Naohiro Hasegawa, and Mikiko Sato. "Acceleration of genetic algorithms for sudoku solution on many-core processors." In Massively Parallel Evolutionary Computation on GPGPUs, pp. 421-444. Springer Berlin Heidelberg, 2013.
1. Monk, Jason, Kevin Hanselman, Robert King, and Raymond Flagg. "Solving sudoku using particle swarm optimization on cuda." In Proceedings of the International Conference on Parallel and Distributed Processing Techniques and Applications (PDPTA), p. 1. The Steering Committee of The World Congress in Computer Science, Computer Engineering and Applied Computing (WorldComp), 2012.