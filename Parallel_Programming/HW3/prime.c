#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

int isprime(int n) {
  long long int i,squareroot;
  if (n>10) {
    squareroot = (int) sqrt(n);
    for (i=3; i<=squareroot; i=i+2)
      if ((n%i)==0)
        return 0;
    return 1;
  }
  else
    return 0;
}

int main(int argc, char *argv[])
{
    int pc, local_pc,               /* prime counter */
        foundone, local_foundone,    /* most recent prime found */
        my_rank,
        size;
    long long int n, limit;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    //start_time = MPI_Wtime();;

    sscanf(argv[1],"%llu",&limit);

    if(my_rank == 0) {
        printf("Starting. Numbers to be scanned= %lld\n",limit);
    }

    local_pc = 0;

    for (n = 11 + 2 * my_rank; n <= limit; n += 2 * size) {
        if (isprime(n)) {
            local_pc++;
            local_foundone = n;
        }
    }

    MPI_Reduce(&local_pc,&pc,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&local_foundone,&foundone,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);

    if(my_rank == 0) {
        pc+=4;     /* Assume (2,3,5,7) are counted here */
        printf("Done. Largest prime is %d Total primes %d\n",foundone,pc);
        //end_time = MPI_Wtime();;
        //printf("time elapsed: %.2lf seconds\n",end_time-start_time);
    }

    MPI_Finalize();

    return 0;
}
