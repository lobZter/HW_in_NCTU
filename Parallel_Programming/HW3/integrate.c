#include <stdio.h>
#include <math.h>
#include "mpi.h"

#define PI 3.1415926535

int main(int argc, char **argv)
{
    int my_rank, size;
    long long i, num_intervals;
    double rect_width, area, sum, local_sum, x_middle, start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    //start_time = MPI_Wtime();;

    sscanf(argv[1],"%llu",&num_intervals);

    rect_width = PI / num_intervals;

    local_sum = 0;
    for(i = 1 + my_rank; i < num_intervals + 1; i += size) {

        /* find the middle of the interval on the X-axis. */

        x_middle = (i - 0.5) * rect_width;
        area = sin(x_middle) * rect_width;
        local_sum = local_sum + area;
    }

    MPI_Reduce(&local_sum,&sum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    if(my_rank == 0) {
        printf("The total area is: %f\n", (float)sum);
        //end_time = MPI_Wtime();;
        //printf("time elapsed: %.2lf seconds\n",end_time-start_time);
    }
    MPI_Finalize();

    return 0;
}
