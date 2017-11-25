/**********************************************************************
 * DESCRIPTION:
 *   Serial Concurrent Wave Equation - C Version
 *   This program implements the concurrent wave equation
 *********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265

int nsteps,                 	/* number of time steps */
    tpoints, 	     		/* total points along string */
    rcode;                  	/* generic return code */
int block_size = 1024;
float  values[MAXPOINTS+2]; 	/* values at time t */
float *values_d;


/**********************************************************************
 *	Checks input values from parameters
 *********************************************************************/
void check_param(void)
{
   char tchar[20];

   /* check number of points, number of iterations */
   while ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS)) {
      printf("Enter number of points along vibrating string [%d-%d]: "
           ,MINPOINTS, MAXPOINTS);
      scanf("%s", tchar);
      tpoints = atoi(tchar);
      if ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS))
         printf("Invalid. Please enter value between %d and %d\n", 
                 MINPOINTS, MAXPOINTS);
   }
   while ((nsteps < 1) || (nsteps > MAXSTEPS)) {
      printf("Enter number of time steps [1-%d]: ", MAXSTEPS);
      scanf("%s", tchar);
      nsteps = atoi(tchar);
      if ((nsteps < 1) || (nsteps > MAXSTEPS))
         printf("Invalid. Please enter value between 1 and %d\n", MAXSTEPS);
   }

   printf("Using points = %d, steps = %d\n", tpoints, nsteps);

}

__global__ void parallel(float *values_d, int tpoints,  int nsteps, int block_size)
{
   int i, k;
   float x, fac, tmp;
   float dtime, c, dx, tau, sqtau;
   float value, newval, oldval;

   // do math
   dtime = 0.3;
   c = 1.0;
   dx = 1.0;
   tau = (c * dtime / dx);
   sqtau = tau * tau;
   
   // init
   fac = 2.0 * PI;
   k = 1 + threadIdx.x + blockIdx.x * block_size;
   tmp = tpoints - 1;
   x = (k - 1) / tmp;
   value = sin (fac * x);  
   oldval = value;

   if(k <= tpoints) {
      for (i = 1; i<= nsteps; i++) {
         if ((k == 1) || (k  == tpoints))
            newval = 0.0;
         else
            newval = (2.0 * value) - oldval + (sqtau *  (-2.0) * value);
         oldval = value;
         value = newval;
      }
      values_d[k] = value;
   }

}

/**********************************************************************
 *     Print final results
 *********************************************************************/
void printfinal()
{
   int i;

   for (i = 1; i <= tpoints; i++) {
      printf("%6.4f ", values[i]);
      if (i%10 == 0)
         printf("\n");
   }
}

/**********************************************************************
 *	Main program
 *********************************************************************/
int main(int argc, char *argv[])
{
   sscanf(argv[1],"%d",&tpoints);
   sscanf(argv[2],"%d",&nsteps);
   check_param();

   int size = (tpoints + 1) * sizeof(float);
   cudaMalloc((void**)&values_d, size);

   printf("Initializing points on the line...\n");
   printf("Updating all points for all time steps...\n");
   int block_num = (tpoints % block_size == 0) ? (tpoints / block_size) : (tpoints / block_size + 1);
   parallel<<<block_num, block_size>>>(values_d, tpoints, nsteps, block_size);

   cudaMemcpy(values, values_d, size, cudaMemcpyDeviceToHost);
   cudaFree(values_d);

   printf("Printing final results...\n");
   printfinal();
   printf("\nDone.\n\n");
   
   return 0;
}