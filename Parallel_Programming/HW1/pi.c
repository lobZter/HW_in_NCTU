#include <stdio.h>
#include <stdlib.h>     /* rand(), srand(), RAND_MAX */
#include <time.h>       /* time(),*/
#include <pthread.h>
#include <sys/sysinfo.h>

unsigned long long num_in_circle;
pthread_mutex_t mutex;

void* Thread_sum(void* thread_num_toss) {

    unsigned toss, seed = time(NULL);
    double x, y;
    long long thread_num_in_circle = 0;

    for ( toss = 0; toss < (unsigned long long)thread_num_toss; toss ++) {
        x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        if ( x * x + y * y <= 1.0)
        thread_num_in_circle ++;
    }

    pthread_mutex_lock(&mutex);
    num_in_circle += thread_num_in_circle;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(int argc, char* argv[]) {

    if(argc!=2) {
        printf("./main [num_toss]\n");
        return 0;
    }

    pthread_t* thread_handles;
    long thread_idx; /* Use long in case of a 64-bit system */
    long thread_count = get_nprocs();
    unsigned long long num_toss = strtoll(argv[1], NULL, 10);
    unsigned long long thread_num_toss = num_toss / thread_count;
    num_in_circle = 0;


    thread_handles = (pthread_t*) malloc (thread_count*sizeof(pthread_t));
    pthread_mutex_init(&mutex, NULL);

    for (thread_idx = 0; thread_idx < thread_count; thread_idx++) {
        if(thread_idx == thread_count - 1)
            thread_num_toss += num_toss % thread_count;
        pthread_create(&thread_handles[thread_idx], NULL, Thread_sum, (void*)thread_num_toss);
    }

    for (thread_idx = 0; thread_idx < thread_count; thread_idx++)
        pthread_join(thread_handles[thread_idx], NULL);

    pthread_mutex_destroy(&mutex);
    free(thread_handles);

    printf("%.15f\n", 4.0 * (double)num_in_circle / (double)num_toss);
    return 0;

}
