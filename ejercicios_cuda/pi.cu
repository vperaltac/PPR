#include "stdio.h"
#include "iostream"
#include <time.h>
#include <assert.h>


double calcular_pi_CPU(double step, int num_steps){
    double x,sum = 0.0;
    double pi;

    step = 1.0/(double) num_steps;
    for(int i=1; i<= num_steps; i++){
        x = (i-0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }

    pi = step * sum;

    return pi;
}

__global__ void calcular_pi_GPU(double *sum, double step, int num_steps,int threads_per_block, int num_blocks){
    double x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=tid;i<=num_steps;i+=threads_per_block*num_blocks){
        x = (i+0.5)*step;
        sum[tid] += 4.0/(1.0+x*x);
    }
}

int main(int argc, char ** argv){
    if(argc != 2){
        std::cerr << "Falta argumento: steps" << std::endl;
        exit(1);
    }

    int num_steps = atoi(argv[1]);
    double step   = 1.0/(double) num_steps;
    int num_blocks        = 12;
    int threads_per_block = 256;
    

    double *h_sum;
    double *d_sum;

    int smemSize = threads_per_block*num_blocks*sizeof(double);
    h_sum = (double*) malloc(smemSize);
    cudaMalloc ((void **) &d_sum, smemSize);

    for (int i=0; i<threads_per_block*num_blocks;i++)
        h_sum[i]= 0;
    cudaMemcpy(d_sum,h_sum,smemSize, cudaMemcpyHostToDevice);

    calcular_pi_GPU<<<num_blocks,threads_per_block>>>(d_sum,step,num_steps,threads_per_block,num_blocks);

    cudaMemcpy(h_sum, d_sum, smemSize,cudaMemcpyDeviceToHost);

    double pi=0;
    for(int i=0; i<threads_per_block*num_blocks; i++)
        pi += h_sum[i];
    pi *= step;

    printf("resultado: %f\n",pi);

    free(h_sum);
    cudaFree(d_sum);
}