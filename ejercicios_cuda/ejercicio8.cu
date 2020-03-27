#include "stdio.h"
#include "iostream"

const int N=32;

__global__ void reduceSum(float *d_V, int N){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
	sdata[tid] = ((i < N) ? d_V[i] : 0.f);
    __syncthreads();

    for (int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s)
            sdata[tid] += sdata[tid + s];
        
        __syncthreads();
    }

    if (tid == 0)
        d_V[blockIdx.x] = sdata[0];
        
}

int main(){
    float *a;
    float *a_d;
    int i;

    a = (float*) malloc(N*sizeof(float));
    cudaMalloc ((void **) &a_d, sizeof(float)*N);

    for (i=0; i<N;i++)
        a[i]= (float) i;

    cudaMemcpy(a_d, a, sizeof(float)*N, cudaMemcpyHostToDevice);

    reduceSum<<< 1, N,sizeof(float)*N>>>(a_d,N);

    cudaMemcpy(a, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

    printf("suma con reducción: %f\n",a[0]);

    free(a);
    cudaFree(a_d);
}