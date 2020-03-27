#include "stdio.h"
#include "iostream"
#include <time.h>
#include <assert.h>

const int N= 1 << 30;

__global__ void reduceSum_v2(float *d_V, int N){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    
	sdata[tid] = ((i < N) ? d_V[i] + d_V[i+blockDim.x] : 0.f);
    __syncthreads();

    for (int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s)
            sdata[tid] += sdata[tid + s];
        
        __syncthreads();
    }

    if (tid == 0)
        d_V[blockIdx.x] = sdata[0];
        
}


__global__ void reduceSum_v1(float *d_V, int N){
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
    float *h_V;
    float *d_V;

    h_V = (float*) malloc(N*sizeof(float));
    cudaMalloc ((void **) &d_V, sizeof(float)*N);
    cudaMalloc ((void ** )&d_V, sizeof(float)*N);

    for (int i=0; i<N;i++)
        h_V[i]= static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    cudaMemcpy(d_V,h_V,sizeof(float)*N, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(128,1);
    dim3 numBlocks( ceil ((float)(N)/threadsPerBlock.x), 1);
    int smemSize = threadsPerBlock.x*sizeof(float);

    // Take initial time
    double inicio_v1=clock();

    reduceSum_v1<<<numBlocks,threadsPerBlock,smemSize>>>(d_V,N);
    cudaMemcpy(h_V, d_V, numBlocks.x*sizeof(float),cudaMemcpyDeviceToHost);

    double fin_v1=clock();
    double tiempo_v1 = (fin_v1-inicio_v1)/CLOCKS_PER_SEC;

    float sum_v1 = 0.0f;
    for(int i=0;i<numBlocks.x;i++)
        sum_v1 += h_V[i];

    double inicio_v2=clock();   

    reduceSum_v2<<<numBlocks,threadsPerBlock,smemSize>>>(d_V,N);
    cudaMemcpy(h_V, d_V, numBlocks.x*sizeof(float),cudaMemcpyDeviceToHost);

    float sum_v2 = 0.0f;
    for(int i=0;i<numBlocks.x;i++)
        sum_v2 += h_V[i];


    double fin_v2=clock();
    double tiempo_v2 = (fin_v2-inicio_v2)/CLOCKS_PER_SEC;


    assert(sum_v1 == sum_v2);
    std::cout << "T1: " << tiempo_v1 << " secs." << std::endl;
    std::cout << "T2: " << tiempo_v2 << " secs." << std::endl;
    std::cout<< "Speedup (T_CPU/T_GPU)= "<<tiempo_v1/tiempo_v2<<std::endl;

    free(h_V);
    cudaFree(d_V);
}