#include "stdio.h"
#include "iostream"

#define  Bsize_addition 256
#define Bsize_minimum   128

const int N=64;

__global__ void reduceSum(float *d_V, int N){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = ((i<N) ? d_V[i] : 0.0f);
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
    int i;

    dim3 dimBlock(Bsize_addition);
    dim3 dimGrid ( ceil((float(N)/(float)dimBlock.x)) );

    dim3 threadsPerBlock(Bsize_minimum, 1);
    dim3 numBlocks( ceil ((float)(N)/threadsPerBlock.x), 1);

    float * vmin;
    vmin = (float*) malloc(N*sizeof(float));

    /* Initialize arrays a and b */
    for (i=0; i<N;i++){
        vmin[i]= (float) i;
    }

    float *vmin_d; 
    cudaMalloc ((void **) &vmin_d, sizeof(float)*N);
    int smemSize = threadsPerBlock.x*sizeof(float);

    // Kernel launch to compute Minimum Vector
    reduceSum<<<numBlocks, threadsPerBlock, smemSize>>>(vmin_d, N);

    /* Copy data from device memory to host memory */
    cudaMemcpy(vmin, vmin_d, numBlocks.x*sizeof(float),cudaMemcpyDeviceToHost);

    /* Print c */
    for (i=0; i<N;i++)
        printf(" vmin[%d]=%f\n",i,vmin[i]);

    /* Free the memory */
    free(vmin);
    cudaFree(vmin_d);
}