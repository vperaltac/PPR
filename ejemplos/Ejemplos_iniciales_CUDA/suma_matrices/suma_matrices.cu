#include "stdio.h"

const int N=40;

__global__ void MatAdd( float *A, float *B, float *C, int N)
{
int j = blockIdx.x * blockDim.x + threadIdx.x;  // Compute row index
int i = blockIdx.y * blockDim.y + threadIdx.y;  // Compute column index
int index=i*N+j; // Compute global 1D index
if (i < N && j < N)
	C[index] = A[index] + B[index]; // Compute C element
}

int main()
{
int i;
const int NN=N*N;
/* pointers to host memory */
/* Allocate arrays A, B and C on host*/
float * A = (float*) malloc(NN*sizeof(float));
float * B = (float*) malloc(NN*sizeof(float));
float * C = (float*) malloc(NN*sizeof(float));

/* pointers to device memory */
float *A_d, *B_d, *C_d;
/* Allocate arrays a_d, b_d and c_d on device*/
cudaMalloc ((void **) &A_d, sizeof(float)*NN);
cudaMalloc ((void **) &B_d, sizeof(float)*NN);
cudaMalloc ((void **) &C_d, sizeof(float)*NN);

/* Initialize arrays a and b */
for (i=0; i<NN;i++)
{
  A[i]= (float) i;
  B[i]= -(float) i;
}


/* Copy data from host memory to device memory */
cudaMemcpy(A_d, A, sizeof(float)*NN, cudaMemcpyHostToDevice);
cudaMemcpy(B_d, B, sizeof(float)*NN, cudaMemcpyHostToDevice);

/* Compute the execution configuration */
dim3 threadsPerBlock (16, 16);
dim3 numBlocks( ceil ((float)(N)/threadsPerBlock.x), ceil ((float)(N)/threadsPerBlock.y) );
MatAdd <<<numBlocks, threadsPerBlock>>> (A_d, B_d, C_d, N);


/* Copy data from deveice memory to host memory */
cudaMemcpy(C, C_d, sizeof(float)*NN, cudaMemcpyDeviceToHost);

/* Print c */
for (i=0; i<NN;i++)
  printf(" c[%d]=%f\n",i,C[i]);

/* Free the memory */
free(A); free(B); free(C);
cudaFree(A_d); cudaFree(B_d);cudaFree(C_d);

}
