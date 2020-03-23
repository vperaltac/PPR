#include "stdio.h"
#include "iostream"

const int   N=1000;

#define  Bsize_addition 256
#define Bsize_minimum   128


using namespace std;

//**************************************************************************
// Vector addition kernel
__global__ void add_arrays_gpu( float *in1, float *in2, float *out, int N)
{
int idx=blockIdx.x*blockDim.x+threadIdx.x;
if (idx<N) 
     out[idx]=in1[idx]+in2[idx];
}
//**************************************************************************



//**************************************************************************
// Vector minimum  kernel

__global__ void reduceMin(float * V_in, float * V_out, const int N) {
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = ((i < N) ? V_in[i] : 100000000.0f);
	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s >>= 1){
	  if (tid < s) 
		if(sdata[tid] > sdata[tid+s]) 
                    sdata[tid] = sdata[tid+s];		
	  __syncthreads();
	}
	if (tid == 0) 
           V_out[blockIdx.x] = sdata[0];
}




//**************************************************************************
int main()
//**************************************************************************

{
/* pointers to host memory */
float *a, *b, *c;
/* pointers to device memory */
float *a_d, *b_d, *c_d;
int i;

/* Allocate arrays a, b and c on host*/
a = (float*) malloc(N*sizeof(float));
b = (float*) malloc(N*sizeof(float));
c = (float*) malloc(N*sizeof(float));

/* Allocate arrays a_d, b_d and c_d on device*/
cudaMalloc ((void **) &a_d, sizeof(float)*N);
cudaMalloc ((void **) &b_d, sizeof(float)*N);
cudaMalloc ((void **) &c_d, sizeof(float)*N);

/* Initialize arrays a and b */
for (i=0; i<N;i++)
{
a[i]= (float) (i%3+1);
b[i]= (float) (i%5-2);
}


/* Copy data from host memory to device memory */
cudaMemcpy(a_d, a, sizeof(float)*N, cudaMemcpyHostToDevice);
cudaMemcpy(b_d, b, sizeof(float)*N, cudaMemcpyHostToDevice);

/* Compute the execution configuration */
dim3 dimBlock(Bsize_addition);
dim3 dimGrid ( ceil((float(N)/(float)dimBlock.x)) );

/* Add arrays a and b, store result in c */
add_arrays_gpu<<< dimGrid, dimBlock >>>(a_d, b_d, c_d, N);


// c_d Minimum computation on GPU
dim3 threadsPerBlock(Bsize_minimum, 1);
dim3 numBlocks( ceil ((float)(N)/threadsPerBlock.x), 1);

// Minimum vector on CPU
float * vmin;
vmin = (float*) malloc(numBlocks.x*sizeof(float));

// Minimum vector  to be computed on GPU
float *vmin_d; 
cudaMalloc ((void **) &vmin_d, sizeof(float)*numBlocks.x);

int smemSize = threadsPerBlock.x*sizeof(float);

// Kernel launch to compute Minimum Vector
reduceMin<<<numBlocks, threadsPerBlock, smemSize>>>(c_d,vmin_d, N);


/* Copy data from device memory to host memory */
cudaMemcpy(vmin, vmin_d, numBlocks.x*sizeof(float),cudaMemcpyDeviceToHost);
cudaMemcpy(c, c_d, sizeof(float)*N,cudaMemcpyDeviceToHost);

// Compute minimum on CPU
float min_cpu=1000000.0f;
for (i=0; i<N;i++){
  //ut<<"c["<<i<<"]="<<c[i]<<endl;
  min_cpu=min(min_cpu, c[i]);
  }



// Perform final reduction in CPU
float min_gpu = 10000000.0f;
for (int i=0; i<numBlocks.x; i++) 
{
  min_gpu =min(min_gpu,vmin[i]); printf(" vmin[%d]=%f\n",i,vmin[i]);
}

  
cout<<" Min on GPU ="<<min_gpu<<"  Min on CPU="<< min_cpu<<endl;


/* Free the memory */
free(a); free(b); free(c);free(vmin);
cudaFree(a_d); cudaFree(b_d);cudaFree(c_d);cudaFree(vmin_d);
}
