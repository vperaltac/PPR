#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <sys/time.h>

namespace cg = cooperative_groups;
using namespace std;

//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}
//**************************************************************************


// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

template <class T>
__global__ void reduceMax(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
		if(sdata[tid] < sdata[tid+s])
			sdata[tid] = sdata[tid+s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduceSum(float *d_V, float*d_out,int N){
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
        d_out[blockIdx.x] = sdata[0];
        
}

__global__ void transformacion(float *A, float *B, float *C, float *D){
    int tid    = threadIdx.x;
    int i      = blockIdx.x * blockDim.x + threadIdx.x;
    int istart = blockIdx.x*blockDim.x;
    int iend   = istart + blockDim.x;

    C[i] = 0.0;

    for(int s=istart; s<iend;s++){
        float a = A[s]*i;

        if((int)ceil(a) % 2 == 0)
            C[i]+= a + B[s];
        else
            C[i]+= a - B[s];
    }

    if (tid == 0)
        D[blockIdx.x] = C[i];
}

__global__ void transformacion_s(float *A, float *B, float *C, float *D){
    // sólo se puede utilizar 1 vector en memoria compartida
    // por tanto es necesario almacenar A,B y C en el mismo vector
    extern __shared__ float sdata[];
    float *sdata_A = sdata;
    float *sdata_B = sdata + blockDim.x;
    float *sdata_C = sdata + blockDim.x*2;

    int tid    = threadIdx.x;
    int i      = blockIdx.x * blockDim.x + threadIdx.x;

    sdata_A[tid] = A[i];
    sdata_B[tid] = B[i];
	sdata_C[tid] = 0.0;
    __syncthreads();

    for(int s=0; s<blockDim.x;s++){
        float a = sdata_A[s]*i;

        if((int)ceil(a) % 2 == 0)
            sdata_C[tid]+= a + sdata_B[s];
        else
            sdata_C[tid]+= a - sdata_B[s];
    }

    C[i] = sdata_C[tid];

    if (tid == 0)
        D[blockIdx.x] = sdata_C[tid];
}


int main(int argc, char *argv[]){
    int Bsize, NBlocks;
    if (argc != 3){ 
        cout << "Uso: transformacion Num_bloques Tam_bloque  "<<endl;
        return(0);
    }
    else{
        NBlocks = atoi(argv[1]);
        Bsize= atoi(argv[2]);
    }

    const int N=Bsize*NBlocks;
    //* pointers to host memory */

    float *A, *B, *C,*D;

    //* Allocate arrays a, b and c on host*/
    A = new float[N];
    B = new float[N];
    C = new float[N];
    D = new float[NBlocks];
    float mx; // maximum of C

    //* Initialize arrays A and B */
    for (int i=0; i<N;i++){
        A[i]= (float) (1  -(i%100)*0.001);
        B[i]= (float) (0.5+(i%10) *0.1  );    
    }

    float *d_A;
    float *d_B;
    float *d_C;
    float *d_D;
    float *d_max;

    cudaMalloc ((void **) &d_A,   sizeof(float)*N);
    cudaMalloc ((void ** )&d_B,   sizeof(float)*N);
    cudaMalloc ((void **) &d_C,   sizeof(float)*N);
    cudaMalloc ((void ** )&d_D,   sizeof(float)*NBlocks);
    cudaMalloc ((void ** )&d_max, sizeof(float)*NBlocks);

    cudaMemcpy(d_A,A,sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,C,sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D,D,sizeof(float)*NBlocks, cudaMemcpyHostToDevice);

	// GPU phase
	double  t1 = cpuSecond();

    int smemSize = Bsize*sizeof(float);
    transformacion<<<NBlocks,Bsize>>>(d_A,d_B,d_C,d_D);
    reduceSum<<<NBlocks,Bsize,smemSize>>>(d_C,d_D,N);
    reduceMax<float><<<NBlocks,Bsize,smemSize>>>(d_C,d_max,N);


    float *res_C   = new float [N];
    float *res_D   = new float [NBlocks];
    float *res_max = new float [NBlocks];
    cudaMemcpy(res_C,d_C,sizeof(float)*N,cudaMemcpyDeviceToHost);
    cudaMemcpy(res_D,d_D,sizeof(float)*NBlocks,cudaMemcpyDeviceToHost);
    cudaMemcpy(res_max,d_max,sizeof(float)*NBlocks,cudaMemcpyDeviceToHost);

    // última reducción en CPU
    float maxGPU = 0.0;
    for(int i=0;i<NBlocks;i++){
        if(res_max[i] > maxGPU)
            maxGPU = res_max[i];
    }
    cout << "El valor máximo con memoria global es: " << maxGPU << endl;


	double Tgpu = cpuSecond()-t1;

    //***************************************************************************************************

    float *d_C2;
    float *d_D2;
    float *d_max2;

    cudaMalloc ((void **) &d_C2,   sizeof(float)*N);
    cudaMalloc ((void ** )&d_D2,   sizeof(float)*NBlocks);
    cudaMalloc ((void ** )&d_max2, sizeof(float)*NBlocks);

    cudaMemcpy(d_C2,C,sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D2,D,sizeof(float)*NBlocks, cudaMemcpyHostToDevice);

	double t3 = cpuSecond();


    int smemSize2 = Bsize*3*sizeof(float);
    transformacion_s<<<NBlocks,Bsize,smemSize2>>>(d_A,d_B,d_C2,d_D2);
    reduceSum<<<NBlocks,Bsize,smemSize>>>(d_C2,d_D2,N);
    reduceMax<float><<<NBlocks,Bsize,smemSize>>>(d_C2,d_max2,N);


    float *res_C2   = new float [N];
    float *res_D2   = new float [NBlocks];
    float *res_max2 = new float [NBlocks];
    cudaMemcpy(res_C2,d_C2,sizeof(float)*N,cudaMemcpyDeviceToHost);
    cudaMemcpy(res_D2,d_D2,sizeof(float)*NBlocks,cudaMemcpyDeviceToHost);
    cudaMemcpy(res_max2,d_max2,sizeof(float)*NBlocks,cudaMemcpyDeviceToHost);

    // última reducción en CPU
    float maxGPU2 = 0.0;
    for(int i=0;i<NBlocks;i++){
        if(res_max2[i] > maxGPU2)
            maxGPU2 = res_max2[i];
    }
    cout << "El valor máximo con memoria compartida es: " << maxGPU2 << endl;

    double t4 = cpuSecond() - t3;
    //***************************************************************************************************
    // Time measurement  
	// CPU phase
	t1 = cpuSecond();

    // Compute C[i], d[K] and mx
    for (int k=0; k<NBlocks;k++){ 
        int istart=k*Bsize;
        int iend  =istart+Bsize;
        D[k]=0.0;

        for (int i=istart; i<iend;i++){
            C[i]=0.0;

            for (int j=istart; j<iend;j++){ 
                float a=A[j]*i;

                if ((int)ceil(a) % 2 ==0)
                    C[i]+= a + B[j];
                else
                    C[i]+= a - B[j];
            }

            D[k]+=C[i];
            mx = (i==1) ? C[0] : max(C[i],mx);
        }
    }

	double t2 = cpuSecond() - t1;

    cout<<"................................."<<endl<<"El valor máximo en C es:  "<<mx<<endl;
    cout<<endl<<"N="<<N<<"= "<<Bsize<<"*"<<NBlocks<<"  ........"<< endl;  

    cout << "Tiempo gastado CPU:                        "<<t2<<endl; 
	cout << "Tiempo gastado GPU con memoria global:     " << Tgpu << endl;
	cout << "Tiempo gastado GPU con memoria compartida: " << t4 << endl;
	cout << "Ganancia para GPU1D: " << t2 / Tgpu << endl;
	cout << "Ganancia para GPU2D: " << t2 / t4   << endl;

    //* Free the memory */
    delete(A); 
    delete(B); 
    delete(C);
    delete(D);
    delete [] res_C;
    delete [] res_D;
    delete [] res_C2;
    delete [] res_D2;
    delete [] res_max;
    delete [] res_max2;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_max);

    cudaFree(d_C2);
    cudaFree(d_D2);
    cudaFree(d_max2);
}