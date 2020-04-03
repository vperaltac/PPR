#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include "Graph.h"
#include <iomanip>      // std::setw

#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;


// CUDA runtime
//#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#define blocksize    1024
#define blocksize_2D 32

using namespace std;

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


//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}
//**************************************************************************

template <class T>
__global__ void reduce2(T *g_idata, T *g_odata, unsigned int n) {
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


/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
template <class T>
__global__ void reduce3(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;

  if (i + blockDim.x < n) mySum += g_idata[i + blockDim.x];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = mySum = mySum + sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = mySum;
}


__global__ void floyd_kernel2D(int *M, const int nverts, const int k){
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < nverts && j < nverts && (i!=j && i!=k && j!=k)){
		int index=i*nverts+j;
		int Mij = M[index];
		
		int Mikj = M[i*nverts+k] + M[k*nverts+j];
		Mij = (Mij > Mikj) ? Mikj : Mij;
		M[index] = Mij;
	}
}


__global__ void floyd_kernel(int * M, const int nverts, const int k) {
	int ij = threadIdx.x + blockDim.x * blockIdx.x;

  	if (ij < nverts * nverts) {
		int Mij = M[ij];
		int i= ij / nverts;
		int j= ij - i * nverts;

		if (i != j && i != k && j != k) {
			int Mikj = M[i * nverts + k] + M[k * nverts + j];
			Mij = (Mij > Mikj) ? Mikj : Mij;
			M[ij] = Mij;
		}
  	}
}

int main (int argc, char *argv[]) {
	if (argc != 2) {
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return(-1);
	}
	
	//Get GPU information
	int devID;
	cudaDeviceProp props;
	cudaError_t err;
	err = cudaGetDevice(&devID);
	if(err != cudaSuccess) {
		cout << "ERRORRR" << endl;
	}


	cudaGetDeviceProperties(&props, devID);
	printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

	Graph G;
	G.lee(argv[1]);// Read the Graph

	//cout << "EL Grafo de entrada es:"<<endl;
	//G.imprime();
	const int nverts = G.vertices;
	const int niters = nverts;

	const int nverts2 = nverts * nverts;

	int *c_Out_M = new int[nverts2];
	int size = nverts2*sizeof(int);
	int * d_In_M = NULL;

	err = cudaMalloc((void **) &d_In_M, size);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	int *A = G.Get_Matrix();

	//**************************************************************************************************************************
	// GPU phase
	double  t1 = cpuSecond();

	err = cudaMemcpy(d_In_M, A, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

 	for(int k = 0; k < niters; k++) {
		//printf("CUDA kernel launch \n");
		int threadsPerBlock = blocksize;
		int blocksPerGrid = (nverts2 + threadsPerBlock - 1) / threadsPerBlock;

		floyd_kernel<<<blocksPerGrid,threadsPerBlock >>>(d_In_M, nverts, k);
		err = cudaGetLastError();

		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
			exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	double Tgpu = cpuSecond()-t1;

	cout << "Tiempo gastado GPU= " << Tgpu << endl << endl;
	//**************************************************************************************************************************

	//**************************************************************************************************************************
	int *c_Out_M_2D = new int[nverts2];
	int size2D = nverts2*sizeof(int);
	int * d_In_M_2D = NULL;

	err = cudaMalloc((void **) &d_In_M_2D, size2D);
	if (err != cudaSuccess) {
		cout << "ERROR RESERVA" << endl;
	}

	int *A_2D = G.Get_Matrix();
	double t3 = cpuSecond();

	err = cudaMemcpy(d_In_M_2D, A_2D, size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		cout << "ERROR COPIA A GPU" << endl;
	}

	dim3 threadsPerBlock (blocksize_2D,blocksize_2D);
	dim3 numBlocks( ceil((float)(nverts)/threadsPerBlock.x),
					ceil((float)(nverts)/threadsPerBlock.y));

	for(int k = 0; k < niters; k++) {
		floyd_kernel2D<<<numBlocks, threadsPerBlock >>>(d_In_M_2D, nverts, k);
		err = cudaGetLastError();

		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to launch kernel! ERROR= %d\n",err);
			exit(EXIT_FAILURE);
		}
	}

	cudaMemcpy(c_Out_M_2D, d_In_M_2D, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	double t4 = cpuSecond() - t3;
	cout << "Tiempo gastado GPU2D= " << t4 << endl << endl;

	for(int i=0;i<nverts;i++){
		for(int j=0;j<nverts;j++){
			if(c_Out_M_2D[j+i*nverts] != c_Out_M[j+i*nverts]){
				cerr << "RESULTADOS DIFERENTES" << endl;
				exit(EXIT_FAILURE);
			}
		}
	}
	//**************************************************************************************************************************



	//**************************************************************************************************************************
	// CPU phase
	t1 = cpuSecond();

	// BUCLE PPAL DEL ALGORITMO
	int inj, in, kn;
	for(int k = 0; k < niters; k++) {
		kn = k * nverts;
	  	for(int i=0;i<nverts;i++) {
			in = i * nverts;
			for(int j = 0; j < nverts; j++){
				if (i!=j && i!=k && j!=k){
					inj = in + j;
					A[inj] = min(A[in+k] + A[kn+j], A[inj]);
	       		}
			}
	   	}
	}

	double t2 = cpuSecond() - t1;
	cout << "Tiempo gastado CPU= " << t2 << endl << endl;
	cout << "Ganancia para GPU1D: " << t2 / Tgpu << endl;
	cout << "Ganancia para GPU2D: " << t2 / t4   << endl;

	for(int i = 0; i < nverts; i++)
		for(int j = 0;j < nverts; j++)
		if (abs(c_Out_M[i*nverts+j] - G.arista(i,j)) > 0)
			cout << "Error (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl;

	for(int i = 0; i < nverts; i++)
		for(int j = 0;j < nverts; j++)
			if (abs(c_Out_M_2D[i*nverts+j] - G.arista(i,j)) > 0)
				cout << "Error (" << i << "," << j << ")   " << c_Out_M_2D[i*nverts+j] << "..." << G.arista(i,j) << endl;
	
	//**************************************************************************************************************************


	// Reducción para calcular la longitud del camino de mayor longitud dentro de los caminos más cortos encontrados.
	int smemSize = 256 * sizeof(int);
	int * d_odata = NULL;
	int * d_idata = NULL;
	cudaMalloc((void **) &d_idata, size);
	cudaMalloc((void **) &d_odata, size);
	
	cudaMemcpy(d_idata,c_Out_M_2D,size, cudaMemcpyHostToDevice);

	reduce2<int><<<64,256,smemSize>>>(d_idata,d_odata,size);

	int *res = new int[nverts2];
	cudaMemcpy(res, d_odata, size, cudaMemcpyDeviceToHost);

	printf("Camino de mayor longitud: %d\n",res[0]);

	delete [] res; delete [] c_Out_M; delete [] c_Out_M_2D;
	cudaFree(d_In_M); cudaFree(d_In_M_2D); cudaFree(d_idata); cudaFree(d_odata);
}