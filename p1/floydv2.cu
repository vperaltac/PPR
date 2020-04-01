#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include "Graph.h"

// CUDA runtime
//#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#define blocksize 256

using namespace std;

//**************************************************************************
double cpuSecond()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}
//**************************************************************************

__global__ void floyd_kernel2D(int *M, const int nverts, const int k){
	__shared__ int sdata[blocksize];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	sdata[i] = M[i*nverts+j];
	int index=i*nverts+j;
	printf("%i \n",sdata[i]);

	if(i < nverts && j < nverts){
		int Mij = M[index];
		
		if(i!=j && i!=k && j!=k){
			int Mikj = M[i*nverts+k] + M[k*nverts+j];
			Mij = (Mij > Mikj) ? Mikj : Mij;
			M[index] = Mij;
		}
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
/* 
	for(int i=0;i<nverts;i++){
		for(int j=0;j<nverts;j++){
			cout << c_Out_M[j+i*nverts] << " ";
		}
		cout << endl;
	} */

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

	dim3 threadsPerBlock (32,32);
	dim3 numBlocks( ceil((float)(nverts)/threadsPerBlock.x),
					ceil((float)(nverts)/threadsPerBlock.y));

	int smemSize = nverts*sizeof(int);

	for(int k = 0; k < niters; k++) {
		floyd_kernel2D<<<numBlocks, threadsPerBlock,smemSize >>>(d_In_M_2D, nverts, k);
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
	cout << "Ganancia= " << t2 / Tgpu << endl;


	for(int i = 0; i < nverts; i++)
		for(int j = 0;j < nverts; j++)
		if (abs(c_Out_M[i*nverts+j] - G.arista(i,j)) > 0)
			cout << "Error (" << i << "," << j << ")   " << c_Out_M[i*nverts+j] << "..." << G.arista(i,j) << endl;

	for(int i = 0; i < nverts; i++)
		for(int j = 0;j < nverts; j++)
			if (abs(c_Out_M_2D[i*nverts+j] - G.arista(i,j)) > 0)
				cout << "Error (" << i << "," << j << ")   " << c_Out_M_2D[i*nverts+j] << "..." << G.arista(i,j) << endl;
	
	//**************************************************************************************************************************
}