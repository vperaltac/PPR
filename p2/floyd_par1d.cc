#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include "Graph.h"
#include "mpi.h" 

using namespace std;

//**************************************************************************

int main (int argc, char *argv[]) {
    MPI::Init(argc, argv); 

    if (argc != 2){
        cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
        return(-1);
	}
	
	Graph G;
    int nverts,rank,size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
        
    // Read the Graph in process 0
	if(rank==0){
        G.lee(argv[1]);
        nverts = G.vertices;
        //G.imprime();
    }   

    // Broadcast the number of vertices to all processes
    MPI_Bcast(&nverts,1,MPI_INT, 0, MPI_COMM_WORLD);

    const int bsize1d= nverts/size;
    const int bsize2d= bsize1d*nverts;
  
    int *A = G.Get_Matrix();

	//Process 0 scatters blocks of matrix A 	
	int * local_A= new int[bsize2d];
	MPI_Scatter(A,bsize2d,MPI_INT,local_A,bsize2d,MPI_INT, 0, MPI_COMM_WORLD);
	
    // Computing the local first and last row indexes
    const int local_i_start=0;
    const int local_i_end= bsize1d;
            // Computing the local first row index for each process
    const int global_i_start=rank*bsize1d;
            
    //Vector storing kth row of the global matrix A 
    int *fila_k= new int[nverts]; 
    int *tmp;
        
	double t1 = MPI_Wtime();
 
	// MAIN LOOP OF THE ALGORITHM
	int inj, in, kn;
	for(int k = 0; k < nverts; k++){    
        // Broadcat global row k to all processes
        int row_k_process=k/bsize1d;
        if (rank==row_k_process){
            const int local_k=k%bsize1d;
            tmp=fila_k;
            fila_k=&(local_A[local_k*nverts]);
        }

        MPI_Bcast(fila_k,nverts,MPI_INT, row_k_process, MPI_COMM_WORLD);

        // Update local martix local_A
	    for(int i=local_i_start;i<local_i_end;i++) {
		    const int global_i = global_i_start+i,
            i_nverts=i*nverts, 
            local_ik=i_nverts+k;
                
            for(int j = 0; j < nverts; j++)
                if (global_i!=j && global_i!=k && j!=k){
                    int local_ij=i_nverts + j;
                    int suma_ikj=local_A[local_ik] + fila_k[j];
		            local_A[local_ij] = min(local_A[local_ij],suma_ikj);
                }
        }

        if (rank==row_k_process) fila_k=tmp;
	}

    MPI_Gather(local_A,bsize2d,MPI_INT,A,bsize2d,MPI_INT, 0, MPI_COMM_WORLD);
    double t2 = MPI_Wtime() - t1;
    
    if (rank==0){
        G.imprime(); 
        cout << "Tiempo gastado= " << t2 << endl << endl;
    }

    MPI::Finalize();
}
