#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include "Graph.h"
#include "mpi.h" 

using namespace std;

//**************************************************************************

int main (int argc, char *argv[]) {

	if (argc != 2) {
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return(-1);
	}
	

	Graph G;
	G.lee(argv[1]);// Read the Graph

	//cout << "EL Grafo de entrada es:"<<endl;
	//G.imprime();
	const int nverts = G.vertices;
	const int niters = nverts;

	const int nverts2 = nverts * nverts;

	int *c_Out_M = new int[nverts2];
	int size = nverts2*sizeof(int);


	int *A = G.Get_Matrix();

	// CPU phase
	double t1 = MPI_Wtime();

	// BUCLE PPAL DEL ALGORITMO
	int in, kn,ik;
	for(int k = 0; k < niters; k++) {
          kn = k * nverts;
	  for(int i=0;i<nverts;i++) {
			in = i * nverts;
	                ik=in+k;
			for(int ij = in; ij < in+nverts; ij++)
	                      {
		                int j=ij-in;
	       			if (i!=j && i!=k && j!=k){		
			 	    A[ij] = min(A[ik] + A[kn+j], A[ij]);
				}
				 
	       }
	   }
	}

  double t2 = MPI_Wtime() - t1;
  G.imprime();
  cout << "Tiempo gastado= " << t2 << endl << endl;

}
