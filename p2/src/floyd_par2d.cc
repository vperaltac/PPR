#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include "Graph.h"
#include "mpi.h" 

using namespace std;

// N = nverts
// P = size

int main (int argc, char *argv[]) {    
    MPI::Init(argc,argv);

        if (argc != 2){
        cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
        return(-1);
	}
	
	Graph G;
    int nverts,rank,size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Datatype MPI_BLOQUE;

    // Read the Graph in process 0
	if(rank==0){
        G.lee(argv[1]);
        nverts = G.vertices;
        //G.imprime();
    }

    // Broadcast the number of vertices to all processes
    MPI_Bcast(&nverts,1,MPI_INT, 0, MPI_COMM_WORLD);        

    int raiz_P = sqrt(size);
    int tam = nverts/raiz_P;

    // Creo buffer de envío para almacenar los datos empaquetados 
    int *buf_envio = new int[nverts*nverts];

    if(rank == 0){
        // Obtiene matriz local a repartir 
        int** matriz_A = new int*[nverts];
        for (int i = 0; i < nverts; ++i)
            matriz_A[i] = new int[nverts];

        // Defino el tipo bloque cuadrado
        MPI_Type_vector(tam,tam,nverts,MPI_INT,&MPI_BLOQUE);

        // Creo el nuevo tipo
        MPI_Type_commit(&MPI_BLOQUE);

        // Empaqueta bloque a bloque en el buffer de envío
        for(int i=0, posicion=0; i<size; ++i){
            // Cálculo la posición de comienzo de cada submatriz
            int fila_P = i/raiz_P;
            int columna_P = i%raiz_P;
            int comienzo=(columna_P*tam)+(fila_P*tam*tam*raiz_P);

            MPI_Pack(matriz_A[comienzo],1,MPI_BLOQUE,buf_envio,sizeof(int)*nverts*nverts,&posicion,MPI_COMM_WORLD);
        }

        // Destruye la matriz local
        free(matriz_A);
        // Libera el tipo bloque
        MPI_Type_free(&MPI_BLOQUE);
    }

    // Creo un buffer de recepción
    int *buf_recep = new int[tam*tam];

    // Distribuimos la matriz entre los procesos
    MPI_Scatter(buf_envio, sizeof(int)*tam*tam, MPI_PACKED, buf_recep, tam*tam, MPI_INT,0,MPI_COMM_WORLD);
    MPI::Finalize();
}