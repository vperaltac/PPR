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

    //***************************************************************************************
    // DISTRIBUCIÓN INICIAL DE MATRIZ DE ENTRADA POR BLOQUES ENTRE LOS PROCESOS
    //***************************************************************************************
    // Broadcast the number of vertices to all processes
    MPI_Bcast(&nverts,1,MPI_INT, 0, MPI_COMM_WORLD);        

    int raiz_P = sqrt(size);
    int tam = nverts/raiz_P;

    // Creo buffer de envío para almacenar los datos empaquetados 
    int *buf_envio = new int[nverts*nverts];

    if(rank == 0){
        // Obtiene matriz local a repartir 
        int *matriz_A = G.Get_Matrix();

        // Defino el tipo bloque cuadrado
        MPI_Type_vector(tam,tam,nverts,MPI_INT,&MPI_BLOQUE);
        MPI_Type_commit(&MPI_BLOQUE);

        // Empaqueta bloque a bloque en el buffer de envío
        for(int i=0, posicion=0; i<size; ++i){
            // Cálculo la posición de comienzo de cada submatriz
            int fila_P = i/raiz_P;
            int columna_P = i%raiz_P;
            int comienzo=(columna_P*tam)+(fila_P*tam*tam*raiz_P);

            MPI_Pack(&matriz_A[comienzo],1,MPI_BLOQUE,buf_envio,sizeof(int)*nverts*nverts,&posicion,MPI_COMM_WORLD);
        }

        // Destruye la matriz local
        free(matriz_A);
    }

    // Creo un buffer de recepción
    int *buf_recep = new int[tam*tam];

    // Distribuimos la matriz entre los procesos
    MPI_Scatter(buf_envio, sizeof(int)*tam*tam, MPI_PACKED, buf_recep, tam*tam, MPI_INT,0,MPI_COMM_WORLD);
    // Cada proceso se queda con una matriz de tamaño tam*tam

    //***************************************************************************************

    //***************************************************************************************
    // ALGORITMO FLOYD PARALELO 2D
    //***************************************************************************************
    // Crear comunicadores comm_fila y comm_columna
    MPI_Comm comm_fila, comm_columna;
    MPI_Comm_split(MPI_COMM_WORLD, rank%raiz_P, rank, &comm_fila);
    MPI_Comm_split(MPI_COMM_WORLD, rank/raiz_P, rank, &comm_columna);

    MPI_Barrier(MPI_COMM_WORLD);
	double t1 = MPI_Wtime();

    // valores para fila y columna inicial de la matriz GLOBAL para cada proceso
    int global_i_start = (rank/raiz_P)*tam;
    int global_j_start = (rank%raiz_P)*tam;

    int *subfila_k = new int[tam];
    int *subcolumna_k = new int [tam];

    for(int k=0;k<nverts;++k){
        // Proceso que hará broadcast (RANK DE COMUNICADOR DE FILA Y COLUMNA) 
        // (en nverts = 4, sólo 0 y 1 valores posibles)
        int proceso = k/tam;

        // si k está entre la primera fila de la submatriz y la última fila
        // quiere decir que la subfila k forma parte de la submatriz actual
        if(k >= global_i_start && k < global_i_start+tam){
            // módulo tam para mantener el valor dentro del tamaño de las submatrices
            // *tam porque es una matriz "vector", cada fila está en aumentos de 2 posiciones
            memcpy(subfila_k, &buf_recep[(k%tam)*tam],sizeof(int)*tam);
        }

        // si k está entre la primera columna de la submatriz y la última columna
        // quiere decir que la subcolumna k forma parte de la submatriz actual
        if(k >= global_j_start && k < global_j_start+tam){
            for(int i=0;i<tam;++i){
                subcolumna_k[i] = buf_recep[i*tam+(k%tam)];
            }
        }

        // el proceso k/tam hace broadcast de su fila y columna
        MPI_Bcast(subfila_k, tam, MPI_INT, proceso, comm_fila);
        MPI_Bcast(subcolumna_k, tam, MPI_INT, proceso, comm_columna);

        for(int i=0;i<tam;++i){
            for(int j=0;j<tam;++j){
                if(global_i_start+i != global_j_start+j && global_i_start+i != k && global_j_start+j != k){
                    int ikj = subcolumna_k[i] + subfila_k[j];
                    buf_recep[i*tam+j] = min(ikj,buf_recep[i*tam+j]);
                }
            }
        }
    }

    free(subfila_k);
    free(subcolumna_k);

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime() - t1;
    //***************************************************************************************


    //***************************************************************************************
    // RECOGIDA DE RESULTADO DE CADA PROCESO EN EL PROCESO 0 Y DESEMPAQUETADO
    //***************************************************************************************
    MPI_Gather(buf_recep,tam*tam,MPI_INT,buf_envio,sizeof(int)*tam*tam,MPI_PACKED,0,MPI_COMM_WORLD);

    int *buf_unpack = new int[nverts*nverts];
    if (rank == 0){
        for (int i=0, posicion=0; i<size; ++i){
            // Cálculo la posición de comienzo de cada submatriz
            int fila_P = i/raiz_P;
            int columna_P = i%raiz_P;
            int comienzo=(columna_P*tam)+(fila_P*tam*tam*raiz_P);

            MPI_Unpack(buf_envio,sizeof(int)*nverts*nverts,&posicion,&buf_unpack[comienzo],1,MPI_BLOQUE,MPI_COMM_WORLD);
        }

        cout << "--------------------------------" << endl;
        for(int i=0;i<nverts;i++){
            cout << "A["<<i << ",*]= ";

            for(int j=0;j<nverts;j++){
                if (buf_unpack[i*nverts+j]==INF) 
                    cout << "INF";
                else  
                    cout << buf_unpack[i*nverts+j];
                
                if (j<nverts-1) 
                    cout << ",";
                else
                    cout << endl;
            }
        }

        // Libera el tipo bloque
        MPI_Type_free(&MPI_BLOQUE);
    }


    if(rank==0)
        cout << "Tiempo gastado= " << t2 << endl << endl;

    free(buf_envio);
    free(buf_recep);
    free(buf_unpack);

    MPI::Finalize();
    //***************************************************************************************
}