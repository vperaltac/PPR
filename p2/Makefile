
all: floyd_par1d floyd_seq

floyd_par1d: floyd_par1d.cc Graph.cc Graph.h
	mpicxx -I./includes  -O3      floyd_par1d.cc Graph.cc -o floyd_par1d

floyd_seq: floyd_seq.cc Graph.cc Graph.h
	mpicxx -I./includes  -O3   floyd_seq.cc Graph.cc -o floyd_seq

clean:
	rm -f floyd_seq floyd_par1d  *.o.

