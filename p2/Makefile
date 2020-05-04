
all: bin/floyd_par1d bin/floyd_seq

bin/floyd_par1d: src/floyd_par1d.cc lib/Graph.cc lib/Graph.h
	mpicxx -I./lib  -O3      src/floyd_par1d.cc lib/Graph.cc -o bin/floyd_par1d

bin/floyd_seq: src/floyd_seq.cc lib/Graph.cc lib/Graph.h
	mpicxx -I./lib  -O3   src/floyd_seq.cc lib/Graph.cc -o bin/floyd_seq

bin/floyd_par2d: src/floyd_par2d.cc lib/Graph.cc lib/Graph.h
	mpicxx -I./lib -O3 src/floyd_par1d.cc lib/Graph.cc -o bin/floyd_par2d

clean:
	rm -f bin/floyd_seq bin/floyd_par1d  *.o.

