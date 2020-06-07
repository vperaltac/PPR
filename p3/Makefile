
bin/bbseq: obj/bbseq.o obj/libbb.o
	mpicxx -I./include -O3 obj/bbseq.o obj/libbb.o -o bin/bbseq
	
obj/bbseq.o: src/bbseq.cc
	mpicxx -I./include -O3 -c src/bbseq.cc -o obj/bbseq.o


obj/libbb.o: src/libbb.cc include/libbb.h
	mpicxx -I./include -O3 -c src/libbb.cc -o obj/libbb.o  


clean:
	/bin/rm -f obj/*.o bin/bbseq