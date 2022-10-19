
#for qemu
CC = aarch64-linux-gnu-gcc
CFLAGS = -static -march=armv8-a+simd

LDLIBS = -lm
 
.PHONY: all
all: main

valgrind: 
	valgrind -v --leak-check=full --track-origins=yes --trace-children=yes --track-fds=yes --show-leak-kinds=all ./main

main: main_c.o create_images_c.o loadImage_c.o loadNet_c.o executeNet_c.o conv_c.o maxpool_c.o flatten_c.o dense_c.o tests_c.o prints_c.o frees_c.o measure_time_c.o loadNumpyImg_c.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

main_c.o: main.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

create_images_c.o: create_images.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

loadImage_c.o: loadImage.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

loadNet_c.o: loadNet.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

conv_c.o: conv.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

maxpool_c.o: maxpool.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

flatten_c.o: flatten.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

dense_c.o: dense.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

executeNet_c.o: executeNet.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

tests_c.o: tests.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

prints_c.o: prints.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

frees_c.o: frees.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

measure_time_c.o: measure_time.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

loadNumpyImg_c.o: loadNumpyImg.c main.h
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o
	rm -f main
