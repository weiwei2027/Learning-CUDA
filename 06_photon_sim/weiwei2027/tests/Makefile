# 测试 Makefile - 仅保留解析器测试
CC = g++
CFLAGS = -O2 -Wall -std=c++11 -I../include

.PHONY: all clean test

all: test

test: test_parser
	./test_parser

test_parser: test_parser.cpp ../src/utils.cpp ../include/utils.h ../include/types.h
	$(CC) $(CFLAGS) test_parser.cpp ../src/utils.cpp -o $@ -lm

clean:
	rm -f test_parser *.o
