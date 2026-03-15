# 测试程序 Makefile

CC = g++
CFLAGS = -O2 -Wall -std=c++11 -I../include -DTESTING

# 源文件目录
SRC_DIR = ../src

.PHONY: all clean test_parser

all: test_parser

# 解析模块单元测试
test_parser: test_parser.cpp $(SRC_DIR)/geometry.cu $(SRC_DIR)/material.cu $(SRC_DIR)/io.cu ../include/types.h
	@echo "Compiling parser tests..."
	$(CC) $(CFLAGS) -x c++ \
		test_parser.cpp \
		$(SRC_DIR)/geometry.cu \
		$(SRC_DIR)/material.cu \
		$(SRC_DIR)/io.cu \
		-o test_parser -lm
	@echo "Running parser tests..."
	./test_parser

# 清理
clean:
	rm -f test_parser
	rm -f *.o
