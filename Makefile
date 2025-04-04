BASE_EXTENSION=version_simple_cython21_adj_categorical_clean_epsilon
NEW_EXTENSION=CART
CC=gcc
INCLUDE=$(shell ./python_include.py)
FLAGS=-shared -pthread -fPIC -fwrapv -O3 -Wall -Wextra -std=c17 ${INCLUDE}

all: ${BASE_EXTENSION}.so ${NEW_EXTENSION}.so

%.so: %.c
	${CC} ${FLAGS} -o $@ $<

%.c: %.pyx
	cythonize $<

clean:
	rm -f *.so *.c

.PHONY: clean
