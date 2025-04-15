BASE_EXTENSION=version_simple_cython21_adj_categorical_clean_epsilon
NEW_EXTENSION=CART
CC=gcc
INCLUDE=$(shell ./python_include.py)
FLAGS=-shared -pthread -fPIC -fwrapv -O3 -Wall -Wextra -std=c17 ${INCLUDE}

all: ${BASE_EXTENSION}.so ${NEW_EXTENSION}.so loss.so

%.so: %.c
	${CC} ${FLAGS} -o $@ $<

%.c: %.pyx
	cythonize $<

CART.so: _CART.h

loss.so: _loss.h

loss.c: loss.pxd

clean:
	rm -f *.so *.c

.PHONY: clean

.SECONDARY: CART.c loss.c
