BASE_EXTENSION=version_simple_cython21_adj_categorical_clean_epsilon
EXTENSIONS=${BASE_EXTENSION}.so CART.so loss.so dataset.so
CC=gcc
INCLUDE=$(shell ./python_include.py)
BASE_FLAGS=-shared -pthread -fPIC -fwrapv -Wall -Wextra -std=c17 ${INCLUDE}
FLAGS=${BASE_FLAGS} -O3  # For release
# FLAGS=${BASE_FLAGS} -g # -fsanitize=address  # For debug

all: ${EXTENSIONS}

%.so: %.c
	${CC} ${FLAGS} -o $@ $<

%.c: %.pyx
	cythonize $<

CART.so: _CART.h

CART.c: CART.pxd

loss.so: _loss.h

loss.c: loss.pxd

dataset.so: _dataset.h

dataset.c: dataset.pxd

clean:
	rm -f *.so *.c

.PHONY: clean

.SECONDARY: CART.c loss.c dataset.c
