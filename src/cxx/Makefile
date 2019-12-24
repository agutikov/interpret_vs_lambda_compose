

CXX_OPT ?= 2
CXX_FLAGS = -O${CXX_OPT} -std=c++17

CXX = g++

APPS = test bench

DEPS = Makefile parser.hh

.PHONY: all
all: ${APPS}

${APPS}: % : %.cxx ${DEPS}
	${CXX} ${CXX_FLAGS} $< -o $@

.PHONY: clean
clean:
	rm -f ${APPS}

.PHONY: rebuild
rebuild: | clean all
