SHELL = /bin/sh


#BOOST = /home/constantin/Work/inst/boost/include
CC = g++
CDEBUG = -g3 -Wall
CCFLAGS = $(CDEBUG) -O -std=c++11
ALL_CFLAGS = -I $(BOOST) $(CCFLAGS)
INCLUDE = /home/constantin/Work/my_stuff/c++/
INCLUDE_PYTHON = /usr/include/python2.7

# defining paths
prefix := .
bindir := /home/constantin/Work/programs/inst/bin
libdir := /home/constantin/Work/programs/inst/lib
pylibdir := /home/constantin/Work/programs/inst/lib/python2.7/site-packages

srcdir := $(prefix)/c++
pydir := $(prefix)/pybindings

testdir := $(prefix)/test

# C++ source files
_SRC1 := $(srcdir)/utilities.cpp
_SRC2 := $(pydir)/pybindings.cpp
SRC := $(_SRC1) $(_SRC2)

# collect object files
_OBJ := utilities.o
OBJ := $(patsubst %,$(libdir)/%,$(_OBJ))

#targets

install: build $(bindir)/my_test.exe $(pylibdir)/mypybindings.so

_installdirs: mkinstalldirs
	@./mkinstalldirs $(bindir) $(libdir)

# target build: build all object files
build: $(SRC) $(OBJ)

$(libdir)/utilities.o: $(srcdir)/utilities.cpp
	$(CC) $(ALL_CFLAGS) -c -o $(libdir)/utilities.o $(srcdir)/utilities.cpp 

pybindings.o: $(pydir)/pybindings.cpp
	$(CC) $(ALL_CFLGAS) -I $(INCLUDE_PYTHON) -I $(INCLUDE) -fpic -c $(pydir)/pybindings.cpp 

$(pylibdir)/mypybindings.so: pybindings.o
	$(CC) -shared -Wl,--export-dynamic pybindings.o -lpython2.7 -lboost_python -L/usr/lib/oython2.7/config -fpic -o $(pylibdir)/mypybindings.so
	rm pybindings.o

$(bindir)/my_test.exe: $(testdir)/test.cpp
	$(CC) $(ALL_CFLAGS) -I $(INCLUDE) -o $(bindir)/my_test.exe $(libdir)/utilities.o $(testdir)/test.cpp


#$(libdir)/%.o: $(srcdir)/%.cpp $(srcdir)/%.h
#	$(CC) $(ALL_CFLAGS) -c -o $@ $<
#	$(CC) $(ALL_CFLAGS) -MM $< > $(libdir)/$*.d

