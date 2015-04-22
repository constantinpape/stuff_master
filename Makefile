SHELL = /bin/sh


BOOST = /home/constantin/Work/programs/boost_1_58_0
CC = g++
CDEBUG = -g3 -Wall
CCFLAGS = $(CDEBUG) -O -std=c++0x
ALL_CFLAGS = -I $(BOOST) $(CFLAGS)

# defining paths
prefix := .
bindir := $(prefix)/bin
libdir := $(prefix)/lib
srcdir := $(prefix)/c++
pydir := $(prefix)/pybindings

# C++ source files
_SRC1 := $(srcdir)/utilities.cp
_SRC2 := $(pydir)/pybindings.cpp
SRC := $(_SRC1) $(_SRC2)

# collect object files
_OBJ := utilities.o
OBJ := $(patsubst %,$(libdir)/%,$(_OBJ))

# target build: build all object files
build: $(SRC) $(OBJ)

-include $(OBJ:.o=.d)

$(libdir)/%.o: $(srcdir)/%.cpp $(srcdir)/%.h
	$(CC) $(ALL_CFLAGS) -c -o $@ $<
	$(CC) $(ALL_CFLAGS) -MM $< > $(libdir)/$*.d

