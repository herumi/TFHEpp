CFLAGS=-DNDEBUG -O2 -g -I ../../include
CFLAGS+=-I ../../../cybozulib/include
LIB_C=fft_processor_spqlios.cpp spqlios-fft-impl-avx512.cpp
LIB_S=spqlios-fft-avx512.s spqlios-ifft-avx512.s
LIB=spqlios.a
SRC=test.cpp $(LIB_C)

LIB_OBJ=$(LIB_C:.cpp=.o) $(LIB_S:.s=.o) fft_avx512.o

OBJ=$(SRC:.cpp=.o)
DEP=$(SRC:.cpp=.d)

fft_avx512.S: gen.py
	python3 gen.py -m gas > $@

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CFLAGS) -MMD -MP -MF $(@:.o=.d)

%.o: %.s
	$(CXX) -c -o $@ $< $(CFLAGS)

%.o: %.S
	$(CXX) -c -o $@ $< $(CFLAGS)

%.exe: %.o $(LIB)
	$(CXX) -o $@ $< $(LDFLAGS) $(LIB)

$(LIB): $(LIB_OBJ)
	$(AR) $(ARFLAGS) $@ $(LIB_OBJ)

test: test.exe
	./test.exe > a.txt
	diff org.txt a.txt

-include $(DEP)

clean:
	$(RM) -rf *.exe *.o *.d $(LIB) *.S

.PHONY: clean

.SECONDARY: $(OBJ)
