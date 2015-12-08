# @todo Add openmp / parallel flags for linux/unix platform.
CC = g++
CCFLAGS = -g -ansi -Wall -W -ansi -pedantic -I ../include/ -I ./unittest++
SED = sed
MV = mv
RM = rm

.SUFFIXES: .o .cpp

lib = UnitTest++.a
test = TestUnitTest++

src = unittest++/AssertException.cpp \
	unittest++/Test.cpp \
	unittest++/Checks.cpp \
	unittest++/TestRunner.cpp \
	unittest++/TestResults.cpp \
	unittest++/TestReporter.cpp \
	unittest++/TestReporterStdout.cpp \
	unittest++/ReportAssert.cpp \
	unittest++/TestList.cpp \
	unittest++/TimeConstraint.cpp \
	unittest++/MemoryOutStream.cpp \
	unittest++/Posix/SignalTranslator.cpp \
	unittest++/Posix/TimeHelpers.cpp

test_src = tests/main.cpp \
	tests/ProximityList.cpp \
	test/ThreadStorageTest.cpp

objects = $(patsubst %.cpp, %.o, $(src))
test_objects = $(patsubst %.cpp, %.o, $(test_src))
dependencies = $(subst .o,.d,$(objects))
test_dependencies = $(subst .o,.d,$(test_objects))

define make-depend
  $(CC) $(CCFLAGS) -M $1 | \
  $(SED) -e 's,\($(notdir $2)\) *:,$(dir $2)\1: ,' > $3.tmp
  $(SED) -e 's/#.*//' \
      -e 's/^[^:]*: *//' \
      -e 's/ *\\$$//' \
      -e '/^$$/ d' \
      -e 's/$$/ :/' $3.tmp >> $3.tmp
  $(MV) $3.tmp $3
endef


all: $(test)


$(lib): $(objects) 
	@echo Creating $(lib) library...
	@ar cr $(lib) $(objects)
    
$(test): $(lib) $(test_objects)
	@echo Linking $(test)...
	@$(CC) -o $(test) $(test_objects) $(lib)
	@echo Running unit tests...
	@./$(test)

clean:
	-@$(RM) $(objects) $(test_objects) $(dependencies) $(test_dependencies) $(test) $(lib) 2> /dev/null

%.o : %.cpp
	@echo $<
	@$(call make-depend,$<,$@,$(subst .o,.d,$@))
	@$(CC) $(CCFLAGS) -c $< -o $(patsubst %.cpp, %.o, $<)


ifneq "$(MAKECMDGOALS)" "clean"
-include $(dependencies)
-include $(test_dependencies)
endif
