#ifdef _WIN32
#pragma warning(disable:4164 4003)
#endif 
# 1 "kernel_kernel.cudafe1.gpu"
#ifndef __stub_type_size_t
# 214 "/usr/lib/gcc/i486-linux-gnu/4.2.4/include/stddef.h" 3
typedef unsigned size_t;
#define __stub_type_size_t
#endif
#include "crt/host_runtime.h"
#ifndef __stub_type__Complex_long_double
#define __stub_type__Complex_long_double
#endif
#ifndef __stub_type__Complex_double
#define __stub_type__Complex_double
#endif
#ifndef __stub_type__Complex_float
#define __stub_type__Complex_float
#endif
#ifndef __stub_type___clock_t
# 145 "/usr/include/bits/types.h" 3
typedef long __clock_t;
#define __stub_type___clock_t
#endif
#ifndef __stub_type_clock_t
# 61 "/usr/include/time.h" 3
typedef __clock_t clock_t;
#define __stub_type_clock_t
#endif
void *memcpy(void*, const void*, size_t); void *memset(void*, int, size_t);
#include "kernel_kernel.cudafe2.stub.h"

#include "kernel_kernel.cudafe2.stub.c"
