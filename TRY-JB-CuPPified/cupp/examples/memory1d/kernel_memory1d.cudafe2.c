#ifdef _WIN32
#pragma warning(disable:4164 4003)
#endif 
# 1 "kernel_memory1d.cudafe1.gpu"
# 31 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/memory1d.h"
struct _ZN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE;
#ifndef __stub_type__ZN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE
# 35 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/memory1d.h"
typedef struct _ZN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE _ZN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE;
#define __stub_type__ZN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE
#endif
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
#ifndef __stub_type__ZN4cupp7deviceT8memory1dIiNS_8memory1dIiEEE9size_typeE
# 43 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/memory1d.h"
typedef size_t _ZN4cupp7deviceT8memory1dIiNS_8memory1dIiEEE9size_typeE;
#define __stub_type__ZN4cupp7deviceT8memory1dIiNS_8memory1dIiEEE9size_typeE
#endif
# 31 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/memory1d.h"
struct _ZN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE {
# 87 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/memory1d.h"
int *device_pointer_;
# 92 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/memory1d.h"
_ZN4cupp7deviceT8memory1dIiNS_8memory1dIiEEE9size_typeE size_;};
void *memcpy(void*, const void*, size_t); void *memset(void*, int, size_t);
# 98 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/memory1d.h"
__asm(".align 2");
#include "kernel_memory1d.cudafe2.stub.h"

#include "kernel_memory1d.cudafe2.stub.c"
