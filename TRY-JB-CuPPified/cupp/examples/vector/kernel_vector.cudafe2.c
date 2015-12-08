#ifdef _WIN32
#pragma warning(disable:4164 4003)
#endif 
# 1 "kernel_vector.cudafe1.gpu"
# 29 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/vector.h"
struct _ZN4cupp7deviceT6vectorIiEE;
#ifndef __stub_type__ZN4cupp7deviceT6vectorIiEE
# 33 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/vector.h"
typedef struct _ZN4cupp7deviceT6vectorIiEE _ZN4cupp7deviceT6vectorIiEE;
#define __stub_type__ZN4cupp7deviceT6vectorIiEE
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
#ifndef __stub_type__ZN4cupp7deviceT8memory1dIiNS_6vectorIiEEE9size_typeE
# 43 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/memory1d.h"
typedef size_t _ZN4cupp7deviceT8memory1dIiNS_6vectorIiEEE9size_typeE;
#define __stub_type__ZN4cupp7deviceT8memory1dIiNS_6vectorIiEEE9size_typeE
#endif
#ifndef __stub_type__ZN4cupp7deviceT6vectorIiE9size_typeE
# 41 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/vector.h"
typedef _ZN4cupp7deviceT8memory1dIiNS_6vectorIiEEE9size_typeE _ZN4cupp7deviceT6vectorIiE9size_typeE;
#define __stub_type__ZN4cupp7deviceT6vectorIiE9size_typeE
#endif
# 29 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/vector.h"
struct _ZN4cupp7deviceT6vectorIiEE {
# 84 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/vector.h"
int *device_pointer_;
# 89 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/vector.h"
_ZN4cupp7deviceT6vectorIiE9size_typeE size_;};
void *memcpy(void*, const void*, size_t); void *memset(void*, int, size_t);
# 94 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/vector.h"
__asm(".align 2");
#include "kernel_vector.cudafe2.stub.h"

#include "kernel_vector.cudafe2.stub.c"
