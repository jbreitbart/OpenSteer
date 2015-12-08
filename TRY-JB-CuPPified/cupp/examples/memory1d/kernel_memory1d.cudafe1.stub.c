#if defined(__cplusplus)
extern "C" {
#endif
#include "kernel_memory1d.fatbin.c"
#include "crt/host_runtime.h"
struct __T20;
struct __T20 {_ZN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE *__par0;int __dummy_field;};
#if defined(__device_emulation)
static void __device_wrapper__Z15global_functionRN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE(char *);
#endif
static void __sti____cudaRegisterAll_23_kernel_memory1d_cpp1_ii_c7d0aa79(void) __attribute__((__constructor__));
void __device_stub__Z15global_functionRN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE(_ZN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE *__par0){auto struct __T20 *__T21;
__cudaInitArgBlock(__T21);__cudaSetupArg(__par0, __T21);__cudaLaunch(((char *)__device_stub__Z15global_functionRN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE));}
#if defined(__device_emulation)
static void __device_wrapper__Z15global_functionRN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE(char *__T22){_Z15global_functionRN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE((((*((struct __T20 *)__T22)).__par0)));}
#endif
static void __sti____cudaRegisterAll_23_kernel_memory1d_cpp1_ii_c7d0aa79(void){__cudaRegisterBinary();__cudaRegisterEntry(_Z15global_functionRN4cupp7deviceT8memory1dIiNS_8memory1dIiEEEE, (-1));}
#if defined(__cplusplus)
}
#endif
