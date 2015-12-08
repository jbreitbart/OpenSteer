#if defined(__cplusplus)
extern "C" {
#endif
#include "kernel_vector.fatbin.c"
#include "crt/host_runtime.h"
struct __T20;
struct __T20 {_ZN4cupp7deviceT6vectorIiEE *__par0;int __dummy_field;};
#if defined(__device_emulation)
static void __device_wrapper__Z15global_functionRN4cupp7deviceT6vectorIiEE(char *);
#endif
static void __sti____cudaRegisterAll_21_kernel_vector_cpp1_ii_4f678c78(void) __attribute__((__constructor__));
void __device_stub__Z15global_functionRN4cupp7deviceT6vectorIiEE(_ZN4cupp7deviceT6vectorIiEE *__par0){auto struct __T20 *__T21;
__cudaInitArgBlock(__T21);__cudaSetupArg(__par0, __T21);__cudaLaunch(((char *)__device_stub__Z15global_functionRN4cupp7deviceT6vectorIiEE));}
#if defined(__device_emulation)
static void __device_wrapper__Z15global_functionRN4cupp7deviceT6vectorIiEE(char *__T22){_Z15global_functionRN4cupp7deviceT6vectorIiEE((((*((struct __T20 *)__T22)).__par0)));}
#endif
static void __sti____cudaRegisterAll_21_kernel_vector_cpp1_ii_4f678c78(void){__cudaRegisterBinary();__cudaRegisterEntry(_Z15global_functionRN4cupp7deviceT6vectorIiEE, (-1));}
#if defined(__cplusplus)
}
#endif
