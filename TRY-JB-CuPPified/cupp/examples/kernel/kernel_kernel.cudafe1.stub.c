#if defined(__cplusplus)
extern "C" {
#endif
#include "kernel_kernel.fatbin.c"
#include "crt/host_runtime.h"
struct __T20;
struct __T20 {int __par0;int *__par1;int __dummy_field;};
#if defined(__device_emulation)
static void __device_wrapper__Z15global_functioniRi(char *);
#endif
static void __sti____cudaRegisterAll_21_kernel_kernel_cpp1_ii_e5318f95(void) __attribute__((__constructor__));
void __device_stub__Z15global_functioniRi(const int __par0, int *__par1){auto struct __T20 *__T21;
__cudaInitArgBlock(__T21);__cudaSetupArg(__par0, __T21);__cudaSetupArg(__par1, __T21);__cudaLaunch(((char *)__device_stub__Z15global_functioniRi));}
#if defined(__device_emulation)
static void __device_wrapper__Z15global_functioniRi(char *__T22){_Z15global_functioniRi((((*((struct __T20 *)__T22)).__par0)), (((*((struct __T20 *)__T22)).__par1)));}
#endif
static void __sti____cudaRegisterAll_21_kernel_kernel_cpp1_ii_e5318f95(void){__cudaRegisterBinary();__cudaRegisterEntry(_Z15global_functioniRi, (-1));}
#if defined(__cplusplus)
}
#endif
