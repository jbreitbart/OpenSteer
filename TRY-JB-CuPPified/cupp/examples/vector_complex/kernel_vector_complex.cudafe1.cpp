# 1 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector_complex/kernel_vector_complex.cu"
# 149 "/usr/include/c++/4.2/i486-linux-gnu/bits/c++config.h" 3
namespace std __attribute__((visibility("default"))) { 
# 161 "/usr/include/c++/4.2/i486-linux-gnu/bits/c++config.h" 3
}
#if 0
# 46 "/usr/local/cuda/bin/../include/device_types.h"
enum cudaRoundMode { 
# 48
cudaRoundNearest, 
# 49
cudaRoundZero, 
# 50
cudaRoundPosInf, 
# 51
cudaRoundMinInf
# 52
}; 
#endif
# 152 "/usr/lib/gcc/i486-linux-gnu/4.2.4/include/stddef.h" 3
typedef int ptrdiff_t; 
# 214 "/usr/lib/gcc/i486-linux-gnu/4.2.4/include/stddef.h" 3
typedef unsigned size_t; 
#include "crt/host_runtime.h"
#if 0
# 59 "/usr/local/cuda/bin/../include/driver_types.h"
enum cudaError { 
# 61
cudaSuccess, 
# 62
cudaErrorMissingConfiguration, 
# 63
cudaErrorMemoryAllocation, 
# 64
cudaErrorInitializationError, 
# 65
cudaErrorLaunchFailure, 
# 66
cudaErrorPriorLaunchFailure, 
# 67
cudaErrorLaunchTimeout, 
# 68
cudaErrorLaunchOutOfResources, 
# 69
cudaErrorInvalidDeviceFunction, 
# 70
cudaErrorInvalidConfiguration, 
# 71
cudaErrorInvalidDevice, 
# 72
cudaErrorInvalidValue, 
# 73
cudaErrorInvalidPitchValue, 
# 74
cudaErrorInvalidSymbol, 
# 75
cudaErrorMapBufferObjectFailed, 
# 76
cudaErrorUnmapBufferObjectFailed, 
# 77
cudaErrorInvalidHostPointer, 
# 78
cudaErrorInvalidDevicePointer, 
# 79
cudaErrorInvalidTexture, 
# 80
cudaErrorInvalidTextureBinding, 
# 81
cudaErrorInvalidChannelDescriptor, 
# 82
cudaErrorInvalidMemcpyDirection, 
# 83
cudaErrorAddressOfConstant, 
# 84
cudaErrorTextureFetchFailed, 
# 85
cudaErrorTextureNotBound, 
# 86
cudaErrorSynchronizationError, 
# 87
cudaErrorInvalidFilterSetting, 
# 88
cudaErrorInvalidNormSetting, 
# 89
cudaErrorMixedDeviceExecution, 
# 90
cudaErrorCudartUnloading, 
# 91
cudaErrorUnknown, 
# 92
cudaErrorNotYetImplemented, 
# 93
cudaErrorMemoryValueTooLarge, 
# 94
cudaErrorInvalidResourceHandle, 
# 95
cudaErrorNotReady, 
# 96
cudaErrorInsufficientDriver, 
# 97
cudaErrorSetOnActiveProcess, 
# 98
cudaErrorStartupFailure = 127, 
# 99
cudaErrorApiFailureBase = 10000
# 100
}; 
#endif
#if 0
enum cudaChannelFormatKind { 
# 105
cudaChannelFormatKindSigned, 
# 106
cudaChannelFormatKindUnsigned, 
# 107
cudaChannelFormatKindFloat, 
# 108
cudaChannelFormatKindNone
# 109
}; 
#endif
#if 0
struct cudaChannelFormatDesc { 
# 114
int x; 
# 115
int y; 
# 116
int z; 
# 117
int w; 
# 118
cudaChannelFormatKind f; 
# 119
}; 
#endif
#if 0
struct cudaArray; 
#endif
#if 0
enum cudaMemcpyKind { 
# 127
cudaMemcpyHostToHost, 
# 128
cudaMemcpyHostToDevice, 
# 129
cudaMemcpyDeviceToHost, 
# 130
cudaMemcpyDeviceToDevice
# 131
}; 
#endif
#if 0
struct cudaPitchedPtr { 
# 136
void *ptr; 
# 137
size_t pitch; 
# 138
size_t xsize; 
# 139
size_t ysize; 
# 140
}; 
#endif
#if 0
struct cudaExtent { 
# 145
size_t width; 
# 146
size_t height; 
# 147
size_t depth; 
# 148
}; 
#endif
#if 0
struct cudaPos { 
# 153
size_t x; 
# 154
size_t y; 
# 155
size_t z; 
# 156
}; 
#endif
#if 0
struct cudaMemcpy3DParms { 
# 161
cudaArray *srcArray; 
# 162
cudaPos srcPos; 
# 163
cudaPitchedPtr srcPtr; 
# 165
cudaArray *dstArray; 
# 166
cudaPos dstPos; 
# 167
cudaPitchedPtr dstPtr; 
# 169
cudaExtent extent; 
# 170
cudaMemcpyKind kind; 
# 171
}; 
#endif
#if 0
struct cudaDeviceProp { 
# 176
char name[256]; 
# 177
size_t totalGlobalMem; 
# 178
size_t sharedMemPerBlock; 
# 179
int regsPerBlock; 
# 180
int warpSize; 
# 181
size_t memPitch; 
# 182
int maxThreadsPerBlock; 
# 183
int maxThreadsDim[3]; 
# 184
int maxGridSize[3]; 
# 185
int clockRate; 
# 186
size_t totalConstMem; 
# 187
int major; 
# 188
int minor; 
# 189
size_t textureAlignment; 
# 190
int deviceOverlap; 
# 191
int multiProcessorCount; 
# 192
int kernelExecTimeoutEnabled; 
# 193
int __cudaReserved[39]; 
# 194
}; 
#endif
#if 0
# 224 "/usr/local/cuda/bin/../include/driver_types.h"
typedef cudaError cudaError_t; 
#endif
#if 0
typedef int cudaStream_t; 
#endif
#if 0
typedef int cudaEvent_t; 
#endif
#if 0
# 54 "/usr/local/cuda/bin/../include/texture_types.h"
enum cudaTextureAddressMode { 
# 56
cudaAddressModeWrap, 
# 57
cudaAddressModeClamp
# 58
}; 
#endif
#if 0
enum cudaTextureFilterMode { 
# 63
cudaFilterModePoint, 
# 64
cudaFilterModeLinear
# 65
}; 
#endif
#if 0
enum cudaTextureReadMode { 
# 70
cudaReadModeElementType, 
# 71
cudaReadModeNormalizedFloat
# 72
}; 
#endif
#if 0
struct textureReference { 
# 77
int normalized; 
# 78
cudaTextureFilterMode filterMode; 
# 79
cudaTextureAddressMode addressMode[3]; 
# 80
cudaChannelFormatDesc channelDesc; 
# 81
int __cudaReserved[16]; 
# 82
}; 
#endif
#if 0
# 54 "/usr/local/cuda/bin/../include/vector_types.h"
struct char1 { 
# 56
signed char x; 
# 57
}; 
#endif
#if 0
struct uchar1 { 
# 62
unsigned char x; 
# 63
}; 
#endif
#if 0
struct __attribute__((__aligned__(2))) char2 { 
# 68
signed char x; signed char y; 
# 69
}; 
#endif
#if 0
struct __attribute__((__aligned__(2))) uchar2 { 
# 74
unsigned char x; unsigned char y; 
# 75
}; 
#endif
#if 0
struct char3 { 
# 80
signed char x; signed char y; signed char z; 
# 81
}; 
#endif
#if 0
struct uchar3 { 
# 86
unsigned char x; unsigned char y; unsigned char z; 
# 87
}; 
#endif
#if 0
struct __attribute__((__aligned__(4))) char4 { 
# 92
signed char x; signed char y; signed char z; signed char w; 
# 93
}; 
#endif
#if 0
struct __attribute__((__aligned__(4))) uchar4 { 
# 98
unsigned char x; unsigned char y; unsigned char z; unsigned char w; 
# 99
}; 
#endif
#if 0
struct short1 { 
# 104
short x; 
# 105
}; 
#endif
#if 0
struct ushort1 { 
# 110
unsigned short x; 
# 111
}; 
#endif
#if 0
struct __attribute__((__aligned__(4))) short2 { 
# 116
short x; short y; 
# 117
}; 
#endif
#if 0
struct __attribute__((__aligned__(4))) ushort2 { 
# 122
unsigned short x; unsigned short y; 
# 123
}; 
#endif
#if 0
struct short3 { 
# 128
short x; short y; short z; 
# 129
}; 
#endif
#if 0
struct ushort3 { 
# 134
unsigned short x; unsigned short y; unsigned short z; 
# 135
}; 
#endif
#if 0
struct __attribute__((__aligned__(8))) short4 { 
# 140
short x; short y; short z; short w; 
# 141
}; 
#endif
#if 0
struct __attribute__((__aligned__(8))) ushort4 { 
# 146
unsigned short x; unsigned short y; unsigned short z; unsigned short w; 
# 147
}; 
#endif
#if 0
struct int1 { 
# 152
int x; 
# 153
}; 
#endif
#if 0
struct uint1 { 
# 158
unsigned x; 
# 159
}; 
#endif
#if 0
struct __attribute__((__aligned__(8))) int2 { 
# 164
int x; int y; 
# 165
}; 
#endif
#if 0
struct __attribute__((__aligned__(8))) uint2 { 
# 170
unsigned x; unsigned y; 
# 171
}; 
#endif
#if 0
struct int3 { 
# 176
int x; int y; int z; 
# 177
}; 
#endif
#if 0
struct uint3 { 
# 182
unsigned x; unsigned y; unsigned z; 
# 183
}; 
#endif
#if 0
struct __attribute__((__aligned__(16))) int4 { 
# 188
int x; int y; int z; int w; 
# 189
}; 
#endif
#if 0
struct __attribute__((__aligned__(16))) uint4 { 
# 194
unsigned x; unsigned y; unsigned z; unsigned w; 
# 195
}; 
#endif
#if 0
struct long1 { 
# 200
long x; 
# 201
}; 
#endif
#if 0
struct ulong1 { 
# 206
unsigned long x; 
# 207
}; 
#endif
#if 0
# 216
struct __attribute__((__aligned__(8))) long2 { 
# 218
long x; long y; 
# 219
}; 
#endif
#if 0
# 228
struct __attribute__((__aligned__(8))) ulong2 { 
# 230
unsigned long x; unsigned long y; 
# 231
}; 
#endif
#if 0
# 236
struct long3 { 
# 238
long x; long y; long z; 
# 239
}; 
#endif
#if 0
struct ulong3 { 
# 244
unsigned long x; unsigned long y; unsigned long z; 
# 245
}; 
#endif
#if 0
struct __attribute__((__aligned__(16))) long4 { 
# 250
long x; long y; long z; long w; 
# 251
}; 
#endif
#if 0
struct __attribute__((__aligned__(16))) ulong4 { 
# 256
unsigned long x; unsigned long y; unsigned long z; unsigned long w; 
# 257
}; 
#endif
#if 0
# 262
struct float1 { 
# 264
float x; 
# 265
}; 
#endif
#if 0
struct __attribute__((__aligned__(8))) float2 { 
# 270
float x; float y; 
# 271
}; 
#endif
#if 0
struct float3 { 
# 276
float x; float y; float z; 
# 277
}; 
#endif
#if 0
struct __attribute__((__aligned__(16))) float4 { 
# 282
float x; float y; float z; float w; 
# 283
}; 
#endif
#if 0
struct longlong1 { 
# 288
long long x; 
# 289
}; 
#endif
#if 0
struct ulonglong1 { 
# 294
unsigned long long x; 
# 295
}; 
#endif
#if 0
struct __attribute__((__aligned__(16))) longlong2 { 
# 300
long long x; long long y; 
# 301
}; 
#endif
#if 0
struct __attribute__((__aligned__(16))) ulonglong2 { 
# 306
unsigned long long x; unsigned long long y; 
# 307
}; 
#endif
#if 0
struct double1 { 
# 312
double x; 
# 313
}; 
#endif
#if 0
struct __attribute__((__aligned__(16))) double2 { 
# 318
double x; double y; 
# 319
}; 
#endif
#if 0
# 328
typedef char1 char1; 
#endif
#if 0
# 330
typedef uchar1 uchar1; 
#endif
#if 0
# 332
typedef char2 char2; 
#endif
#if 0
# 334
typedef uchar2 uchar2; 
#endif
#if 0
# 336
typedef char3 char3; 
#endif
#if 0
# 338
typedef uchar3 uchar3; 
#endif
#if 0
# 340
typedef char4 char4; 
#endif
#if 0
# 342
typedef uchar4 uchar4; 
#endif
#if 0
# 344
typedef short1 short1; 
#endif
#if 0
# 346
typedef ushort1 ushort1; 
#endif
#if 0
# 348
typedef short2 short2; 
#endif
#if 0
# 350
typedef ushort2 ushort2; 
#endif
#if 0
# 352
typedef short3 short3; 
#endif
#if 0
# 354
typedef ushort3 ushort3; 
#endif
#if 0
# 356
typedef short4 short4; 
#endif
#if 0
# 358
typedef ushort4 ushort4; 
#endif
#if 0
# 360
typedef int1 int1; 
#endif
#if 0
# 362
typedef uint1 uint1; 
#endif
#if 0
# 364
typedef int2 int2; 
#endif
#if 0
# 366
typedef uint2 uint2; 
#endif
#if 0
# 368
typedef int3 int3; 
#endif
#if 0
# 370
typedef uint3 uint3; 
#endif
#if 0
# 372
typedef int4 int4; 
#endif
#if 0
# 374
typedef uint4 uint4; 
#endif
#if 0
# 376
typedef long1 long1; 
#endif
#if 0
# 378
typedef ulong1 ulong1; 
#endif
#if 0
# 380
typedef long2 long2; 
#endif
#if 0
# 382
typedef ulong2 ulong2; 
#endif
#if 0
# 384
typedef long3 long3; 
#endif
#if 0
# 386
typedef ulong3 ulong3; 
#endif
#if 0
# 388
typedef long4 long4; 
#endif
#if 0
# 390
typedef ulong4 ulong4; 
#endif
#if 0
# 392
typedef float1 float1; 
#endif
#if 0
# 394
typedef float2 float2; 
#endif
#if 0
# 396
typedef float3 float3; 
#endif
#if 0
# 398
typedef float4 float4; 
#endif
#if 0
# 400
typedef longlong1 longlong1; 
#endif
#if 0
# 402
typedef ulonglong1 ulonglong1; 
#endif
#if 0
# 404
typedef longlong2 longlong2; 
#endif
#if 0
# 406
typedef ulonglong2 ulonglong2; 
#endif
#if 0
# 408
typedef double1 double1; 
#endif
#if 0
# 410
typedef double2 double2; 
#endif
#if 0
#endif
#if 0
# 419
typedef struct dim3 dim3; 
#endif
#if 0
struct dim3 { 
# 424
unsigned x; unsigned y; unsigned z; 
# 430
}; 
#endif
# 88 "/usr/local/cuda/bin/../include/cuda_runtime_api.h"
extern "C" cudaError_t cudaMalloc3D(cudaPitchedPtr *, cudaExtent); 
# 89
extern "C" cudaError_t cudaMalloc3DArray(cudaArray **, const cudaChannelFormatDesc *, cudaExtent); 
# 90
extern "C" cudaError_t cudaMemset3D(cudaPitchedPtr, int, cudaExtent); 
# 91
extern "C" cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *); 
# 92
extern "C" cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *, cudaStream_t); 
# 101
extern "C" cudaError_t cudaMalloc(void **, size_t); 
# 102
extern "C" cudaError_t cudaMallocHost(void **, size_t); 
# 103
extern "C" cudaError_t cudaMallocPitch(void **, size_t *, size_t, size_t); 
# 104
extern "C" cudaError_t cudaMallocArray(cudaArray **, const cudaChannelFormatDesc *, size_t, size_t = (1)); 
# 105
extern "C" cudaError_t cudaFree(void *); 
# 106
extern "C" cudaError_t cudaFreeHost(void *); 
# 107
extern "C" cudaError_t cudaFreeArray(cudaArray *); 
# 116
extern "C" cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind); 
# 117
extern "C" cudaError_t cudaMemcpyToArray(cudaArray *, size_t, size_t, const void *, size_t, cudaMemcpyKind); 
# 118
extern "C" cudaError_t cudaMemcpyFromArray(void *, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind); 
# 119
extern "C" cudaError_t cudaMemcpyArrayToArray(cudaArray *, size_t, size_t, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind = cudaMemcpyDeviceToDevice); 
# 120
extern "C" cudaError_t cudaMemcpy2D(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind); 
# 121
extern "C" cudaError_t cudaMemcpy2DToArray(cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind); 
# 122
extern "C" cudaError_t cudaMemcpy2DFromArray(void *, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind); 
# 123
extern "C" cudaError_t cudaMemcpy2DArrayToArray(cudaArray *, size_t, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind = cudaMemcpyDeviceToDevice); 
# 124
extern "C" cudaError_t cudaMemcpyToSymbol(const char *, const void *, size_t, size_t = (0), cudaMemcpyKind = cudaMemcpyHostToDevice); 
# 125
extern "C" cudaError_t cudaMemcpyFromSymbol(void *, const char *, size_t, size_t = (0), cudaMemcpyKind = cudaMemcpyDeviceToHost); 
# 133
extern "C" cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t); 
# 134
extern "C" cudaError_t cudaMemcpyToArrayAsync(cudaArray *, size_t, size_t, const void *, size_t, cudaMemcpyKind, cudaStream_t); 
# 135
extern "C" cudaError_t cudaMemcpyFromArrayAsync(void *, const cudaArray *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t); 
# 136
extern "C" cudaError_t cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t); 
# 137
extern "C" cudaError_t cudaMemcpy2DToArrayAsync(cudaArray *, size_t, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t); 
# 138
extern "C" cudaError_t cudaMemcpy2DFromArrayAsync(void *, size_t, const cudaArray *, size_t, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t); 
# 139
extern "C" cudaError_t cudaMemcpyToSymbolAsync(const char *, const void *, size_t, size_t, cudaMemcpyKind, cudaStream_t); 
# 140
extern "C" cudaError_t cudaMemcpyFromSymbolAsync(void *, const char *, size_t, size_t, cudaMemcpyKind, cudaStream_t); 
# 148
extern "C" cudaError_t cudaMemset(void *, int, size_t); 
# 149
extern "C" cudaError_t cudaMemset2D(void *, size_t, int, size_t, size_t); 
# 157
extern "C" cudaError_t cudaGetSymbolAddress(void **, const char *); 
# 158
extern "C" cudaError_t cudaGetSymbolSize(size_t *, const char *); 
# 166
extern "C" cudaError_t cudaGetDeviceCount(int *); 
# 167
extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp *, int); 
# 168
extern "C" cudaError_t cudaChooseDevice(int *, const cudaDeviceProp *); 
# 169
extern "C" cudaError_t cudaSetDevice(int); 
# 170
extern "C" cudaError_t cudaGetDevice(int *); 
# 178
extern "C" cudaError_t cudaBindTexture(size_t *, const textureReference *, const void *, const cudaChannelFormatDesc *, size_t = (((2147483647) * 2U) + 1U)); 
# 179
extern "C" cudaError_t cudaBindTextureToArray(const textureReference *, const cudaArray *, const cudaChannelFormatDesc *); 
# 180
extern "C" cudaError_t cudaUnbindTexture(const textureReference *); 
# 181
extern "C" cudaError_t cudaGetTextureAlignmentOffset(size_t *, const textureReference *); 
# 182
extern "C" cudaError_t cudaGetTextureReference(const textureReference **, const char *); 
# 190
extern "C" cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *, const cudaArray *); 
# 191
extern "C" cudaChannelFormatDesc cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind); 
# 199
extern "C" cudaError_t cudaGetLastError(); 
# 200
extern "C" const char *cudaGetErrorString(cudaError_t); 
# 208
extern "C" cudaError_t cudaConfigureCall(dim3, dim3, size_t = (0), cudaStream_t = (0)); 
# 209
extern "C" cudaError_t cudaSetupArgument(const void *, size_t, size_t); 
# 210
extern "C" cudaError_t cudaLaunch(const char *); 
# 218
extern "C" cudaError_t cudaStreamCreate(cudaStream_t *); 
# 219
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t); 
# 220
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t); 
# 221
extern "C" cudaError_t cudaStreamQuery(cudaStream_t); 
# 229
extern "C" cudaError_t cudaEventCreate(cudaEvent_t *); 
# 230
extern "C" cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t); 
# 231
extern "C" cudaError_t cudaEventQuery(cudaEvent_t); 
# 232
extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t); 
# 233
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t); 
# 234
extern "C" cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t); 
# 242
extern "C" cudaError_t cudaSetDoubleForDevice(double *); 
# 243
extern "C" cudaError_t cudaSetDoubleForHost(double *); 
# 251
extern "C" cudaError_t cudaThreadExit(); 
# 252
extern "C" cudaError_t cudaThreadSynchronize(); 
# 58 "/usr/local/cuda/bin/../include/channel_descriptor.h"
template<class T> inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 59
{ 
# 60
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 61
} 
# 63
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
# 64
{ 
# 65
auto int e = (((int)sizeof(char)) * 8); 
# 70
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 72
} 
# 74
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
# 75
{ 
# 76
auto int e = (((int)sizeof(signed char)) * 8); 
# 78
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 79
} 
# 81
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
# 82
{ 
# 83
auto int e = (((int)sizeof(unsigned char)) * 8); 
# 85
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 86
} 
# 88
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
# 89
{ 
# 90
auto int e = (((int)sizeof(signed char)) * 8); 
# 92
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 93
} 
# 95
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
# 96
{ 
# 97
auto int e = (((int)sizeof(unsigned char)) * 8); 
# 99
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 100
} 
# 102
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
# 103
{ 
# 104
auto int e = (((int)sizeof(signed char)) * 8); 
# 106
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 107
} 
# 109
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
# 110
{ 
# 111
auto int e = (((int)sizeof(unsigned char)) * 8); 
# 113
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 114
} 
# 116
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
# 117
{ 
# 118
auto int e = (((int)sizeof(signed char)) * 8); 
# 120
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 121
} 
# 123
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
# 124
{ 
# 125
auto int e = (((int)sizeof(unsigned char)) * 8); 
# 127
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 128
} 
# 130
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
# 131
{ 
# 132
auto int e = (((int)sizeof(short)) * 8); 
# 134
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 135
} 
# 137
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
# 138
{ 
# 139
auto int e = (((int)sizeof(unsigned short)) * 8); 
# 141
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 142
} 
# 144
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
# 145
{ 
# 146
auto int e = (((int)sizeof(short)) * 8); 
# 148
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 149
} 
# 151
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
# 152
{ 
# 153
auto int e = (((int)sizeof(unsigned short)) * 8); 
# 155
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 156
} 
# 158
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
# 159
{ 
# 160
auto int e = (((int)sizeof(short)) * 8); 
# 162
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 163
} 
# 165
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
# 166
{ 
# 167
auto int e = (((int)sizeof(unsigned short)) * 8); 
# 169
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 170
} 
# 172
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
# 173
{ 
# 174
auto int e = (((int)sizeof(short)) * 8); 
# 176
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 177
} 
# 179
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
# 180
{ 
# 181
auto int e = (((int)sizeof(unsigned short)) * 8); 
# 183
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 184
} 
# 186
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
# 187
{ 
# 188
auto int e = (((int)sizeof(int)) * 8); 
# 190
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 191
} 
# 193
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
# 194
{ 
# 195
auto int e = (((int)sizeof(unsigned)) * 8); 
# 197
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 198
} 
# 200
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
# 201
{ 
# 202
auto int e = (((int)sizeof(int)) * 8); 
# 204
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 205
} 
# 207
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
# 208
{ 
# 209
auto int e = (((int)sizeof(unsigned)) * 8); 
# 211
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 212
} 
# 214
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
# 215
{ 
# 216
auto int e = (((int)sizeof(int)) * 8); 
# 218
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 219
} 
# 221
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
# 222
{ 
# 223
auto int e = (((int)sizeof(unsigned)) * 8); 
# 225
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 226
} 
# 228
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
# 229
{ 
# 230
auto int e = (((int)sizeof(int)) * 8); 
# 232
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 233
} 
# 235
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
# 236
{ 
# 237
auto int e = (((int)sizeof(unsigned)) * 8); 
# 239
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 240
} 
# 244
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< long> () 
# 245
{ 
# 246
auto int e = (((int)sizeof(long)) * 8); 
# 248
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 249
} 
# 251
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned long> () 
# 252
{ 
# 253
auto int e = (((int)sizeof(unsigned long)) * 8); 
# 255
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 256
} 
# 258
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< long1> () 
# 259
{ 
# 260
auto int e = (((int)sizeof(long)) * 8); 
# 262
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 263
} 
# 265
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong1> () 
# 266
{ 
# 267
auto int e = (((int)sizeof(unsigned long)) * 8); 
# 269
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 270
} 
# 272
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< long2> () 
# 273
{ 
# 274
auto int e = (((int)sizeof(long)) * 8); 
# 276
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 277
} 
# 279
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong2> () 
# 280
{ 
# 281
auto int e = (((int)sizeof(unsigned long)) * 8); 
# 283
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 284
} 
# 286
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< long4> () 
# 287
{ 
# 288
auto int e = (((int)sizeof(long)) * 8); 
# 290
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 291
} 
# 293
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong4> () 
# 294
{ 
# 295
auto int e = (((int)sizeof(unsigned long)) * 8); 
# 297
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 298
} 
# 302
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
# 303
{ 
# 304
auto int e = (((int)sizeof(float)) * 8); 
# 306
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 307
} 
# 309
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
# 310
{ 
# 311
auto int e = (((int)sizeof(float)) * 8); 
# 313
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 314
} 
# 316
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
# 317
{ 
# 318
auto int e = (((int)sizeof(float)) * 8); 
# 320
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 321
} 
# 323
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
# 324
{ 
# 325
auto int e = (((int)sizeof(float)) * 8); 
# 327
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 328
} 
# 54 "/usr/local/cuda/bin/../include/driver_functions.h"
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
# 55
{ 
# 56
auto cudaPitchedPtr s; 
# 58
(s.ptr) = d; 
# 59
(s.pitch) = p; 
# 60
(s.xsize) = xsz; 
# 61
(s.ysize) = ysz; 
# 63
return s; 
# 64
} 
# 66
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
# 67
{ 
# 68
auto cudaPos p; 
# 70
(p.x) = x; 
# 71
(p.y) = y; 
# 72
(p.z) = z; 
# 74
return p; 
# 75
} 
# 77
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
# 78
{ 
# 79
auto cudaExtent e; 
# 81
(e.width) = w; 
# 82
(e.height) = h; 
# 83
(e.depth) = d; 
# 85
return e; 
# 86
} 
# 54 "/usr/local/cuda/bin/../include/vector_functions.h"
static inline char1 make_char1(signed char x) 
# 55
{ 
# 56
auto char1 t; (t.x) = x; return t; 
# 57
} 
# 59
static inline uchar1 make_uchar1(unsigned char x) 
# 60
{ 
# 61
auto uchar1 t; (t.x) = x; return t; 
# 62
} 
# 64
static inline char2 make_char2(signed char x, signed char y) 
# 65
{ 
# 66
auto char2 t; (t.x) = x; (t.y) = y; return t; 
# 67
} 
# 69
static inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
# 70
{ 
# 71
auto uchar2 t; (t.x) = x; (t.y) = y; return t; 
# 72
} 
# 74
static inline char3 make_char3(signed char x, signed char y, signed char z) 
# 75
{ 
# 76
auto char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 77
} 
# 79
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
# 80
{ 
# 81
auto uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 82
} 
# 84
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
# 85
{ 
# 86
auto char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 87
} 
# 89
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
# 90
{ 
# 91
auto uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 92
} 
# 94
static inline short1 make_short1(short x) 
# 95
{ 
# 96
auto short1 t; (t.x) = x; return t; 
# 97
} 
# 99
static inline ushort1 make_ushort1(unsigned short x) 
# 100
{ 
# 101
auto ushort1 t; (t.x) = x; return t; 
# 102
} 
# 104
static inline short2 make_short2(short x, short y) 
# 105
{ 
# 106
auto short2 t; (t.x) = x; (t.y) = y; return t; 
# 107
} 
# 109
static inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
# 110
{ 
# 111
auto ushort2 t; (t.x) = x; (t.y) = y; return t; 
# 112
} 
# 114
static inline short3 make_short3(short x, short y, short z) 
# 115
{ 
# 116
auto short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 117
} 
# 119
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
# 120
{ 
# 121
auto ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 122
} 
# 124
static inline short4 make_short4(short x, short y, short z, short w) 
# 125
{ 
# 126
auto short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 127
} 
# 129
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
# 130
{ 
# 131
auto ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 132
} 
# 134
static inline int1 make_int1(int x) 
# 135
{ 
# 136
auto int1 t; (t.x) = x; return t; 
# 137
} 
# 139
static inline uint1 make_uint1(unsigned x) 
# 140
{ 
# 141
auto uint1 t; (t.x) = x; return t; 
# 142
} 
# 144
static inline int2 make_int2(int x, int y) 
# 145
{ 
# 146
auto int2 t; (t.x) = x; (t.y) = y; return t; 
# 147
} 
# 149
static inline uint2 make_uint2(unsigned x, unsigned y) 
# 150
{ 
# 151
auto uint2 t; (t.x) = x; (t.y) = y; return t; 
# 152
} 
# 154
static inline int3 make_int3(int x, int y, int z) 
# 155
{ 
# 156
auto int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 157
} 
# 159
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
# 160
{ 
# 161
auto uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 162
} 
# 164
static inline int4 make_int4(int x, int y, int z, int w) 
# 165
{ 
# 166
auto int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 167
} 
# 169
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
# 170
{ 
# 171
auto uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 172
} 
# 174
static inline long1 make_long1(long x) 
# 175
{ 
# 176
auto long1 t; (t.x) = x; return t; 
# 177
} 
# 179
static inline ulong1 make_ulong1(unsigned long x) 
# 180
{ 
# 181
auto ulong1 t; (t.x) = x; return t; 
# 182
} 
# 184
static inline long2 make_long2(long x, long y) 
# 185
{ 
# 186
auto long2 t; (t.x) = x; (t.y) = y; return t; 
# 187
} 
# 189
static inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
# 190
{ 
# 191
auto ulong2 t; (t.x) = x; (t.y) = y; return t; 
# 192
} 
# 196
static inline long3 make_long3(long x, long y, long z) 
# 197
{ 
# 198
auto long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 199
} 
# 201
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
# 202
{ 
# 203
auto ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 204
} 
# 206
static inline long4 make_long4(long x, long y, long z, long w) 
# 207
{ 
# 208
auto long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 209
} 
# 211
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
# 212
{ 
# 213
auto ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 214
} 
# 218
static inline float1 make_float1(float x) 
# 219
{ 
# 220
auto float1 t; (t.x) = x; return t; 
# 221
} 
# 223
static inline float2 make_float2(float x, float y) 
# 224
{ 
# 225
auto float2 t; (t.x) = x; (t.y) = y; return t; 
# 226
} 
# 228
static inline float3 make_float3(float x, float y, float z) 
# 229
{ 
# 230
auto float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 231
} 
# 233
static inline float4 make_float4(float x, float y, float z, float w) 
# 234
{ 
# 235
auto float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 236
} 
# 238
static inline longlong1 make_longlong1(long long x) 
# 239
{ 
# 240
auto longlong1 t; (t.x) = x; return t; 
# 241
} 
# 243
static inline ulonglong1 make_ulonglong1(unsigned long long x) 
# 244
{ 
# 245
auto ulonglong1 t; (t.x) = x; return t; 
# 246
} 
# 248
static inline longlong2 make_longlong2(long long x, long long y) 
# 249
{ 
# 250
auto longlong2 t; (t.x) = x; (t.y) = y; return t; 
# 251
} 
# 253
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) 
# 254
{ 
# 255
auto ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
# 256
} 
# 258
static inline double1 make_double1(double x) 
# 259
{ 
# 260
auto double1 t; (t.x) = x; return t; 
# 261
} 
# 263
static inline double2 make_double2(double x, double y) 
# 264
{ 
# 265
auto double2 t; (t.x) = x; (t.y) = y; return t; 
# 266
} 
# 31 "/usr/include/bits/types.h" 3
extern "C" { typedef unsigned char __u_char; }
# 32
extern "C" { typedef unsigned short __u_short; }
# 33
extern "C" { typedef unsigned __u_int; }
# 34
extern "C" { typedef unsigned long __u_long; }
# 37
extern "C" { typedef signed char __int8_t; }
# 38
extern "C" { typedef unsigned char __uint8_t; }
# 39
extern "C" { typedef signed short __int16_t; }
# 40
extern "C" { typedef unsigned short __uint16_t; }
# 41
extern "C" { typedef signed int __int32_t; }
# 42
extern "C" { typedef unsigned __uint32_t; }
# 47
extern "C" { typedef signed long long __int64_t; }
# 48
extern "C" { typedef unsigned long long __uint64_t; }
# 56
extern "C" { typedef long long __quad_t; }
# 57
extern "C" { typedef unsigned long long __u_quad_t; }
# 134 "/usr/include/bits/types.h" 3
extern "C" { typedef __u_quad_t __dev_t; }
# 135
extern "C" { typedef unsigned __uid_t; }
# 136
extern "C" { typedef unsigned __gid_t; }
# 137
extern "C" { typedef unsigned long __ino_t; }
# 138
extern "C" { typedef __u_quad_t __ino64_t; }
# 139
extern "C" { typedef unsigned __mode_t; }
# 140
extern "C" { typedef unsigned __nlink_t; }
# 141
extern "C" { typedef long __off_t; }
# 142
extern "C" { typedef __quad_t __off64_t; }
# 143
extern "C" { typedef int __pid_t; }
# 144
extern "C" { typedef struct __fsid_t { int __val[2]; } __fsid_t; }
# 145
extern "C" { typedef long __clock_t; }
# 146
extern "C" { typedef unsigned long __rlim_t; }
# 147
extern "C" { typedef __u_quad_t __rlim64_t; }
# 148
extern "C" { typedef unsigned __id_t; }
# 149
extern "C" { typedef long __time_t; }
# 150
extern "C" { typedef unsigned __useconds_t; }
# 151
extern "C" { typedef long __suseconds_t; }
# 153
extern "C" { typedef int __daddr_t; }
# 154
extern "C" { typedef long __swblk_t; }
# 155
extern "C" { typedef int __key_t; }
# 158
extern "C" { typedef int __clockid_t; }
# 161
extern "C" { typedef void *__timer_t; }
# 164
extern "C" { typedef long __blksize_t; }
# 169
extern "C" { typedef long __blkcnt_t; }
# 170
extern "C" { typedef __quad_t __blkcnt64_t; }
# 173
extern "C" { typedef unsigned long __fsblkcnt_t; }
# 174
extern "C" { typedef __u_quad_t __fsblkcnt64_t; }
# 177
extern "C" { typedef unsigned long __fsfilcnt_t; }
# 178
extern "C" { typedef __u_quad_t __fsfilcnt64_t; }
# 180
extern "C" { typedef int __ssize_t; }
# 184
extern "C" { typedef __off64_t __loff_t; }
# 185
extern "C" { typedef __quad_t *__qaddr_t; }
# 186
extern "C" { typedef char *__caddr_t; }
# 189
extern "C" { typedef int __intptr_t; }
# 192
extern "C" { typedef unsigned __socklen_t; }
# 61 "/usr/include/time.h" 3
extern "C" { typedef __clock_t clock_t; }
# 77 "/usr/include/time.h" 3
extern "C" { typedef __time_t time_t; }
# 93 "/usr/include/time.h" 3
extern "C" { typedef __clockid_t clockid_t; }
# 105 "/usr/include/time.h" 3
extern "C" { typedef __timer_t timer_t; }
# 121 "/usr/include/time.h" 3
extern "C" { struct timespec { 
# 123
__time_t tv_sec; 
# 124
long tv_nsec; 
# 125
}; }
# 134
extern "C" { struct tm { 
# 136
int tm_sec; 
# 137
int tm_min; 
# 138
int tm_hour; 
# 139
int tm_mday; 
# 140
int tm_mon; 
# 141
int tm_year; 
# 142
int tm_wday; 
# 143
int tm_yday; 
# 144
int tm_isdst; 
# 147
long tm_gmtoff; 
# 148
const char *tm_zone; 
# 153
}; }
# 162
extern "C" { struct itimerspec { 
# 164
timespec it_interval; 
# 165
timespec it_value; 
# 166
}; }
# 169
struct sigevent; 
# 175
extern "C" { typedef __pid_t pid_t; }
# 184
extern "C"  __attribute__((__weak__)) clock_t clock() throw(); 
# 187
extern "C" time_t time(time_t *) throw(); 
# 190
extern "C" double difftime(time_t, time_t) throw() __attribute__((__const__)); 
# 194
extern "C" time_t mktime(tm *) throw(); 
# 200
extern "C" size_t strftime(char *__restrict__, size_t, const char *__restrict__, const tm *__restrict__) throw(); 
# 208
extern "C" char *strptime(const char *__restrict__, const char *__restrict__, tm *) throw(); 
# 40 "/usr/include/xlocale.h" 3
extern "C" { typedef 
# 28
struct __locale_struct { 
# 31
struct locale_data *__locales[13]; 
# 34
const unsigned short *__ctype_b; 
# 35
const int *__ctype_tolower; 
# 36
const int *__ctype_toupper; 
# 39
const char *__names[13]; 
# 40
} *__locale_t; }
# 218 "/usr/include/time.h" 3
extern "C" size_t strftime_l(char *__restrict__, size_t, const char *__restrict__, const tm *__restrict__, __locale_t) throw(); 
# 223
extern "C" char *strptime_l(const char *__restrict__, const char *__restrict__, tm *, __locale_t) throw(); 
# 232
extern "C" tm *gmtime(const time_t *) throw(); 
# 236
extern "C" tm *localtime(const time_t *) throw(); 
# 242
extern "C" tm *gmtime_r(const time_t *__restrict__, tm *__restrict__) throw(); 
# 247
extern "C" tm *localtime_r(const time_t *__restrict__, tm *__restrict__) throw(); 
# 254
extern "C" char *asctime(const tm *) throw(); 
# 257
extern "C" char *ctime(const time_t *) throw(); 
# 265
extern "C" char *asctime_r(const tm *__restrict__, char *__restrict__) throw(); 
# 269
extern "C" char *ctime_r(const time_t *__restrict__, char *__restrict__) throw(); 
# 275
extern "C" { extern char *__tzname[2]; } 
# 276
extern "C" { extern int __daylight; } 
# 277
extern "C" { extern long __timezone; } 
# 282
extern "C" { extern char *tzname[2]; } 
# 286
extern "C" void tzset() throw(); 
# 290
extern "C" { extern int daylight; } 
# 291
extern "C" { extern long timezone; } 
# 297
extern "C" int stime(const time_t *) throw(); 
# 312
extern "C" time_t timegm(tm *) throw(); 
# 315
extern "C" time_t timelocal(tm *) throw(); 
# 318
extern "C" int dysize(int) throw() __attribute__((__const__)); 
# 327
extern "C" int nanosleep(const timespec *, timespec *); 
# 332
extern "C" int clock_getres(clockid_t, timespec *) throw(); 
# 335
extern "C" int clock_gettime(clockid_t, timespec *) throw(); 
# 338
extern "C" int clock_settime(clockid_t, const timespec *) throw(); 
# 346
extern "C" int clock_nanosleep(clockid_t, int, const timespec *, timespec *); 
# 351
extern "C" int clock_getcpuclockid(pid_t, clockid_t *) throw(); 
# 356
extern "C" int timer_create(clockid_t, sigevent *__restrict__, timer_t *__restrict__) throw(); 
# 361
extern "C" int timer_delete(timer_t) throw(); 
# 364
extern "C" int timer_settime(timer_t, int, const itimerspec *__restrict__, itimerspec *__restrict__) throw(); 
# 369
extern "C" int timer_gettime(timer_t, itimerspec *) throw(); 
# 373
extern "C" int timer_getoverrun(timer_t) throw(); 
# 389
extern "C" { extern int getdate_err; } 
# 398
extern "C" tm *getdate(const char *); 
# 412
extern "C" int getdate_r(const char *__restrict__, tm *__restrict__); 
# 38 "/usr/include/string.h" 3
extern "C"  __attribute__((__weak__)) void *memcpy(void *__restrict__, const void *__restrict__, size_t) throw(); 
# 43
extern "C" void *memmove(void *, const void *, size_t) throw(); 
# 51
extern "C" void *memccpy(void *__restrict__, const void *__restrict__, int, size_t) throw(); 
# 59
extern "C"  __attribute__((__weak__)) void *memset(void *, int, size_t) throw(); 
# 62
extern "C" int memcmp(const void *, const void *, size_t) throw() __attribute__((__pure__)); 
# 66
extern "C" void *memchr(const void *, int, size_t) throw() __attribute__((__pure__)); 
# 73
extern "C" void *rawmemchr(const void *, int) throw() __attribute__((__pure__)); 
# 77
extern "C" void *memrchr(const void *, int, size_t) throw() __attribute__((__pure__)); 
# 84
extern "C" char *strcpy(char *__restrict__, const char *__restrict__) throw(); 
# 87
extern "C" char *strncpy(char *__restrict__, const char *__restrict__, size_t) throw(); 
# 92
extern "C" char *strcat(char *__restrict__, const char *__restrict__) throw(); 
# 95
extern "C" char *strncat(char *__restrict__, const char *__restrict__, size_t) throw(); 
# 99
extern "C" int strcmp(const char *, const char *) throw() __attribute__((__pure__)); 
# 102
extern "C" int strncmp(const char *, const char *, size_t) throw() __attribute__((__pure__)); 
# 106
extern "C" int strcoll(const char *, const char *) throw() __attribute__((__pure__)); 
# 109
extern "C" size_t strxfrm(char *__restrict__, const char *__restrict__, size_t) throw(); 
# 121 "/usr/include/string.h" 3
extern "C" int strcoll_l(const char *, const char *, __locale_t) throw() __attribute__((__pure__)); 
# 124
extern "C" size_t strxfrm_l(char *, const char *, size_t, __locale_t) throw(); 
# 130
extern "C" char *strdup(const char *) throw() __attribute__((__malloc__)); 
# 138
extern "C" char *strndup(const char *, size_t) throw() __attribute__((__malloc__)); 
# 167 "/usr/include/string.h" 3
extern "C" char *strchr(const char *, int) throw() __attribute__((__pure__)); 
# 170
extern "C" char *strrchr(const char *, int) throw() __attribute__((__pure__)); 
# 177
extern "C" char *strchrnul(const char *, int) throw() __attribute__((__pure__)); 
# 184
extern "C" size_t strcspn(const char *, const char *) throw() __attribute__((__pure__)); 
# 188
extern "C" size_t strspn(const char *, const char *) throw() __attribute__((__pure__)); 
# 191
extern "C" char *strpbrk(const char *, const char *) throw() __attribute__((__pure__)); 
# 194
extern "C" char *strstr(const char *, const char *) throw() __attribute__((__pure__)); 
# 199
extern "C" char *strtok(char *__restrict__, const char *__restrict__) throw(); 
# 205
extern "C" char *__strtok_r(char *__restrict__, const char *__restrict__, char **__restrict__) throw(); 
# 210
extern "C" char *strtok_r(char *__restrict__, const char *__restrict__, char **__restrict__) throw(); 
# 217
extern "C" char *strcasestr(const char *, const char *) throw() __attribute__((__pure__)); 
# 225
extern "C" void *memmem(const void *, size_t, const void *, size_t) throw() __attribute__((__pure__)); 
# 231
extern "C" void *__mempcpy(void *__restrict__, const void *__restrict__, size_t) throw(); 
# 234
extern "C" void *mempcpy(void *__restrict__, const void *__restrict__, size_t) throw(); 
# 242
extern "C" size_t strlen(const char *) throw() __attribute__((__pure__)); 
# 249
extern "C" size_t strnlen(const char *, size_t) throw() __attribute__((__pure__)); 
# 256
extern "C" char *strerror(int) throw(); 
# 281 "/usr/include/string.h" 3
extern "C" char *strerror_r(int, char *, size_t) throw(); 
# 288
extern "C" char *strerror_l(int, __locale_t) throw(); 
# 294
extern "C" void __bzero(void *, size_t) throw(); 
# 298
extern "C" void bcopy(const void *, void *, size_t) throw(); 
# 302
extern "C" void bzero(void *, size_t) throw(); 
# 305
extern "C" int bcmp(const void *, const void *, size_t) throw() __attribute__((__pure__)); 
# 309
extern "C" char *index(const char *, int) throw() __attribute__((__pure__)); 
# 313
extern "C" char *rindex(const char *, int) throw() __attribute__((__pure__)); 
# 318
extern "C" int ffs(int) throw() __attribute__((__const__)); 
# 323
extern "C" int ffsl(long) throw() __attribute__((__const__)); 
# 325
extern "C" int ffsll(long long) throw() __attribute__((__const__)); 
# 331
extern "C" int strcasecmp(const char *, const char *) throw() __attribute__((__pure__)); 
# 335
extern "C" int strncasecmp(const char *, const char *, size_t) throw() __attribute__((__pure__)); 
# 342
extern "C" int strcasecmp_l(const char *, const char *, __locale_t) throw() __attribute__((__pure__)); 
# 346
extern "C" int strncasecmp_l(const char *, const char *, size_t, __locale_t) throw() __attribute__((__pure__)); 
# 354
extern "C" char *strsep(char **__restrict__, const char *__restrict__) throw(); 
# 361
extern "C" int strverscmp(const char *, const char *) throw() __attribute__((__pure__)); 
# 365
extern "C" char *strsignal(int) throw(); 
# 368
extern "C" char *__stpcpy(char *__restrict__, const char *__restrict__) throw(); 
# 370
extern "C" char *stpcpy(char *__restrict__, const char *__restrict__) throw(); 
# 375
extern "C" char *__stpncpy(char *__restrict__, const char *__restrict__, size_t) throw(); 
# 378
extern "C" char *stpncpy(char *__restrict__, const char *__restrict__, size_t) throw(); 
# 383
extern "C" char *strfry(char *) throw(); 
# 386
extern "C" void *memfrob(void *, size_t) throw(); 
# 393
extern "C" char *basename(const char *) throw(); 
# 56 "/usr/local/cuda/bin/../include/common_functions.h"
extern "C"  __attribute__((__weak__)) clock_t clock() throw(); 
# 59
extern "C"  __attribute__((__weak__)) void *memset(void *, int, size_t) throw(); 
# 62
extern "C"  __attribute__((__weak__)) void *memcpy(void *, const void *, size_t) throw(); 
# 65 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C"  __attribute__((__weak__)) int abs(int) throw() __attribute__((__const__)); 
# 67
extern "C"  __attribute__((__weak__)) long labs(long) throw() __attribute__((__const__)); 
# 69
extern "C"  __attribute__((__weak__)) long long llabs(long long) throw() __attribute__((__const__)); 
# 71
extern "C"  __attribute__((__weak__)) double fabs(double) throw() __attribute__((__const__)); 
# 73
extern "C"  __attribute__((__weak__)) float fabsf(float) throw() __attribute__((__const__)); 
# 76
extern "C"  __attribute__((__weak__)) int min(int, int); 
# 78
extern "C"  __attribute__((__weak__)) unsigned umin(unsigned, unsigned); 
# 80
extern "C"  __attribute__((__weak__)) float fminf(float, float) throw(); 
# 82
extern "C"  __attribute__((__weak__)) double fmin(double, double) throw(); 
# 85
extern "C"  __attribute__((__weak__)) int max(int, int); 
# 87
extern "C"  __attribute__((__weak__)) unsigned umax(unsigned, unsigned); 
# 89
extern "C"  __attribute__((__weak__)) float fmaxf(float, float) throw(); 
# 91
extern "C"  __attribute__((__weak__)) double fmax(double, double) throw(); 
# 94
extern "C"  __attribute__((__weak__)) double sin(double) throw(); 
# 96
extern "C"  __attribute__((__weak__)) float sinf(float) throw(); 
# 99
extern "C"  __attribute__((__weak__)) double cos(double) throw(); 
# 101
extern "C"  __attribute__((__weak__)) float cosf(float) throw(); 
# 104
extern "C"  __attribute__((__weak__)) void sincos(double, double *, double *) throw(); 
# 106
extern "C"  __attribute__((__weak__)) void sincosf(float, float *, float *) throw(); 
# 109
extern "C"  __attribute__((__weak__)) double tan(double) throw(); 
# 111
extern "C"  __attribute__((__weak__)) float tanf(float) throw(); 
# 114
extern "C"  __attribute__((__weak__)) double sqrt(double) throw(); 
# 116
extern "C"  __attribute__((__weak__)) float sqrtf(float) throw(); 
# 119
extern "C"  __attribute__((__weak__)) double rsqrt(double); 
# 121
extern "C"  __attribute__((__weak__)) float rsqrtf(float); 
# 124
extern "C"  __attribute__((__weak__)) double exp2(double) throw(); 
# 126
extern "C"  __attribute__((__weak__)) float exp2f(float) throw(); 
# 129
extern "C"  __attribute__((__weak__)) double exp10(double) throw(); 
# 131
extern "C"  __attribute__((__weak__)) float exp10f(float) throw(); 
# 134
extern "C"  __attribute__((__weak__)) double expm1(double) throw(); 
# 136
extern "C"  __attribute__((__weak__)) float expm1f(float) throw(); 
# 139
extern "C"  __attribute__((__weak__)) double log2(double) throw(); 
# 141
extern "C"  __attribute__((__weak__)) float log2f(float) throw(); 
# 144
extern "C"  __attribute__((__weak__)) double log10(double) throw(); 
# 146
extern "C"  __attribute__((__weak__)) float log10f(float) throw(); 
# 149
extern "C"  __attribute__((__weak__)) double log(double) throw(); 
# 151
extern "C"  __attribute__((__weak__)) float logf(float) throw(); 
# 154
extern "C"  __attribute__((__weak__)) double log1p(double) throw(); 
# 156
extern "C"  __attribute__((__weak__)) float log1pf(float) throw(); 
# 159
extern "C"  __attribute__((__weak__)) double floor(double) throw() __attribute__((__const__)); 
# 161
extern "C"  __attribute__((__weak__)) float floorf(float) throw() __attribute__((__const__)); 
# 164
extern "C"  __attribute__((__weak__)) double exp(double) throw(); 
# 166
extern "C"  __attribute__((__weak__)) float expf(float) throw(); 
# 169
extern "C"  __attribute__((__weak__)) double cosh(double) throw(); 
# 171
extern "C"  __attribute__((__weak__)) float coshf(float) throw(); 
# 174
extern "C"  __attribute__((__weak__)) double sinh(double) throw(); 
# 176
extern "C"  __attribute__((__weak__)) float sinhf(float) throw(); 
# 179
extern "C"  __attribute__((__weak__)) double tanh(double) throw(); 
# 181
extern "C"  __attribute__((__weak__)) float tanhf(float) throw(); 
# 184
extern "C"  __attribute__((__weak__)) double acosh(double) throw(); 
# 186
extern "C"  __attribute__((__weak__)) float acoshf(float) throw(); 
# 189
extern "C"  __attribute__((__weak__)) double asinh(double) throw(); 
# 191
extern "C"  __attribute__((__weak__)) float asinhf(float) throw(); 
# 194
extern "C"  __attribute__((__weak__)) double atanh(double) throw(); 
# 196
extern "C"  __attribute__((__weak__)) float atanhf(float) throw(); 
# 199
extern "C"  __attribute__((__weak__)) double ldexp(double, int) throw(); 
# 201
extern "C"  __attribute__((__weak__)) float ldexpf(float, int) throw(); 
# 204
extern "C"  __attribute__((__weak__)) double logb(double) throw(); 
# 206
extern "C"  __attribute__((__weak__)) float logbf(float) throw(); 
# 209
extern "C"  __attribute__((__weak__)) int ilogb(double) throw(); 
# 211
extern "C"  __attribute__((__weak__)) int ilogbf(float) throw(); 
# 214
extern "C"  __attribute__((__weak__)) double scalbn(double, int) throw(); 
# 216
extern "C"  __attribute__((__weak__)) float scalbnf(float, int) throw(); 
# 219
extern "C"  __attribute__((__weak__)) double scalbln(double, long) throw(); 
# 221
extern "C"  __attribute__((__weak__)) float scalblnf(float, long) throw(); 
# 224
extern "C"  __attribute__((__weak__)) double frexp(double, int *) throw(); 
# 226
extern "C"  __attribute__((__weak__)) float frexpf(float, int *) throw(); 
# 229
extern "C"  __attribute__((__weak__)) double round(double) throw() __attribute__((__const__)); 
# 231
extern "C"  __attribute__((__weak__)) float roundf(float) throw() __attribute__((__const__)); 
# 234
extern "C"  __attribute__((__weak__)) long lround(double) throw(); 
# 236
extern "C"  __attribute__((__weak__)) long lroundf(float) throw(); 
# 239
extern "C"  __attribute__((__weak__)) long long llround(double) throw(); 
# 241
extern "C"  __attribute__((__weak__)) long long llroundf(float) throw(); 
# 244
extern "C"  __attribute__((__weak__)) double rint(double) throw(); 
# 246
extern "C"  __attribute__((__weak__)) float rintf(float) throw(); 
# 249
extern "C"  __attribute__((__weak__)) long lrint(double) throw(); 
# 251
extern "C"  __attribute__((__weak__)) long lrintf(float) throw(); 
# 254
extern "C"  __attribute__((__weak__)) long long llrint(double) throw(); 
# 256
extern "C"  __attribute__((__weak__)) long long llrintf(float) throw(); 
# 259
extern "C"  __attribute__((__weak__)) double nearbyint(double) throw(); 
# 261
extern "C"  __attribute__((__weak__)) float nearbyintf(float) throw(); 
# 264
extern "C"  __attribute__((__weak__)) double ceil(double) throw() __attribute__((__const__)); 
# 266
extern "C"  __attribute__((__weak__)) float ceilf(float) throw() __attribute__((__const__)); 
# 269
extern "C"  __attribute__((__weak__)) double trunc(double) throw() __attribute__((__const__)); 
# 271
extern "C"  __attribute__((__weak__)) float truncf(float) throw() __attribute__((__const__)); 
# 274
extern "C"  __attribute__((__weak__)) double fdim(double, double) throw(); 
# 276
extern "C"  __attribute__((__weak__)) float fdimf(float, float) throw(); 
# 279
extern "C"  __attribute__((__weak__)) double atan2(double, double) throw(); 
# 281
extern "C"  __attribute__((__weak__)) float atan2f(float, float) throw(); 
# 284
extern "C"  __attribute__((__weak__)) double atan(double) throw(); 
# 286
extern "C"  __attribute__((__weak__)) float atanf(float) throw(); 
# 289
extern "C"  __attribute__((__weak__)) double asin(double) throw(); 
# 291
extern "C"  __attribute__((__weak__)) float asinf(float) throw(); 
# 294
extern "C"  __attribute__((__weak__)) double acos(double) throw(); 
# 296
extern "C"  __attribute__((__weak__)) float acosf(float) throw(); 
# 299
extern "C"  __attribute__((__weak__)) double hypot(double, double) throw(); 
# 301
extern "C"  __attribute__((__weak__)) float hypotf(float, float) throw(); 
# 304
extern "C"  __attribute__((__weak__)) double cbrt(double) throw(); 
# 306
extern "C"  __attribute__((__weak__)) float cbrtf(float) throw(); 
# 309
extern "C"  __attribute__((__weak__)) double pow(double, double) throw(); 
# 311
extern "C"  __attribute__((__weak__)) float powf(float, float) throw(); 
# 314
extern "C"  __attribute__((__weak__)) double modf(double, double *) throw(); 
# 316
extern "C"  __attribute__((__weak__)) float modff(float, float *) throw(); 
# 319
extern "C"  __attribute__((__weak__)) double fmod(double, double) throw(); 
# 321
extern "C"  __attribute__((__weak__)) float fmodf(float, float) throw(); 
# 324
extern "C"  __attribute__((__weak__)) double remainder(double, double) throw(); 
# 326
extern "C"  __attribute__((__weak__)) float remainderf(float, float) throw(); 
# 329
extern "C"  __attribute__((__weak__)) double remquo(double, double, int *) throw(); 
# 331
extern "C"  __attribute__((__weak__)) float remquof(float, float, int *) throw(); 
# 334
extern "C"  __attribute__((__weak__)) double erf(double) throw(); 
# 336
extern "C"  __attribute__((__weak__)) float erff(float) throw(); 
# 339
extern "C"  __attribute__((__weak__)) double erfc(double) throw(); 
# 341
extern "C"  __attribute__((__weak__)) float erfcf(float) throw(); 
# 344
extern "C"  __attribute__((__weak__)) double lgamma(double) throw(); 
# 346
extern "C"  __attribute__((__weak__)) float lgammaf(float) throw(); 
# 349
extern "C"  __attribute__((__weak__)) double tgamma(double) throw(); 
# 351
extern "C"  __attribute__((__weak__)) float tgammaf(float) throw(); 
# 354
extern "C"  __attribute__((__weak__)) double copysign(double, double) throw() __attribute__((__const__)); 
# 356
extern "C"  __attribute__((__weak__)) float copysignf(float, float) throw() __attribute__((__const__)); 
# 359
extern "C"  __attribute__((__weak__)) double nextafter(double, double) throw() __attribute__((__const__)); 
# 361
extern "C"  __attribute__((__weak__)) float nextafterf(float, float) throw() __attribute__((__const__)); 
# 364
extern "C"  __attribute__((__weak__)) double nan(const char *) throw() __attribute__((__const__)); 
# 366
extern "C"  __attribute__((__weak__)) float nanf(const char *) throw() __attribute__((__const__)); 
# 369
extern "C"  __attribute__((__weak__)) int __isinf(double) throw() __attribute__((__const__)); 
# 371
extern "C"  __attribute__((__weak__)) int __isinff(float) throw() __attribute__((__const__)); 
# 374
extern "C"  __attribute__((__weak__)) int __isnan(double) throw() __attribute__((__const__)); 
# 376
extern "C"  __attribute__((__weak__)) int __isnanf(float) throw() __attribute__((__const__)); 
# 390 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C"  __attribute__((__weak__)) int __finite(double) throw() __attribute__((__const__)); 
# 392
extern "C"  __attribute__((__weak__)) int __finitef(float) throw() __attribute__((__const__)); 
# 394
extern "C"  __attribute__((__weak__)) int __signbit(double) throw() __attribute__((__const__)); 
# 399
extern "C"  __attribute__((__weak__)) int __signbitf(float) throw() __attribute__((__const__)); 
# 402
extern "C"  __attribute__((__weak__)) double fma(double, double, double) throw(); 
# 404
extern "C"  __attribute__((__weak__)) float fmaf(float, float, float) throw(); 
# 38 "/usr/include/bits/mathdef.h" 3
extern "C" { typedef long double float_t; }
# 40
extern "C" { typedef long double double_t; }
# 55 "/usr/include/bits/mathcalls.h" 3
extern "C"  __attribute__((__weak__)) double acos(double) throw(); extern "C" double __acos(double) throw(); 
# 57
extern "C"  __attribute__((__weak__)) double asin(double) throw(); extern "C" double __asin(double) throw(); 
# 59
extern "C"  __attribute__((__weak__)) double atan(double) throw(); extern "C" double __atan(double) throw(); 
# 61
extern "C"  __attribute__((__weak__)) double atan2(double, double) throw(); extern "C" double __atan2(double, double) throw(); 
# 64
extern "C"  __attribute__((__weak__)) double cos(double) throw(); extern "C" double __cos(double) throw(); 
# 66
extern "C"  __attribute__((__weak__)) double sin(double) throw(); extern "C" double __sin(double) throw(); 
# 68
extern "C"  __attribute__((__weak__)) double tan(double) throw(); extern "C" double __tan(double) throw(); 
# 73
extern "C"  __attribute__((__weak__)) double cosh(double) throw(); extern "C" double __cosh(double) throw(); 
# 75
extern "C"  __attribute__((__weak__)) double sinh(double) throw(); extern "C" double __sinh(double) throw(); 
# 77
extern "C"  __attribute__((__weak__)) double tanh(double) throw(); extern "C" double __tanh(double) throw(); 
# 82
extern "C"  __attribute__((__weak__)) void sincos(double, double *, double *) throw(); extern "C" void __sincos(double, double *, double *) throw(); 
# 89
extern "C"  __attribute__((__weak__)) double acosh(double) throw(); extern "C" double __acosh(double) throw(); 
# 91
extern "C"  __attribute__((__weak__)) double asinh(double) throw(); extern "C" double __asinh(double) throw(); 
# 93
extern "C"  __attribute__((__weak__)) double atanh(double) throw(); extern "C" double __atanh(double) throw(); 
# 101
extern "C"  __attribute__((__weak__)) double exp(double) throw(); extern "C" double __exp(double) throw(); 
# 104
extern "C"  __attribute__((__weak__)) double frexp(double, int *) throw(); extern "C" double __frexp(double, int *) throw(); 
# 107
extern "C"  __attribute__((__weak__)) double ldexp(double, int) throw(); extern "C" double __ldexp(double, int) throw(); 
# 110
extern "C"  __attribute__((__weak__)) double log(double) throw(); extern "C" double __log(double) throw(); 
# 113
extern "C"  __attribute__((__weak__)) double log10(double) throw(); extern "C" double __log10(double) throw(); 
# 116
extern "C"  __attribute__((__weak__)) double modf(double, double *) throw(); extern "C" double __modf(double, double *) throw(); 
# 121
extern "C"  __attribute__((__weak__)) double exp10(double) throw(); extern "C" double __exp10(double) throw(); 
# 123
extern "C" double pow10(double) throw(); extern "C" double __pow10(double) throw(); 
# 129
extern "C"  __attribute__((__weak__)) double expm1(double) throw(); extern "C" double __expm1(double) throw(); 
# 132
extern "C"  __attribute__((__weak__)) double log1p(double) throw(); extern "C" double __log1p(double) throw(); 
# 135
extern "C"  __attribute__((__weak__)) double logb(double) throw(); extern "C" double __logb(double) throw(); 
# 142
extern "C"  __attribute__((__weak__)) double exp2(double) throw(); extern "C" double __exp2(double) throw(); 
# 145
extern "C"  __attribute__((__weak__)) double log2(double) throw(); extern "C" double __log2(double) throw(); 
# 154
extern "C"  __attribute__((__weak__)) double pow(double, double) throw(); extern "C" double __pow(double, double) throw(); 
# 157
extern "C"  __attribute__((__weak__)) double sqrt(double) throw(); extern "C" double __sqrt(double) throw(); 
# 163
extern "C"  __attribute__((__weak__)) double hypot(double, double) throw(); extern "C" double __hypot(double, double) throw(); 
# 170
extern "C"  __attribute__((__weak__)) double cbrt(double) throw(); extern "C" double __cbrt(double) throw(); 
# 179
extern "C"  __attribute__((__weak__)) double ceil(double) throw() __attribute__((__const__)); extern "C" double __ceil(double) throw() __attribute__((__const__)); 
# 182
extern "C"  __attribute__((__weak__)) double fabs(double) throw() __attribute__((__const__)); extern "C" double __fabs(double) throw() __attribute__((__const__)); 
# 185
extern "C"  __attribute__((__weak__)) double floor(double) throw() __attribute__((__const__)); extern "C" double __floor(double) throw() __attribute__((__const__)); 
# 188
extern "C"  __attribute__((__weak__)) double fmod(double, double) throw(); extern "C" double __fmod(double, double) throw(); 
# 193
extern "C"  __attribute__((__weak__)) int __isinf(double) throw() __attribute__((__const__)); 
# 196
extern "C"  __attribute__((__weak__)) int __finite(double) throw() __attribute__((__const__)); 
# 202
extern "C" int isinf(double) throw() __attribute__((__const__)); 
# 205
extern "C" int finite(double) throw() __attribute__((__const__)); 
# 208
extern "C" double drem(double, double) throw(); extern "C" double __drem(double, double) throw(); 
# 212
extern "C" double significand(double) throw(); extern "C" double __significand(double) throw(); 
# 218
extern "C"  __attribute__((__weak__)) double copysign(double, double) throw() __attribute__((__const__)); extern "C" double __copysign(double, double) throw() __attribute__((__const__)); 
# 225
extern "C"  __attribute__((__weak__)) double nan(const char *) throw() __attribute__((__const__)); extern "C" double __nan(const char *) throw() __attribute__((__const__)); 
# 231
extern "C"  __attribute__((__weak__)) int __isnan(double) throw() __attribute__((__const__)); 
# 235
extern "C" int isnan(double) throw() __attribute__((__const__)); 
# 238
extern "C" double j0(double) throw(); extern "C" double __j0(double) throw(); 
# 239
extern "C" double j1(double) throw(); extern "C" double __j1(double) throw(); 
# 240
extern "C" double jn(int, double) throw(); extern "C" double __jn(int, double) throw(); 
# 241
extern "C" double y0(double) throw(); extern "C" double __y0(double) throw(); 
# 242
extern "C" double y1(double) throw(); extern "C" double __y1(double) throw(); 
# 243
extern "C" double yn(int, double) throw(); extern "C" double __yn(int, double) throw(); 
# 250
extern "C"  __attribute__((__weak__)) double erf(double) throw(); extern "C" double __erf(double) throw(); 
# 251
extern "C"  __attribute__((__weak__)) double erfc(double) throw(); extern "C" double __erfc(double) throw(); 
# 252
extern "C"  __attribute__((__weak__)) double lgamma(double) throw(); extern "C" double __lgamma(double) throw(); 
# 259
extern "C"  __attribute__((__weak__)) double tgamma(double) throw(); extern "C" double __tgamma(double) throw(); 
# 265
extern "C" double gamma(double) throw(); extern "C" double __gamma(double) throw(); 
# 272
extern "C" double lgamma_r(double, int *) throw(); extern "C" double __lgamma_r(double, int *) throw(); 
# 280
extern "C"  __attribute__((__weak__)) double rint(double) throw(); extern "C" double __rint(double) throw(); 
# 283
extern "C"  __attribute__((__weak__)) double nextafter(double, double) throw() __attribute__((__const__)); extern "C" double __nextafter(double, double) throw() __attribute__((__const__)); 
# 285
extern "C" double nexttoward(double, long double) throw() __attribute__((__const__)); extern "C" double __nexttoward(double, long double) throw() __attribute__((__const__)); 
# 289
extern "C"  __attribute__((__weak__)) double remainder(double, double) throw(); extern "C" double __remainder(double, double) throw(); 
# 293
extern "C"  __attribute__((__weak__)) double scalbn(double, int) throw(); extern "C" double __scalbn(double, int) throw(); 
# 297
extern "C"  __attribute__((__weak__)) int ilogb(double) throw(); extern "C" int __ilogb(double) throw(); 
# 302
extern "C"  __attribute__((__weak__)) double scalbln(double, long) throw(); extern "C" double __scalbln(double, long) throw(); 
# 306
extern "C"  __attribute__((__weak__)) double nearbyint(double) throw(); extern "C" double __nearbyint(double) throw(); 
# 310
extern "C"  __attribute__((__weak__)) double round(double) throw() __attribute__((__const__)); extern "C" double __round(double) throw() __attribute__((__const__)); 
# 314
extern "C"  __attribute__((__weak__)) double trunc(double) throw() __attribute__((__const__)); extern "C" double __trunc(double) throw() __attribute__((__const__)); 
# 319
extern "C"  __attribute__((__weak__)) double remquo(double, double, int *) throw(); extern "C" double __remquo(double, double, int *) throw(); 
# 326
extern "C"  __attribute__((__weak__)) long lrint(double) throw(); extern "C" long __lrint(double) throw(); 
# 327
extern "C"  __attribute__((__weak__)) long long llrint(double) throw(); extern "C" long long __llrint(double) throw(); 
# 331
extern "C"  __attribute__((__weak__)) long lround(double) throw(); extern "C" long __lround(double) throw(); 
# 332
extern "C"  __attribute__((__weak__)) long long llround(double) throw(); extern "C" long long __llround(double) throw(); 
# 336
extern "C"  __attribute__((__weak__)) double fdim(double, double) throw(); extern "C" double __fdim(double, double) throw(); 
# 339
extern "C"  __attribute__((__weak__)) double fmax(double, double) throw(); extern "C" double __fmax(double, double) throw(); 
# 342
extern "C"  __attribute__((__weak__)) double fmin(double, double) throw(); extern "C" double __fmin(double, double) throw(); 
# 346
extern "C" int __fpclassify(double) throw() __attribute__((__const__)); 
# 350
extern "C"  __attribute__((__weak__)) int __signbit(double) throw() __attribute__((__const__)); 
# 355
extern "C"  __attribute__((__weak__)) double fma(double, double, double) throw(); extern "C" double __fma(double, double, double) throw(); 
# 364
extern "C" double scalb(double, double) throw(); extern "C" double __scalb(double, double) throw(); 
# 55 "/usr/include/bits/mathcalls.h" 3
extern "C"  __attribute__((__weak__)) float acosf(float) throw(); extern "C" float __acosf(float) throw(); 
# 57
extern "C"  __attribute__((__weak__)) float asinf(float) throw(); extern "C" float __asinf(float) throw(); 
# 59
extern "C"  __attribute__((__weak__)) float atanf(float) throw(); extern "C" float __atanf(float) throw(); 
# 61
extern "C"  __attribute__((__weak__)) float atan2f(float, float) throw(); extern "C" float __atan2f(float, float) throw(); 
# 64
extern "C"  __attribute__((__weak__)) float cosf(float) throw(); 
# 66
extern "C"  __attribute__((__weak__)) float sinf(float) throw(); 
# 68
extern "C"  __attribute__((__weak__)) float tanf(float) throw(); 
# 73
extern "C"  __attribute__((__weak__)) float coshf(float) throw(); extern "C" float __coshf(float) throw(); 
# 75
extern "C"  __attribute__((__weak__)) float sinhf(float) throw(); extern "C" float __sinhf(float) throw(); 
# 77
extern "C"  __attribute__((__weak__)) float tanhf(float) throw(); extern "C" float __tanhf(float) throw(); 
# 82
extern "C"  __attribute__((__weak__)) void sincosf(float, float *, float *) throw(); 
# 89
extern "C"  __attribute__((__weak__)) float acoshf(float) throw(); extern "C" float __acoshf(float) throw(); 
# 91
extern "C"  __attribute__((__weak__)) float asinhf(float) throw(); extern "C" float __asinhf(float) throw(); 
# 93
extern "C"  __attribute__((__weak__)) float atanhf(float) throw(); extern "C" float __atanhf(float) throw(); 
# 101
extern "C"  __attribute__((__weak__)) float expf(float) throw(); 
# 104
extern "C"  __attribute__((__weak__)) float frexpf(float, int *) throw(); extern "C" float __frexpf(float, int *) throw(); 
# 107
extern "C"  __attribute__((__weak__)) float ldexpf(float, int) throw(); extern "C" float __ldexpf(float, int) throw(); 
# 110
extern "C"  __attribute__((__weak__)) float logf(float) throw(); 
# 113
extern "C"  __attribute__((__weak__)) float log10f(float) throw(); 
# 116
extern "C"  __attribute__((__weak__)) float modff(float, float *) throw(); extern "C" float __modff(float, float *) throw(); 
# 121
extern "C"  __attribute__((__weak__)) float exp10f(float) throw(); 
# 123
extern "C" float pow10f(float) throw(); extern "C" float __pow10f(float) throw(); 
# 129
extern "C"  __attribute__((__weak__)) float expm1f(float) throw(); extern "C" float __expm1f(float) throw(); 
# 132
extern "C"  __attribute__((__weak__)) float log1pf(float) throw(); extern "C" float __log1pf(float) throw(); 
# 135
extern "C"  __attribute__((__weak__)) float logbf(float) throw(); extern "C" float __logbf(float) throw(); 
# 142
extern "C"  __attribute__((__weak__)) float exp2f(float) throw(); extern "C" float __exp2f(float) throw(); 
# 145
extern "C"  __attribute__((__weak__)) float log2f(float) throw(); 
# 154
extern "C"  __attribute__((__weak__)) float powf(float, float) throw(); 
# 157
extern "C"  __attribute__((__weak__)) float sqrtf(float) throw(); extern "C" float __sqrtf(float) throw(); 
# 163
extern "C"  __attribute__((__weak__)) float hypotf(float, float) throw(); extern "C" float __hypotf(float, float) throw(); 
# 170
extern "C"  __attribute__((__weak__)) float cbrtf(float) throw(); extern "C" float __cbrtf(float) throw(); 
# 179
extern "C"  __attribute__((__weak__)) float ceilf(float) throw() __attribute__((__const__)); extern "C" float __ceilf(float) throw() __attribute__((__const__)); 
# 182
extern "C"  __attribute__((__weak__)) float fabsf(float) throw() __attribute__((__const__)); extern "C" float __fabsf(float) throw() __attribute__((__const__)); 
# 185
extern "C"  __attribute__((__weak__)) float floorf(float) throw() __attribute__((__const__)); extern "C" float __floorf(float) throw() __attribute__((__const__)); 
# 188
extern "C"  __attribute__((__weak__)) float fmodf(float, float) throw(); extern "C" float __fmodf(float, float) throw(); 
# 193
extern "C"  __attribute__((__weak__)) int __isinff(float) throw() __attribute__((__const__)); 
# 196
extern "C"  __attribute__((__weak__)) int __finitef(float) throw() __attribute__((__const__)); 
# 202
extern "C" int isinff(float) throw() __attribute__((__const__)); 
# 205
extern "C" int finitef(float) throw() __attribute__((__const__)); 
# 208
extern "C" float dremf(float, float) throw(); extern "C" float __dremf(float, float) throw(); 
# 212
extern "C" float significandf(float) throw(); extern "C" float __significandf(float) throw(); 
# 218
extern "C"  __attribute__((__weak__)) float copysignf(float, float) throw() __attribute__((__const__)); extern "C" float __copysignf(float, float) throw() __attribute__((__const__)); 
# 225
extern "C"  __attribute__((__weak__)) float nanf(const char *) throw() __attribute__((__const__)); extern "C" float __nanf(const char *) throw() __attribute__((__const__)); 
# 231
extern "C"  __attribute__((__weak__)) int __isnanf(float) throw() __attribute__((__const__)); 
# 235
extern "C" int isnanf(float) throw() __attribute__((__const__)); 
# 238
extern "C" float j0f(float) throw(); extern "C" float __j0f(float) throw(); 
# 239
extern "C" float j1f(float) throw(); extern "C" float __j1f(float) throw(); 
# 240
extern "C" float jnf(int, float) throw(); extern "C" float __jnf(int, float) throw(); 
# 241
extern "C" float y0f(float) throw(); extern "C" float __y0f(float) throw(); 
# 242
extern "C" float y1f(float) throw(); extern "C" float __y1f(float) throw(); 
# 243
extern "C" float ynf(int, float) throw(); extern "C" float __ynf(int, float) throw(); 
# 250
extern "C"  __attribute__((__weak__)) float erff(float) throw(); extern "C" float __erff(float) throw(); 
# 251
extern "C"  __attribute__((__weak__)) float erfcf(float) throw(); extern "C" float __erfcf(float) throw(); 
# 252
extern "C"  __attribute__((__weak__)) float lgammaf(float) throw(); extern "C" float __lgammaf(float) throw(); 
# 259
extern "C"  __attribute__((__weak__)) float tgammaf(float) throw(); extern "C" float __tgammaf(float) throw(); 
# 265
extern "C" float gammaf(float) throw(); extern "C" float __gammaf(float) throw(); 
# 272
extern "C" float lgammaf_r(float, int *) throw(); extern "C" float __lgammaf_r(float, int *) throw(); 
# 280
extern "C"  __attribute__((__weak__)) float rintf(float) throw(); extern "C" float __rintf(float) throw(); 
# 283
extern "C"  __attribute__((__weak__)) float nextafterf(float, float) throw() __attribute__((__const__)); extern "C" float __nextafterf(float, float) throw() __attribute__((__const__)); 
# 285
extern "C" float nexttowardf(float, long double) throw() __attribute__((__const__)); extern "C" float __nexttowardf(float, long double) throw() __attribute__((__const__)); 
# 289
extern "C"  __attribute__((__weak__)) float remainderf(float, float) throw(); extern "C" float __remainderf(float, float) throw(); 
# 293
extern "C"  __attribute__((__weak__)) float scalbnf(float, int) throw(); extern "C" float __scalbnf(float, int) throw(); 
# 297
extern "C"  __attribute__((__weak__)) int ilogbf(float) throw(); extern "C" int __ilogbf(float) throw(); 
# 302
extern "C"  __attribute__((__weak__)) float scalblnf(float, long) throw(); extern "C" float __scalblnf(float, long) throw(); 
# 306
extern "C"  __attribute__((__weak__)) float nearbyintf(float) throw(); extern "C" float __nearbyintf(float) throw(); 
# 310
extern "C"  __attribute__((__weak__)) float roundf(float) throw() __attribute__((__const__)); extern "C" float __roundf(float) throw() __attribute__((__const__)); 
# 314
extern "C"  __attribute__((__weak__)) float truncf(float) throw() __attribute__((__const__)); extern "C" float __truncf(float) throw() __attribute__((__const__)); 
# 319
extern "C"  __attribute__((__weak__)) float remquof(float, float, int *) throw(); extern "C" float __remquof(float, float, int *) throw(); 
# 326
extern "C"  __attribute__((__weak__)) long lrintf(float) throw(); extern "C" long __lrintf(float) throw(); 
# 327
extern "C"  __attribute__((__weak__)) long long llrintf(float) throw(); extern "C" long long __llrintf(float) throw(); 
# 331
extern "C"  __attribute__((__weak__)) long lroundf(float) throw(); extern "C" long __lroundf(float) throw(); 
# 332
extern "C"  __attribute__((__weak__)) long long llroundf(float) throw(); extern "C" long long __llroundf(float) throw(); 
# 336
extern "C"  __attribute__((__weak__)) float fdimf(float, float) throw(); extern "C" float __fdimf(float, float) throw(); 
# 339
extern "C"  __attribute__((__weak__)) float fmaxf(float, float) throw(); extern "C" float __fmaxf(float, float) throw(); 
# 342
extern "C"  __attribute__((__weak__)) float fminf(float, float) throw(); extern "C" float __fminf(float, float) throw(); 
# 346
extern "C" int __fpclassifyf(float) throw() __attribute__((__const__)); 
# 350
extern "C"  __attribute__((__weak__)) int __signbitf(float) throw() __attribute__((__const__)); 
# 355
extern "C"  __attribute__((__weak__)) float fmaf(float, float, float) throw(); extern "C" float __fmaf(float, float, float) throw(); 
# 364
extern "C" float scalbf(float, float) throw(); extern "C" float __scalbf(float, float) throw(); 
# 55 "/usr/include/bits/mathcalls.h" 3
extern "C" long double acosl(long double) throw(); extern "C" long double __acosl(long double) throw(); 
# 57
extern "C" long double asinl(long double) throw(); extern "C" long double __asinl(long double) throw(); 
# 59
extern "C" long double atanl(long double) throw(); extern "C" long double __atanl(long double) throw(); 
# 61
extern "C" long double atan2l(long double, long double) throw(); extern "C" long double __atan2l(long double, long double) throw(); 
# 64
extern "C" long double cosl(long double) throw(); extern "C" long double __cosl(long double) throw(); 
# 66
extern "C" long double sinl(long double) throw(); extern "C" long double __sinl(long double) throw(); 
# 68
extern "C" long double tanl(long double) throw(); extern "C" long double __tanl(long double) throw(); 
# 73
extern "C" long double coshl(long double) throw(); extern "C" long double __coshl(long double) throw(); 
# 75
extern "C" long double sinhl(long double) throw(); extern "C" long double __sinhl(long double) throw(); 
# 77
extern "C" long double tanhl(long double) throw(); extern "C" long double __tanhl(long double) throw(); 
# 82
extern "C" void sincosl(long double, long double *, long double *) throw(); extern "C" void __sincosl(long double, long double *, long double *) throw(); 
# 89
extern "C" long double acoshl(long double) throw(); extern "C" long double __acoshl(long double) throw(); 
# 91
extern "C" long double asinhl(long double) throw(); extern "C" long double __asinhl(long double) throw(); 
# 93
extern "C" long double atanhl(long double) throw(); extern "C" long double __atanhl(long double) throw(); 
# 101
extern "C" long double expl(long double) throw(); extern "C" long double __expl(long double) throw(); 
# 104
extern "C" long double frexpl(long double, int *) throw(); extern "C" long double __frexpl(long double, int *) throw(); 
# 107
extern "C" long double ldexpl(long double, int) throw(); extern "C" long double __ldexpl(long double, int) throw(); 
# 110
extern "C" long double logl(long double) throw(); extern "C" long double __logl(long double) throw(); 
# 113
extern "C" long double log10l(long double) throw(); extern "C" long double __log10l(long double) throw(); 
# 116
extern "C" long double modfl(long double, long double *) throw(); extern "C" long double __modfl(long double, long double *) throw(); 
# 121
extern "C" long double exp10l(long double) throw(); extern "C" long double __exp10l(long double) throw(); 
# 123
extern "C" long double pow10l(long double) throw(); extern "C" long double __pow10l(long double) throw(); 
# 129
extern "C" long double expm1l(long double) throw(); extern "C" long double __expm1l(long double) throw(); 
# 132
extern "C" long double log1pl(long double) throw(); extern "C" long double __log1pl(long double) throw(); 
# 135
extern "C" long double logbl(long double) throw(); extern "C" long double __logbl(long double) throw(); 
# 142
extern "C" long double exp2l(long double) throw(); extern "C" long double __exp2l(long double) throw(); 
# 145
extern "C" long double log2l(long double) throw(); extern "C" long double __log2l(long double) throw(); 
# 154
extern "C" long double powl(long double, long double) throw(); extern "C" long double __powl(long double, long double) throw(); 
# 157
extern "C" long double sqrtl(long double) throw(); extern "C" long double __sqrtl(long double) throw(); 
# 163
extern "C" long double hypotl(long double, long double) throw(); extern "C" long double __hypotl(long double, long double) throw(); 
# 170
extern "C" long double cbrtl(long double) throw(); extern "C" long double __cbrtl(long double) throw(); 
# 179
extern "C" long double ceill(long double) throw() __attribute__((__const__)); extern "C" long double __ceill(long double) throw() __attribute__((__const__)); 
# 182
extern "C" long double fabsl(long double) throw() __attribute__((__const__)); extern "C" long double __fabsl(long double) throw() __attribute__((__const__)); 
# 185
extern "C" long double floorl(long double) throw() __attribute__((__const__)); extern "C" long double __floorl(long double) throw() __attribute__((__const__)); 
# 188
extern "C" long double fmodl(long double, long double) throw(); extern "C" long double __fmodl(long double, long double) throw(); 
# 193
extern "C"  __attribute__((__weak__)) int __isinfl(long double) throw() __attribute__((__const__)); 
# 196
extern "C"  __attribute__((__weak__)) int __finitel(long double) throw() __attribute__((__const__)); 
# 202
extern "C" int isinfl(long double) throw() __attribute__((__const__)); 
# 205
extern "C" int finitel(long double) throw() __attribute__((__const__)); 
# 208
extern "C" long double dreml(long double, long double) throw(); extern "C" long double __dreml(long double, long double) throw(); 
# 212
extern "C" long double significandl(long double) throw(); extern "C" long double __significandl(long double) throw(); 
# 218
extern "C" long double copysignl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __copysignl(long double, long double) throw() __attribute__((__const__)); 
# 225
extern "C" long double nanl(const char *) throw() __attribute__((__const__)); extern "C" long double __nanl(const char *) throw() __attribute__((__const__)); 
# 231
extern "C"  __attribute__((__weak__)) int __isnanl(long double) throw() __attribute__((__const__)); 
# 235
extern "C" int isnanl(long double) throw() __attribute__((__const__)); 
# 238
extern "C" long double j0l(long double) throw(); extern "C" long double __j0l(long double) throw(); 
# 239
extern "C" long double j1l(long double) throw(); extern "C" long double __j1l(long double) throw(); 
# 240
extern "C" long double jnl(int, long double) throw(); extern "C" long double __jnl(int, long double) throw(); 
# 241
extern "C" long double y0l(long double) throw(); extern "C" long double __y0l(long double) throw(); 
# 242
extern "C" long double y1l(long double) throw(); extern "C" long double __y1l(long double) throw(); 
# 243
extern "C" long double ynl(int, long double) throw(); extern "C" long double __ynl(int, long double) throw(); 
# 250
extern "C" long double erfl(long double) throw(); extern "C" long double __erfl(long double) throw(); 
# 251
extern "C" long double erfcl(long double) throw(); extern "C" long double __erfcl(long double) throw(); 
# 252
extern "C" long double lgammal(long double) throw(); extern "C" long double __lgammal(long double) throw(); 
# 259
extern "C" long double tgammal(long double) throw(); extern "C" long double __tgammal(long double) throw(); 
# 265
extern "C" long double gammal(long double) throw(); extern "C" long double __gammal(long double) throw(); 
# 272
extern "C" long double lgammal_r(long double, int *) throw(); extern "C" long double __lgammal_r(long double, int *) throw(); 
# 280
extern "C" long double rintl(long double) throw(); extern "C" long double __rintl(long double) throw(); 
# 283
extern "C" long double nextafterl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __nextafterl(long double, long double) throw() __attribute__((__const__)); 
# 285
extern "C" long double nexttowardl(long double, long double) throw() __attribute__((__const__)); extern "C" long double __nexttowardl(long double, long double) throw() __attribute__((__const__)); 
# 289
extern "C" long double remainderl(long double, long double) throw(); extern "C" long double __remainderl(long double, long double) throw(); 
# 293
extern "C" long double scalbnl(long double, int) throw(); extern "C" long double __scalbnl(long double, int) throw(); 
# 297
extern "C" int ilogbl(long double) throw(); extern "C" int __ilogbl(long double) throw(); 
# 302
extern "C" long double scalblnl(long double, long) throw(); extern "C" long double __scalblnl(long double, long) throw(); 
# 306
extern "C" long double nearbyintl(long double) throw(); extern "C" long double __nearbyintl(long double) throw(); 
# 310
extern "C" long double roundl(long double) throw() __attribute__((__const__)); extern "C" long double __roundl(long double) throw() __attribute__((__const__)); 
# 314
extern "C" long double truncl(long double) throw() __attribute__((__const__)); extern "C" long double __truncl(long double) throw() __attribute__((__const__)); 
# 319
extern "C" long double remquol(long double, long double, int *) throw(); extern "C" long double __remquol(long double, long double, int *) throw(); 
# 326
extern "C" long lrintl(long double) throw(); extern "C" long __lrintl(long double) throw(); 
# 327
extern "C" long long llrintl(long double) throw(); extern "C" long long __llrintl(long double) throw(); 
# 331
extern "C" long lroundl(long double) throw(); extern "C" long __lroundl(long double) throw(); 
# 332
extern "C" long long llroundl(long double) throw(); extern "C" long long __llroundl(long double) throw(); 
# 336
extern "C" long double fdiml(long double, long double) throw(); extern "C" long double __fdiml(long double, long double) throw(); 
# 339
extern "C" long double fmaxl(long double, long double) throw(); extern "C" long double __fmaxl(long double, long double) throw(); 
# 342
extern "C" long double fminl(long double, long double) throw(); extern "C" long double __fminl(long double, long double) throw(); 
# 346
extern "C" int __fpclassifyl(long double) throw() __attribute__((__const__)); 
# 350
extern "C"  __attribute__((__weak__)) int __signbitl(long double) throw() __attribute__((__const__)); 
# 355
extern "C" long double fmal(long double, long double, long double) throw(); extern "C" long double __fmal(long double, long double, long double) throw(); 
# 364
extern "C" long double scalbl(long double, long double) throw(); extern "C" long double __scalbl(long double, long double) throw(); 
# 157 "/usr/include/math.h" 3
extern "C" { extern int signgam; } 
# 199
enum __cuda_FP_NAN { 
# 200
FP_NAN, 
# 202
FP_INFINITE, 
# 204
FP_ZERO, 
# 206
FP_SUBNORMAL, 
# 208
FP_NORMAL
# 210
}; 
# 291 "/usr/include/math.h" 3
extern "C" { typedef 
# 285
enum { 
# 286
_IEEE_ = (-1), 
# 287
_SVID_, 
# 288
_XOPEN_, 
# 289
_POSIX_, 
# 290
_ISOC_
# 291
} _LIB_VERSION_TYPE; }
# 296
extern "C" { extern _LIB_VERSION_TYPE _LIB_VERSION; } 
# 307
extern "C" { struct __exception { 
# 312
int type; 
# 313
char *name; 
# 314
double arg1; 
# 315
double arg2; 
# 316
double retval; 
# 317
}; }
# 320
extern "C" int matherr(__exception *) throw(); 
# 67 "/usr/include/bits/waitstatus.h" 3
extern "C" { union wait { 
# 69
int w_status; 
# 71
struct { 
# 73
unsigned __w_termsig:7; 
# 74
unsigned __w_coredump:1; 
# 75
unsigned __w_retcode:8; 
# 76
unsigned:16; 
# 84
} __wait_terminated; 
# 86
struct { 
# 88
unsigned __w_stopval:8; 
# 89
unsigned __w_stopsig:8; 
# 90
unsigned:16; 
# 97
} __wait_stopped; 
# 98
}; }
# 102 "/usr/include/stdlib.h" 3
extern "C" { typedef 
# 99
struct div_t { 
# 100
int quot; 
# 101
int rem; 
# 102
} div_t; }
# 110
extern "C" { typedef 
# 107
struct ldiv_t { 
# 108
long quot; 
# 109
long rem; 
# 110
} ldiv_t; }
# 122
extern "C" { typedef 
# 119
struct lldiv_t { 
# 120
long long quot; 
# 121
long long rem; 
# 122
} lldiv_t; }
# 140
extern "C" size_t __ctype_get_mb_cur_max() throw(); 
# 145
extern "C" double atof(const char *) throw() __attribute__((__pure__)); 
# 148
extern "C" int atoi(const char *) throw() __attribute__((__pure__)); 
# 151
extern "C" long atol(const char *) throw() __attribute__((__pure__)); 
# 158
extern "C" long long atoll(const char *) throw() __attribute__((__pure__)); 
# 165
extern "C" double strtod(const char *__restrict__, char **__restrict__) throw(); 
# 173
extern "C" float strtof(const char *__restrict__, char **__restrict__) throw(); 
# 176
extern "C" long double strtold(const char *__restrict__, char **__restrict__) throw(); 
# 184
extern "C" long strtol(const char *__restrict__, char **__restrict__, int) throw(); 
# 188
extern "C" unsigned long strtoul(const char *__restrict__, char **__restrict__, int) throw(); 
# 196
extern "C" long long strtoq(const char *__restrict__, char **__restrict__, int) throw(); 
# 201
extern "C" unsigned long long strtouq(const char *__restrict__, char **__restrict__, int) throw(); 
# 210
extern "C" long long strtoll(const char *__restrict__, char **__restrict__, int) throw(); 
# 215
extern "C" unsigned long long strtoull(const char *__restrict__, char **__restrict__, int) throw(); 
# 240 "/usr/include/stdlib.h" 3
extern "C" long strtol_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw(); 
# 244
extern "C" unsigned long strtoul_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw(); 
# 250
extern "C" long long strtoll_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw(); 
# 256
extern "C" unsigned long long strtoull_l(const char *__restrict__, char **__restrict__, int, __locale_t) throw(); 
# 261
extern "C" double strtod_l(const char *__restrict__, char **__restrict__, __locale_t) throw(); 
# 265
extern "C" float strtof_l(const char *__restrict__, char **__restrict__, __locale_t) throw(); 
# 269
extern "C" long double strtold_l(const char *__restrict__, char **__restrict__, __locale_t) throw(); 
# 311 "/usr/include/stdlib.h" 3
extern "C" char *l64a(long) throw(); 
# 314
extern "C" long a64l(const char *) throw() __attribute__((__pure__)); 
# 35 "/usr/include/sys/types.h" 3
extern "C" { typedef __u_char u_char; }
# 36
extern "C" { typedef __u_short u_short; }
# 37
extern "C" { typedef __u_int u_int; }
# 38
extern "C" { typedef __u_long u_long; }
# 39
extern "C" { typedef __quad_t quad_t; }
# 40
extern "C" { typedef __u_quad_t u_quad_t; }
# 41
extern "C" { typedef __fsid_t fsid_t; }
# 46
extern "C" { typedef __loff_t loff_t; }
# 50
extern "C" { typedef __ino_t ino_t; }
# 57
extern "C" { typedef __ino64_t ino64_t; }
# 62
extern "C" { typedef __dev_t dev_t; }
# 67
extern "C" { typedef __gid_t gid_t; }
# 72
extern "C" { typedef __mode_t mode_t; }
# 77
extern "C" { typedef __nlink_t nlink_t; }
# 82
extern "C" { typedef __uid_t uid_t; }
# 88
extern "C" { typedef __off_t off_t; }
# 95
extern "C" { typedef __off64_t off64_t; }
# 105 "/usr/include/sys/types.h" 3
extern "C" { typedef __id_t id_t; }
# 110
extern "C" { typedef __ssize_t ssize_t; }
# 116
extern "C" { typedef __daddr_t daddr_t; }
# 117
extern "C" { typedef __caddr_t caddr_t; }
# 123
extern "C" { typedef __key_t key_t; }
# 137 "/usr/include/sys/types.h" 3
extern "C" { typedef __useconds_t useconds_t; }
# 141
extern "C" { typedef __suseconds_t suseconds_t; }
# 151 "/usr/include/sys/types.h" 3
extern "C" { typedef unsigned long ulong; }
# 152
extern "C" { typedef unsigned short ushort; }
# 153
extern "C" { typedef unsigned uint; }
# 195 "/usr/include/sys/types.h" 3
extern "C" { typedef signed char int8_t; }
# 196
extern "C" { typedef short int16_t; }
# 197
extern "C" { typedef int int32_t; }
# 198
extern "C" { typedef long long int64_t; }
# 201
extern "C" { typedef unsigned char u_int8_t; }
# 202
extern "C" { typedef unsigned short u_int16_t; }
# 203
extern "C" { typedef unsigned u_int32_t; }
# 204
extern "C" { typedef unsigned long long u_int64_t; }
# 206
extern "C" { typedef int register_t; }
# 24 "/usr/include/bits/sigset.h" 3
extern "C" { typedef int __sig_atomic_t; }
# 32
extern "C" { typedef 
# 30
struct __sigset_t { 
# 31
unsigned long __val[((1024) / ((8) * sizeof(unsigned long)))]; 
# 32
} __sigset_t; }
# 38 "/usr/include/sys/select.h" 3
extern "C" { typedef __sigset_t sigset_t; }
# 69 "/usr/include/bits/time.h" 3
extern "C" { struct timeval { 
# 71
__time_t tv_sec; 
# 72
__suseconds_t tv_usec; 
# 73
}; }
# 55 "/usr/include/sys/select.h" 3
extern "C" { typedef long __fd_mask; }
# 78
extern "C" { typedef 
# 68
struct fd_set { 
# 72
__fd_mask fds_bits[((1024) / ((8) * sizeof(__fd_mask)))]; 
# 78
} fd_set; }
# 85
extern "C" { typedef __fd_mask fd_mask; }
# 109
extern "C" int select(int, fd_set *__restrict__, fd_set *__restrict__, fd_set *__restrict__, timeval *__restrict__); 
# 121
extern "C" int pselect(int, fd_set *__restrict__, fd_set *__restrict__, fd_set *__restrict__, const timespec *__restrict__, const __sigset_t *__restrict__); 
# 31 "/usr/include/sys/sysmacros.h" 3
extern "C" unsigned gnu_dev_major(unsigned long long) throw(); 
# 34
extern "C" unsigned gnu_dev_minor(unsigned long long) throw(); 
# 37
extern "C" unsigned long long gnu_dev_makedev(unsigned, unsigned) throw(); 
# 228 "/usr/include/sys/types.h" 3
extern "C" { typedef __blksize_t blksize_t; }
# 235
extern "C" { typedef __blkcnt_t blkcnt_t; }
# 239
extern "C" { typedef __fsblkcnt_t fsblkcnt_t; }
# 243
extern "C" { typedef __fsfilcnt_t fsfilcnt_t; }
# 262 "/usr/include/sys/types.h" 3
extern "C" { typedef __blkcnt64_t blkcnt64_t; }
# 263
extern "C" { typedef __fsblkcnt64_t fsblkcnt64_t; }
# 264
extern "C" { typedef __fsfilcnt64_t fsfilcnt64_t; }
# 50 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef unsigned long pthread_t; }
# 57
extern "C" { typedef 
# 54
union pthread_attr_t { 
# 55
char __size[36]; 
# 56
long __align; 
# 57
} pthread_attr_t; }
# 70 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef 
# 67
struct __pthread_internal_slist { 
# 69
__pthread_internal_slist *__next; 
# 70
} __pthread_slist_t; }
# 104
extern "C" { typedef 
# 77
union pthread_mutex_t { 
# 78
struct __pthread_mutex_s { 
# 80
int __lock; 
# 81
unsigned __count; 
# 82
int __owner; 
# 88
int __kind; 
# 94
unsigned __nusers; 
# 96
union { 
# 97
int __spins; 
# 98
__pthread_slist_t __list; 
# 99
}; 
# 101
} __data; 
# 102
char __size[24]; 
# 103
long __align; 
# 104
} pthread_mutex_t; }
# 110
extern "C" { typedef 
# 107
union pthread_mutexattr_t { 
# 108
char __size[4]; 
# 109
int __align; 
# 110
} pthread_mutexattr_t; }
# 130
extern "C" { typedef 
# 116
union pthread_cond_t { 
# 118
struct { 
# 119
int __lock; 
# 120
unsigned __futex; 
# 121
unsigned long long __total_seq; 
# 122
unsigned long long __wakeup_seq; 
# 123
unsigned long long __woken_seq; 
# 124
void *__mutex; 
# 125
unsigned __nwaiters; 
# 126
unsigned __broadcast_seq; 
# 127
} __data; 
# 128
char __size[48]; 
# 129
long long __align; 
# 130
} pthread_cond_t; }
# 136
extern "C" { typedef 
# 133
union pthread_condattr_t { 
# 134
char __size[4]; 
# 135
int __align; 
# 136
} pthread_condattr_t; }
# 140
extern "C" { typedef unsigned pthread_key_t; }
# 144
extern "C" { typedef int pthread_once_t; }
# 189 "/usr/include/bits/pthreadtypes.h" 3
extern "C" { typedef 
# 151 "/usr/include/bits/pthreadtypes.h" 3
union pthread_rwlock_t { 
# 171 "/usr/include/bits/pthreadtypes.h" 3
struct { 
# 172
int __lock; 
# 173
unsigned __nr_readers; 
# 174
unsigned __readers_wakeup; 
# 175
unsigned __writer_wakeup; 
# 176
unsigned __nr_readers_queued; 
# 177
unsigned __nr_writers_queued; 
# 180
unsigned char __flags; 
# 181
unsigned char __shared; 
# 182
unsigned char __pad1; 
# 183
unsigned char __pad2; 
# 184
int __writer; 
# 185
} __data; 
# 187
char __size[32]; 
# 188
long __align; 
# 189
} pthread_rwlock_t; }
# 195
extern "C" { typedef 
# 192
union pthread_rwlockattr_t { 
# 193
char __size[8]; 
# 194
long __align; 
# 195
} pthread_rwlockattr_t; }
# 201
extern "C" { typedef volatile int pthread_spinlock_t; }
# 210
extern "C" { typedef 
# 207
union pthread_barrier_t { 
# 208
char __size[20]; 
# 209
long __align; 
# 210
} pthread_barrier_t; }
# 216
extern "C" { typedef 
# 213
union pthread_barrierattr_t { 
# 214
char __size[4]; 
# 215
int __align; 
# 216
} pthread_barrierattr_t; }
# 327 "/usr/include/stdlib.h" 3
extern "C" long random() throw(); 
# 330
extern "C" void srandom(unsigned) throw(); 
# 336
extern "C" char *initstate(unsigned, char *, size_t) throw(); 
# 341
extern "C" char *setstate(char *) throw(); 
# 349
extern "C" { struct random_data { 
# 351
int32_t *fptr; 
# 352
int32_t *rptr; 
# 353
int32_t *state; 
# 354
int rand_type; 
# 355
int rand_deg; 
# 356
int rand_sep; 
# 357
int32_t *end_ptr; 
# 358
}; }
# 360
extern "C" int random_r(random_data *__restrict__, int32_t *__restrict__) throw(); 
# 363
extern "C" int srandom_r(unsigned, random_data *) throw(); 
# 366
extern "C" int initstate_r(unsigned, char *__restrict__, size_t, random_data *__restrict__) throw(); 
# 371
extern "C" int setstate_r(char *__restrict__, random_data *__restrict__) throw(); 
# 380
extern "C" int rand() throw(); 
# 382
extern "C" void srand(unsigned) throw(); 
# 387
extern "C" int rand_r(unsigned *) throw(); 
# 395
extern "C" double drand48() throw(); 
# 396
extern "C" double erand48(unsigned short [3]) throw(); 
# 399
extern "C" long lrand48() throw(); 
# 400
extern "C" long nrand48(unsigned short [3]) throw(); 
# 404
extern "C" long mrand48() throw(); 
# 405
extern "C" long jrand48(unsigned short [3]) throw(); 
# 409
extern "C" void srand48(long) throw(); 
# 410
extern "C" unsigned short *seed48(unsigned short [3]) throw(); 
# 412
extern "C" void lcong48(unsigned short [7]) throw(); 
# 418
extern "C" { struct drand48_data { 
# 420
unsigned short __x[3]; 
# 421
unsigned short __old_x[3]; 
# 422
unsigned short __c; 
# 423
unsigned short __init; 
# 424
unsigned long long __a; 
# 425
}; }
# 428
extern "C" int drand48_r(drand48_data *__restrict__, double *__restrict__) throw(); 
# 430
extern "C" int erand48_r(unsigned short [3], drand48_data *__restrict__, double *__restrict__) throw(); 
# 435
extern "C" int lrand48_r(drand48_data *__restrict__, long *__restrict__) throw(); 
# 438
extern "C" int nrand48_r(unsigned short [3], drand48_data *__restrict__, long *__restrict__) throw(); 
# 444
extern "C" int mrand48_r(drand48_data *__restrict__, long *__restrict__) throw(); 
# 447
extern "C" int jrand48_r(unsigned short [3], drand48_data *__restrict__, long *__restrict__) throw(); 
# 453
extern "C" int srand48_r(long, drand48_data *) throw(); 
# 456
extern "C" int seed48_r(unsigned short [3], drand48_data *) throw(); 
# 459
extern "C" int lcong48_r(unsigned short [7], drand48_data *) throw(); 
# 471
extern "C" void *malloc(size_t) throw() __attribute__((__malloc__)); 
# 473
extern "C" void *calloc(size_t, size_t) throw() __attribute__((__malloc__)); 
# 485
extern "C" void *realloc(void *, size_t) throw(); 
# 488
extern "C" void free(void *) throw(); 
# 493
extern "C" void cfree(void *) throw(); 
# 33 "/usr/include/alloca.h" 3
extern "C" void *alloca(size_t) throw(); 
# 502 "/usr/include/stdlib.h" 3
extern "C" void *valloc(size_t) throw() __attribute__((__malloc__)); 
# 507
extern "C" int posix_memalign(void **, size_t, size_t) throw(); 
# 513
extern "C" void abort() throw() __attribute__((__noreturn__)); 
# 517
extern "C" int atexit(void (*)(void)) throw(); 
# 523
extern "C" int on_exit(void (*)(int, void *), void *) throw(); 
# 531
extern "C" void exit(int) throw() __attribute__((__noreturn__)); 
# 538
extern "C" void _Exit(int) throw() __attribute__((__noreturn__)); 
# 545
extern "C" char *getenv(const char *) throw(); 
# 550
extern "C" char *__secure_getenv(const char *) throw(); 
# 557
extern "C" int putenv(char *) throw(); 
# 563
extern "C" int setenv(const char *, const char *, int) throw(); 
# 567
extern "C" int unsetenv(const char *) throw(); 
# 574
extern "C" int clearenv() throw(); 
# 583
extern "C" char *mktemp(char *) throw(); 
# 594
extern "C" int mkstemp(char *); 
# 604 "/usr/include/stdlib.h" 3
extern "C" int mkstemp64(char *); 
# 614
extern "C" char *mkdtemp(char *) throw(); 
# 625
extern "C" int mkostemp(char *, int); 
# 635 "/usr/include/stdlib.h" 3
extern "C" int mkostemp64(char *, int); 
# 645
extern "C" int system(const char *); 
# 652
extern "C" char *canonicalize_file_name(const char *) throw(); 
# 662
extern "C" char *realpath(const char *__restrict__, char *__restrict__) throw(); 
# 670
extern "C" { typedef int (*__compar_fn_t)(const void *, const void *); }
# 673
extern "C" { typedef __compar_fn_t comparison_fn_t; }
# 680
extern "C" void *bsearch(const void *, const void *, size_t, size_t, __compar_fn_t); 
# 686
extern "C" void qsort(void *, size_t, size_t, __compar_fn_t); 
# 691
extern "C"  __attribute__((__weak__)) int abs(int) throw() __attribute__((__const__)); 
# 692
extern "C"  __attribute__((__weak__)) long labs(long) throw() __attribute__((__const__)); 
# 696
extern "C"  __attribute__((__weak__)) long long llabs(long long) throw() __attribute__((__const__)); 
# 705
extern "C" div_t div(int, int) throw() __attribute__((__const__)); 
# 707
extern "C" ldiv_t ldiv(long, long) throw() __attribute__((__const__)); 
# 713
extern "C" lldiv_t lldiv(long long, long long) throw() __attribute__((__const__)); 
# 727
extern "C" char *ecvt(double, int, int *__restrict__, int *__restrict__) throw(); 
# 733
extern "C" char *fcvt(double, int, int *__restrict__, int *__restrict__) throw(); 
# 739
extern "C" char *gcvt(double, int, char *) throw(); 
# 745
extern "C" char *qecvt(long double, int, int *__restrict__, int *__restrict__) throw(); 
# 748
extern "C" char *qfcvt(long double, int, int *__restrict__, int *__restrict__) throw(); 
# 751
extern "C" char *qgcvt(long double, int, char *) throw(); 
# 757
extern "C" int ecvt_r(double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw(); 
# 760
extern "C" int fcvt_r(double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw(); 
# 764
extern "C" int qecvt_r(long double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw(); 
# 768
extern "C" int qfcvt_r(long double, int, int *__restrict__, int *__restrict__, char *__restrict__, size_t) throw(); 
# 779
extern "C" int mblen(const char *, size_t) throw(); 
# 782
extern "C" int mbtowc(wchar_t *__restrict__, const char *__restrict__, size_t) throw(); 
# 786
extern "C" int wctomb(char *, wchar_t) throw(); 
# 790
extern "C" size_t mbstowcs(wchar_t *__restrict__, const char *__restrict__, size_t) throw(); 
# 793
extern "C" size_t wcstombs(char *__restrict__, const wchar_t *__restrict__, size_t) throw(); 
# 804
extern "C" int rpmatch(const char *) throw(); 
# 815
extern "C" int getsubopt(char **__restrict__, char *const *__restrict__, char **__restrict__) throw(); 
# 824
extern "C" void setkey(const char *) throw(); 
# 832
extern "C" int posix_openpt(int); 
# 840
extern "C" int grantpt(int) throw(); 
# 844
extern "C" int unlockpt(int) throw(); 
# 849
extern "C" char *ptsname(int) throw(); 
# 856
extern "C" int ptsname_r(int, char *, size_t) throw(); 
# 860
extern "C" int getpt(); 
# 867
extern "C" int getloadavg(double [], int) throw(); 
# 74 "/usr/include/c++/4.2/bits/cpp_type_traits.h" 3
namespace __gnu_cxx __attribute__((visibility("default"))) { 
# 76
template<class _Iterator, class _Container> class __normal_iterator; 
# 79
}
# 81
namespace std __attribute__((visibility("default"))) { 
# 83
namespace __detail { 
# 87
typedef char __one; 
# 88
typedef char __two[2]; 
# 90
template<class _Tp> extern __one __test_type(int _Tp::*); 
# 92
template<class _Tp> extern __two &__test_type(...); 
# 94
}
# 97
struct __true_type { }; 
# 98
struct __false_type { }; 
# 100
template<bool > 
# 101
struct __truth_type { 
# 102
typedef __false_type __type; }; 
# 105
template<> struct __truth_type< true>  { 
# 106
typedef __true_type __type; }; 
# 110
template<class _Sp, class _Tp> 
# 111
struct __traitor { 
# 113
enum __cuda___value { __value = (((bool)_Sp::__value) || ((bool)_Tp::__value))}; 
# 114
typedef typename __truth_type< (((bool)_Sp::__value) || ((bool)_Tp::__value))> ::__type __type; 
# 115
}; 
# 118
template<class , class > 
# 119
struct __are_same { 
# 121
enum __cuda___value { __value}; 
# 122
typedef __false_type __type; 
# 123
}; 
# 125
template<class _Tp> 
# 126
struct __are_same< _Tp, _Tp>  { 
# 128
enum __cuda___value { __value = 1}; 
# 129
typedef __true_type __type; 
# 130
}; 
# 133
template<class _Tp> 
# 134
struct __is_void { 
# 136
enum __cuda___value { __value}; 
# 137
typedef __false_type __type; 
# 138
}; 
# 141
template<> struct __is_void< void>  { 
# 143
enum __cuda___value { __value = 1}; 
# 144
typedef __true_type __type; 
# 145
}; 
# 150
template<class _Tp> 
# 151
struct __is_integer { 
# 153
enum __cuda___value { __value}; 
# 154
typedef __false_type __type; 
# 155
}; 
# 161
template<> struct __is_integer< bool>  { 
# 163
enum __cuda___value { __value = 1}; 
# 164
typedef __true_type __type; 
# 165
}; 
# 168
template<> struct __is_integer< char>  { 
# 170
enum __cuda___value { __value = 1}; 
# 171
typedef __true_type __type; 
# 172
}; 
# 175
template<> struct __is_integer< signed char>  { 
# 177
enum __cuda___value { __value = 1}; 
# 178
typedef __true_type __type; 
# 179
}; 
# 182
template<> struct __is_integer< unsigned char>  { 
# 184
enum __cuda___value { __value = 1}; 
# 185
typedef __true_type __type; 
# 186
}; 
# 190
template<> struct __is_integer< wchar_t>  { 
# 192
enum __cuda___value { __value = 1}; 
# 193
typedef __true_type __type; 
# 194
}; 
# 198
template<> struct __is_integer< short>  { 
# 200
enum __cuda___value { __value = 1}; 
# 201
typedef __true_type __type; 
# 202
}; 
# 205
template<> struct __is_integer< unsigned short>  { 
# 207
enum __cuda___value { __value = 1}; 
# 208
typedef __true_type __type; 
# 209
}; 
# 212
template<> struct __is_integer< int>  { 
# 214
enum __cuda___value { __value = 1}; 
# 215
typedef __true_type __type; 
# 216
}; 
# 219
template<> struct __is_integer< unsigned>  { 
# 221
enum __cuda___value { __value = 1}; 
# 222
typedef __true_type __type; 
# 223
}; 
# 226
template<> struct __is_integer< long>  { 
# 228
enum __cuda___value { __value = 1}; 
# 229
typedef __true_type __type; 
# 230
}; 
# 233
template<> struct __is_integer< unsigned long>  { 
# 235
enum __cuda___value { __value = 1}; 
# 236
typedef __true_type __type; 
# 237
}; 
# 240
template<> struct __is_integer< long long>  { 
# 242
enum __cuda___value { __value = 1}; 
# 243
typedef __true_type __type; 
# 244
}; 
# 247
template<> struct __is_integer< unsigned long long>  { 
# 249
enum __cuda___value { __value = 1}; 
# 250
typedef __true_type __type; 
# 251
}; 
# 256
template<class _Tp> 
# 257
struct __is_floating { 
# 259
enum __cuda___value { __value}; 
# 260
typedef __false_type __type; 
# 261
}; 
# 265
template<> struct __is_floating< float>  { 
# 267
enum __cuda___value { __value = 1}; 
# 268
typedef __true_type __type; 
# 269
}; 
# 272
template<> struct __is_floating< double>  { 
# 274
enum __cuda___value { __value = 1}; 
# 275
typedef __true_type __type; 
# 276
}; 
# 279
template<> struct __is_floating< long double>  { 
# 281
enum __cuda___value { __value = 1}; 
# 282
typedef __true_type __type; 
# 283
}; 
# 288
template<class _Tp> 
# 289
struct __is_pointer { 
# 291
enum __cuda___value { __value}; 
# 292
typedef __false_type __type; 
# 293
}; 
# 295
template<class _Tp> 
# 296
struct __is_pointer< _Tp *>  { 
# 298
enum __cuda___value { __value = 1}; 
# 299
typedef __true_type __type; 
# 300
}; 
# 305
template<class _Tp> 
# 306
struct __is_normal_iterator { 
# 308
enum __cuda___value { __value}; 
# 309
typedef __false_type __type; 
# 310
}; 
# 312
template<class _Iterator, class _Container> 
# 313
struct __is_normal_iterator< __gnu_cxx::__normal_iterator< _Iterator, _Container> >  { 
# 316
enum __cuda___value { __value = 1}; 
# 317
typedef __true_type __type; 
# 318
}; 
# 323
template<class _Tp> 
# 324
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> >  { 
# 326
}; 
# 331
template<class _Tp> 
# 332
struct __is_fundamental : public __traitor< __is_void< _Tp> , __is_arithmetic< _Tp> >  { 
# 334
}; 
# 339
template<class _Tp> 
# 340
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> >  { 
# 342
}; 
# 345
template<class _Tp> 
# 346
struct __is_pod { 
# 349
enum __cuda___value { 
# 350
__value = (sizeof((__detail::__test_type< _Tp> (0))) != sizeof(__detail::__one))
# 352
}; 
# 353
}; 
# 358
template<class _Tp> 
# 359
struct __is_empty { 
# 363
private: 
# 362
template<class > 
# 363
struct __first { }; 
# 364
template<class _Up> 
# 365
struct __second : public _Up { 
# 366
}; 
# 370
public: enum __cuda___value { 
# 371
__value = (sizeof(__first< _Tp> ) == sizeof(__second< _Tp> ))
# 372
}; 
# 373
}; 
# 378
template<class _Tp> 
# 379
struct __is_char { 
# 381
enum __cuda___value { __value}; 
# 382
typedef __false_type __type; 
# 383
}; 
# 386
template<> struct __is_char< char>  { 
# 388
enum __cuda___value { __value = 1}; 
# 389
typedef __true_type __type; 
# 390
}; 
# 394
template<> struct __is_char< wchar_t>  { 
# 396
enum __cuda___value { __value = 1}; 
# 397
typedef __true_type __type; 
# 398
}; 
# 401
}
# 53 "/usr/include/c++/4.2/cstddef" 3
namespace std __attribute__((visibility("default"))) { 
# 55
using ::ptrdiff_t;
# 56
using ::size_t;
# 58
}
# 74 "/usr/include/c++/4.2/bits/stl_relops.h" 3
namespace std __attribute__((visibility("default"))) { 
# 76
namespace rel_ops { 
# 90
template<class _Tp> inline bool 
# 92
operator!=(const _Tp &__x, const _Tp &__y) 
# 93
{ return !(__x == __y); } 
# 103
template<class _Tp> inline bool 
# 105
operator>(const _Tp &__x, const _Tp &__y) 
# 106
{ return __y < __x; } 
# 116
template<class _Tp> inline bool 
# 118
operator<=(const _Tp &__x, const _Tp &__y) 
# 119
{ return !(__y < __x); } 
# 129
template<class _Tp> inline bool 
# 131
operator>=(const _Tp &__x, const _Tp &__y) 
# 132
{ return !(__x < __y); } 
# 134
}
# 136
}
# 64 "/usr/include/c++/4.2/bits/stl_pair.h" 3
namespace std __attribute__((visibility("default"))) { 
# 67
template<class _T1, class _T2> 
# 68
struct pair { 
# 70
typedef _T1 first_type; 
# 71
typedef _T2 second_type; 
# 73
_T1 first; 
# 74
_T2 second; 
# 80
pair() : first(), second() 
# 81
{ } 
# 84
pair(const _T1 &__a, const _T2 &__b) : first(__a), second(__b) 
# 85
{ } 
# 88
template<class _U1, class _U2> 
# 89
pair(const std::pair< _U1, _U2>  &__p) : first((__p.first)), second((__p.second)) 
# 90
{ } 
# 91
}; 
# 94
template<class _T1, class _T2> inline bool 
# 96
operator==(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 97
{ return ((__x.first) == (__y.first)) && ((__x.second) == (__y.second)); } 
# 100
template<class _T1, class _T2> inline bool 
# 102
operator<(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 103
{ return ((__x.first) < (__y.first)) || ((!((__y.first) < (__x.first))) && ((__x.second) < (__y.second))); 
# 104
} 
# 107
template<class _T1, class _T2> inline bool 
# 109
operator!=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 110
{ return !(__x == __y); } 
# 113
template<class _T1, class _T2> inline bool 
# 115
operator>(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 116
{ return __y < __x; } 
# 119
template<class _T1, class _T2> inline bool 
# 121
operator<=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 122
{ return !(__y < __x); } 
# 125
template<class _T1, class _T2> inline bool 
# 127
operator>=(const pair< _T1, _T2>  &__x, const pair< _T1, _T2>  &__y) 
# 128
{ return !(__x < __y); } 
# 142
template<class _T1, class _T2> inline pair< _T1, _T2>  
# 144
make_pair(_T1 __x, _T2 __y) 
# 145
{ return pair< _T1, _T2> (__x, __y); } 
# 147
}
# 44 "/usr/include/c++/4.2/ext/type_traits.h" 3
namespace __gnu_cxx __attribute__((visibility("default"))) { 
# 47
template<bool , class > 
# 48
struct __enable_if { 
# 49
}; 
# 51
template<class _Tp> 
# 52
struct __enable_if< true, _Tp>  { 
# 53
typedef _Tp __type; }; 
# 57
template<bool _Cond, class _Iftrue, class _Iffalse> 
# 58
struct __conditional_type { 
# 59
typedef _Iftrue __type; }; 
# 61
template<class _Iftrue, class _Iffalse> 
# 62
struct __conditional_type< false, _Iftrue, _Iffalse>  { 
# 63
typedef _Iffalse __type; }; 
# 67
template<class _Tp> 
# 68
struct __add_unsigned { 
# 71
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 74
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 75
}; 
# 78
template<> struct __add_unsigned< char>  { 
# 79
typedef unsigned char __type; }; 
# 82
template<> struct __add_unsigned< signed char>  { 
# 83
typedef unsigned char __type; }; 
# 86
template<> struct __add_unsigned< short>  { 
# 87
typedef unsigned short __type; }; 
# 90
template<> struct __add_unsigned< int>  { 
# 91
typedef unsigned __type; }; 
# 94
template<> struct __add_unsigned< long>  { 
# 95
typedef unsigned long __type; }; 
# 98
template<> struct __add_unsigned< long long>  { 
# 99
typedef unsigned long long __type; }; 
# 103
template<> struct __add_unsigned< bool> ; 
# 106
template<> struct __add_unsigned< wchar_t> ; 
# 110
template<class _Tp> 
# 111
struct __remove_unsigned { 
# 114
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 117
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 118
}; 
# 121
template<> struct __remove_unsigned< char>  { 
# 122
typedef signed char __type; }; 
# 125
template<> struct __remove_unsigned< unsigned char>  { 
# 126
typedef signed char __type; }; 
# 129
template<> struct __remove_unsigned< unsigned short>  { 
# 130
typedef short __type; }; 
# 133
template<> struct __remove_unsigned< unsigned>  { 
# 134
typedef int __type; }; 
# 137
template<> struct __remove_unsigned< unsigned long>  { 
# 138
typedef long __type; }; 
# 141
template<> struct __remove_unsigned< unsigned long long>  { 
# 142
typedef long long __type; }; 
# 146
template<> struct __remove_unsigned< bool> ; 
# 149
template<> struct __remove_unsigned< wchar_t> ; 
# 151
}
# 82 "/usr/include/c++/4.2/cmath" 3
namespace std __attribute__((visibility("default"))) { 
# 86
template<class _Tp> extern inline _Tp __cmath_power(_Tp, unsigned); 
# 89
inline double abs(double __x) 
# 90
{ return __builtin_fabs(__x); } 
# 93
inline float abs(float __x) 
# 94
{ return __builtin_fabsf(__x); } 
# 97
inline long double abs(long double __x) 
# 98
{ return __builtin_fabsl(__x); } 
# 100
using ::acos;
# 103
inline float acos(float __x) 
# 104
{ return __builtin_acosf(__x); } 
# 107
inline long double acos(long double __x) 
# 108
{ return __builtin_acosl(__x); } 
# 110
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 113
acos(_Tp __x) 
# 114
{ return __builtin_acos(__x); } 
# 116
using ::asin;
# 119
inline float asin(float __x) 
# 120
{ return __builtin_asinf(__x); } 
# 123
inline long double asin(long double __x) 
# 124
{ return __builtin_asinl(__x); } 
# 126
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 129
asin(_Tp __x) 
# 130
{ return __builtin_asin(__x); } 
# 132
using ::atan;
# 135
inline float atan(float __x) 
# 136
{ return __builtin_atanf(__x); } 
# 139
inline long double atan(long double __x) 
# 140
{ return __builtin_atanl(__x); } 
# 142
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 145
atan(_Tp __x) 
# 146
{ return __builtin_atan(__x); } 
# 148
using ::atan2;
# 151
inline float atan2(float __y, float __x) 
# 152
{ return __builtin_atan2f(__y, __x); } 
# 155
inline long double atan2(long double __y, long double __x) 
# 156
{ return __builtin_atan2l(__y, __x); } 
# 158
template<class _Tp, class _Up> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value && __is_integer< _Up> ::__value), double> ::__type 
# 162
atan2(_Tp __y, _Up __x) 
# 163
{ return __builtin_atan2(__y, __x); } 
# 165
using ::ceil;
# 168
inline float ceil(float __x) 
# 169
{ return __builtin_ceilf(__x); } 
# 172
inline long double ceil(long double __x) 
# 173
{ return __builtin_ceill(__x); } 
# 175
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 178
ceil(_Tp __x) 
# 179
{ return __builtin_ceil(__x); } 
# 181
using ::cos;
# 184
inline float cos(float __x) 
# 185
{ return __builtin_cosf(__x); } 
# 188
inline long double cos(long double __x) 
# 189
{ return __builtin_cosl(__x); } 
# 191
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 194
cos(_Tp __x) 
# 195
{ return __builtin_cos(__x); } 
# 197
using ::cosh;
# 200
inline float cosh(float __x) 
# 201
{ return __builtin_coshf(__x); } 
# 204
inline long double cosh(long double __x) 
# 205
{ return __builtin_coshl(__x); } 
# 207
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 210
cosh(_Tp __x) 
# 211
{ return __builtin_cosh(__x); } 
# 213
using ::exp;
# 216
inline float exp(float __x) 
# 217
{ return __builtin_expf(__x); } 
# 220
inline long double exp(long double __x) 
# 221
{ return __builtin_expl(__x); } 
# 223
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 226
exp(_Tp __x) 
# 227
{ return __builtin_exp(__x); } 
# 229
using ::fabs;
# 232
inline float fabs(float __x) 
# 233
{ return __builtin_fabsf(__x); } 
# 236
inline long double fabs(long double __x) 
# 237
{ return __builtin_fabsl(__x); } 
# 239
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 242
fabs(_Tp __x) 
# 243
{ return __builtin_fabs(__x); } 
# 245
using ::floor;
# 248
inline float floor(float __x) 
# 249
{ return __builtin_floorf(__x); } 
# 252
inline long double floor(long double __x) 
# 253
{ return __builtin_floorl(__x); } 
# 255
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 258
floor(_Tp __x) 
# 259
{ return __builtin_floor(__x); } 
# 261
using ::fmod;
# 264
inline float fmod(float __x, float __y) 
# 265
{ return __builtin_fmodf(__x, __y); } 
# 268
inline long double fmod(long double __x, long double __y) 
# 269
{ return __builtin_fmodl(__x, __y); } 
# 271
using ::frexp;
# 274
inline float frexp(float __x, int *__exp) 
# 275
{ return __builtin_frexpf(__x, __exp); } 
# 278
inline long double frexp(long double __x, int *__exp) 
# 279
{ return __builtin_frexpl(__x, __exp); } 
# 281
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 284
frexp(_Tp __x, int *__exp) 
# 285
{ return __builtin_frexp(__x, __exp); } 
# 287
using ::ldexp;
# 290
inline float ldexp(float __x, int __exp) 
# 291
{ return __builtin_ldexpf(__x, __exp); } 
# 294
inline long double ldexp(long double __x, int __exp) 
# 295
{ return __builtin_ldexpl(__x, __exp); } 
# 297
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 300
ldexp(_Tp __x, int __exp) 
# 301
{ return __builtin_ldexp(__x, __exp); } 
# 303
using ::log;
# 306
inline float log(float __x) 
# 307
{ return __builtin_logf(__x); } 
# 310
inline long double log(long double __x) 
# 311
{ return __builtin_logl(__x); } 
# 313
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 316
log(_Tp __x) 
# 317
{ return __builtin_log(__x); } 
# 319
using ::log10;
# 322
inline float log10(float __x) 
# 323
{ return __builtin_log10f(__x); } 
# 326
inline long double log10(long double __x) 
# 327
{ return __builtin_log10l(__x); } 
# 329
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 332
log10(_Tp __x) 
# 333
{ return __builtin_log10(__x); } 
# 335
using ::modf;
# 338
inline float modf(float __x, float *__iptr) 
# 339
{ return __builtin_modff(__x, __iptr); } 
# 342
inline long double modf(long double __x, long double *__iptr) 
# 343
{ return __builtin_modfl(__x, __iptr); } 
# 345
template<class _Tp> inline _Tp 
# 347
__pow_helper(_Tp __x, int __n) 
# 348
{ 
# 349
return (__n < 0) ? (((_Tp)(1)) / __cmath_power(__x, -__n)) : (__cmath_power(__x, __n)); 
# 352
} 
# 354
using ::pow;
# 357
inline float pow(float __x, float __y) 
# 358
{ return __builtin_powf(__x, __y); } 
# 361
inline long double pow(long double __x, long double __y) 
# 362
{ return __builtin_powl(__x, __y); } 
# 365
inline double pow(double __x, int __i) 
# 366
{ return __builtin_powi(__x, __i); } 
# 369
inline float pow(float __x, int __n) 
# 370
{ return __builtin_powif(__x, __n); } 
# 373
inline long double pow(long double __x, int __n) 
# 374
{ return __builtin_powil(__x, __n); } 
# 376
using ::sin;
# 379
inline float sin(float __x) 
# 380
{ return __builtin_sinf(__x); } 
# 383
inline long double sin(long double __x) 
# 384
{ return __builtin_sinl(__x); } 
# 386
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 389
sin(_Tp __x) 
# 390
{ return __builtin_sin(__x); } 
# 392
using ::sinh;
# 395
inline float sinh(float __x) 
# 396
{ return __builtin_sinhf(__x); } 
# 399
inline long double sinh(long double __x) 
# 400
{ return __builtin_sinhl(__x); } 
# 402
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 405
sinh(_Tp __x) 
# 406
{ return __builtin_sinh(__x); } 
# 408
using ::sqrt;
# 411
inline float sqrt(float __x) 
# 412
{ return __builtin_sqrtf(__x); } 
# 415
inline long double sqrt(long double __x) 
# 416
{ return __builtin_sqrtl(__x); } 
# 418
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 421
sqrt(_Tp __x) 
# 422
{ return __builtin_sqrt(__x); } 
# 424
using ::tan;
# 427
inline float tan(float __x) 
# 428
{ return __builtin_tanf(__x); } 
# 431
inline long double tan(long double __x) 
# 432
{ return __builtin_tanl(__x); } 
# 434
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 437
tan(_Tp __x) 
# 438
{ return __builtin_tan(__x); } 
# 440
using ::tanh;
# 443
inline float tanh(float __x) 
# 444
{ return __builtin_tanhf(__x); } 
# 447
inline long double tanh(long double __x) 
# 448
{ return __builtin_tanhl(__x); } 
# 450
template<class _Tp> inline typename __gnu_cxx::__enable_if< (__is_integer< _Tp> ::__value), double> ::__type 
# 453
tanh(_Tp __x) 
# 454
{ return __builtin_tanh(__x); } 
# 456
}
# 464
namespace __gnu_cxx __attribute__((visibility("default"))) { 
# 466
template<class _Tp> inline int 
# 468
__capture_fpclassify(_Tp __f) { return (sizeof(__f) == sizeof(float)) ? (__fpclassifyf(__f)) : ((sizeof(__f) == sizeof(double)) ? (__fpclassify(__f)) : (__fpclassifyl(__f))); } 
# 470
template<class _Tp> inline int 
# 472
__capture_isfinite(_Tp __f) { return (sizeof(__f) == sizeof(float)) ? (__finitef(__f)) : ((sizeof(__f) == sizeof(double)) ? (__finite(__f)) : (__finitel(__f))); } 
# 474
template<class _Tp> inline int 
# 476
__capture_isinf(_Tp __f) { return (sizeof(__f) == sizeof(float)) ? (__isinff(__f)) : ((sizeof(__f) == sizeof(double)) ? (__isinf(__f)) : (__isinfl(__f))); } 
# 478
template<class _Tp> inline int 
# 480
__capture_isnan(_Tp __f) { return (sizeof(__f) == sizeof(float)) ? (__isnanf(__f)) : ((sizeof(__f) == sizeof(double)) ? (__isnan(__f)) : (__isnanl(__f))); } 
# 482
template<class _Tp> inline int 
# 484
__capture_isnormal(_Tp __f) { return ((sizeof(__f) == sizeof(float)) ? (__fpclassifyf(__f)) : ((sizeof(__f) == sizeof(double)) ? (__fpclassify(__f)) : (__fpclassifyl(__f)))) == FP_NORMAL; } 
# 486
template<class _Tp> inline int 
# 488
__capture_signbit(_Tp __f) { return (sizeof(__f) == sizeof(float)) ? (__signbitf(__f)) : ((sizeof(__f) == sizeof(double)) ? (__signbit(__f)) : (__signbitl(__f))); } 
# 490
template<class _Tp> inline int 
# 492
__capture_isgreater(_Tp __f1, _Tp __f2) 
# 493
{ return __builtin_isgreater(__f1, __f2); } 
# 495
template<class _Tp> inline int 
# 497
__capture_isgreaterequal(_Tp __f1, _Tp __f2) 
# 498
{ return __builtin_isgreaterequal(__f1, __f2); } 
# 500
template<class _Tp> inline int 
# 502
__capture_isless(_Tp __f1, _Tp __f2) { return __builtin_isless(__f1, __f2); } 
# 504
template<class _Tp> inline int 
# 506
__capture_islessequal(_Tp __f1, _Tp __f2) 
# 507
{ return __builtin_islessequal(__f1, __f2); } 
# 509
template<class _Tp> inline int 
# 511
__capture_islessgreater(_Tp __f1, _Tp __f2) 
# 512
{ return __builtin_islessgreater(__f1, __f2); } 
# 514
template<class _Tp> inline int 
# 516
__capture_isunordered(_Tp __f1, _Tp __f2) 
# 517
{ return __builtin_isunordered(__f1, __f2); } 
# 519
}
# 535 "/usr/include/c++/4.2/cmath" 3
namespace std __attribute__((visibility("default"))) { 
# 537
template<class _Tp> inline int 
# 539
fpclassify(_Tp __f) { return __gnu_cxx::__capture_fpclassify(__f); } 
# 541
template<class _Tp> inline int 
# 543
isfinite(_Tp __f) { return __gnu_cxx::__capture_isfinite(__f); } 
# 545
template<class _Tp> inline int 
# 547
isinf(_Tp __f) { return __gnu_cxx::__capture_isinf(__f); } 
# 549
template<class _Tp> inline int 
# 551
isnan(_Tp __f) { return __gnu_cxx::__capture_isnan(__f); } 
# 553
template<class _Tp> inline int 
# 555
isnormal(_Tp __f) { return __gnu_cxx::__capture_isnormal(__f); } 
# 557
template<class _Tp> inline int 
# 559
signbit(_Tp __f) { return __gnu_cxx::__capture_signbit(__f); } 
# 561
template<class _Tp> inline int 
# 563
isgreater(_Tp __f1, _Tp __f2) 
# 564
{ return __gnu_cxx::__capture_isgreater(__f1, __f2); } 
# 566
template<class _Tp> inline int 
# 568
isgreaterequal(_Tp __f1, _Tp __f2) 
# 569
{ return __gnu_cxx::__capture_isgreaterequal(__f1, __f2); } 
# 571
template<class _Tp> inline int 
# 573
isless(_Tp __f1, _Tp __f2) 
# 574
{ return __gnu_cxx::__capture_isless(__f1, __f2); } 
# 576
template<class _Tp> inline int 
# 578
islessequal(_Tp __f1, _Tp __f2) 
# 579
{ return __gnu_cxx::__capture_islessequal(__f1, __f2); } 
# 581
template<class _Tp> inline int 
# 583
islessgreater(_Tp __f1, _Tp __f2) 
# 584
{ return __gnu_cxx::__capture_islessgreater(__f1, __f2); } 
# 586
template<class _Tp> inline int 
# 588
isunordered(_Tp __f1, _Tp __f2) 
# 589
{ return __gnu_cxx::__capture_isunordered(__f1, __f2); } 
# 591
}
# 39 "/usr/include/c++/4.2/bits/cmath.tcc" 3
namespace std __attribute__((visibility("default"))) { 
# 41
template<class _Tp> inline _Tp 
# 43
__cmath_power(_Tp __x, unsigned __n) 
# 44
{ 
# 45
auto _Tp __y = ((__n % (2)) ? __x : 1); 
# 47
while (__n >>= 1) 
# 48
{ 
# 49
__x = __x * __x; 
# 50
if (__n % (2)) { 
# 51
__y = __y * __x; }  
# 52
}  
# 54
return __y; 
# 55
} 
# 57
}
# 104 "/usr/include/c++/4.2/cstdlib" 3
namespace std __attribute__((visibility("default"))) { 
# 106
using ::div_t;
# 107
using ::ldiv_t;
# 109
using ::abort;
# 110
using ::abs;
# 111
using ::atexit;
# 112
using ::atof;
# 113
using ::atoi;
# 114
using ::atol;
# 115
using ::bsearch;
# 116
using ::calloc;
# 117
using ::div;
# 118
using ::exit;
# 119
using ::free;
# 120
using ::getenv;
# 121
using ::labs;
# 122
using ::ldiv;
# 123
using ::malloc;
# 125
using ::mblen;
# 126
using ::mbstowcs;
# 127
using ::mbtowc;
# 129
using ::qsort;
# 130
using ::rand;
# 131
using ::realloc;
# 132
using ::srand;
# 133
using ::strtod;
# 134
using ::strtol;
# 135
using ::strtoul;
# 136
using ::system;
# 138
using ::wcstombs;
# 139
using ::wctomb;
# 143
inline long abs(long __i) { return labs(__i); } 
# 146
inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); } 
# 148
}
# 161 "/usr/include/c++/4.2/cstdlib" 3
namespace __gnu_cxx __attribute__((visibility("default"))) { 
# 164
using ::lldiv_t;
# 170
using ::_Exit;
# 174
inline long long abs(long long __x) { return (__x >= (0)) ? __x : (-__x); } 
# 177
using ::llabs;
# 180
inline lldiv_t div(long long __n, long long __d) 
# 181
{ auto lldiv_t __q; (__q.quot) = __n / __d; (__q.rem) = __n % __d; return __q; } 
# 183
using ::lldiv;
# 194 "/usr/include/c++/4.2/cstdlib" 3
using ::atoll;
# 195
using ::strtoll;
# 196
using ::strtoull;
# 198
using ::strtof;
# 199
using ::strtold;
# 201
}
# 203
namespace std __attribute__((visibility("default"))) { 
# 206
using __gnu_cxx::lldiv_t;
# 208
using __gnu_cxx::_Exit;
# 209
using __gnu_cxx::abs;
# 211
using __gnu_cxx::llabs;
# 212
using __gnu_cxx::div;
# 213
using __gnu_cxx::lldiv;
# 215
using __gnu_cxx::atoll;
# 216
using __gnu_cxx::strtof;
# 217
using __gnu_cxx::strtoll;
# 218
using __gnu_cxx::strtoull;
# 219
using __gnu_cxx::strtold;
# 221
}
# 424 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C"  __attribute__((__weak__)) int __signbitl(long double) throw() __attribute__((__const__)); 
# 426
extern "C"  __attribute__((__weak__)) int __isinfl(long double) throw() __attribute__((__const__)); 
# 428
extern "C"  __attribute__((__weak__)) int __isnanl(long double) throw() __attribute__((__const__)); 
# 438 "/usr/local/cuda/bin/../include/math_functions.h"
extern "C"  __attribute__((__weak__)) int __finitel(long double) throw() __attribute__((__const__)); 
# 463 "/usr/local/cuda/bin/../include/math_functions.h"
namespace __gnu_cxx { 
# 465
extern inline long long abs(long long) __attribute__((visibility("default"))); 
# 466
}
# 468
namespace std { 
# 470
template<class T> extern inline T __pow_helper(T, int); 
# 471
template<class T> extern inline T __cmath_power(T, unsigned); 
# 472
}
# 474
using std::abs;
# 475
using std::fabs;
# 476
using std::ceil;
# 477
using std::floor;
# 478
using std::sqrt;
# 479
using std::pow;
# 480
using std::log;
# 481
using std::log10;
# 482
using std::fmod;
# 483
using std::modf;
# 484
using std::exp;
# 485
using std::frexp;
# 486
using std::ldexp;
# 487
using std::asin;
# 488
using std::sin;
# 489
using std::sinh;
# 490
using std::acos;
# 491
using std::cos;
# 492
using std::cosh;
# 493
using std::atan;
# 494
using std::atan2;
# 495
using std::tan;
# 496
using std::tanh;
# 550 "/usr/local/cuda/bin/../include/math_functions.h"
namespace std { 
# 553
extern inline long abs(long) __attribute__((visibility("default"))); 
# 554
extern inline float abs(float) __attribute__((visibility("default"))); 
# 555
extern inline double abs(double) __attribute__((visibility("default"))); 
# 556
extern inline float fabs(float) __attribute__((visibility("default"))); 
# 557
extern inline float ceil(float) __attribute__((visibility("default"))); 
# 558
extern inline float floor(float) __attribute__((visibility("default"))); 
# 559
extern inline float sqrt(float) __attribute__((visibility("default"))); 
# 560
extern inline float pow(float, float) __attribute__((visibility("default"))); 
# 561
extern inline float pow(float, int) __attribute__((visibility("default"))); 
# 562
extern inline double pow(double, int) __attribute__((visibility("default"))); 
# 563
extern inline float log(float) __attribute__((visibility("default"))); 
# 564
extern inline float log10(float) __attribute__((visibility("default"))); 
# 565
extern inline float fmod(float, float) __attribute__((visibility("default"))); 
# 566
extern inline float modf(float, float *) __attribute__((visibility("default"))); 
# 567
extern inline float exp(float) __attribute__((visibility("default"))); 
# 568
extern inline float frexp(float, int *) __attribute__((visibility("default"))); 
# 569
extern inline float ldexp(float, int) __attribute__((visibility("default"))); 
# 570
extern inline float asin(float) __attribute__((visibility("default"))); 
# 571
extern inline float sin(float) __attribute__((visibility("default"))); 
# 572
extern inline float sinh(float) __attribute__((visibility("default"))); 
# 573
extern inline float acos(float) __attribute__((visibility("default"))); 
# 574
extern inline float cos(float) __attribute__((visibility("default"))); 
# 575
extern inline float cosh(float) __attribute__((visibility("default"))); 
# 576
extern inline float atan(float) __attribute__((visibility("default"))); 
# 577
extern inline float atan2(float, float) __attribute__((visibility("default"))); 
# 578
extern inline float tan(float) __attribute__((visibility("default"))); 
# 579
extern inline float tanh(float) __attribute__((visibility("default"))); 
# 582
}
# 585
static inline float logb(float a) 
# 586
{ 
# 587
return logbf(a); 
# 588
} 
# 590
static inline int ilogb(float a) 
# 591
{ 
# 592
return ilogbf(a); 
# 593
} 
# 595
static inline float scalbn(float a, int b) 
# 596
{ 
# 597
return scalbnf(a, b); 
# 598
} 
# 600
static inline float scalbln(float a, long b) 
# 601
{ 
# 602
return scalblnf(a, b); 
# 603
} 
# 605
static inline float exp2(float a) 
# 606
{ 
# 607
return exp2f(a); 
# 608
} 
# 610
static inline float exp10(float a) 
# 611
{ 
# 612
return exp10f(a); 
# 613
} 
# 615
static inline float expm1(float a) 
# 616
{ 
# 617
return expm1f(a); 
# 618
} 
# 620
static inline float log2(float a) 
# 621
{ 
# 622
return log2f(a); 
# 623
} 
# 625
static inline float log1p(float a) 
# 626
{ 
# 627
return log1pf(a); 
# 628
} 
# 630
static inline float rsqrt(float a) 
# 631
{ 
# 632
return rsqrtf(a); 
# 633
} 
# 635
static inline float acosh(float a) 
# 636
{ 
# 637
return acoshf(a); 
# 638
} 
# 640
static inline float asinh(float a) 
# 641
{ 
# 642
return asinhf(a); 
# 643
} 
# 645
static inline float atanh(float a) 
# 646
{ 
# 647
return atanhf(a); 
# 648
} 
# 650
static inline float hypot(float a, float b) 
# 651
{ 
# 652
return hypotf(a, b); 
# 653
} 
# 655
static inline float cbrt(float a) 
# 656
{ 
# 657
return cbrtf(a); 
# 658
} 
# 660
static inline void sincos(float a, float *sptr, float *cptr) 
# 661
{ 
# 662
sincosf(a, sptr, cptr); 
# 663
} 
# 665
static inline float erf(float a) 
# 666
{ 
# 667
return erff(a); 
# 668
} 
# 670
static inline float erfc(float a) 
# 671
{ 
# 672
return erfcf(a); 
# 673
} 
# 675
static inline float lgamma(float a) 
# 676
{ 
# 677
return lgammaf(a); 
# 678
} 
# 680
static inline float tgamma(float a) 
# 681
{ 
# 682
return tgammaf(a); 
# 683
} 
# 685
static inline float copysign(float a, float b) 
# 686
{ 
# 687
return copysignf(a, b); 
# 688
} 
# 690
static inline double copysign(double a, float b) 
# 691
{ 
# 692
return copysign(a, (double)b); 
# 693
} 
# 695
static inline float copysign(float a, double b) 
# 696
{ 
# 697
return copysignf(a, (float)b); 
# 698
} 
# 700
static inline float nextafter(float a, float b) 
# 701
{ 
# 702
return nextafterf(a, b); 
# 703
} 
# 705
static inline float remainder(float a, float b) 
# 706
{ 
# 707
return remainderf(a, b); 
# 708
} 
# 710
static inline float remquo(float a, float b, int *quo) 
# 711
{ 
# 712
return remquof(a, b, quo); 
# 713
} 
# 715
static inline float round(float a) 
# 716
{ 
# 717
return roundf(a); 
# 718
} 
# 720
static inline long lround(float a) 
# 721
{ 
# 722
return lroundf(a); 
# 723
} 
# 725
static inline long long llround(float a) 
# 726
{ 
# 727
return llroundf(a); 
# 728
} 
# 730
static inline float trunc(float a) 
# 731
{ 
# 732
return truncf(a); 
# 733
} 
# 735
static inline float rint(float a) 
# 736
{ 
# 737
return rintf(a); 
# 738
} 
# 740
static inline long lrint(float a) 
# 741
{ 
# 742
return lrintf(a); 
# 743
} 
# 745
static inline long long llrint(float a) 
# 746
{ 
# 747
return llrintf(a); 
# 748
} 
# 750
static inline float nearbyint(float a) 
# 751
{ 
# 752
return nearbyintf(a); 
# 753
} 
# 755
static inline float fdim(float a, float b) 
# 756
{ 
# 757
return fdimf(a, b); 
# 758
} 
# 760
static inline float fma(float a, float b, float c) 
# 761
{ 
# 762
return fmaf(a, b, c); 
# 763
} 
# 765
static inline unsigned min(unsigned a, unsigned b) 
# 766
{ 
# 767
return umin(a, b); 
# 768
} 
# 770
static inline unsigned min(int a, unsigned b) 
# 771
{ 
# 772
return umin((unsigned)a, b); 
# 773
} 
# 775
static inline unsigned min(unsigned a, int b) 
# 776
{ 
# 777
return umin(a, (unsigned)b); 
# 778
} 
# 780
static inline float min(float a, float b) 
# 781
{ 
# 782
return fminf(a, b); 
# 783
} 
# 785
static inline double min(double a, double b) 
# 786
{ 
# 787
return fmin(a, b); 
# 788
} 
# 790
static inline double min(float a, double b) 
# 791
{ 
# 792
return fmin((double)a, b); 
# 793
} 
# 795
static inline double min(double a, float b) 
# 796
{ 
# 797
return fmin(a, (double)b); 
# 798
} 
# 800
static inline unsigned max(unsigned a, unsigned b) 
# 801
{ 
# 802
return umax(a, b); 
# 803
} 
# 805
static inline unsigned max(int a, unsigned b) 
# 806
{ 
# 807
return umax((unsigned)a, b); 
# 808
} 
# 810
static inline unsigned max(unsigned a, int b) 
# 811
{ 
# 812
return umax(a, (unsigned)b); 
# 813
} 
# 815
static inline float max(float a, float b) 
# 816
{ 
# 817
return fmaxf(a, b); 
# 818
} 
# 820
static inline double max(double a, double b) 
# 821
{ 
# 822
return fmax(a, b); 
# 823
} 
# 825
static inline double max(float a, double b) 
# 826
{ 
# 827
return fmax((double)a, b); 
# 828
} 
# 830
static inline double max(double a, float b) 
# 831
{ 
# 832
return fmax(a, (double)b); 
# 833
} 
# 59 "/usr/local/cuda/bin/../include/cuda_texture_types.h"
template<class T, int dim = 1, cudaTextureReadMode  = cudaReadModeElementType> 
# 60
struct texture : public textureReference { 
# 62
texture(int norm = 0, cudaTextureFilterMode 
# 63
fMode = cudaFilterModePoint, cudaTextureAddressMode 
# 64
aMode = cudaAddressModeClamp) 
# 65
{ 
# 66
(this->normalized) = norm; 
# 67
(this->filterMode) = fMode; 
# 68
((this->addressMode)[0]) = aMode; 
# 69
((this->addressMode)[1]) = aMode; 
# 70
((this->addressMode)[2]) = aMode; 
# 71
(this->channelDesc) = cudaCreateChannelDesc< T> (); 
# 72
} 
# 74
texture(int norm, cudaTextureFilterMode 
# 75
fMode, cudaTextureAddressMode 
# 76
aMode, cudaChannelFormatDesc 
# 77
desc) 
# 78
{ 
# 79
(this->normalized) = norm; 
# 80
(this->filterMode) = fMode; 
# 81
((this->addressMode)[0]) = aMode; 
# 82
((this->addressMode)[1]) = aMode; 
# 83
((this->addressMode)[2]) = aMode; 
# 84
(this->channelDesc) = desc; 
# 85
} 
# 86
}; 
#if 0
#endif
#if 0
#endif
#if 0
#endif
#if 0
#endif
#if 0
#endif
#if 0
#endif
#if 0
# 53 "/usr/local/cuda/bin/../include/device_launch_parameters.h"
extern "C" { extern const uint3 threadIdx; } 
#endif
#if 0
# 55
extern "C" { extern const uint3 blockIdx; } 
#endif
#if 0
# 57
extern "C" { extern const dim3 blockDim; } 
#endif
#if 0
# 59
extern "C" { extern const dim3 gridDim; } 
#endif
#if 0
# 61
extern "C" { extern const int warpSize; } 
#endif
# 77 "/usr/local/cuda/bin/../include/cuda_runtime.h"
template<class T> inline cudaError_t 
# 78
cudaSetupArgument(T 
# 79
arg, size_t 
# 80
offset) 
# 82
{ 
# 83
return cudaSetupArgument((const void *)(&arg), sizeof(T), offset); 
# 84
} 
# 94
static inline cudaError_t cudaMemcpyToSymbol(char *
# 95
symbol, const void *
# 96
src, size_t 
# 97
count, size_t 
# 98
offset = (0), cudaMemcpyKind 
# 99
kind = cudaMemcpyHostToDevice) 
# 101
{ 
# 102
return cudaMemcpyToSymbol((const char *)symbol, src, count, offset, kind); 
# 103
} 
# 105
template<class T> inline cudaError_t 
# 106
cudaMemcpyToSymbol(const T &
# 107
symbol, const void *
# 108
src, size_t 
# 109
count, size_t 
# 110
offset = (0), cudaMemcpyKind 
# 111
kind = cudaMemcpyHostToDevice) 
# 113
{ 
# 114
return cudaMemcpyToSymbol((const char *)(&symbol), src, count, offset, kind); 
# 115
} 
# 117
static inline cudaError_t cudaMemcpyToSymbolAsync(char *
# 118
symbol, const void *
# 119
src, size_t 
# 120
count, size_t 
# 121
offset, cudaMemcpyKind 
# 122
kind, cudaStream_t 
# 123
stream) 
# 125
{ 
# 126
return cudaMemcpyToSymbolAsync((const char *)symbol, src, count, offset, kind, stream); 
# 127
} 
# 129
template<class T> inline cudaError_t 
# 130
cudaMemcpyToSymbolAsync(const T &
# 131
symbol, const void *
# 132
src, size_t 
# 133
count, size_t 
# 134
offset, cudaMemcpyKind 
# 135
kind, cudaStream_t 
# 136
stream) 
# 138
{ 
# 139
return cudaMemcpyToSymbolAsync((const char *)(&symbol), src, count, offset, kind, stream); 
# 140
} 
# 148
static inline cudaError_t cudaMemcpyFromSymbol(void *
# 149
dst, char *
# 150
symbol, size_t 
# 151
count, size_t 
# 152
offset = (0), cudaMemcpyKind 
# 153
kind = cudaMemcpyDeviceToHost) 
# 155
{ 
# 156
return cudaMemcpyFromSymbol(dst, (const char *)symbol, count, offset, kind); 
# 157
} 
# 159
template<class T> inline cudaError_t 
# 160
cudaMemcpyFromSymbol(void *
# 161
dst, const T &
# 162
symbol, size_t 
# 163
count, size_t 
# 164
offset = (0), cudaMemcpyKind 
# 165
kind = cudaMemcpyDeviceToHost) 
# 167
{ 
# 168
return cudaMemcpyFromSymbol(dst, (const char *)(&symbol), count, offset, kind); 
# 169
} 
# 171
static inline cudaError_t cudaMemcpyFromSymbolAsync(void *
# 172
dst, char *
# 173
symbol, size_t 
# 174
count, size_t 
# 175
offset, cudaMemcpyKind 
# 176
kind, cudaStream_t 
# 177
stream) 
# 179
{ 
# 180
return cudaMemcpyFromSymbolAsync(dst, (const char *)symbol, count, offset, kind, stream); 
# 181
} 
# 183
template<class T> inline cudaError_t 
# 184
cudaMemcpyFromSymbolAsync(void *
# 185
dst, const T &
# 186
symbol, size_t 
# 187
count, size_t 
# 188
offset, cudaMemcpyKind 
# 189
kind, cudaStream_t 
# 190
stream) 
# 192
{ 
# 193
return cudaMemcpyFromSymbolAsync(dst, (const char *)(&symbol), count, offset, kind, stream); 
# 194
} 
# 196
static inline cudaError_t cudaGetSymbolAddress(void **
# 197
devPtr, char *
# 198
symbol) 
# 200
{ 
# 201
return cudaGetSymbolAddress(devPtr, (const char *)symbol); 
# 202
} 
# 204
template<class T> inline cudaError_t 
# 205
cudaGetSymbolAddress(void **
# 206
devPtr, const T &
# 207
symbol) 
# 209
{ 
# 210
return cudaGetSymbolAddress(devPtr, (const char *)(&symbol)); 
# 211
} 
# 219
static inline cudaError_t cudaGetSymbolSize(size_t *
# 220
size, char *
# 221
symbol) 
# 223
{ 
# 224
return cudaGetSymbolSize(size, (const char *)symbol); 
# 225
} 
# 227
template<class T> inline cudaError_t 
# 228
cudaGetSymbolSize(size_t *
# 229
size, const T &
# 230
symbol) 
# 232
{ 
# 233
return cudaGetSymbolSize(size, (const char *)(&symbol)); 
# 234
} 
# 242
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t 
# 243
cudaBindTexture(size_t *
# 244
offset, const texture< T, dim, readMode>  &
# 245
tex, const void *
# 246
devPtr, const cudaChannelFormatDesc &
# 247
desc, size_t 
# 248
size = (((2147483647) * 2U) + 1U)) 
# 250
{ 
# 251
return cudaBindTexture(offset, &tex, devPtr, (&desc), size); 
# 252
} 
# 254
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t 
# 255
cudaBindTexture(size_t *
# 256
offset, const texture< T, dim, readMode>  &
# 257
tex, const void *
# 258
devPtr, size_t 
# 259
size = (((2147483647) * 2U) + 1U)) 
# 261
{ 
# 262
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size); 
# 263
} 
# 265
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t 
# 266
cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 267
tex, const cudaArray *
# 268
array, const cudaChannelFormatDesc &
# 269
desc) 
# 271
{ 
# 272
return cudaBindTextureToArray(&tex, array, (&desc)); 
# 273
} 
# 275
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t 
# 276
cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 277
tex, const cudaArray *
# 278
array) 
# 280
{ 
# 281
auto cudaChannelFormatDesc desc; 
# 282
auto cudaError_t err = cudaGetChannelDesc(&desc, array); 
# 284
return (err == (cudaSuccess)) ? (cudaBindTextureToArray(tex, array, desc)) : err; 
# 285
} 
# 293
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t 
# 294
cudaUnbindTexture(const texture< T, dim, readMode>  &
# 295
tex) 
# 297
{ 
# 298
return cudaUnbindTexture(&tex); 
# 299
} 
# 307
template<class T, int dim, cudaTextureReadMode readMode> inline cudaError_t 
# 308
cudaGetTextureAlignmentOffset(size_t *
# 309
offset, const texture< T, dim, readMode>  &
# 310
tex) 
# 312
{ 
# 313
return cudaGetTextureAlignmentOffset(offset, &tex); 
# 314
} 
# 322
template<class T> inline cudaError_t 
# 323
cudaLaunch(T *
# 324
symbol) 
# 326
{ 
# 327
return cudaLaunch((const char *)symbol); 
# 328
} 
# 16 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/kernel_type_binding.h"
namespace cupp { 
# 18
namespace impl { 
# 27
template<class T> 
# 28
class has_typdefs { 
# 30
typedef char one; 
# 31
typedef char (&two)[2]; 
# 33
template<class R> struct helper; 
# 35
template<class S> static one check(helper< typename S::host_type>  *, helper< typename S::device_type>  *); 
# 36
template<class S> static two check(...); 
# 39
public: enum __cuda_value { value = (sizeof((check< T> (0, 0))) == sizeof(char))}; 
# 40
}; 
# 50
template<bool POD, class T> 
# 51
struct get_type { 
# 52
typedef typename T::host_type host_type; 
# 53
typedef typename T::device_type device_type; 
# 54
}; 
# 59
template<class T> 
# 60
struct get_type< false, T>  { 
# 61
typedef T host_type; 
# 62
typedef T device_type; 
# 63
}; 
# 65
}
# 74
template<class T> 
# 75
struct get_type { 
# 76
typedef typename impl::get_type< impl::has_typdefs< T> ::value, T> ::host_type host_type; 
# 77
typedef typename impl::get_type< impl::has_typdefs< T> ::value, T> ::device_type device_type; 
# 78
}; 
# 160 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/kernel_type_binding.h"
}
# 14 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/memory1d.h"
namespace cupp { 
# 16
template<class T> class memory1d; 
# 19
namespace deviceT { 
# 30
template<class T, class host_type_ = cupp::memory1d< T> > 
# 31
class memory1d { 
# 36
public: typedef deviceT::memory1d< T>  device_type; 
# 37
typedef host_type_ host_type; 
# 43
typedef size_t size_type; 
# 49
typedef T value_type; 
# 58
inline size_type size() const; 
# 78
void set_device_pointer(T *); 
# 81
void set_size(const size_type); 
# 87
T *device_pointer_; 
# 92
size_type size_; 
# 94
}; 
# 108
template<class T, class host_type> inline typename memory1d< T, host_type> ::size_type 
# 109
memory1d< T, host_type> ::size() const { 
# 110
return this->size_; 
# 111
} 
# 113
template<class T, class host_type> void 
# 114
memory1d< T, host_type> ::set_device_pointer(T *device_pointer) { 
# 115
(this->device_pointer_) = device_pointer; 
# 116
} 
# 118
template<class T, class host_type> void 
# 119
memory1d< T, host_type> ::set_size(const size_type size) { 
# 120
(this->size_) = size; 
# 121
} 
# 123
}
# 124
}
# 13 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/include/cupp/deviceT/vector.h"
namespace cupp { 
# 15
template<class T> class vector; 
# 18
namespace deviceT { 
# 28
template<class T> 
# 29
class vector { 
# 34
public: typedef deviceT::vector< T>  device_type; 
# 35
typedef cupp::vector< typename get_type< T> ::host_type>  host_type; 
# 41
typedef typename memory1d< T, cupp::vector< T> > ::size_type size_type; 
# 47
typedef typename memory1d< T, cupp::vector< T> > ::value_type value_type; 
# 55
inline size_type size() const; 
# 75
void set_device_pointer(T *); 
# 78
void set_size(const size_type); 
# 84
T *device_pointer_; 
# 89
size_type size_; 
# 91
}; 
# 104
template<class T> inline typename vector< T> ::size_type 
# 105
vector< T> ::size() const { 
# 106
return this->size_; 
# 107
} 
# 109
template<class T> void 
# 110
vector< T> ::set_device_pointer(T *device_pointer) { 
# 111
(this->device_pointer_) = device_pointer; 
# 112
} 
# 114
template<class T> void 
# 115
vector< T> ::set_size(const typename memory1d< T, cupp::vector< T> > ::size_type size) { 
# 116
(this->size_) = size; 
# 117
} 
# 119
}
# 120
}
# 11 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector_complex/kernel_t.h"
typedef void (*kernelT)(cupp::deviceT::vector< cupp::deviceT::vector< int> >  &); 
# 14
extern kernelT get_kernel(); 
# 9 "/home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector_complex/kernel_vector_complex.cu"
using namespace cupp;
#define __include___device_stub__Z15global_functionRN4cupp7deviceT6vectorINS1_IiEEEE 1
#include "kernel_vector_complex.cudafe1.stub.h"
#undef __include___device_stub__Z15global_functionRN4cupp7deviceT6vectorINS1_IiEEEE
# 15
kernelT get_kernel() { 
# 16
return (kernelT)__device_stub__Z15global_functionRN4cupp7deviceT6vectorINS1_IiEEEE; 
# 17
} 

#include "kernel_vector_complex.cudafe1.stub.c"
