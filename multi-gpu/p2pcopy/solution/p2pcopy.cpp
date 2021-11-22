#include "stdio.h"
#include "stdint.h"
#include <time.h>
#include <hip/hip_runtime.h>

//HIP error handling macro
#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
	if (err != hipSuccess) {
		printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

void copyP2P(int p2p, int gpu0, int gpu1, int* dA_0, int* dA_1, int size){
    
  //Enable peer access for GPUs
  if(p2p)
  {
    HIP_ERRCHK(hipSetDevice(gpu0));
    HIP_ERRCHK(hipDeviceEnablePeerAccess(gpu1, 0));
    HIP_ERRCHK(hipSetDevice(gpu1));
    HIP_ERRCHK(hipDeviceEnablePeerAccess(gpu0, 0));
  }

  //Do the first copy without timing to remove the impact of the first memcpy
  HIP_ERRCHK(hipMemcpy(dA_0, dA_1, size, hipMemcpyDefault));

  //Do a timed P2P memcpys
  int copies = 10;
  clock_t tStart = clock();
  for (int i = 0; i < copies; ++i) {
    HIP_ERRCHK(hipMemcpy(dA_0, dA_1, size, hipMemcpyDefault));
  }
  HIP_ERRCHK(hipStreamSynchronize(0));
  clock_t tStop = clock();
  
  //Calcute time and bandwith
  double time_s = (double)(tStop - tStart) / CLOCKS_PER_SEC;
  double bandwidth = (double)size * (double)copies / (double)1e9 / time_s;
  
  //Disable peer access for GPUs
  if(p2p) 
  {
    HIP_ERRCHK(hipSetDevice(gpu0));
    HIP_ERRCHK(hipDeviceDisablePeerAccess(gpu1));
    HIP_ERRCHK(hipSetDevice(gpu1));
    HIP_ERRCHK(hipDeviceDisablePeerAccess(gpu0));
    printf("P2P enabled - Bandwith: %.3f (GB/s), Time: %.3f s\n", bandwidth, time_s);
  }
  else
  {
    printf("P2P disabled - Bandwith: %.3f (GB/s), Time: %.3f s\n", bandwidth, time_s);
  }
}

int main(int argc, char *argv[])
{
    
  //Check device count
  int devCount; 
  HIP_ERRCHK(hipGetDeviceCount(&devCount));
  if(devCount < 2)
  {
    printf("Need at least two GPUs!\n");
    exit(EXIT_FAILURE);
  }
    
  // Allocate memory for both GPUs
  int gpu0 = 0, gpu1 = 1, *dA_0, *dA_1;
  int size = pow(2, 28);

  HIP_ERRCHK(hipSetDevice(gpu0));
  HIP_ERRCHK(hipMalloc((void**)&dA_0, size));

  HIP_ERRCHK(hipSetDevice(gpu1));
  HIP_ERRCHK(hipMalloc((void**)&dA_1, size));

  //Check peer accessibility between GPUs 0 and 1
  int peerAccess01;
  int peerAccess10;
  HIP_ERRCHK(hipDeviceCanAccessPeer(&peerAccess01, gpu0, gpu1));
  HIP_ERRCHK(hipDeviceCanAccessPeer(&peerAccess10, gpu1, gpu0));
  printf("hipDeviceCanAccessPeer: %d (GPU %d to GPU %d)\n", peerAccess01, gpu0, gpu1);
  printf("hipDeviceCanAccessPeer: %d (GPU %d to GPU %d)\n", peerAccess10, gpu1, gpu0);
 
  //Memcopy, P2P enabled
  if (peerAccess01 && peerAccess10)
    copyP2P(1, gpu0, gpu1, dA_0, dA_1, size);

  //Memcopy, P2P disabled
  copyP2P(0, gpu0, gpu1, dA_0, dA_1, size);
  
  //Free device allocations
  HIP_ERRCHK(hipFree(dA_0));
  HIP_ERRCHK(hipFree(dA_1));
}
