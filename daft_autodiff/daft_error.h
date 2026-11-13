// daft_error.h
#ifndef DAFT_ERROR_H
#define DAFT_ERROR_H
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

using namespace std;

namespace DA {
  #define cudaErrCk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
  inline void cudaAssert(cudaError_t err, const char *file, int line) {
      if(err != cudaSuccess) {
          fprintf(stderr, "CudaAssert: %s %s %d\n", cudaGetErrorString(err), file, line);
          exit(err);
      }
  }
  
  #define cublasErrCk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
  inline void cublasAssert(cublasStatus_t err, const char *file, int line) {
      if(err != CUBLAS_STATUS_SUCCESS) {
        const char* errMsg;
        switch(err) {
           case CUBLAS_STATUS_NOT_INITIALIZED: errMsg = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
           case CUBLAS_STATUS_ALLOC_FAILED: errMsg = "CUBLAS_STATUS_ALLOC_FAILED"; break;
           case CUBLAS_STATUS_INVALID_VALUE: errMsg = "CUBLAS_STATUS_INVALID_VALUE"; break;
           case CUBLAS_STATUS_ARCH_MISMATCH: errMsg = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
           case CUBLAS_STATUS_MAPPING_ERROR: errMsg = "CUBLAS_STATUS_MAPPING_ERROR"; break;
           case CUBLAS_STATUS_EXECUTION_FAILED: errMsg = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
           case CUBLAS_STATUS_INTERNAL_ERROR: errMsg = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
           case CUBLAS_STATUS_NOT_SUPPORTED: errMsg = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
           case CUBLAS_STATUS_LICENSE_ERROR: errMsg = "CUBLAS_STATUS_LICENSE_ERROR"; break;
           default: errMsg = "Unknown CUBLAS error"; break;
        }
        fprintf(stderr,"CUBLAS assert: %s %s %d\n", errMsg, file, line); 
        exit(err);
      }
  }
}

#endif /* DAFT_ERROR_H */
