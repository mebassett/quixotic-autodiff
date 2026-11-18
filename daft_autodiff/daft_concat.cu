#include "daft_autodiff.h"
#include "daft_error.h"
#include "daft_concat.h"
#include "daft_kernels.cuh"

using namespace DA;

namespace DA {

  // so we are assuming the results of each of the targets are just a big 
  // continuous block in memory.  otherwise this won't work.  so be careful
  // using it.  it's mostly used to push the gradients down.
  // UPDATE: no longer to a big continuous block of memory.  now we do copying
  // and interleaving, which is slower.  concat adds a step.
  Operation Operation::concat(string name, const vector<string>& targets, uint size) {
      return { .opType=OperationType::Concat
             , .workingSize = 0
             , .resultSize = size // we assume the user set this correctly.
             , .gradSize = 0
             , .rows = size 
             , .cols = 1
             , .name{name}
             , .config { (ConcatConfig) {
                   .targets{targets}
                 , .size=size
             }}
      };
  }

  void Concat_computeGrad(Function* f, float* seed, Operation& op) {
    ConcatConfig opCnfg = get<ConcatConfig>(op.config);
    int targetSizes[32];
    int d_size = opCnfg.targets.size();
    int memIndex = 0;
    float* d_newSeed = f->getGrad(op.name);

    for (int i=0; i < d_size; i++) {
        Operation* targetOp;
        try {
            targetOp = f->opsMap.at(opCnfg.targets[i]);
        } catch (const out_of_range&) {
            break;
        }
        targetSizes[i] = targetOp->resultSize;
    }

     
    setConcatSizes(targetSizes, d_size);

    dim3 bd(16,16,4);
    dim3 gd(ceil((float)targetSizes[0] / 16.0), ceil((float)d_size / 16.0), ceil(f->batchSize / 4.0));


    doConcatDeinterleave<<<gd, bd>>>(seed, d_size, d_newSeed, opCnfg.size, f->batchSize);
    cudaErrCk( cudaPeekAtLastError() );


    for (auto target : opCnfg.targets) {
      Operation* targetOp;
      try {
        targetOp = f->opsMap.at(target);
      } catch (const out_of_range&) {
        break;
      }
      f->computeGrad(target, d_newSeed + memIndex);
      memIndex += targetOp->rows * targetOp->cols * f->batchSize;
    }
  }

  void Concat_compute(Function* f, Operation& op) {
    // ~no operation here, we could do a copy to get all the targets~
    // ~into one continuous block in memory, but we're just going~
    // ~to assume that's the case. ~
    // ~a smart compiler might enforce that, actually!~
    // with batch processing this gets tricky. if we compute A and B
    // then we store them as
    // A_1 ... A_n B_1 ... B_n
    // A smart compile could ensure that, but with batch processing
    // we actually expect this tob e
    // A_1 B_1 A_2 B_2 ... A_n B_n.
    // so our operation here is actually just to do the interleaving.
    // this is trivial is batchSize = 1.
    ConcatConfig opConfig = get<ConcatConfig>(op.config);

    vector<float*> hostTargets;
    float** d_targets = f->pointerLocs[op.name];
    int targetSizes[32];
    int d_size = opConfig.targets.size();
    for(int i=0;i<opConfig.targets.size();i++) {
        Operation* targetOp;
        try {
            targetOp = f->opsMap.at(opConfig.targets[i]);
        } catch (const out_of_range&) {
            cout << "compute(Concat) cannot find op " <<
              opConfig.targets[i];

            exit(1);
            return;
        }
        hostTargets.push_back(f->getResult(opConfig.targets[i]));
        targetSizes[i] = targetOp->resultSize;
    }

    setConcatSizes(targetSizes, d_size);
    cudaMemcpy(d_targets, &(hostTargets[0]), d_size*sizeof(float*), cudaMemcpyHostToDevice); 

    dim3 bd(16,16,4);
    dim3 gd(ceil((float)targetSizes[0] / 16.0), ceil((float)d_size / 16.0), ceil(f->batchSize / 4.0));

    doConcatInterleave<<<gd, bd>>>(d_targets, d_size, f->getResult(op.name), opConfig.size, f->batchSize);
    cudaErrCk( cudaPeekAtLastError() );

  }

}
