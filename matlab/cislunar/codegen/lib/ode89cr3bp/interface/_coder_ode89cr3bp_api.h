/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: _coder_ode89cr3bp_api.h
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 13:49:44
 */

#ifndef _CODER_ODE89CR3BP_API_H
#define _CODER_ODE89CR3BP_API_H

/* Include Files */
#include "emlrt.h"
#include "mex.h"
#include "tmwtypes.h"
#include <string.h>

/* Type Definitions */
#ifndef struct_emxArray_real_T
#define struct_emxArray_real_T
struct emxArray_real_T {
  real_T *data;
  int32_T *size;
  int32_T allocatedSize;
  int32_T numDimensions;
  boolean_T canFreeData;
};
#endif /* struct_emxArray_real_T */
#ifndef typedef_emxArray_real_T
#define typedef_emxArray_real_T
typedef struct emxArray_real_T emxArray_real_T;
#endif /* typedef_emxArray_real_T */

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
void ode89cr3bp(real_T x0[6], real_T tEnd, real_T numPeriods,
                emxArray_real_T *tResult, emxArray_real_T *XODE);

void ode89cr3bp_api(const mxArray *const prhs[3], int32_T nlhs,
                    const mxArray *plhs[2]);

void ode89cr3bp_atexit(void);

void ode89cr3bp_initialize(void);

void ode89cr3bp_terminate(void);

void ode89cr3bp_xil_shutdown(void);

void ode89cr3bp_xil_terminate(void);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for _coder_ode89cr3bp_api.h
 *
 * [EOF]
 */
