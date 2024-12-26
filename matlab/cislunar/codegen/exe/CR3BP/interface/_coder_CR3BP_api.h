/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: _coder_CR3BP_api.h
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 10:23:51
 */

#ifndef _CODER_CR3BP_API_H
#define _CODER_CR3BP_API_H

/* Include Files */
#include "emlrt.h"
#include "mex.h"
#include "tmwtypes.h"
#include <string.h>

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
void CR3BP(void);

void CR3BP_api(void);

void CR3BP_atexit(void);

void CR3BP_initialize(void);

void CR3BP_terminate(void);

void CR3BP_xil_shutdown(void);

void CR3BP_xil_terminate(void);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for _coder_CR3BP_api.h
 *
 * [EOF]
 */
