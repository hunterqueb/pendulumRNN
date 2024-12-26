/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: mainODE_emxutil.h
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 12:23:29
 */

#ifndef MAINODE_EMXUTIL_H
#define MAINODE_EMXUTIL_H

/* Include Files */
#include "mainODE_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
extern void emxEnsureCapacity_real_T(emxArray_real_T *emxArray, int oldNumel);

extern void emxFree_real_T(emxArray_real_T **pEmxArray);

extern void emxInit_real_T(emxArray_real_T **pEmxArray, int numDimensions);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for mainODE_emxutil.h
 *
 * [EOF]
 */
