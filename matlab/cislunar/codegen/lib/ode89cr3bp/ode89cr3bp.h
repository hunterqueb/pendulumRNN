/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ode89cr3bp.h
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 13:49:44
 */

#ifndef ODE89CR3BP_H
#define ODE89CR3BP_H

/* Include Files */
#include "ode89cr3bp_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
extern void ode89cr3bp(const double x0[6], double tEnd, double numPeriods,
                       emxArray_real_T *tResult, emxArray_real_T *XODE);

void ode89cr3bp_anonFcn1(const double x[6], double varargout_1[6]);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for ode89cr3bp.h
 *
 * [EOF]
 */
