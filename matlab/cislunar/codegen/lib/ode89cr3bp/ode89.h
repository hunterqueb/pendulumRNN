/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ode89.h
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 13:49:44
 */

#ifndef ODE89_H
#define ODE89_H

/* Include Files */
#include "ode89cr3bp_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
void ode89(const emxArray_real_T *tspan, const double b_y0[6],
           emxArray_real_T *varargout_1, emxArray_real_T *varargout_2);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for ode89.h
 *
 * [EOF]
 */
