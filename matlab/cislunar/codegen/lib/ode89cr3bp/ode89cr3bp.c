/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ode89cr3bp.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 13:49:44
 */

/* Include Files */
#include "ode89cr3bp.h"
#include "ode89.h"
#include "ode89cr3bp_emxutil.h"
#include "ode89cr3bp_rtwutil.h"
#include "ode89cr3bp_types.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */
/*
 * Arguments    : const double x0[6]
 *                double tEnd
 *                double numPeriods
 *                emxArray_real_T *tResult
 *                emxArray_real_T *XODE
 * Return Type  : void
 */
void ode89cr3bp(const double x0[6], double tEnd, double numPeriods,
                emxArray_real_T *tResult, emxArray_real_T *XODE)
{
  emxArray_real_T *t;
  double kd;
  double *t_data;
  int k;
  kd = numPeriods * tEnd;
  emxInit_real_T(&t, 2);
  if (rtIsNaN(kd)) {
    int nm1d2;
    nm1d2 = t->size[0] * t->size[1];
    t->size[0] = 1;
    t->size[1] = 1;
    emxEnsureCapacity_real_T(t, nm1d2);
    t_data = t->data;
    t_data[0] = rtNaN;
  } else if (kd < 0.0) {
    t->size[0] = 1;
    t->size[1] = 0;
  } else {
    double apnd;
    double cdiff;
    double ndbl;
    int nm1d2;
    ndbl = floor(kd / 0.01 + 0.5);
    apnd = ndbl * 0.01;
    cdiff = apnd - kd;
    if (fabs(cdiff) < 4.4408920985006262E-16 * kd) {
      ndbl++;
      apnd = kd;
    } else if (cdiff > 0.0) {
      apnd = (ndbl - 1.0) * 0.01;
    } else {
      ndbl++;
    }
    nm1d2 = t->size[0] * t->size[1];
    t->size[0] = 1;
    t->size[1] = (int)ndbl;
    emxEnsureCapacity_real_T(t, nm1d2);
    t_data = t->data;
    if ((int)ndbl > 0) {
      t_data[0] = 0.0;
      if ((int)ndbl > 1) {
        t_data[(int)ndbl - 1] = apnd;
        nm1d2 = ((int)ndbl - 1) / 2;
        for (k = 0; k <= nm1d2 - 2; k++) {
          kd = ((double)k + 1.0) * 0.01;
          t_data[k + 1] = kd;
          t_data[((int)ndbl - k) - 2] = apnd - kd;
        }
        if (nm1d2 << 1 == (int)ndbl - 1) {
          t_data[nm1d2] = apnd / 2.0;
        } else {
          kd = (double)nm1d2 * 0.01;
          t_data[nm1d2] = kd;
          t_data[nm1d2 + 1] = apnd - kd;
        }
      }
    }
  }
  /* tolerance  */
  ode89(t, x0, tResult, XODE);
  emxFree_real_T(&t);
}

/*
 * Arguments    : const double x[6]
 *                double varargout_1[6]
 * Return Type  : void
 */
void ode89cr3bp_anonFcn1(const double x[6], double varargout_1[6])
{
  double b_r1_tmp;
  double r1_tmp;
  double varargout_1_tmp;
  /*  Solve the CR3BP in nondimensional coordinates. */
  /*  */
  /*  The state vector is Y, with the first two components as the */
  /*  position of m, and the second two components its velocity. */
  /*  */
  /*  The solution is parameterized on mu, the mass ratio. */
  /*  */
  /*  Arguments: */
  /*  t: current time */
  /*  Y: current state vector */
  /*  mu: mass ratio (default value is mu = 0.012277471) */
  /*  */
  /*  Returns: */
  /*  dydt: derivative vector */
  /*  Get the position and velocity from the solution vector */
  r1_tmp = x[1] * x[1];
  b_r1_tmp = x[2] * x[2];
  varargout_1[0] = x[3];
  varargout_1[1] = x[4];
  varargout_1[2] = x[5];
  varargout_1_tmp = rt_powd_snf(
      sqrt(((x[0] + 0.012150515586657583) * (x[0] + 0.012150515586657583) +
            r1_tmp) +
           b_r1_tmp),
      3.0);
  r1_tmp = rt_powd_snf(sqrt((((x[0] - 1.0) + 0.012150515586657583) *
                                 ((x[0] - 1.0) + 0.012150515586657583) +
                             r1_tmp) +
                            b_r1_tmp),
                       3.0);
  varargout_1[3] =
      ((2.0 * x[4] + x[0]) -
       0.98784948441334242 * (x[0] + 0.012150515586657583) / varargout_1_tmp) -
      0.012150515586657583 * ((x[0] - 1.0) + 0.012150515586657583) / r1_tmp;
  varargout_1[4] =
      ((-2.0 * x[3] + x[1]) - 0.98784948441334242 * x[1] / varargout_1_tmp) -
      0.012150515586657583 * x[1] / r1_tmp;
  varargout_1[5] = -0.98784948441334242 * x[2] / varargout_1_tmp -
                   0.012150515586657583 * x[2] / r1_tmp;
}

/*
 * File trailer for ode89cr3bp.c
 *
 * [EOF]
 */
