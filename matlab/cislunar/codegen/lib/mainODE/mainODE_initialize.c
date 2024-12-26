/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: mainODE_initialize.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 12:23:29
 */

/* Include Files */
#include "mainODE_initialize.h"
#include "CoderTimeAPI.h"
#include "mainODE_data.h"
#include "rt_nonfinite.h"
#include "timeKeeper.h"

/* Function Definitions */
/*
 * Arguments    : void
 * Return Type  : void
 */
void mainODE_initialize(void)
{
  c_CoderTimeAPI_callCoderClockGe();
  timeKeeper_init();
  isInitialized_mainODE = true;
}

/*
 * File trailer for mainODE_initialize.c
 *
 * [EOF]
 */
