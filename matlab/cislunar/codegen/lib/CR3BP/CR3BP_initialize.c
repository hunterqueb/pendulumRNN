/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: CR3BP_initialize.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 10:26:05
 */

/* Include Files */
#include "CR3BP_initialize.h"
#include "CR3BP_data.h"
#include "CoderTimeAPI.h"
#include "timeKeeper.h"

/* Function Definitions */
/*
 * Arguments    : void
 * Return Type  : void
 */
void CR3BP_initialize(void)
{
  c_CoderTimeAPI_callCoderClockGe();
  timeKeeper_init();
  isInitialized_CR3BP = true;
}

/*
 * File trailer for CR3BP_initialize.c
 *
 * [EOF]
 */
