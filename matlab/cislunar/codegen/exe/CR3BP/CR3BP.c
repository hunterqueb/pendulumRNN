/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: CR3BP.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 10:23:51
 */

/* Include Files */
#include "CR3BP.h"
#include "CR3BP_data.h"
#include "CR3BP_initialize.h"
#include "tic.h"
#include "toc.h"
#include "coder_posix_time.h"

/* Function Definitions */
/*
 * Arguments    : void
 * Return Type  : void
 */
void CR3BP(void)
{
  coderTimespec savedTime;
  if (!isInitialized_CR3BP) {
    CR3BP_initialize();
  }
  /*  tEnd = 2.187543568657557; */
  /* tolerance  */
  tic(&savedTime);
  toc(&savedTime);
}

/*
 * File trailer for CR3BP.c
 *
 * [EOF]
 */
