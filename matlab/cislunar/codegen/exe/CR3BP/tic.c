/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: tic.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 10:23:51
 */

/* Include Files */
#include "tic.h"
#include "CR3BP_data.h"
#include "timeKeeper.h"
#include "coder_posix_time.h"

/* Function Definitions */
/*
 * Arguments    : coderTimespec *savedTime
 * Return Type  : void
 */
void tic(coderTimespec *savedTime)
{
  coderTimespec b_timespec;
  if (!freq_not_empty) {
    freq_not_empty = true;
    coderInitTimeFunctions(&freq);
  }
  coderTimeClockGettimeMonotonic(&b_timespec, freq);
  timeKeeper(b_timespec.tv_sec, b_timespec.tv_nsec, savedTime);
}

/*
 * File trailer for tic.c
 *
 * [EOF]
 */
