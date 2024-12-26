/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: toc.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 12:23:29
 */

/* Include Files */
#include "toc.h"
#include "mainODE_data.h"
#include "rt_nonfinite.h"
#include "timeKeeper.h"
#include "coder_posix_time.h"
#include <stdio.h>

/* Function Definitions */
/*
 * Arguments    : const coderTimespec *savedTime
 * Return Type  : void
 */
void toc(const coderTimespec *savedTime)
{
  coderTimespec b_timespec;
  double tstart_tv_nsec;
  double tstart_tv_sec;
  tstart_tv_sec = b_timeKeeper(savedTime, &tstart_tv_nsec);
  if (!freq_not_empty) {
    freq_not_empty = true;
    coderInitTimeFunctions(&freq);
  }
  coderTimeClockGettimeMonotonic(&b_timespec, freq);
  printf("Elapsed time is %f seconds\n",
         (b_timespec.tv_sec - tstart_tv_sec) +
             (b_timespec.tv_nsec - tstart_tv_nsec) / 1.0E+9);
  fflush(stdout);
}

/*
 * File trailer for toc.c
 *
 * [EOF]
 */
