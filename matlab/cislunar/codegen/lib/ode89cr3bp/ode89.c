/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ode89.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 13:49:44
 */

/* Include Files */
#include "ode89.h"
#include "ode89cr3bp.h"
#include "ode89cr3bp_emxutil.h"
#include "ode89cr3bp_rtwutil.h"
#include "ode89cr3bp_types.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"
#include <math.h>
#include <string.h>

/* Function Definitions */
/*
 * Arguments    : const emxArray_real_T *tspan
 *                const double b_y0[6]
 *                emxArray_real_T *varargout_1
 *                emxArray_real_T *varargout_2
 * Return Type  : void
 */
void ode89(const emxArray_real_T *tspan, const double b_y0[6],
           emxArray_real_T *varargout_1, emxArray_real_T *varargout_2)
{
  static const double BI2[14] = {
      -12.753040692823889, -0.72057856025085987, -48.069691071487554,
      16.323457884253532,  -5.888504109270885,   -69.228211006868563,
      -38.046680725851893, -75.215988996101871,  -19.465886391170532,
      22.252769646169018,  14.38227638804284,    94.927562972880509,
      63.977571285183132,  57.524943377297021};
  static const double BI3[14] = {
      68.544701138311623,  6.5591194520909966,  451.82800481387454,
      -118.70005449434305, 89.447041137159417,  627.44028835681524,
      340.98943797822335,  670.55517565639661,  172.84424194045155,
      -202.22394933405374, -301.88629212849133, -819.38105212649634,
      -398.47103694661422, -587.54562544332475};
  static const double BI4[14] = {
      -194.80866105296525, -23.654171833483552, -1652.4971812128811,
      379.7566858308295,   -380.52125191333255, -2258.3514339668986,
      -1221.0896373201583, -2395.4252530528292, -616.3040486605513,
      741.13805869267912,  1781.1588155601307,  2811.2877107277009,
      716.59668699058045,  2312.7136812111785};
  static const double BI5[14] = {
      317.73924400589175,  44.168956092815563,  3109.7406407590311,
      -658.71105350578171, 770.99372724733769,  4212.3958002154923,
      2271.1215335552592,  4449.1432316271021,  1143.4836374440097,
      -1419.3459915439014, -4706.7855554044454, -4973.9945835413719,
      52.8905216646184,    -4612.8401086160557};
  static const double BI6[14] = {
      -299.51103170165538, -45.39036357573746,  -3211.322751886536,
      644.280520604847,    -831.471689354838,   -4325.9161491924906,
      -2328.0991205857131, -4556.7690501793295, -1170.3576696386438,
      1505.7234531151748,  6291.6458318775594,  4810.8679804449812,
      -1470.5008449627007, 4986.820885035082};
  static const double BI7[14] = {
      151.95042488515421,  24.434613891115724,  1734.4328663747124,
      -334.87518257678943, 461.93871778719159,  2327.6482614562428,
      1251.1326318309245,  2447.3653500866567,  628.2905498131297,
      -841.35803757610779, -4164.4324574681059, -2417.5631703814524,
      1524.4331339911969,  -2793.3977021138694};
  static const double BI8[14] = {
      -32.147047729128992, -5.3955512686625244, -383.89408306825595,
      72.05311579106953,   -104.27357901970106, -513.80983041316631,
      -275.93222128510257, -539.52398055397464, -138.46134705961285,
      193.81369700003981,  1085.9173811753094,  493.85555190375771,
      -488.92603202226388, 636.72392654969224};
  emxArray_real_T *tout;
  emxArray_real_T *yout;
  double f[84];
  double f0[6];
  double y[6];
  const double *tspan_data;
  double absh;
  double absx;
  double err;
  double hmax;
  double t;
  double tdir;
  double tfinal_tmp;
  double twidth;
  double *tout_data;
  double *varargout_1_data;
  double *yout_data;
  int b_i;
  int exponent;
  int i;
  int k;
  int next;
  int nnxt;
  int nout;
  bool Done;
  bool MinStepExit;
  tspan_data = tspan->data;
  tfinal_tmp = tspan_data[tspan->size[1] - 1];
  ode89cr3bp_anonFcn1(b_y0, f0);
  emxInit_real_T(&tout, 2);
  i = tout->size[0] * tout->size[1];
  tout->size[0] = 1;
  next = tspan->size[1];
  tout->size[1] = tspan->size[1];
  emxEnsureCapacity_real_T(tout, i);
  tout_data = tout->data;
  for (i = 0; i < next; i++) {
    tout_data[i] = 0.0;
  }
  emxInit_real_T(&yout, 2);
  i = yout->size[0] * yout->size[1];
  yout->size[0] = 6;
  yout->size[1] = tspan->size[1];
  emxEnsureCapacity_real_T(yout, i);
  yout_data = yout->data;
  next = 6 * tspan->size[1];
  for (i = 0; i < next; i++) {
    yout_data[i] = 0.0;
  }
  nout = 1;
  tout_data[0] = tspan_data[0];
  for (i = 0; i < 6; i++) {
    yout_data[i] = b_y0[i];
  }
  tdir = tfinal_tmp - tspan_data[0];
  twidth = fabs(tdir);
  absx = fabs(tspan_data[0]);
  hmax = fmin(twidth, fmax(0.1 * twidth, 3.5527136788005009E-15 *
                                             fmax(absx, fabs(tfinal_tmp))));
  if (rtIsInf(absx) || rtIsNaN(absx)) {
    err = rtNaN;
  } else if (absx < 4.4501477170144028E-308) {
    err = 4.94065645841247E-324;
  } else {
    frexp(absx, &nnxt);
    err = ldexp(1.0, nnxt - 53);
  }
  absh = fmin(hmax, fabs(tspan_data[1] - tspan_data[0]));
  twidth = 0.0;
  for (k = 0; k < 6; k++) {
    absx = fabs(f0[k] / fmax(fabs(b_y0[k]), 0.045035996273704963));
    if (rtIsNaN(absx) || (absx > twidth)) {
      twidth = absx;
    }
  }
  twidth /= 0.024323596469704865;
  if (absh * twidth > 1.0) {
    absh = 1.0 / twidth;
  }
  absh = fmax(absh, 16.0 * err);
  t = tspan_data[0];
  for (b_i = 0; b_i < 6; b_i++) {
    y[b_i] = b_y0[b_i];
  }
  memset(&f[0], 0, 84U * sizeof(double));
  for (i = 0; i < 6; i++) {
    f[i] = f0[i];
  }
  if (!rtIsNaN(tdir)) {
    if (tdir < 0.0) {
      tdir = -1.0;
    } else {
      tdir = (tdir > 0.0);
    }
  }
  next = 0;
  MinStepExit = false;
  Done = false;
  int exitg1;
  do {
    double b_y[6];
    double f4[6];
    double d;
    double h;
    double hmin;
    double tnew;
    bool NoFailedAttempts;
    exitg1 = 0;
    absx = fabs(t);
    if (rtIsInf(absx) || rtIsNaN(absx)) {
      err = rtNaN;
    } else if (absx < 4.4501477170144028E-308) {
      err = 4.94065645841247E-324;
    } else {
      frexp(absx, &exponent);
      err = ldexp(1.0, exponent - 53);
    }
    hmin = 16.0 * err;
    absh = fmin(hmax, fmax(hmin, absh));
    h = tdir * absh;
    d = tfinal_tmp - t;
    twidth = fabs(d);
    if (1.1 * absh >= twidth) {
      h = d;
      absh = twidth;
      Done = true;
    }
    NoFailedAttempts = true;
    int exitg2;
    do {
      double f3[6];
      double f6[6];
      double f3_tmp;
      exitg2 = 0;
      if (t == tspan_data[0]) {
        for (i = 0; i < 6; i++) {
          f[i] = f0[i];
        }
      } else if (NoFailedAttempts) {
        ode89cr3bp_anonFcn1(y, &f[0]);
      }
      twidth = h * 0.04;
      for (i = 0; i < 6; i++) {
        b_y[i] = y[i] + twidth * f[i];
      }
      ode89cr3bp_anonFcn1(b_y, f3);
      for (i = 0; i < 6; i++) {
        b_y[i] = y[i] + h * (-0.01988527319182291 * f[i] +
                             0.11637263332969652 * f3[i]);
      }
      ode89cr3bp_anonFcn1(b_y, f3);
      for (i = 0; i < 6; i++) {
        b_y[i] = y[i] +
                 h * (0.0361827600517026 * f[i] + 0.10854828015510781 * f3[i]);
      }
      ode89cr3bp_anonFcn1(b_y, f4);
      for (i = 0; i < 6; i++) {
        b_y[i] =
            y[i] +
            h * ((2.2721142642901775 * f[i] + -8.5268864479763984 * f3[i]) +
                 6.8307721836862214 * f4[i]);
      }
      ode89cr3bp_anonFcn1(b_y, f3);
      for (i = 0; i < 6; i++) {
        b_y[i] =
            y[i] +
            h * ((0.050943855353893744 * f[i] + 0.17558650498090711 * f4[i]) +
                 0.00070229612707574678 * f3[i]);
      }
      ode89cr3bp_anonFcn1(b_y, f6);
      for (i = 0; i < 6; i++) {
        f3[i] =
            y[i] +
            h * (((0.14247836686832849 * f[i] + -0.35417994346686843 * f4[i]) +
                  0.075953154502951009 * f3[i]) +
                 0.6765157656337123 * f6[i]);
      }
      ode89cr3bp_anonFcn1(f3, f4);
      for (i = 0; i < 6; i++) {
        b_y[i] =
            y[i] +
            h * ((0.071111111111111111 * f[i] + 0.32799092876058983 * f6[i]) +
                 0.24089796012829906 * f4[i]);
      }
      ode89cr3bp_anonFcn1(b_y, &f[6]);
      for (i = 0; i < 6; i++) {
        b_y[i] = y[i] + h * (((0.07125 * f[i] + 0.32688424515752457 * f6[i]) +
                              0.11561575484247544 * f4[i]) +
                             -0.03375 * f[i + 6]);
      }
      ode89cr3bp_anonFcn1(b_y, &f[12]);
      for (i = 0; i < 6; i++) {
        b_y[i] =
            y[i] +
            h * ((((0.048226773224658105 * f[i] + 0.039485599804954 * f6[i]) +
                   0.10588511619346581 * f4[i]) +
                  -0.021520063204743093 * f[i + 6]) +
                 -0.10453742601833482 * f[i + 12]);
      }
      ode89cr3bp_anonFcn1(b_y, &f[18]);
      for (i = 0; i < 6; i++) {
        b_y[i] = y[i] + h * (((((-0.026091134357549235 * f[i] +
                                 0.033333333333333333 * f6[i]) +
                                -0.1652504006638105 * f4[i]) +
                               0.034346641183686168 * f[i + 6]) +
                              0.1595758283215209 * f[i + 12]) +
                             0.21408573218281934 * f[i + 18]);
      }
      ode89cr3bp_anonFcn1(b_y, &f[24]);
      for (i = 0; i < 6; i++) {
        b_y[i] = y[i] + h * ((((((-0.036284233962556589 * f[i] +
                                  -1.0961675974272087 * f6[i]) +
                                 0.18260355043213311 * f4[i]) +
                                0.070822544441706839 * f[i + 6]) +
                               -0.023136470184824311 * f[i + 12]) +
                              0.27112047263209327 * f[i + 18]) +
                             1.3081337494229808 * f[i + 24]);
      }
      ode89cr3bp_anonFcn1(b_y, &f[30]);
      for (i = 0; i < 6; i++) {
        b_y[i] = y[i] + h * (((((((-0.50746350564169751 * f[i] +
                                   -6.6313421986572374 * f6[i]) +
                                  -0.2527480100908801 * f4[i]) +
                                 -0.49526123800360955 * f[i + 6]) +
                                0.29325255452538868 * f[i + 12]) +
                               1.440108693768281 * f[i + 18]) +
                              6.2379344986470562 * f[i + 24]) +
                             0.72701920545269871 * f[i + 30]);
      }
      ode89cr3bp_anonFcn1(b_y, &f[36]);
      for (i = 0; i < 6; i++) {
        b_y[i] = y[i] + h * ((((((((0.6130118256955932 * f[i] +
                                    9.0888038916404632 * f6[i]) +
                                   -0.40737881562934486 * f4[i]) +
                                  1.7907333894903747 * f[i + 6]) +
                                 0.714927166761755 * f[i + 12]) +
                                -1.438580857841723 * f[i + 18]) +
                               -8.26332931206474 * f[i + 24]) +
                              -1.5375705708088652 * f[i + 30]) +
                             0.34538328275648716 * f[i + 36]);
      }
      ode89cr3bp_anonFcn1(b_y, &f[42]);
      for (i = 0; i < 6; i++) {
        b_y[i] = y[i] + h * (((((((((-1.2116979103438739 * f[i] +
                                     -19.055818715595954 * f6[i]) +
                                    1.2630606753898752 * f4[i]) +
                                   -6.9139169691784579 * f[i + 6]) +
                                  -0.676462266509498 * f[i + 12]) +
                                 3.3678604450266079 * f[i + 18]) +
                                18.006751643125909 * f[i + 24]) +
                               6.83882892679428 * f[i + 30]) +
                              -1.0315164519219504 * f[i + 36]) +
                             0.41291062321306227 * f[i + 42]);
      }
      ode89cr3bp_anonFcn1(b_y, &f[48]);
      for (i = 0; i < 6; i++) {
        f6[i] = y[i] + h * ((((((((2.1573890074940536 * f[i] +
                                   23.807122198095804 * f6[i]) +
                                  0.88627792492165558 * f4[i]) +
                                 13.139130397598764 * f[i + 6]) +
                                -2.6044157092877147 * f[i + 12]) +
                               -5.1938599497838727 * f[i + 18]) +
                              -20.412340711541507 * f[i + 24]) +
                             -12.300856252505723 * f[i + 30]) +
                            1.5215530950085394 * f[i + 36]);
      }
      ode89cr3bp_anonFcn1(f6, f3);
      for (i = 0; i < 6; i++) {
        double b_f3_tmp;
        double c_f3_tmp;
        double d_f3_tmp;
        twidth = f[i + 6];
        absx = f[i + 12];
        err = f[i + 18];
        f3_tmp = f[i + 24];
        tnew = f[i + 30];
        b_f3_tmp = f[i + 36];
        c_f3_tmp = f[i + 42];
        d_f3_tmp = f[i + 48];
        d = f[i];
        f3[i] =
            ((((((((0.0057578137681889487 * d + 1.0675934530948108 * twidth) +
                   -0.14099636134393978 * absx) +
                  -0.014411715396914925 * err) +
                 0.030796961251883033 * f3_tmp) +
                -1.1613152578179067 * tnew) +
               0.32221113486118586 * b_f3_tmp) +
              -0.12948458791975614 * c_f3_tmp) +
             -0.029477447612619417 * d_f3_tmp) +
            0.04932600711506839 * f3[i];
        f4[i] = y[i] + h * ((((((((0.014588852784055396 * d +
                                   0.0020241978878893325 * twidth) +
                                  0.21780470845697167 * absx) +
                                 0.12748953408543898 * err) +
                                0.22446177454631319 * f3_tmp) +
                               0.17872544912599031 * tnew) +
                              0.075943447580965578 * b_f3_tmp) +
                             0.12948458791975614 * c_f3_tmp) +
                            0.029477447612619417 * d_f3_tmp);
      }
      tnew = t + h;
      if (Done) {
        tnew = tfinal_tmp;
      }
      h = tnew - t;
      if (NoFailedAttempts) {
        f3_tmp = 0.0;
        for (k = 0; k < 6; k++) {
          twidth = fabs(f3[k]);
          absx = fabs(y[k]);
          err = fabs(f4[k]);
          if ((absx > err) || rtIsNaN(err)) {
            if (absx > 0.045035996273704963) {
              twidth /= absx;
            } else {
              twidth /= 0.045035996273704963;
            }
          } else if (err > 0.045035996273704963) {
            twidth /= err;
          } else {
            twidth /= 0.045035996273704963;
          }
          if ((twidth > f3_tmp) || rtIsNaN(twidth)) {
            f3_tmp = twidth;
          }
        }
        err = absh * f3_tmp;
      } else {
        f3_tmp = 0.0;
        for (k = 0; k < 6; k++) {
          twidth = fabs(f3[k]);
          absx = fabs(y[k]);
          if (absx > 0.045035996273704963) {
            twidth /= absx;
          } else {
            twidth /= 0.045035996273704963;
          }
          if ((twidth > f3_tmp) || rtIsNaN(twidth)) {
            f3_tmp = twidth;
          }
        }
        err = absh * f3_tmp;
      }
      if (!(err <= 2.2204460492503131E-14)) {
        if (absh <= hmin) {
          MinStepExit = true;
          exitg2 = 1;
        } else {
          if (NoFailedAttempts) {
            NoFailedAttempts = false;
            absh = fmax(
                hmin,
                absh * fmax(0.1, 0.8 * rt_powd_snf(2.2204460492503131E-14 / err,
                                                   0.1111111111111111)));
          } else {
            absh = fmax(hmin, 0.5 * absh);
          }
          h = tdir * absh;
          Done = false;
        }
      } else {
        exitg2 = 1;
      }
    } while (exitg2 == 0);
    if (MinStepExit) {
      exitg1 = 1;
    } else {
      int noutnew;
      nnxt = next;
      while ((nnxt + 2 <= tspan->size[1]) &&
             (tdir * (tnew - tspan_data[nnxt + 1]) >= 0.0)) {
        nnxt++;
      }
      noutnew = nnxt - next;
      if (noutnew > 0) {
        double b[14];
        for (i = 0; i < 6; i++) {
          b_y[i] = y[i] + h * ((((((((0.014588852784055396 * f[i] +
                                      0.0020241978878893325 * f[i + 6]) +
                                     0.21780470845697167 * f[i + 12]) +
                                    0.12748953408543898 * f[i + 18]) +
                                   0.22446177454631319 * f[i + 24]) +
                                  0.17872544912599031 * f[i + 30]) +
                                 0.075943447580965578 * f[i + 36]) +
                                0.12948458791975614 * f[i + 42]) +
                               0.029477447612619417 * f[i + 48]);
        }
        ode89cr3bp_anonFcn1(b_y, &f[54]);
        for (i = 0; i < 6; i++) {
          b_y[i] = y[i] + h * (((((((((0.015601405261088616 * f[i] +
                                       0.26811643933275847 * f[i + 6]) +
                                      0.1883053124587791 * f[i + 12]) +
                                     0.12491991374610308 * f[i + 18]) +
                                    0.2302302127814522 * f[i + 24]) +
                                   -0.13603122161327985 * f[i + 30]) +
                                  0.074886599713069532 * f[i + 36]) +
                                 -0.028128400297956289 * f[i + 42]) +
                                -0.023144557264819496 * f[i + 48]) +
                               0.027345304241113474 * f[i + 54]);
        }
        ode89cr3bp_anonFcn1(b_y, &f[60]);
        for (i = 0; i < 6; i++) {
          b_y[i] = y[i] + h * ((((((((((0.013111957218440684 * f[i] +
                                        -0.14640242659698269 * f[i + 6]) +
                                       0.2471264389666796 * f[i + 12]) +
                                      0.13113752030800324 * f[i + 18]) +
                                     0.21705603469825827 * f[i + 24]) +
                                    0.286753671376032 * f[i + 30]) +
                                   0.023233113391494219 * f[i + 36]) +
                                  0.052506772641993958 * f[i + 42]) +
                                 0.0028339515860099506 * f[i + 48]) +
                                -0.0085024038519957122 * f[i + 54]) +
                               0.069145370262066491 * f[i + 60]);
        }
        ode89cr3bp_anonFcn1(b_y, &f[66]);
        for (i = 0; i < 6; i++) {
          b_y[i] = y[i] + h * (((((((((((0.013989212133617684 * f[i] +
                                         -0.031574065179505 * f[i + 6]) +
                                        0.22718125132721581 * f[i + 12]) +
                                       0.12894864109967866 * f[i + 18]) +
                                      0.2216682589135277 * f[i + 24]) +
                                     0.19483682365424806 * f[i + 30]) +
                                    0.05740088404417653 * f[i + 36]) +
                                   0.090083665426759552 * f[i + 42]) +
                                  0.015791532088442122 * f[i + 48]) +
                                 -0.018991315059091858 * f[i + 54]) +
                                -0.08830926811918835 * f[i + 60]) +
                               -0.11502562032988092 * f[i + 66]);
        }
        ode89cr3bp_anonFcn1(b_y, &f[72]);
        for (i = 0; i < 6; i++) {
          b_y[i] = y[i] + h * ((((((((((((0.016151472919007624 * f[i] +
                                          0.080986850032429059 * f[i + 6]) +
                                         0.12769162943069304 * f[i + 12]) +
                                        0.12348143593834805 * f[i + 18]) +
                                       0.233985125914011 * f[i + 24]) +
                                      -0.065959956833573682 * f[i + 30]) +
                                     -0.025652768594064328 * f[i + 36]) +
                                    -0.12589734638192471 * f[i + 42]) +
                                   -0.043076724903648438 * f[i + 48]) +
                                  0.04973042479196705 * f[i + 54]) +
                                 0.10004735401793927 * f[i + 60]) +
                                0.13786588067636232 * f[i + 66]) +
                               -0.12235337700754625 * f[i + 72]);
        }
        ode89cr3bp_anonFcn1(b_y, &f[78]);
        for (k = next + 2; k <= nnxt; k++) {
          d = tspan_data[k - 1];
          tout_data[k - 1] = d;
          twidth = (d - t) / h;
          absx = twidth * twidth;
          for (b_i = 0; b_i < 14; b_i++) {
            b[b_i] = ((((((BI8[b_i] * twidth + BI7[b_i]) * twidth + BI6[b_i]) *
                             twidth +
                         BI5[b_i]) *
                            twidth +
                        BI4[b_i]) *
                           twidth +
                       BI3[b_i]) *
                          twidth +
                      BI2[b_i]) *
                     absx;
          }
          b[0] += twidth;
          for (i = 0; i < 6; i++) {
            d = 0.0;
            for (b_i = 0; b_i < 14; b_i++) {
              d += f[i + 6 * b_i] * b[b_i];
            }
            yout_data[i + 6 * (k - 1)] = y[i] + h * d;
          }
        }
        tout_data[nnxt] = tspan_data[nnxt];
        if (tspan_data[nnxt] == tnew) {
          for (i = 0; i < 6; i++) {
            yout_data[i + 6 * nnxt] = f4[i];
          }
        } else {
          twidth = (tspan_data[nnxt] - t) / h;
          absx = twidth * twidth;
          for (b_i = 0; b_i < 14; b_i++) {
            b[b_i] = ((((((BI8[b_i] * twidth + BI7[b_i]) * twidth + BI6[b_i]) *
                             twidth +
                         BI5[b_i]) *
                            twidth +
                        BI4[b_i]) *
                           twidth +
                       BI3[b_i]) *
                          twidth +
                      BI2[b_i]) *
                     absx;
          }
          b[0] += twidth;
          for (i = 0; i < 6; i++) {
            d = 0.0;
            for (b_i = 0; b_i < 14; b_i++) {
              d += f[i + 6 * b_i] * b[b_i];
            }
            yout_data[i + 6 * nnxt] = y[i] + h * d;
          }
        }
        nout += noutnew;
        next = nnxt;
      }
      if (Done) {
        exitg1 = 1;
      } else {
        if (NoFailedAttempts) {
          twidth = 1.25 * rt_powd_snf(err / 2.2204460492503131E-14,
                                      0.1111111111111111);
          if (twidth > 0.2) {
            absh /= twidth;
          } else {
            absh *= 5.0;
          }
        }
        t = tnew;
        for (b_i = 0; b_i < 6; b_i++) {
          y[b_i] = f4[b_i];
        }
      }
    }
  } while (exitg1 == 0);
  i = varargout_1->size[0];
  varargout_1->size[0] = nout;
  emxEnsureCapacity_real_T(varargout_1, i);
  varargout_1_data = varargout_1->data;
  for (i = 0; i < nout; i++) {
    varargout_1_data[i] = tout_data[i];
  }
  emxFree_real_T(&tout);
  i = varargout_2->size[0] * varargout_2->size[1];
  varargout_2->size[0] = nout;
  varargout_2->size[1] = 6;
  emxEnsureCapacity_real_T(varargout_2, i);
  varargout_1_data = varargout_2->data;
  for (i = 0; i < 6; i++) {
    for (b_i = 0; b_i < nout; b_i++) {
      varargout_1_data[b_i + varargout_2->size[0] * i] = yout_data[i + 6 * b_i];
    }
  }
  emxFree_real_T(&yout);
}

/*
 * File trailer for ode89.c
 *
 * [EOF]
 */
