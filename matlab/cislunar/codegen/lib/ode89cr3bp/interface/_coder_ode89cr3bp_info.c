/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: _coder_ode89cr3bp_info.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 13:49:44
 */

/* Include Files */
#include "_coder_ode89cr3bp_info.h"
#include "emlrt.h"
#include "tmwtypes.h"

/* Function Declarations */
static const mxArray *c_emlrtMexFcnResolvedFunctionsI(void);

/* Function Definitions */
/*
 * Arguments    : void
 * Return Type  : const mxArray *
 */
static const mxArray *c_emlrtMexFcnResolvedFunctionsI(void)
{
  const mxArray *nameCaptureInfo;
  const char_T *data[4] = {
      "789cc553cb4ac340149d482c6eaa59b9f11f3a942028e8c6d880a8b5b455282276928c4d"
      "c83cca2423f5135cf90b7e869fa731efc2d048b5decdcde170e7dc73"
      "6602b48b6b0d00b00bd29ab6d2deceb091f52d50af655e53f4bcb6815e9bcbf9b7acbb9c"
      "c57811a780218a8b498fd38021168f5fe618081c71f28cbd6fe62920",
      "781c503caa827e82a85da10a9050c9b7e563371c490a841f951b922a28f2f850f8d51be6"
      "3154e4612cf1f7bd07781b6111415f7e2521a0cd4508e7987992483a"
      "ecf7214531410e748388488604e41e3e3a7685e9cc3b542bc123629cd92eebd67d4cd7f4"
      "b1a3f49132d6d03c1bdc9cf77e4bafa5d44b198f4b87e0d2dffb9a7a",
      "274abd3afff37bca93e9d0647e553e7b0df755fd67edec3ec46938dba4dec1ddabbe49bd"
      "bcfe4b6fa138afe97bdb57e8194bfc444c66e125113e3abc2256e474"
      "5d535a56b9c76085ceaa3d8002fff5f99ff526718a",
      ""};
  nameCaptureInfo = NULL;
  emlrtNameCaptureMxArrayR2016a(&data[0], 1656U, &nameCaptureInfo);
  return nameCaptureInfo;
}

/*
 * Arguments    : void
 * Return Type  : mxArray *
 */
mxArray *emlrtMexFcnProperties(void)
{
  mxArray *xEntryPoints;
  mxArray *xInputs;
  mxArray *xResult;
  const char_T *propFieldName[9] = {"Version",
                                    "ResolvedFunctions",
                                    "Checksum",
                                    "EntryPoints",
                                    "CoverageInfo",
                                    "IsPolymorphic",
                                    "PropertyList",
                                    "UUID",
                                    "ClassEntryPointIsHandle"};
  const char_T *epFieldName[8] = {
      "Name",     "NumberOfInputs", "NumberOfOutputs", "ConstantInputs",
      "FullPath", "TimeStamp",      "Constructor",     "Visible"};
  xEntryPoints =
      emlrtCreateStructMatrix(1, 1, 8, (const char_T **)&epFieldName[0]);
  xInputs = emlrtCreateLogicalMatrix(1, 3);
  emlrtSetField(xEntryPoints, 0, "Name", emlrtMxCreateString("ode89cr3bp"));
  emlrtSetField(xEntryPoints, 0, "NumberOfInputs",
                emlrtMxCreateDoubleScalar(3.0));
  emlrtSetField(xEntryPoints, 0, "NumberOfOutputs",
                emlrtMxCreateDoubleScalar(2.0));
  emlrtSetField(xEntryPoints, 0, "ConstantInputs", xInputs);
  emlrtSetField(
      xEntryPoints, 0, "FullPath",
      emlrtMxCreateString(
          "/Users/hunter/Fork/pendulumRNN/matlab/cislunar/ode89cr3bp.m"));
  emlrtSetField(xEntryPoints, 0, "TimeStamp",
                emlrtMxCreateDoubleScalar(739612.52034722222));
  emlrtSetField(xEntryPoints, 0, "Constructor",
                emlrtMxCreateLogicalScalar(false));
  emlrtSetField(xEntryPoints, 0, "Visible", emlrtMxCreateLogicalScalar(true));
  xResult =
      emlrtCreateStructMatrix(1, 1, 9, (const char_T **)&propFieldName[0]);
  emlrtSetField(xResult, 0, "Version",
                emlrtMxCreateString("24.1.0.2537033 (R2024a)"));
  emlrtSetField(xResult, 0, "ResolvedFunctions",
                (mxArray *)c_emlrtMexFcnResolvedFunctionsI());
  emlrtSetField(xResult, 0, "Checksum",
                emlrtMxCreateString("LzUIFEzDJhidgHORARXU8"));
  emlrtSetField(xResult, 0, "EntryPoints", xEntryPoints);
  return xResult;
}

/*
 * File trailer for _coder_ode89cr3bp_info.c
 *
 * [EOF]
 */
