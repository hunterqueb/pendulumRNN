/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: _coder_CR3BP_info.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 26-Dec-2024 10:26:05
 */

/* Include Files */
#include "_coder_CR3BP_info.h"
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
      "789cc554cb4e8340141d0c366eaaacdcf80f9d348d3bdd88251a159bb69a34c6e8006321"
      "cc0ccdc098fa09aefc05977ea6e559209940acd6bbb99c9cdc3967ce"
      "05807279a30000f6415aa34edabb19d6b2be03aa55e71549cf6b17a895b99cffc8ba1db0"
      "082fa31430447131e904d4638845d3b705061c870179c54ec2bc7804",
      "4f3d8a276560c6881a25aa0031153feb2eb6fd89a080bbe1da212983228f2fc97dd59679"
      "5c48f2d06afcc3f011de859887d015ab24383402eec305668e20828e"
      "4d135214116441db0b896088437d3c381bf5a892f427c40266d8ac9ffb7eded0f79ed477"
      "ca24aab7e7c3dfd2eb48f552c6098445f07a2f9f1bea9d48f5aafc0f",
      "f7b24aa647e3f9a67c0e5afa957d57dd6c1ffcd49f6f53efe8fe5ddda65e5effa5b7949c"
      "d7f67d3b94e869357ec66773ff8a70171d5f133db4faf640e87ae9ff"
      "dca0d3e40348f05f9fff0db44c6d04",
      ""};
  nameCaptureInfo = NULL;
  emlrtNameCaptureMxArrayR2016a(&data[0], 1640U, &nameCaptureInfo);
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
  xInputs = emlrtCreateLogicalMatrix(1, 0);
  emlrtSetField(xEntryPoints, 0, "Name", emlrtMxCreateString("CR3BP"));
  emlrtSetField(xEntryPoints, 0, "NumberOfInputs",
                emlrtMxCreateDoubleScalar(0.0));
  emlrtSetField(xEntryPoints, 0, "NumberOfOutputs",
                emlrtMxCreateDoubleScalar(0.0));
  emlrtSetField(xEntryPoints, 0, "ConstantInputs", xInputs);
  emlrtSetField(xEntryPoints, 0, "FullPath",
                emlrtMxCreateString(
                    "/Users/hunter/Fork/pendulumRNN/matlab/cislunar/CR3BP.m"));
  emlrtSetField(xEntryPoints, 0, "TimeStamp",
                emlrtMxCreateDoubleScalar(739612.42268518521));
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
 * File trailer for _coder_CR3BP_info.c
 *
 * [EOF]
 */
