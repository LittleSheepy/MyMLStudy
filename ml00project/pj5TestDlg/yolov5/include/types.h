#pragma once
#include "config.h"

#ifdef yolov5_7_0_EXPORTS
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT __declspec(dllimport)
#endif

struct YoloKernel {
  int width;
  int height;
  float anchors[kNumAnchor * 2];
};

struct alignas(float) Detection {
  float bbox[4];  // center_x center_y w h
  float conf;  // bbox_conf * cls_conf
  float class_id;
  float mask[32];
};

