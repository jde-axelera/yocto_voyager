// yolo_postproc.h
//
// YOLOv11 postprocess:
//   raw int8 AIPU outputs  →  classify each output by stride (8/16/32) and kind (bbox/cls)
//   → dequantize → DFL decode → sigmoid argmax across 80 classes → letterbox-undo
//   → NMS → final boxes in original image coords.
#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include "axruntime/axruntime.h"

namespace yvm {

/// One detection, in original image coordinates.
struct Detection {
    float x1, y1, x2, y2;
    float score;
    int   cls;
};

/// Build a (stride_index, kind) → output_index lookup table for the 6
/// outputs of the yolo11 head (3 strides × {bbox 64-ch, class 80-ch}).
/// stride_index: 0=stride 8, 1=stride 16, 2=stride 32
/// kind:         0=bbox, 1=class
std::array<std::array<int, 2>, 3> classify_outputs(
    const std::vector<axrTensorInfo>& outs);

/// Decode + filter detections from the 6 raw AIPU output tensors.
///
/// `bufs[i]` points to the int8 buffer for output `i`; `infos[i]` is its
/// tensor info (used for dequantization and shape).
///
/// `scale_letterbox`, `padx_640`, `pady_640` come from preprocess() and let
/// us map boxes from 640x640 letterbox space back to original (orig_w, orig_h).
void decode_dfl_sigmoid_filter(
    const std::vector<const int8_t*>& bufs,
    const std::vector<axrTensorInfo>& infos,
    const std::array<std::array<int, 2>, 3>& tbl,
    float conf_thresh,
    float scale_letterbox, int padx_640, int pady_640,
    int orig_w, int orig_h,
    std::vector<Detection>& out);

/// Class-aware non-maximum suppression.
std::vector<Detection> nms(std::vector<Detection> dets,
                           float iou_thresh,
                           int max_out = 300);

}  // namespace yvm
