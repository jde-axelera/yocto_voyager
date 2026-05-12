// tasks/embed.cpp
#include "tasks/embed.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace yvm {

void EmbedTask::init(axrModel* /*model*/,
                     const std::vector<axrTensorInfo>& in_infos,
                     const std::vector<axrTensorInfo>& out_infos) {
    ctx_ = make_preproc(in_infos[0]);
    const auto& info = out_infos[0];
    size_t n = 1;
    for (size_t d = 0; d < info.ndims; ++d) n *= info.dims[d];
    if (info.ndims >= 1 && info.dims[0] > 0) n /= info.dims[0];
    vec_len_ = n;
}

void EmbedTask::preproc(const uint8_t* bgr, int sw, int sh,
                        int8_t* dst,
                        float& lscale, int& padx, int& pady) const {
    yvm::preprocess(bgr, sw, sh, dst, ctx_, lscale, padx, pady);
}

std::unique_ptr<TaskResult> EmbedTask::postproc(
        const std::vector<const int8_t*>& out_ptrs,
        const std::vector<axrTensorInfo>& /*out_infos*/,
        float, float, float, int, int, int, int) const {
    auto r = std::make_unique<EmbedResult>();
    if (out_ptrs.empty() || !out_ptrs[0]) return r;
    const int8_t* v = out_ptrs[0];
    r->dim = (int)vec_len_;
    double sumsq = 0.0;
    for (size_t i = 0; i < vec_len_; ++i) {
        double s = (double)v[i];
        sumsq += s * s;
    }
    r->l2_norm = std::sqrt(sumsq);
    int take = (int)std::min<size_t>(8, vec_len_);
    for (int i = 0; i < take; ++i) r->first_few[i] = v[i];
    return r;
}

void EmbedTask::draw(Image& canvas, const TaskResult& result,
                     const FontAtlas* label_font) const {
    const auto& er = static_cast<const EmbedResult&>(result);
    char buf[128];
    std::snprintf(buf, sizeof buf, "embed[%d]  |v|=%.1f  [%d %d %d %d %d %d %d %d ...]",
                  er.dim, er.l2_norm,
                  er.first_few[0], er.first_few[1], er.first_few[2], er.first_few[3],
                  er.first_few[4], er.first_few[5], er.first_few[6], er.first_few[7]);

    int x = 12, y = 12;
    int tw = label_font ? text_width(*label_font, buf) : (int)std::strlen(buf) * 8;
    int th = label_font ? label_font->line_height : 14;
    fill_rect_alpha(canvas.data, canvas.w, canvas.h,
                    x, y, x + tw + 16, y + th + 8, 0, 0, 0, 200);
    if (label_font)
        draw_text_shadow(canvas.data, canvas.w, canvas.h, x + 8, y + 4,
                         buf, 255, 255, 255, *label_font);
}

}  // namespace yvm
