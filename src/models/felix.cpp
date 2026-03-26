#include "models.h"

llm_build_felix::llm_build_felix(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    const int n_spokes = hparams.n_spokes;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // --- Hub: attention ---
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // --- Hub: SwiGLU FFN ---
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // Residual after FFN
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        // --- Spoke: gated low-rank projections ---
        if (model.layers[il].spoke_norm) {
            // RMSNorm the hub state for spoke input
            ggml_tensor * h_norm = build_norm(cur,
                    model.layers[il].spoke_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(h_norm, "spoke_norm", il);

            // Compute spoke projections: SiLU(h_norm @ w_down) @ w_up, then mean
            ggml_tensor * spoke_sum = nullptr;
            for (int s = 0; s < n_spokes; ++s) {
                ggml_tensor * down = ggml_mul_mat(ctx0, model.layers[il].spoke_w_down[s], h_norm);
                ggml_tensor * act  = ggml_silu(ctx0, down);
                ggml_tensor * up   = ggml_mul_mat(ctx0, model.layers[il].spoke_w_up[s], act);

                spoke_sum = spoke_sum ? ggml_add(ctx0, spoke_sum, up) : up;
            }

            // Mean over spokes
            ggml_tensor * spoke_mean = ggml_scale(ctx0, spoke_sum, 1.0f / n_spokes);

            // Gated residual: h = h + sigmoid(gate_bias) * mean_update
            // Cast gate_bias from F16 to F32 to match computation precision
            ggml_tensor * gate_f32 = ggml_cast(ctx0, model.layers[il].spoke_gate_bias, GGML_TYPE_F32);
            ggml_tensor * gate = ggml_sigmoid(ctx0, gate_f32);
            // Broadcast scalar gate over the mean update
            ggml_tensor * gated = ggml_mul(ctx0, spoke_mean, ggml_repeat(ctx0, gate, spoke_mean));

            cur = ggml_add(ctx0, cur, gated);
            cb(cur, "spoke_out", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
