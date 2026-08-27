#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path> [--mtp]\n", argv[0]);
        fprintf(stderr, "  --mtp : Enable Multi-Token Prediction (MTP) mode testing.\n");
        return 1;
    }

    const std::string model_path = argv[1];

    // Detect the --mtp flag from the command line arguments (this will switch to mtp model test)
    bool bIsMtp = false;
    if (argc >= 3 && std::string(argv[2]) == "--mtp") {
        bIsMtp = true;
    }

    // 1. Initialize Backend
    llama_backend_init();

    // 2. Load Model & Vocab
    llama_model_params mparams = llama_model_default_params();

    // Explicitly command the loader to read MTP projection tensors into VRAM if testing MTP mode
    mparams.load_mtp           = bIsMtp;
    llama_model *      model   = llama_model_load_from_file(model_path.c_str(), mparams);
    GGML_ASSERT(model != nullptr && "Failed to load model");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    GGML_ASSERT(vocab != nullptr && "Failed to load vocab");

    // 3. Initialize Speculative Parameters
    llama_speculative_params spec_params = llama_speculative_default_params();

    // A budget of 3. For MTP, this optimally matches the model's `nextn_predict_layers` count.
    spec_params.n_draft = 3;

    // Toggle this flag to test standard Self-Speculation (false) vs Multi-Token Prediction (true)
    spec_params.is_mtp = bIsMtp;

    // 4. Create Contexts Dynamically Based on Mode
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = 1024;
    cparams.n_batch              = 512;

    // Speculative decoding dynamically forks sequences during the draft phase.
    // 1 Base Sequence + 3 Draft Tokens = 4 concurrent sequences max.
    cparams.n_seq_max            = 4;  // Allows Prompt (0) + 3 Draft Branches

    llama_context * ctx_tgt = llama_init_from_model(model, cparams);
    GGML_ASSERT(ctx_tgt != nullptr && "Failed to create target context");

    // If testing MTP, flag the parameter before initializing the draft context!
    // This is strictly required to route the hidden states correctly through the projection heads.
    if (spec_params.is_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    // For this CI test, we use the SAME model for both target and draft (Self-Speculation).
    // This tests the engine's mechanics without needing to download two separate physical models.
    llama_context * ctx_dft = llama_init_from_model(model, cparams);
    GGML_ASSERT(ctx_dft != nullptr && "Failed to create draft context");

    // 5. Initialize Greedy Sampler
    // Speculative verification relies on strict mathematical equality. Stochastic samplers will fail validation here.
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler *            sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    // 6. Initialize Speculative Context
    llama_speculative_context * spec_ctx = llama_speculative_init(ctx_tgt, ctx_dft, sampler, &spec_params);
    GGML_ASSERT(spec_ctx != nullptr && "Failed to initialize speculative context");

    // 7. Tokenize Prompt
    llama_batch              batch  = llama_batch_init(512, 0, 1);
    const char *             prompt = "The capital of France is";
    std::vector<llama_token> tokens(32);

    int32_t n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false);
    GGML_ASSERT(n_tokens > 0 && "Failed to tokenize prompt");

    for (int i = 0; i < n_tokens; i++) {
        batch.token[batch.n_tokens]     = tokens[i];
        batch.pos[batch.n_tokens]       = i;
        batch.n_seq_id[batch.n_tokens]  = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens]    = false;
        batch.n_tokens++;
    }
    batch.logits[n_tokens - 1] = true;  // Evaluate logits for the final prompt token

    // 8. Execute Continuous Speculative Decoding Loop
    llama_speculative_result results[1];
    int32_t                  max_generation   = 20;
    int32_t                  generated_tokens = 0;
    llama_pos                current_pos      = n_tokens;

    fprintf(stderr, "Generating: ");

    while (generated_tokens < max_generation) {
        // Run the full pipeline (Draft -> Verify -> Clean/Rollback) internally
        int32_t n_results = llama_speculative_decode(spec_ctx, &batch, results, 1);
        GGML_ASSERT(n_results == 1 && "Expected exactly 1 sequence result");
        GGML_ASSERT(results[0].count > 0 && "Expected at least 1 accepted token");

        // Print the accepted tokens cleanly
        for (int i = 0; i < results[0].count; i++) {
            char    buf[128] = { 0 };  // Ensure buffer is empty
            int32_t n_chars  = llama_token_to_piece(vocab, results[0].tokens[i], buf, sizeof(buf) - 1, 0, true);
            if (n_chars >= 0) {
                buf[n_chars] = '\0';  // Explicitly add null terminator
                fprintf(stderr, "%s", buf);
            }
        }

        generated_tokens += results[0].count;

        // Prepare the batch for the next step.
        // Because the speculative engine handles its own KV cache and context synchronization,
        // we only need to pass the final accepted token back in to trigger the next loop.
        llama_token last_token = results[0].tokens[results[0].count - 1];
        current_pos += results[0].count;

        batch.n_tokens     = 1;
        batch.token[0]     = last_token;
        batch.pos[0]       = current_pos - 1;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = true;
    }
    fprintf(stderr, "\n\n");

    // 9. Assertions & Summary
    GGML_ASSERT(generated_tokens >= max_generation && "Failed to generate the target number of tokens");
    fprintf(stderr, "Successfully generated %d tokens speculatively across multiple iterations!\n", generated_tokens);

    // 10. Memory Cleanup
    llama_batch_free(batch);
    llama_speculative_free(spec_ctx);
    llama_sampler_free(sampler);

    // Unconditionally free ctx_dft, since we are now correctly allocating it in both modes!
    llama_free(ctx_dft);

    llama_free(ctx_tgt);
    llama_free_model(model);
    llama_backend_free();

    fprintf(stderr, "Speculative decoding C API test passed successfully.\n");
    return 0;
}

