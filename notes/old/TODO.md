# TODO

## Phase 1: Small transformer
- [x]Load tiny-stories dataset
- [x]Configure a basic decoder only transformer (d_model: 512, nlayers: 3, nhead: 4, d_hidden: 512*4)
- [x]Apply RoPE embeddings
- [x]Choose a pretrained tokenizer.
- [x]Train tokenizer (optional).
- [x]Set up training loop (muon or similar)
    - [x]Research optimizations that can be done in the training loop
- [x]Integrate detailed W&B logging for analysis and sweeps
    - [ ]Use loss over total tokens during training as primary metric
- [x]Integrate basic evaluation pipeline for larger models
- [ ]Training
    - [ ]

## Phase 1.5: Evals
- [ ] Zipf lens: understand how the model performs for low-frequency tokens
- [x] loss vs flops: estimate flops/step 
- [ ] evaluations like lambda, wikitext ppl, hellaswag, piqa, arc-easy: for multiple-choice we feed the answer but track the logprobs. for others, it is just text completion.
- [ ] ppl vs context length: cut the output sequences to various lengths, and visualise perplexity on data
- [ ] moe expert routing etc: look at how the expert loading actually works during training. what % of tokens do the experts get? (we can prune experts that get no tokens, or change params to improve routing)


## Phase 2: Improving small transformer
- [x]Add top-2 MoE in MLP block (3 experts total) variant
- [x]Ensure easy argparse 'swap' between model variants
- [ ]Add (second) math or similar logical dataset
- [ ]Train, and inspect proper allocation of experts
- [ ]Perform ablations of experts to visualize performance
- [ ]Optional: Do an interpretability analysis of the experts in comparison to a base model

## Phase x: Scaling to more data
