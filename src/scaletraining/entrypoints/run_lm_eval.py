"""
Integration with lm-evaluation-harness for standardized benchmarks.
Runs zero-shot (or few-shot) evaluations on ScaleTraining checkpoints.

Usage:
  python -m scaletraining.entrypoints.run_lm_eval --model_path=outputs/latest/model.pt --tasks=hellaswag,mmlu
"""

import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

import hydra
from omegaconf import DictConfig
from lm_eval import simple_evaluate
from lm_eval.api.model import LM

from scaletraining.config import load_project_config
from scaletraining.reporting import refresh_run_report
from scaletraining.util import resolve_device
from scaletraining.util.eval_utils import (
    load_pretrained_model_and_tokenizer,
    write_lm_eval_result,
)

LOGGER = logging.getLogger(__name__)

class ScaleTrainingLM(LM):
    """
    Wrapper for ScaleTraining TransformerNetwork to fit the lm-eval API.
    """
    def __init__(self, model, tokenizer, device, batch_size=1):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._batch_size = int(batch_size) if batch_size else 1
        
        # Ensure model is in eval mode
        self._model.eval()

    @property
    def max_length(self):
        return self._model.transformer_blocks[0].attention.max_seq_len

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        # ScaleTraining tokenizer wrapper usually handles encoding
        # We need raw IDs.
        # Check if it's the custom tokenizer wrapper or raw HF tokenizer
        if hasattr(self._tokenizer, "encode"):
            # Raw HF or wrapping compliant
            return self._tokenizer.encode(string, add_special_tokens=False)
        elif hasattr(self._tokenizer, "tokenizer"):
             # It's likely the scaletraining TextTokenizer wrapper
             return self._tokenizer.tokenizer.encode(string, add_special_tokens=False)
        return []

    def tok_decode(self, tokens):
        if hasattr(self._tokenizer, "decode"):
             return self._tokenizer.decode(tokens)
        elif hasattr(self._tokenizer, "tokenizer"):
             return self._tokenizer.tokenizer.decode(tokens)
        return ""

    def _prefix_token_id(self):
        for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
            token_id = getattr(self._tokenizer, attr, None)
            if token_id is not None:
                return int(token_id)
        return 0

    def loglikelihood(self, requests):
        """
        Compute log-likelihood of a list of Instance objects.
        
        Each Instance has .args = (context, continuation)
        Return: list of (logprob, is_greedy)
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(requests), self.batch_size), desc="Evaluating"):
            batch = requests[i : i + self.batch_size]
            
            inputs = []
            ctx_lens = []
            cont_lens = []
            
            for req in batch:
                # lm_eval 0.4+ uses Instance objects with .args
                context, continuation = req.args
                
                # 1. Encode context and continuation
                # Note: We concatenate them to get P(continuation | context)
                ctx_enc = self.tok_encode(context) or [self._prefix_token_id()]
                cont_enc = self.tok_encode(continuation)
                
                # Setup full sequence
                full_enc = ctx_enc + cont_enc
                if len(full_enc) > self.max_length:
                    overflow = len(full_enc) - self.max_length
                    ctx_enc = ctx_enc[overflow:] if overflow < len(ctx_enc) else []
                    if not ctx_enc:
                        ctx_enc = [self._prefix_token_id()]
                    full_enc = ctx_enc + cont_enc
                
                inputs.append(torch.tensor(full_enc, dtype=torch.long))
                ctx_lens.append(len(ctx_enc))
                cont_lens.append(len(cont_enc))

            # Pad batch
            max_len = max(len(x) for x in inputs)
            padded_inputs = torch.full(
                (len(inputs), max_len), 
                self._tokenizer.pad_token_id if hasattr(self._tokenizer, 'pad_token_id') and self._tokenizer.pad_token_id is not None else 0,
                dtype=torch.long
            )
            
            for j, seq in enumerate(inputs):
                padded_inputs[j, :len(seq)] = seq
                
            padded_inputs = padded_inputs.to(self.device)
            
            with torch.no_grad():
                # Get logits for the full sequence
                # ScaleTraining model returns [B, T, V]
                logits = self._model(padded_inputs)
                
                # Log softmax
                logits = F.log_softmax(logits, dim=-1)

            # Extract logprobs for the continuation portion
            for j, req in enumerate(batch):
                # The logits at index t predict token at t+1.
                # We want P(token[k] | tokens[0...k-1]).
                # So for a sequence of length L, logits at L-2 predict L-1.
                
                # Sequence: [C0, C1... Cn, T0, T1... Tm]
                # Logits:   [L0, L1... Ln, M0, M1... Mm]
                # We want prediction of T0 (given Context) up to Tm.
                
                # Indices in `padded_inputs`:
                # Context ends at ctx_lens[j] - 1.
                # First continuation token is at ctx_lens[j].
                # We need logits at `ctx_lens[j] - 1` to predict `padded_inputs[j, ctx_lens[j]]`.
                
                start_idx = ctx_lens[j]
                length = cont_lens[j]
                end_idx = start_idx + length
                
                # Slice logits: from context_end-1 to full_end-2
                # This gives predictions for tokens at start_idx to end_idx-1
                relevant_logits = logits[j, start_idx-1 : end_idx-1, :]
                
                # Slice targets: actual continuation tokens
                relevant_targets = padded_inputs[j, start_idx : end_idx]
                
                # Gather logprobs of the correct tokens
                # relevant_logits: [Length, Vocab]
                # relevant_targets: [Length]
                
                token_logprobs = torch.gather(
                    relevant_logits, 
                    1, 
                    relevant_targets.unsqueeze(-1)
                ).squeeze(-1)
                
                greedy_tokens = relevant_logits.argmax(dim=-1)
                is_greedy = (greedy_tokens == relevant_targets).all().item()
                
                sum_logprob = token_logprobs.sum().item()
                
                results.append((sum_logprob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests):
        """
        Compute rolling log-likelihood for unconditional perplexity tasks.

        lm-eval passes Instance objects with .args = (text,). We score each text
        in fixed windows, using the first token in each window only as context
        and summing logprobs for the remaining tokens.
        """
        results = []
        window = max(2, int(self.max_length))

        for req in tqdm(requests, desc="Rolling eval"):
            (text,) = req.args
            token_ids = self.tok_encode(text)
            if len(token_ids) < 2:
                results.append(0.0)
                continue

            total = 0.0
            for start in range(0, len(token_ids) - 1, window - 1):
                chunk = token_ids[start : start + window]
                if len(chunk) < 2:
                    continue
                input_ids = torch.tensor([chunk], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits = F.log_softmax(self._model(input_ids), dim=-1)
                targets = input_ids[:, 1:]
                token_logprobs = torch.gather(
                    logits[:, :-1, :],
                    2,
                    targets.unsqueeze(-1),
                ).squeeze(-1)
                total += float(token_logprobs.sum().item())
            results.append(total)

        return results

    def generate_until(self, requests):
        """
        Generate text until a stop condition is met.
        
        Each Instance has .args = (context, gen_kwargs)
        gen_kwargs contains 'until' (stop strings) and optionally 'max_gen_toks'
        Return: list of generated strings
        """
        results = []
        
        for req in tqdm(requests, desc="Generating"):
            context, gen_kwargs = req.args
            stop_sequences = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            
            # Encode context
            input_ids = self.tok_encode(context)
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            
            # Autoregressive generation
            generated_ids = []
            for _ in range(max_gen_toks):
                with torch.no_grad():
                    logits = self._model(input_ids)
                    next_token_logits = logits[:, -1, :]
                    next_token = next_token_logits.argmax(dim=-1)
                    
                generated_ids.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Check stop conditions
                generated_text = self.tok_decode(generated_ids)
                should_stop = False
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        # Truncate at stop sequence
                        generated_text = generated_text.split(stop_seq)[0]
                        should_stop = True
                        break
                
                # Also stop on EOS token
                if hasattr(self._tokenizer, 'eos_token_id') and next_token.item() == self._tokenizer.eos_token_id:
                    should_stop = True
                    
                if should_stop:
                    break
            
            generated_text = self.tok_decode(generated_ids)
            # Final truncation at stop sequences
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
                    break
                    
            results.append(generated_text)
        
        return results


@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent.parent.parent / "conf"), config_name="config")
def main(cfg: DictConfig):
    cfg = load_project_config(cfg)
    device = resolve_device(cfg)
    
    # 1. Load Model & Tokenizer
    # We cheat and use the existing util which handles finding 'latest' etc.
    print(f"Loading model from {cfg.generation.model_path}...")
    model, tokenizer = load_pretrained_model_and_tokenizer(cfg)
    
    # 2. Wrap
    lm = ScaleTrainingLM(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=cfg.training.batch_size  # Reuse training batch size
    )
    
    # 3. Parse tasks: env var > config > default
    # Usage: LM_EVAL_TASKS=hellaswag,mmlu python -m scaletraining.entrypoints.run_lm_eval ...
    # Or:    python -m scaletraining.entrypoints.run_lm_eval eval.tasks=hellaswag,mmlu ...
    import os
    tasks_str = os.environ.get("LM_EVAL_TASKS", "")
    if not tasks_str:
        # Fall back to config
        tasks_str = getattr(cfg.eval, "tasks", "") if hasattr(cfg, "eval") else ""
    tasks_list = [t.strip() for t in tasks_str.split(",") if t.strip()]
        
    if not tasks_list:
        print("No tasks specified. Defaulting to 'hellaswag'.")
        tasks_list = ["hellaswag"]

    print(f"Evaluating on tasks: {tasks_list}")
    
    # 4. Run Eval
    results = simple_evaluate(
        model=lm,
        tasks=tasks_list,
        batch_size=lm.batch_size,
        device=device
    )
    
    # 5. Print Results
    from lm_eval.utils import make_table
    print(make_table(results))
    if bool(getattr(cfg.eval, "write_results", True)):
        result_path = write_lm_eval_result(cfg, tasks_list, results)
        print(f"lm-eval results written to: {result_path}")
        json_report, markdown_report = refresh_run_report(result_path.parent)
        print(f"Run evidence refreshed: {json_report} and {markdown_report}")

if __name__ == "__main__":
    main()
