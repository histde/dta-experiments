"""Pretrain a character-level xLSTM language model on the DTA documents.

The neural sibling of train_dta_model.py (KenLM): an autoregressive character
LM over clean historical German, usable as an OCR quality scorer via character
perplexity - but with whole-page context instead of 6-gram windows.

It trains a character-based xLSTM on the numpy shards written by
prepare_xlstm_data.py, which are encoded with the character tokenizer of
build_tokenizer.py (normalization included) instead of a subword tokenizer.

The following hyper-parameters are used: AdamW (0.99, 0.95), weight decay 0.1
excluded for embeddings + LM head, grad clip 0.5, linear warmup -> cosine
decay to 10% of peak. If --warmup_steps is not given it is derived from the
OLMo ladder rule (non-embedding parameter count in tokens). Pass exactly one
of --token_budget / --max_steps; the DTA stream is ~1.37B characters, so
--token_budget 2500000000 is about 1.8 epochs.

Checkpoints are self-contained HF models (config.json + safetensors + the
character tokenizer, so AutoTokenizer works), saved in inference mode:
    tok = AutoTokenizer.from_pretrained(ckpt); model = AutoModelForCausalLM.from_pretrained(ckpt)
    loss = model(**tok(page, return_tensors="pt"), labels=...).loss  # bpc = loss / ln 2
The loop itself runs the model in train mode (chunkwise kernel, sequence
length a multiple of 64); resume with --resume_from_checkpoint <dir>. The
output_dir/data_manifest.json + ResumableSampler mechanism from
run_pretraining.py is kept, so a resume continues at exactly the next sample.

Training metrics (loss, bits-per-character, grad norm, learning rate, chars
seen, epoch, steps/s) are logged to TensorBoard under
<output_dir>/tensorboard/<run_name> (override with --logging_dir); the run
arguments are stored as a text summary and the final loss/bpc in the HParams
tab. Monitor with:
    tensorboard --logdir <output_dir>/tensorboard

On GPU pass --bf16 --chunkwise_kernel chunkwise--triton_xl_chunk; the default
chunkwise--native_autograd kernel also runs on CPU (for smoke tests only).
Keep --hidden_size a multiple of 128.

Usage (~20M param model):
    python build_tokenizer.py
    python prepare_xlstm_data.py
    python train_xlstm.py --data-dir dta_xlstm/data --output_dir dta_xlstm/model \\
        --per_device_train_batch_size 32 --gradient_accumulation_steps 2 \\
        --token_budget 2500000000 --bf16 \\
        --chunkwise_kernel chunkwise--triton_xl_chunk --run_name dta-xlstm-20m

AI Disclosure:
    Models:         Claude Fable 5 (claude-fable-5)
    AI-Generated:   fully         # fully | mostly | partially | none
    Human-Reviewed: minimally     # fully | partially | minimally | none
"""

import argparse
import glob
import json
import logging
import math
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from transformers import AutoTokenizer, xLSTMConfig, xLSTMForCausalLM


logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    p.add_argument("--data-dir", default="dta_xlstm/data", help="Output directory of prepare_xlstm_data.py.")
    p.add_argument("--block_size", type=int, default=2048, help="Sequence length (characters) shards are windowed into.")
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    # Model (xLSTM 7B architecture; shape default ~20M params at char vocab,
    # mirroring the width-384 setting of arXiv:2606.12364 Tab. 16)
    p.add_argument("--hidden_size", type=int, default=384, help="Multiple of 128 (see docstring).")
    p.add_argument("--num_blocks", type=int, default=11)
    p.add_argument("--num_heads", type=int, default=3)
    p.add_argument("--chunkwise_kernel", default=None, help="'chunkwise--triton_xl_chunk' on GPU.")
    # Batch / schedule length
    p.add_argument("--per_device_train_batch_size", type=int, required=True)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None, help="Mutually exclusive with --token_budget.")
    p.add_argument("--token_budget", type=int, default=None,
                   help="Total training characters; derives --max_steps. Mutually exclusive with --max_steps.")
    # Optimizer / schedule (SWEEP.md recipe)
    p.add_argument("--learning_rate", type=float, default=3e-3, help="Peak LR (SWEEP.md small-size default).")
    p.add_argument("--min_lr_rate", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=None,
                   help="Default: derived as non-embedding params / global batch tokens (OLMo ladder rule).")
    p.add_argument("--adam_beta1", type=float, default=0.99)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    # Precision / performance
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--torch_compile", action="store_true")
    # Logging / checkpointing
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default=None,
                   help="TensorBoard run name (default: basename of --output_dir).")
    p.add_argument("--logging_dir", default=None,
                   help="TensorBoard log directory (default: <output_dir>/tensorboard/<run_name>).")
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=2500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if (args.max_steps is None) == (args.token_budget is None):
        raise ValueError("Pass exactly one of --max_steps or --token_budget.")
    return args


class NumpyFSLDataset(torch.utils.data.Dataset):
    """Windows raw numpy character-id shards into fixed-length instances at load
    time (per shard, dropping each shard's remainder)."""

    def __init__(self, paths, sequence_length, dtype):
        self.paths = paths
        self.sequence_length = sequence_length
        self.dtype = dtype
        item_size = np.dtype(dtype).itemsize
        shard_lengths = [os.path.getsize(p) // item_size // sequence_length for p in paths]
        self.offsets = np.cumsum([0] + shard_lengths)
        self._mmaps = {}

    def __len__(self):
        return int(self.offsets[-1])

    def _shard(self, shard_idx):
        if shard_idx not in self._mmaps:  # opened lazily per DataLoader worker
            self._mmaps[shard_idx] = np.memmap(self.paths[shard_idx], mode="r", dtype=self.dtype)
        return self._mmaps[shard_idx]

    def __getitem__(self, idx):
        shard_idx = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        start = (idx - int(self.offsets[shard_idx])) * self.sequence_length
        ids = self._shard(shard_idx)[start : start + self.sequence_length].astype(np.int64)
        return {"input_ids": ids}


def collate(examples):
    input_ids = torch.from_numpy(np.stack([example["input_ids"] for example in examples])).long()
    return {"input_ids": input_ids, "labels": input_ids.clone()}


class ResumableSampler(torch.utils.data.Sampler):
    """Infinite deterministic per-epoch-reshuffled index stream; a fixed seed plus
    a samples-consumed count fully reconstructs the position."""

    def __init__(self, dataset_len, seed, start_position=0):
        self.dataset_len = dataset_len
        self.seed = seed
        self.start_position = start_position

    def __iter__(self):
        position = self.start_position
        while True:
            epoch, offset = divmod(position, self.dataset_len)
            generator = torch.Generator()
            generator.manual_seed(self.seed + epoch)
            permutation = torch.randperm(self.dataset_len, generator=generator).tolist()
            for idx in permutation[offset:]:
                yield idx
                position += 1

    def __len__(self):
        return self.dataset_len


def build_config(args, tokenizer, metadata):
    """xLSTM 7B architecture settings, char vocab."""
    return xLSTMConfig(
        vocab_size=64 * math.ceil(len(tokenizer) / 64),
        embedding_dim=args.hidden_size,
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        num_hidden_layers=args.num_blocks,
        num_heads=args.num_heads,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        ffn_proj_factor=2.667,
        ffn_round_up_to_multiple_of=64,
        gate_soft_cap=15.0,
        output_logit_soft_cap=30.0,
        use_bias=False,
        tie_word_embeddings=False,
        add_out_norm=True,
        norm_reduction_force_float32=True,
        autocast_kernel_dtype="bfloat16",
        inference_state_dtype="float32",
        eps=1e-6,
        norm_eps=1e-6,
        chunk_size=64,
        mode="train",
        return_last_states=True,  # must stay True: mLSTMLayer always unpacks (h, state)
        use_cache=False,
        chunkwise_kernel=args.chunkwise_kernel or "chunkwise--native_autograd",
        sequence_kernel="native_sequence__native",
        step_kernel="native",
        weight_mode="single",
        bos_token_id=metadata["eos_token_id"],
        eos_token_id=metadata["eos_token_id"],
        pad_token_id=metadata["eos_token_id"],
    )


def build_optimizer(model, args):
    """AdamW with weight decay excluded for embeddings and the LM head (xLSTM App. B.2)."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay_names = {
        f"{name}.{param_name}"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Embedding)
        for param_name, _ in module.named_parameters(recurse=False)
    }
    no_decay_names.update(n for n, _ in named if n.endswith("lm_head.weight"))
    return torch.optim.AdamW(
        [
            {"params": [p for n, p in named if n not in no_decay_names], "weight_decay": args.weight_decay},
            {"params": [p for n, p in named if n in no_decay_names], "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )


def lr_at_step(step, args, total_steps):
    """Linear warmup to peak, then cosine decay to min_lr_rate of peak."""
    if step < args.warmup_steps:
        return args.learning_rate * step / max(1, args.warmup_steps)
    progress = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
    min_lr = args.learning_rate * args.min_lr_rate
    return min_lr + (args.learning_rate - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(output_dir, step, samples_consumed, model, tokenizer, optimizer, save_total_limit):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Saved config is in inference mode (recurrent kernels, any sequence length,
    # generate() works); resume switches back to train mode explicitly.
    train_mode = model.config.mode
    model.config.mode = "inference"
    model.save_pretrained(ckpt_dir)  # HF format: config.json + safetensors
    model.config.mode = train_mode
    tokenizer.save_pretrained(ckpt_dir)  # tokenizer.json -> checkpoint is a self-contained HF model
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step, "samples_consumed": samples_consumed},
        os.path.join(ckpt_dir, "training_state.pt"),
    )
    logger.info(f"Saved checkpoint to {ckpt_dir}")
    if save_total_limit:
        existing = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
        for stale in existing[:-save_total_limit]:
            shutil.rmtree(stale)
            logger.info(f"Removed old checkpoint {stale} (--save_total_limit {save_total_limit})")


def self_test(model, tokenizer, device, chunk_size=64):
    """Character perplexity of a clean vs a noisy sentence (lower = cleaner).

    Goes through the HF tokenizer (normalization included), exactly like a
    downstream scorer would. The train-mode chunkwise kernel needs sequence
    lengths divisible by chunk_size, so inputs are right-padded with the pad
    token (<eos>) and the loss is computed over the real positions only.
    """
    model.eval()
    print("Self-test (character perplexity, lower = cleaner):")
    for label, sample in [
        ("clean", "Die Zeitung berichtet über die neuesten Ereignisse in der Hauptstadt."),
        ("noisy", "Dje Zc1tur.g ber/chtet üb#r dle n?uest3n Ereign1sse jn d€r Hai;ptstadt."),
    ]:
        real = tokenizer(sample, add_special_tokens=False)["input_ids"]
        padded = real + [tokenizer.pad_token_id] * (-len(real) % chunk_size)
        ids = torch.tensor([padded], device=device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        nll = torch.nn.functional.cross_entropy(logits[0, : len(real) - 1], ids[0, 1 : len(real)])
        print(f"  {label}: ppl {math.exp(nll.item()):.2f}")


def main():
    args = parse_args()
    console = Console()
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=console, show_path=False)])

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = args.data_dir
    with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(data_dir)  # copied there by prepare_xlstm_data.py
    if len(tokenizer) != metadata["vocab_size"] or tokenizer.eos_token_id != metadata["eos_token_id"]:
        raise SystemExit(f"tokenizer in {data_dir} does not match metadata.json - rerun prepare_xlstm_data.py")
    shard_dtype = np.dtype(metadata["shard_dtype"])

    # Pin train_files/block_size/seed at run start; a resume reads the manifest back
    # so dataset index i keeps meaning the same example (see run_pretraining.py).
    manifest_path = os.path.join(args.output_dir, "data_manifest.json")
    if args.resume_from_checkpoint:
        with open(manifest_path) as f:
            manifest = json.load(f)
        train_files, block_size, data_seed = manifest["train_files"], manifest["block_size"], manifest["seed"]
        if block_size != args.block_size:
            logger.warning(f"--block_size {args.block_size} ignored; resuming with pinned block_size {block_size}")
    else:
        train_files = sorted(glob.glob(os.path.join(data_dir, "dta_*.npy")))
        if not train_files:
            raise SystemExit(f"No dta_*.npy shards in {data_dir}/ - run prepare_xlstm_data.py first.")
        block_size, data_seed = args.block_size, args.seed
        with open(manifest_path, "w") as f:
            json.dump({"train_files": train_files, "block_size": block_size, "seed": data_seed}, f, indent=2)

    samples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
    global_batch_tokens = samples_per_step * block_size
    logger.info(f"Global batch: {global_batch_tokens:,} characters ({samples_per_step} sequences)")
    max_steps = max(1, args.token_budget // global_batch_tokens) if args.token_budget else args.max_steps
    if args.token_budget:
        logger.info(f"Token budget {args.token_budget:,} characters -> max_steps {max_steps:,}")

    config = build_config(args, tokenizer, metadata)
    start_step = 0
    if args.resume_from_checkpoint:
        model = xLSTMForCausalLM.from_pretrained(
            args.resume_from_checkpoint, mode="train", chunkwise_kernel=config.chunkwise_kernel)
    else:
        model = xLSTMForCausalLM(config)
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.torch_compile:
        model = torch.compile(model)

    input_embedding_params = model.get_input_embeddings().weight.numel()
    total_params = sum(p.numel() for p in model.parameters())
    non_emb_params = total_params - input_embedding_params
    logger.info(f"Parameters: total={total_params:,}, non-embedding={non_emb_params:,}")

    if args.warmup_steps is None:
        args.warmup_steps = max(1, non_emb_params // global_batch_tokens)
        logger.info(f"Warmup (OLMo ladder rule, non-emb params in tokens): {args.warmup_steps} steps")

    optimizer = build_optimizer(model, args)
    samples_consumed = 0
    if args.resume_from_checkpoint:
        state = torch.load(os.path.join(args.resume_from_checkpoint, "training_state.pt"), map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        start_step, samples_consumed = state["step"], state["samples_consumed"]
        logger.info(f"Resumed from {args.resume_from_checkpoint} at step {start_step:,}")

    train_dataset = NumpyFSLDataset(train_files, sequence_length=block_size, dtype=shard_dtype)
    sampler = ResumableSampler(len(train_dataset), seed=data_seed, start_position=samples_consumed)
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=args.dataloader_num_workers,
        pin_memory=(device.type == "cuda"),
    )
    data_iter = iter(loader)

    run_name = args.run_name or os.path.basename(os.path.normpath(args.output_dir))
    logging_dir = args.logging_dir or os.path.join(args.output_dir, "tensorboard", run_name)
    writer = SummaryWriter(log_dir=logging_dir, purge_step=start_step if start_step else None)
    writer.add_text("args", "```\n" + json.dumps(vars(args), indent=2) + "\n```", global_step=start_step)
    logger.info(f"TensorBoard logs: {logging_dir}  (tensorboard --logdir {os.path.join(args.output_dir, 'tensorboard')})")
    model.train()
    global_step = start_step
    window_loss_sum, window_micro_steps = 0.0, 0
    window_start = time.monotonic()
    train_start = time.monotonic()

    total_tokens = max_steps * global_batch_tokens
    columns = (TextColumn("[progress.description]{task.description}"), BarColumn(),
               TextColumn("[progress.percentage]{task.percentage:>3.1f}%"), TextColumn("•"),
               TimeElapsedColumn(), TextColumn("•ETA"), TimeRemainingColumn())
    with Progress(*columns, console=console) as progress:
        task_id = progress.add_task("", total=total_tokens, completed=start_step * global_batch_tokens)

        while global_step < max_steps:
            optimizer.zero_grad()
            for _ in range(args.gradient_accumulation_steps):
                batch = {k: v.to(device, non_blocking=True) for k, v in next(data_iter).items()}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
                    loss = model(**batch).loss
                window_loss_sum += loss.item()
                window_micro_steps += 1
                (loss / args.gradient_accumulation_steps).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) if args.max_grad_norm > 0 else None
            lr = lr_at_step(global_step, args, max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            global_step += 1
            progress.update(task_id, completed=global_step * global_batch_tokens,
                            description=f"{global_step * global_batch_tokens / 1e6:,.0f}M/{total_tokens / 1e6:,.0f}M chars")

            if global_step % args.logging_steps == 0 or global_step == max_steps:
                elapsed = time.monotonic() - window_start
                logs = {
                    "loss": window_loss_sum / window_micro_steps,
                    "bpc": window_loss_sum / window_micro_steps / math.log(2),
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "learning_rate": lr,
                    "num_chars": global_step * global_batch_tokens,
                    "epoch": global_step * samples_per_step / len(train_dataset),
                    "steps_per_second": (global_step % args.logging_steps or args.logging_steps) / elapsed,
                }
                logger.info(str(logs))
                for key, value in logs.items():
                    if value is not None:
                        writer.add_scalar(f"train/{key}", value, global_step)
                window_loss_sum, window_micro_steps = 0.0, 0
                window_start = time.monotonic()

            if global_step % args.save_steps == 0 or global_step == max_steps:
                save_checkpoint(args.output_dir, global_step, global_step * samples_per_step,
                                model, tokenizer, optimizer, args.save_total_limit)

    runtime = time.monotonic() - train_start
    metrics = {"train_runtime": runtime, "train_steps": global_step,
               "train_steps_per_second": (global_step - start_step) / runtime,
               "num_chars": global_step * global_batch_tokens}
    logger.info(f"Done: {metrics}")
    with open(os.path.join(args.output_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    tokenizer.save_pretrained(args.output_dir)
    writer.add_hparams(
        {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool)) and v is not None},
        {"final/loss": logs["loss"], "final/bpc": logs["bpc"]}, run_name=".")
    writer.close()
    self_test(model, tokenizer, device)


if __name__ == "__main__":
    main()
