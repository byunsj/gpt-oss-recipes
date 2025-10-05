#!/usr/bin/env python3
"""
Crisis Classification Fine-tuning Script for GPT-OSS 120B
Optimized for mental health crisis detection using LoRA/QLoRA
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default="lmsys/gpt-oss-120b-bf16",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for model weights"}
    )
    attn_implementation: Optional[str] = field(
        default="eager",
        metadata={"help": "Attention implementation to use"}
    )
    use_bnb: bool = field(
        default=False,
        metadata={"help": "Use BitsAndBytes quantization"}
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type for 4-bit"}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "Use double quantization"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration"""
    dataset_name: str = field(
        default="./crisis_dataset",
        metadata={"help": "Path to dataset"}
    )
    max_length: int = field(
        default=1536,
        metadata={"help": "Maximum sequence length"}
    )
    dataset_num_proc: int = field(
        default=4,
        metadata={"help": "Number of processes for dataset processing"}
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration"""
    use_peft: bool = field(
        default=False,
        metadata={"help": "Use PEFT (LoRA)"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Target modules for LoRA (comma-separated or 'all-linear')"}
    )


def get_bnb_config(model_args: ModelArguments) -> Optional[BitsAndBytesConfig]:
    """Get BitsAndBytes configuration"""
    if not model_args.use_bnb:
        return None
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, model_args.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
    )


def get_peft_config(lora_args: LoRAArguments) -> Optional[LoraConfig]:
    """Get PEFT configuration"""
    if not lora_args.use_peft:
        return None
    
    # Handle target modules
    if lora_args.lora_target_modules == "all-linear":
        target_modules = "all-linear"
    elif lora_args.lora_target_modules:
        target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",")]
    else:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    
    return LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main():
    """Main training function"""
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # Load from YAML config
        model_args, data_args, lora_args, training_args = parser.parse_yaml_file(yaml_file=sys.argv[1])
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    print("🚀 Starting Crisis Classification Training")
    print(f"📊 Model: {model_args.model_name_or_path}")
    print(f"📊 Dataset: {data_args.dataset_name}")
    print(f"📊 Max Length: {data_args.max_length}")
    print(f"📊 Use PEFT: {lora_args.use_peft}")
    print(f"📊 Use BnB: {model_args.use_bnb}")
    
    # Load dataset
    print(f"\n📂 Loading dataset from {data_args.dataset_name}...")
    try:
        dataset = load_from_disk(data_args.dataset_name)
        print(f"✅ Dataset loaded: {dataset}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("💡 Make sure to run prepare_crisis_data.py or prepare_user_data.py first")
        sys.exit(1)
    
    # Load tokenizer
    print(f"\n🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Load model
    print(f"\n🤖 Loading model...")
    bnb_config = get_bnb_config(model_args)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else None,
        trust_remote_code=True,
        attn_implementation=model_args.attn_implementation,
        device_map="auto" if model_args.use_bnb else None,
    )
    
    print(f"✅ Model loaded")
    
    # Setup PEFT
    peft_config = get_peft_config(lora_args)
    if peft_config:
        print(f"\n🔧 Setting up PEFT...")
        if model_args.use_bnb:
            model = prepare_model_for_kbit_training(model)
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print(f"✅ PEFT setup complete")
    
    # Disable caching for training
    model.config.use_cache = False
    
    # Initialize trainer
    print(f"\n🏋️ Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if "train" in dataset else dataset,
        eval_dataset=dataset["test"] if "test" in dataset else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print(f"✅ Trainer initialized")
    
    # Start training
    print(f"\n🚀 Starting training...")
    trainer.train()
    
    # Save model
    print(f"\n💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"✅ Training completed! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()