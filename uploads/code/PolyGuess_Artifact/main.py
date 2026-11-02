import argparse
import configparser
import os

from peft import LoraConfig, TaskType
from transformers import TrainingArguments

from src.model.train import Basic_Config_For_Training, Trainer_Config, TPG_Trainer
from src.model.knowledge_distillation import run_distill_using_config
from src.model.eval import (
    Basic_Config_For_Evaluation,
    Guesser_Config,
    GuessLLM_Evaluator,
    Targeted_Search_Config
)
from src.search.generation import (
    DivideSearchRuntimeConfig,
    run_divide_search_from_config,
    WidthSearchRuntimeConfig,
    run_width_search_from_config
)


def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def load_training_config(config_path):
    config = load_config(config_path)
    do_train = config["global"].getboolean("do_train")
    do_eval = config["global"].getboolean("do_eval")
    seo = config["global"].get("seo")

    trainer = None
    evaluater = None

    if do_train:
        trainer_basic_config = Basic_Config_For_Training(
            base_model=config.get('train.basic', 'model', fallback=''),
            tokenizer=config.get('train.basic', 'tokenizer', fallback=''),
            train_data_path=config.get('train.basic', 'train_data', fallback=''),
            validation_path=config.get('train.basic', 'validation_data', fallback=''),
        )

        if not trainer_basic_config.base_model:
            raise ValueError("base_model must not be empty.")
        if not trainer_basic_config.tokenizer:
            raise ValueError("tokenizer must not be empty.")
        if not trainer_basic_config.train_data_path:
            raise ValueError("train_data_path must not be empty.")
        # prompt_template_id 0; precision half
        trainer_config = Trainer_Config(
            prompt_template_id=config.getint('train.trainer', 'prompt_template_id', fallback=0),
            precision=config.get('train.trainer', 'precision', fallback='half'),
            tokenizer_padding_side=config.get('train.trainer', 'tokenizer_padding_side', fallback='right')
        )

        train_args = TrainingArguments(
            output_dir=config.get('train.hyperparameters', 'output_dir', fallback=''),
            per_device_train_batch_size=config.getint('train.hyperparameters', 'per_device_train_batch_size', fallback=4),
            gradient_accumulation_steps=config.getint('train.hyperparameters', 'gradient_accumulation_steps', fallback=64),
            weight_decay=config.getfloat('train.hyperparameters', 'weight_decay', fallback=0.01),
            learning_rate=config.getfloat('train.hyperparameters', 'learning_rate', fallback=5e-4),
            num_train_epochs=config.getint('train.hyperparameters', 'num_train_epochs', fallback=2),
            warmup_ratio=config.getfloat('train.hyperparameters', 'warmup_ratio', fallback=0.1),
            logging_steps=config.getint('train.hyperparameters', 'logging_steps', fallback=1),
            save_steps=config.getint('train.hyperparameters', 'save_steps', fallback=100),
            data_seed=config.getint('train.hyperparameters', 'data_seed', fallback=42),
            seed=config.getint('train.hyperparameters', 'seed', fallback=42),
            bf16=config.getboolean('train.hyperparameters', 'bf16', fallback=True),
            eval_strategy=config.get('train.hyperparameters', 'evaluation_strategy', fallback='steps'),
            eval_steps=config.getint('train.hyperparameters', 'eval_steps', fallback=150),
            per_device_eval_batch_size=config.getint('train.hyperparameters', 'per_device_eval_batch_size', fallback=32),
            eval_delay=config.getint('train.hyperparameters', 'eval_delay', fallback=1),
            optim=config.get('train.hyperparameters', 'optim', fallback='adamw_torch'),
            label_smoothing_factor=config.getfloat('train.hyperparameters', 'label_smoothing_factor', fallback=0),
            # label_smoothing_factor=0.1,  # enable label smoothing
            save_total_limit=3,
            eval_on_start=True,
            push_to_hub=False,
            report_to="tensorboard"
        )

        if not train_args.output_dir:
            raise ValueError("output_dir must not be empty.")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.getint('train.lora', 'r', fallback=16),
            target_modules=eval(config.get('train.lora', 'target_modules', fallback="['q_proj','k_proj']")),
            lora_alpha=config.getint('train.lora', 'lora_alpha', fallback=32),
            lora_dropout=config.getfloat('train.lora', 'lora_dropout', fallback=0.05),
            bias=config.get('train.lora', 'bias', fallback='none'),
            init_lora_weights=(
                config.get('train.lora', 'init_lora_weights', fallback=True)
                if config.get('train.lora', 'init_lora_weights', fallback=True) not in ["true", "false"]
                else config.getboolean('train.lora', 'init_lora_weights', fallback=True)
            )
        )

        trainer = TPG_Trainer(
            basic_config=trainer_basic_config,
            train_args=train_args,
            lora_config=lora_config,
            trainer_config=trainer_config
        )

    if do_eval:
        eval_basic_config = Basic_Config_For_Evaluation(
            base_model=config.get('eval.basic', 'base_model', fallback=''),
            tokenizer=config.get('eval.basic', 'tokenizer', fallback=''),
            lora_path=config.get('eval.basic', 'lora_path', fallback=''),
            test_path=config.get('eval.basic', 'test_path', fallback=''),
        )

        if not eval_basic_config.base_model:
            raise ValueError("base_model must not be empty.")
        if not eval_basic_config.tokenizer:
            raise ValueError("tokenizer must not be empty.")
        if not eval_basic_config.test_path:
            raise NotImplementedError

        search_config = Targeted_Search_Config(
            prompt_template_id=config.getint('eval.search', 'prompt_template_id', fallback=0),
            batch_size=config.getint('eval.search', 'batch_size', fallback=64),
            beam_width_list=eval(config.get('eval.search', 'beam_width_list', fallback="[95,95**2/2]+[1000]*16)")),
            eos_threshold=config.getfloat('eval.search', 'eos_threshold', fallback=0.001),
            vocab_limit=config.getboolean('eval.search', 'vocab_limit', fallback=True),
            max_guess_number=config.getint('eval.search', 'max_guess_number', fallback=1000)
        )

        guesser_config = Guesser_Config(
            result_path=config.get('eval.guesser', 'result_path', fallback=''),
            log_interval=config.getint('eval.guesser', 'log_interval', fallback=1),
            log_guess_number=eval(config.get('eval.guesser', 'log_guess_number', fallback="[1,10,100,1000]"))
        )

        evaluater = GuessLLM_Evaluator(
            basic_conf=eval_basic_config,
            eval_conf=search_config,
            guesser_conf=guesser_config
        )

        if not guesser_config.result_path:
            raise ValueError("result_path must not be empty.")

    if do_train and do_eval:
        assert train_args.output_dir + "final/" == eval_basic_config.lora_path

    return trainer, evaluater, seo


def load_divide_generaiotn_config(ini_path: str) -> DivideSearchRuntimeConfig:
    config = configparser.ConfigParser()
    config.read(ini_path)

    section = config['DivideSearch']

    return DivideSearchRuntimeConfig(
        base_model_path=section['base_model_path'],
        tokenizer_path=section['tokenizer_path'],
        lora_path=section['lora_path'],
        output_file=section['output_file'],
        threshold=float(section.get('threshold', 1e-6)),
        batch_size=int(section.get('batch_size', 2000)),
        max_length=int(section.get('max_length', 30)),
        max_prefix=int(section.get('max_prefix', 5000)),
        prompt_id=int(section.get('prompt_id', 1)),
        offload_to_cpu=section.getboolean('offload_to_cpu', False),
        log_interval=int(section.get('log_interval', 1000)),
        sort_by_prob=section.getboolean('sort_by_prob', True),
        chunk_output=section.getboolean('chunk_output', True),
    )


def load_width_search_config(ini_path: str) -> WidthSearchRuntimeConfig:
    config = configparser.ConfigParser()
    config.read(ini_path)

    section = config['WidthSearch']

    return WidthSearchRuntimeConfig(
        base_model_path=section['base_model_path'],
        tokenizer_path=section['tokenizer_path'],
        lora_path=section['lora_path'],
        output_file=section['output_file'],
        threshold=float(section.get('threshold', 1e-6)),
        batch_size=int(section.get('batch_size', 2000)),
        max_length=int(section.get('max_length', 30)),
        prompt_id=int(section.get('prompt_id', 1)),
        offload_to_cpu=section.getboolean('offload_to_cpu', False),
        sort_by_prob=section.getboolean('sort_by_prob', True),
        log_interval=int(section.get('log_interval', 100)),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Targeted password guessing with LLM")
    parser.add_argument("--config", type=str, required=True, help="Path of config file.")
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "distill", "eval", "wsgen", "dsgen"],
        help="Run mode: train | distill | eval"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "distill":
        run_distill_using_config(args.config)

    elif args.mode == "dsgen":
        config = load_divide_generaiotn_config(args.config)
        run_divide_search_from_config(config)

    elif args.mode == "wsgen":
        config = load_width_search_config(args.config)
        run_width_search_from_config(config)

    elif args.mode in ["train", "eval"]:
        trainer, evaluater, seo = load_training_config(args.config)
        if trainer:
            trainer.train(seo)
        if evaluater:
            evaluater.eval()

    else:
        raise NotImplementedError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    main()
