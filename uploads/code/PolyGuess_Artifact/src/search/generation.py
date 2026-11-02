from dataclasses import dataclass


from src.model.eval import Basic_Config_For_Evaluation,load_model
from src.search.search import divide_search,width_search

@dataclass
class WidthSearchRuntimeConfig:
    base_model_path: str
    tokenizer_path: str
    lora_path: str
    output_file: str
    threshold: float = 1e-6
    batch_size: int = 2000
    max_length: int = 30
    prompt_id: int = 1
    offload_to_cpu: bool = False
    sort_by_prob: bool = True
    log_interval: int = 100

@dataclass
class DivideSearchRuntimeConfig:
    base_model_path: str
    tokenizer_path: str
    lora_path: str
    output_file: str
    threshold: float = 1e-6
    batch_size: int = 2000
    max_length: int = 30
    max_prefix: int = 5000
    prompt_id: int = 1
    offload_to_cpu: bool = False
    log_interval: int = 1000
    sort_by_prob: bool = True
    chunk_output: bool = True


def run_divide_search_from_config(config: DivideSearchRuntimeConfig):
    # Model loading
    base_config = Basic_Config_For_Evaluation(
        base_model=config.base_model_path,
        tokenizer=config.tokenizer_path,
        lora_path=config.lora_path,
        test_path= None
    )
    model, tokenizer, pw_vocab = load_model(base_config)

    return divide_search(
        model=model,
        tokenizer=tokenizer,
        output_file=config.output_file,
        prompt_id=config.prompt_id,
        threshold=config.threshold,
        batch_size=config.batch_size,
        max_length=config.max_length,
        max_prefix=config.max_prefix,
        vocab=pw_vocab,
        offload_to_cpu=config.offload_to_cpu,
        log_interval=config.log_interval,
        sort_by_prob=config.sort_by_prob,
        chunk_output=config.chunk_output
    )


def run_width_search_from_config(config: WidthSearchRuntimeConfig):

    base_config = Basic_Config_For_Evaluation(
        base_model = config.base_model_path,
        tokenizer = config.tokenizer_path,
        lora_path = config.lora_path,
        test_path = None
    )

    model, tokenizer, pw_vocab = load_model(base_config)

    return width_search(
        model=model,
        tokenizer=tokenizer,
        output_file=config.output_file,
        threshold=config.threshold,
        batch_size=config.batch_size,
        max_length=config.max_length,
        vocab=pw_vocab,
        offload_to_cpu=config.offload_to_cpu,
        prompt_id=config.prompt_id,
        sort_by_prob=config.sort_by_prob,
        log_interval=config.log_interval
    )