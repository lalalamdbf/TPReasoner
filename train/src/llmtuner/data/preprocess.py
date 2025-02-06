from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Tuple

from ..extras.constants import IGNORE_INDEX
from ..extras.logging import get_logger
from .utils import Role


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ..hparams import DataArguments
    from .template import Template


logger = get_logger(__name__)


def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
) -> Dict[str, List[List[int]]]:
    # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
    text_examples = [messages[0]["content"] + tokenizer.eos_token for messages in examples["prompt"]]
    if not data_args.packing:
        return tokenizer(text_examples, add_special_tokens=False, max_length=data_args.cutoff_len)

    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
    concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    block_size = data_args.cutoff_len
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (total_length // block_size) * block_size
    # split by chunks of cutoff_len
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    if data_args.template == "gemma":
        for i in range(len(result["input_ids"])):
            result["input_ids"][i][0] = tokenizer.bos_token_id

    return result


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            continue

        messages = examples["prompt"][i] + examples["response"][i]
        input_ids, labels = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(
            template.encode_multiturn(
                tokenizer,
                messages,
                examples["system"][i],
                examples["tools"][i],
                data_args.cutoff_len,
                data_args.reserved_label_len,
            )
        ):
            if data_args.train_on_prompt:
                source_mask = source_ids
            elif turn_idx != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    input_ids, labels = [], []
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            continue

        messages = examples["prompt"][i] + examples["response"][i]
        for source_ids, target_ids in template.encode_multiturn(
            tokenizer, messages, examples["system"][i], examples["tools"][i]
        ):
            if data_args.train_on_prompt:
                source_mask = source_ids
            elif len(input_ids) != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    total_length = len(input_ids)
    block_size = data_args.cutoff_len
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (total_length // block_size) * block_size
    # split by chunks of cutoff_len
    for i in range(0, total_length, block_size):
        if not all(label == IGNORE_INDEX for label in labels[i : i + block_size]):
            model_inputs["input_ids"].append(input_ids[i : i + block_size])
            model_inputs["attention_mask"].append([1] * block_size)
            model_inputs["labels"].append(labels[i : i + block_size])

    return model_inputs


def preprocess_unsupervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X` and labels with format `Y <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1:
            continue

        if len(examples["response"][i]) == 1:
            messages = examples["prompt"][i] + examples["response"][i]
        else:
            messages = examples["prompt"][i] + [{"role": Role.ASSISTANT.value, "content": ""}]

        input_ids, labels = template.encode_oneturn(
            tokenizer,
            messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )

        if template.efficient_eos:
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) < 2:
            continue
        chosen_messages = examples["prompt"][i] + [examples["response"][i][0]]
        rejected_messages = examples["prompt"][i] + [examples["response"][i][1]]
        prompt_ids, chosen_ids = template.encode_oneturn(
            tokenizer,
            chosen_messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )
        _, rejected_ids = template.encode_oneturn(
            tokenizer,
            rejected_messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )

        if template.efficient_eos:
            chosen_ids += [tokenizer.eos_token_id]
            rejected_ids += [tokenizer.eos_token_id]

        model_inputs["prompt_ids"].append(prompt_ids)
        model_inputs["chosen_ids"].append(chosen_ids)
        model_inputs["rejected_ids"].append(rejected_ids)

    return model_inputs

def preprocess_cpo_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build input pairs with format `<bos> X1`, `Y1 <eos>`; `<bos> X2`, `Y2 <eos>`
        
    def extract_4path_index(sentence_ids, extract_ids, is_start = True):
        matching_indices = []
        for i in range(len(sentence_ids) - len(extract_ids) + 1):
            if sentence_ids[i:i+len(extract_ids)] == extract_ids:
                matching_indices.append(i + len(extract_ids) - 1 if is_start else i - 2)
        if len(matching_indices) != 4:
            print(sentence_ids, extract_ids)
            raise ValueError("error")
        return matching_indices[-4:]
            
        
    mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3} 
    model_inputs = {"prompt_ids": [], "response_ids": [], "extract_indices": [], "anchor_num": [], "positive_num_0": [], "positive_num_1": []}
    for i in range(len(examples["prompt"])):
    
        example_0 = [examples["prompt"][i][0]] + [examples["response"][i][0]]
        example_0_answer = mapping[examples["response"][i][0]['content'][-3]]
        example_1 = [examples["prompt"][i][1]] + [examples["response"][i][1]]
        example_1_answer = mapping[examples["response"][i][1]['content'][-3]]
        
        temp = [0, 1, 2, 3]
        anchor_num_0 = [example_0_answer]
        anchor_num_1 = [example_1_answer]
        positive_num = [item for index, item in enumerate(temp) if index not in [example_0_answer, example_1_answer]]
        positive_num_0 = [positive_num[0]]
        positive_num_1 = [positive_num[1]]
        # negative_num = [example_1_answer]
        # negative_num_0 = [example_0_answer]
        # negative_num_1 = [example_1_answer]
        # negative_num_0 = [item for index, item in enumerate(temp) if index not in anchor_num]
        # negative_num_1 = [item for index, item in enumerate(temp) if index not in positive_num]
        
        # Mapping model names to their specific token IDs for start and end extraction points
        # start: "Analysis:"  end:'.\n\n'
        model_token_ids = {
            "Mistral": {
                "start_ids": [22039, 28747],
                # "end_ids": [28723, 13, 13]
                # "end_ids": [7648, 1598, 6097, 4637, 29901]
                "end_ids": [8876, 1575, 7809, 3900, 28747]
            },
            "Llama-2": {
                "start_ids": [21067, 4848, 29901],
                # "end_ids": [29889, 13, 13]
                "end_ids": [7648, 1598, 6097, 4637, 29901]
            },
            "Llama-3": {
                "start_ids": [27671, 25],
                # "end_ids": [382]
                "end_ids": [29401, 1463, 12029, 5014, 25]
            }
        }

        # Check the model type from tokenizer's name or path and get corresponding token IDs
        if any(model in tokenizer.name_or_path for model in model_token_ids):
            model_type = next(model for model in model_token_ids if model in tokenizer.name_or_path)
            extract_start_ids = model_token_ids[model_type]['start_ids']
            extract_end_ids = model_token_ids[model_type]['end_ids']
        else:
            raise ValueError("The current support is only for Mistral, Llama-2, and Llama-3.")


        prompt_ids_0, response_id_0 = template.encode_oneturn(
            tokenizer,
            example_0,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )
        
        prompt_ids_1, response_id_1 = template.encode_oneturn(
            tokenizer,
            example_1,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )

        if template.efficient_eos:
            response_id_0 += [tokenizer.eos_token_id]
            response_id_1 += [tokenizer.eos_token_id]
            
        sentence_id_0 = prompt_ids_0 + response_id_0
        sentence_id_1 = prompt_ids_1 + response_id_1
        
        extract_start_indices_0 = extract_4path_index(sentence_id_0, extract_start_ids, True) 
        extract_start_indices_1 = extract_4path_index(sentence_id_1, extract_start_ids, True) 
        extract_end_indices_0 = extract_4path_index(sentence_id_0, extract_end_ids, False) 
        extract_end_indices_1 = extract_4path_index(sentence_id_1, extract_end_ids, False) 
        
        extract_indices_range_0 = [list(pair) for pair in zip(extract_start_indices_0, extract_end_indices_0)]
        extract_indices_range_1 = [list(pair) for pair in zip(extract_start_indices_1, extract_end_indices_1)]
        
        extract_indices_0 = [list(range(start, end)) for start, end in extract_indices_range_0]
        extract_indices_1 = [list(range(start, end)) for start, end in extract_indices_range_1]

        # raise ValueError("test.")
        model_inputs["prompt_ids"].append([prompt_ids_0, prompt_ids_1])
        model_inputs["response_ids"].append([response_id_0, response_id_1])
        model_inputs["extract_indices"].append([extract_indices_0, extract_indices_1])
        model_inputs["anchor_num"].append([anchor_num_0, anchor_num_1])
        model_inputs["positive_num_0"].append([positive_num_0, positive_num_0])
        model_inputs["positive_num_1"].append([positive_num_1, positive_num_1])
        # model_inputs["negative_num"].append([negative_num_0, negative_num_1])

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print(
        "labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        )
    )


def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("prompt_ids:\n{}".format(example["prompt_ids"]))
    print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
    print("chosen_ids:\n{}".format(example["chosen_ids"]))
    print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"], skip_special_tokens=False)))
    print("rejected_ids:\n{}".format(example["rejected_ids"]))
    print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"], skip_special_tokens=False)))
    
def print_cpo_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("prompt_ids:\n{}".format(example["prompt_ids"][0]))
    print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"][0], skip_special_tokens=False)))
    # print("chosen_ids:\n{}".format(example["chosen_ids"][0]))
    # print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"][0], skip_special_tokens=False)))
    # print("rejected_ids:\n{}".format(example["rejected_ids"][1]))
    # print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"][1], skip_special_tokens=False)))
    print("prompt_ids:\n{}".format(example["prompt_ids"][1]))
    print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"][1], skip_special_tokens=False)))
    # print("chosen_ids:\n{}".format(example["chosen_ids"][1]))
    # print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"][1], skip_special_tokens=False)))
    # print("rejected_ids:\n{}".format(example["rejected_ids"][0]))
    # print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"][0], skip_special_tokens=False)))
    


def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))


def get_preprocess_and_print_func(
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo"],
) -> Tuple[Callable, Callable]:
    if stage == "pt":
        preprocess_func = partial(preprocess_pretrain_dataset, tokenizer=tokenizer, data_args=data_args)
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)
    elif stage == "sft" and not training_args.predict_with_generate:
        if data_args.packing:
            preprocess_func = partial(
                preprocess_packed_supervised_dataset, tokenizer=tokenizer, template=template, data_args=data_args
            )
        else:
            preprocess_func = partial(
                preprocess_supervised_dataset, tokenizer=tokenizer, template=template, data_args=data_args
            )

        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
    elif stage == "rm":
        preprocess_func = partial(
            preprocess_pairwise_dataset, tokenizer=tokenizer, template=template, data_args=data_args
        )
        print_function = partial(print_pairwise_dataset_example, tokenizer=tokenizer)
    elif stage == "cpo":
        preprocess_func = partial(
            preprocess_cpo_dataset, tokenizer=tokenizer, template=template, data_args=data_args
        )
        print_function = partial(print_cpo_dataset_example, tokenizer=tokenizer)
    else:
        preprocess_func = partial(
            preprocess_unsupervised_dataset, tokenizer=tokenizer, template=template, data_args=data_args
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)

    return preprocess_func, print_function
