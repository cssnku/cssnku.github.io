"""
Author: Yifei Zhang

Exports the tokenizer and model configuration for the browser demo.
"""

import argparse
import json
import pickle
import string

from word2keypress import Keyboard

import hyper_param as param


def export_all(exp: int, output_dir: str):
    """Export all configuration files needed by the browser client."""

    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1. Export the keyboard charset (charset used by pw2token)
    kb = Keyboard()
    charset_full = string.printable[:-5]
    kb_charlist = kb.word_to_keyseq(charset_full)
    kb_charset = []
    for c in kb_charlist:
        if c not in kb_charset:
            kb_charset.append(c)
    charset = ''.join(kb_charset)
    print(f"[INFO] Keyboard charset ({len(charset)} unique chars): {charset}")

    with open(f"{output_dir}/keyboard_charset.json", 'w', encoding='utf-8') as f:
        json.dump({"charset": charset, "size": len(charset)}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Exported keyboard_charset.json")

    # 2. Export the edit operation tokenizer
    tokenizer_path = f"./ckpts/exp{exp}/edit_tokenizer.pkl"
    with open(tokenizer_path, 'rb') as f:
        op_tokenizer = pickle.load(f)

    # edit2id: maps edit operation strings to integer IDs
    # Note: JSON keys must be strings, and the edit2id keys are already strings
    edit2id = dict(op_tokenizer.edit2id)
    print(f"[INFO] Edit tokenizer vocab size: {len(edit2id)}")

    with open(f"{output_dir}/edit_op_tokenizer.json", 'w', encoding='utf-8') as f:
        json.dump({"edit2id": edit2id}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Exported edit_op_tokenizer.json")

    # 3. Export the model configuration
    config = {
        "MAX_LEN": param.MAX_LEN,
        "SEQ_LEN": param.MAX_LEN + 2,  # actual sequence length including BOS/EOS
        "num_embeddings": param.num_embeddings,
        "type_num": len(op_tokenizer.edit2id) + 5,
        "BOS_TOKEN": 0,
        "EOS_TOKEN": 1,
        "PAD_TOKEN": 2,
        "DEL_TOKEN": 3,  # \xde delete marker
        "CHAR_OFFSET": 4,  # actual character tokens start at 4
    }
    with open(f"{output_dir}/model_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Exported model_config.json")

    print(f"\n[DONE] All files exported to {output_dir}/")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Export tokenizers and config for browser demo")
    # parser.add_argument("--exp", type=int, default=1, help="Experiment number (1-10)")
    # parser.add_argument("--output_dir", type=str, default="./browser_demo",
    #                     help="Output directory for JSON files")
    # args = parser.parse_args()
    test_model={1 : 2,
            2 : 1,
            3 : 4,
            4 : 3,
            5 : 7,
            6 : 5,
            7 : 10,
            8 : 6}

    for exp in range(1, 9):
        output_dir = f"./browser_demo/tokenizers/exp{exp}"
        export_all(test_model[exp], output_dir)
