"""
Author: Originally Yifei Zhang, later modified by DeepSeek
Exports the Pass2Edit model to ONNX format for WebAssembly inference in the browser.

Usage:
    python onnx_convert.py --exp 1 --ckpt_name local_to_global --output simple_p2e.onnx

Model inputs:
    - src:  (batch=1, seq_len=32) int64  -- token sequence of the password before editing (pw2token format)
    - trg:  (batch=1, seq_len=32) int64  -- token sequence of the password after editing
    - len:  (batch=1,)              int64  -- valid length (excluding padding)

Model output:
    - output: (batch=1, num_classes) float32 -- logits of the edit operation (without softmax)
"""

import argparse
import pickle
import torch

# internal .py files
import hyper_param as param
import model as mt


def export_onnx(exp: int, ckpt_name: str, output_path: str):
    """Load the checkpoint and export it to ONNX."""

    # 1. Load the edit tokenizer to determine type_num
    tokenizer_path = f"./ckpts/exp{exp}/edit_tokenizer.pkl"
    with open(tokenizer_path, 'rb') as f:
        op_tokenizer = pickle.load(f)

    type_num = len(op_tokenizer.edit2id) + 5  # leave some headroom for OOV
    print(f"[INFO] Edit tokenizer vocab size: {len(op_tokenizer.edit2id)}, type_num={type_num}")

    # 2. Build the model
    model = mt.SimpleP2E(type_num=type_num)
    model.to('cpu')
    model.eval()

    # 3. Load the weights
    ckpt_path = f"./ckpts/exp{exp}/{ckpt_name}.pth"
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # handle different save formats
    if 'Model' in state_dict:
        model.load_state_dict(state_dict['Model'])
    elif 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)
    print(f"[INFO] Loaded checkpoint from {ckpt_path}")

    # 4. Construct example inputs
    # pw2token format: [BOS=0, char_tokens..., EOS=1, PAD=2, ...] fixed length 32
    max_len = param.MAX_LEN + 2  # 32

    # Use a simple example: a step from source="abc" to target="abd"
    # Construct the input with random valid tokens; only the shape matters
    dummy_src = torch.zeros((1, max_len), dtype=torch.int64)
    dummy_trg = torch.zeros((1, max_len), dtype=torch.int64)
    dummy_len = torch.tensor([5], dtype=torch.int64)  # assume a valid length of 5

    # Fill in example tokens (BOS=0, valid char tokens start at 4)
    for i in range(5):
        dummy_src[0, i] = 4 + (i % 40)
        dummy_trg[0, i] = 4 + ((i + 1) % 40)

    print(f"[INFO] Dummy input shapes: src={dummy_src.shape}, trg={dummy_trg.shape}, len={dummy_len.shape}")

    # 5. Verify the model forward pass works
    with torch.no_grad():
        test_out = model(dummy_src, dummy_trg, dummy_len)
    print(f"[INFO] Test forward pass OK, output shape: {test_out.shape}")

    # 6. Export to ONNX
    torch.onnx.export(
        model,
        (dummy_src, dummy_trg, dummy_len),
        output_path,
        export_params=True,
        opset_version=14,
        input_names=["src", "trg", "len"],
        output_names=["output"],
        dynamic_axes={
            "src": {0: "batch_size"},
            "trg": {0: "batch_size"},
            "len": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"[INFO] ONNX model exported to {output_path}")

    # 7. Validate the ONNX model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[INFO] ONNX model validation passed!")

    # 8. Run an inference verification with onnxruntime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        ort_inputs = {
            "src": dummy_src.numpy(),
            "trg": dummy_trg.numpy(),
            "len": dummy_len.numpy(),
        }
        ort_outputs = session.run(None, ort_inputs)
        print(f"[INFO] ONNX Runtime inference OK, output shape: {ort_outputs[0].shape}")

        # Compare the PyTorch and ONNX outputs
        torch_output = test_out.cpu().numpy()
        onnx_output = ort_outputs[0]
        diff = abs(torch_output - onnx_output).max()
        print(f"[INFO] Max difference between PyTorch and ONNX output: {diff:.6f}")
    except ImportError:
        print("[WARN] onnxruntime not installed, skipping inference verification")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Export Pass2Edit SimpleP2E model to ONNX")
    # parser.add_argument("--exp", type=int, default=2, help="Experiment number (1-10)")
    # parser.add_argument("--ckpt_name", type=str, default="local_to_global",
    #                     help="Checkpoint name (e.g. local, global, local_to_global, etc.)")
    # parser.add_argument("--output", type=str, default="./browser_demo/simple_p2e.onnx",
    #                     help="Output ONNX file path")
    # args = parser.parse_args()

    # export_onnx(args.exp, args.ckpt_name, args.output)\
    test_model={1 : 2,
            2 : 1,
            3 : 4,
            4 : 3,
            5 : 7,
            6 : 5,
            7 : 10,
            8 : 6}

    for exp in range(1, 9):
        ckpt_name = f"local_to_global"
        output_path = f"./browser_demo/onnx_models/exp{exp}_model.onnx"
        export_onnx(test_model[exp], ckpt_name, output_path)
