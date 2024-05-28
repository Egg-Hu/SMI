

# [ICML 2024] Sparse Model Inversion: Efficient Inversion of Vision Transformers with Less Hallucination

Official code of Sparse Model Inversion: Efficient Inversion of Vision Transformers with Less Hallucination

## Requirements
  
```bash
pip install -r requirements.txt
```

## Model Quantization

- Quantize a full-precision model to a low-precision one.
```bash
python test_quant.py [--model] [--dataset] [--datapool] [--mode] [--w_bit] [--a_bit] [--prune_it] [--prune_ratio]

optional arguments:
--model: model architecture, the choises can be: deit_base_16_imagenet and deit_tiny_16_imagenet.
--dataset: path to ImageNet dataset.
--datapool: path to store inverted data.
--mode: mode of calibration data,
        0: Inverted data using sparse model inversion
        1: Gaussian noise
--w_bit: bit-precision of weights, default=8.
--a_bit: bit-precision of activation, default=8.
--prune_it: the iteration indexes for inversion stopping
            -1: to densely invert data
            t1 t2 ... tn: to sparsely invert data and perform inversion stopping at t1, t2, ..., tn
--prune_ratio: the proportion of patches to be pruned relative to the current remaining patches
            0: to densely invert data
            r1 r2 ... rn: progressively stopping the inversion of a fraction (r1, r2, ..., rn)$$ of patches at iterations (t1, t2, ..., tn), respectively
```

- Example: Quantize (W8/A8) DeiT/16-Base with sparsely inverted data **(Sparse Model Inversion)**.

```bash
python test_quant.py --model deit_base_16_imagenet --prune_it 50 100 200 300 --prune_ratio 0.3 0.3 0.3 0.3 --dataset <YOUR_DATA_DIR> --datapool <YOUR_DATAPOOL_DIR> --mode 0 --w_bit 8 --a_bit 8
```

- Example: Quantize (W8/A8) DeiT/16-Base with densely inverted data **(DeepInversion)**.

```bash
python test_quant.py --model deit_base_16_imagenet  --prune_it -1 --prune_ratio 0 --dataset <YOUR_DATA_DIR> --datapool <YOUR_DATAPOOL_DIR> --mode 0 --w_bit 8 --a_bit 8
```

- Example: Quantize (W8/A8) DeiT/16-Base with **(Gaussian noise)**.

```bash
python test_quant.py --model deit_base_16_imagenet  --dataset <YOUR_DATA_DIR> --mode 1
```
## Knowledge Transfer
- Transfer the specific knowledge of one teacher model to the other student model using inverted data.
```bash
python test_quant.py [--model] [--dataset] [--model_path] [--datapool] [--prune_it] [--prune_ratio]

optional arguments:
--model: model architecture of teacher and student, the choises can be: deit_tiny_16_cifar10/deit_base_16_cifar10/deit_tiny_16_cifar100/deit_base_16_cifar100.
--dataset: path to CIFAR10 and CIFAR100 dataset.
--model_path: path to teacher model.
--datapool: path to store inverted data.
--prune_it: the iteration indexes for inversion stopping
            -1: to densely invert data
            t1 t2 ... tn: to sparsely invert data and perform inversion stopping at t1, t2, ..., tn
--prune_ratio: the proportion of patches to be pruned relative to the current remaining patches
            0: to densely invert data
            r1 r2 ... rn: progressively stopping the inversion of a fraction (r1, r2, ..., rn)$$ of patches at iterations (t1, t2, ..., tn), respectively
```

- Example: Transfer knowledge of CIFAR10 from teacher (deit_tiny_16_cifar10) to student using sparsely inverted data from **(Sparse Model Inversion)**.

```bash
python test_kt.py --model deit_tiny_16_cifar10 --prune_it 50 100 200 300 --prune_ratio 0.3 0.3 0.3 0.3 --dataset <YOUR_DATA_DIR> --model_path <YOUR_TEACHER_DIR> --datapool <YOUR_DATAPOOL_DIR>
```

- Example: Transfer knowledge of CIFAR10 from teacher (deit_tiny_16_cifar10) to student using densely inverted data from **(DeepInversion)**.

```bash
python test_kt.py --model deit_tiny_16_cifar10 --prune_it -1 --prune_ratio 0 --dataset <YOUR_DATA_DIR> --model_path <YOUR_TEACHER_DIR> --datapool <YOUR_DATAPOOL_DIR>
```

## Acknowledge
