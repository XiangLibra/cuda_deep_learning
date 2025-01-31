# CUDA practice for deep learning

# 一般 NN

```bash
cd cuda_nn
```

## cuda核心編譯版
```python
python cuda_nn.py
```

## pytorch呼叫cuda編譯板
```python
python pytorch_nn.py
```

## 當中可以發現cuda核心編譯板的執行速度比pytorch呼叫cuda編譯板還快

# Transformer訓練版
```bash
cd cuda_transformer
```

## cuda核心transformer編譯板
```python
python cuda_transformer.py
```

## pytorch呼叫cuda transformer編譯板
```python
python pytorch_transformer.py
```

## 當中可以發現cuda核心編譯板的執行速度比pytorch呼叫cuda編譯板還快


# PTX cuda之一

```bash
cd cuda_ptx_api
```

```bash
bash cuda_driver_api.sh
```

# PTX cuda之二

```bash
cd ptx_launcher
```

```bash
bash ptx_launcher.sh
```