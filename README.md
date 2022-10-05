# 광운대학교 컴퓨터 비전 AI 경진대회 1등 솔루션

## 1. Introduction
본 프로젝트는 *광운대학교 컴퓨터 비전 AI 경진대회* 리더보드 1위 솔루션 내용을 담고 있습니다. 아래의 문서를 참고하여 새로운 모델을 학습하거나 예측을 수행할 수 있습니다. 모든 학습된 모델 가중치는 [여기서](https://github.com/affjljoo3581/KW-Computer-Vision-AI-1st-Solution/releases/tag/v1.0) 다운로드 받을 수 있습니다.

## 2. Requirements
본 프로젝트를 원활히 실행하기 위해서는 다음의 라이브러리가 필요합니다.

* albumentations
* numpy
* omegaconf
* opencv_python_headless
* pandas
* pytorch_lightning
* scikit_learn
* sentencepiece
* timm
* torch
* tqdm
* wandb

혹은 다음의 스크립트를 실행하여 필요한 모듈을 한번에 설치할 수 있습니다.

```bash
$ pip install -r requirements.txt 
```

또한 [NVIDIA의 Apex 라이브러리](https://github.com/NVIDIA/apex)를 사용하여 학습을 진행했습니다. 해당 라이브러리를 설치하는 것을 권장드립니다.
```bash
$ git clone https://github.com/NVIDIA/apex
$ sed -i "s/or (bare_metal_minor != torch_binary_minor)//g" apex/setup.py
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex/
$ rm -rf apex
```

마지막으로 [wandb](https://wandb.ai/home)에 가입하여 터미널에서 로그인하세요. 방법은 다음과 같습니다.
```bash
$ wandb login [your wandb api]
```

### Using docker to setup the environment
빠른 학습과 클라우드 환경설정을 위해 [nvcr.io](https://catalog.ngc.nvidia.com/containers)의 [PyTorch 이미지](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)를 사용하는 것을 권장드립니다. 본 프로젝트는 [GCP](https://www.google.com/cloud) 환경에서 `pytorch:22.08-py3` 이미지로 학습하였으며, 다음의 코드를 통해 실행 가능합니다.
```bash
$ docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:22.08-py3
```
도커 환경을 사용하는 경우에는 Apex 라이브러리를 설치하지 않아도 됩니다.

## 3. Implementations
본 프로젝트는 여러 스크립트 파일을 포함하고 있습니다. 각 프로그램별 사용 방법은 다음과 같습니다.

### Train new model
[config](./config) 폴더 안에 학습하고자 하는 config 파일을 생성하세요. config 파일에 대한 자세한 설명은 [여기](#4-how-to-reproduce-the-leaderboard-score)서 확인하세요.
다음의 명령을 통해 모델을 학습할 수 있습니다.
```bash
$ python src/train.py config/[config name]
```

학습이 중단되어 지속하고자 한다면, 다음의 명령을 사용하세요. wandb의 `run-id`와 checkpoint file 경로를 확인해 두세요. 다음의 명령으로 모델 학습을 지속할 수 있습니다.
```bash
$ python src/train.py config/[config name] --resume-id [wandb id] --resume-from [checkpoint file]
```

### Inference with the model
모델이 성공적으로 학습된다면 `*.pt` 형태의 파일이 생성됩니다. 이를 통해 이미지에 대한 예측을 수행할 수 있습니다. 다음의 명령으로 추론을 실행하세요.
```bash
$ python src/predict.py [model file] [image directory]
```

이미지 크기를 변경하려면 `--image-size` 인자를 사용하세요. batch size와 tta를 적용하기 위해서는 `--batch-size`와 `--use-tta` 인자를 설정하면 됩니다. 앙상블을 위해 확률을 출력하기 위해서는 `--return-probs` 인자를 설정하세요.

자세한 사용 방법은 다음과 같습니다.
```bash
$ python src/predict.py --help
usage: predict.py [-h] [--image-size IMAGE_SIZE] [--batch-size BATCH_SIZE]
                       [--use-tta] [--return-probs] model imagedir

positional arguments:
  model
  imagedir

optional arguments:
  -h, --help            show this help message and exit
  --image-size IMAGE_SIZE
  --batch-size BATCH_SIZE
  --use-tta
  --return-probs
```

### Synthesize new images
새로운 이미지를 생성하기 위해서는 다음의 명령을 수행하세요.
```bash
$ python utils/synthesize_images.py --mnist-data [mnist data directory] --output-dir [output directory] --output-csv [output csv file] --num-images [number of images] --index-offset [offset of indices]
```
자세한 내용은 [아래](#4-how-to-reproduce-the-leaderboard-score)를 참고하세요. 사용법은 다음과 같습니다.
```bash
$ python utils/synthesize_images.py --help
usage: synthesize_images.py [-h] [--mnist-data MNIST_DATA] [--output-dir OUTPUT_DIR] 
                                 [--output-csv OUTPUT_CSV] [--num-images NUM_IMAGES] 
                                 [--index-offset INDEX_OFFSET]

optional arguments:
  -h, --help            show this help message and exit
  --mnist-data MNIST_DATA
  --output-dir OUTPUT_DIR
  --output-csv OUTPUT_CSV
  --num-images NUM_IMAGES
  --index-offset INDEX_OFFSET
```

## 4. How to reproduce the leaderboard score?
본 팀의 점수를 재현하기 위해서는 여러 환경설정 및 전처리를 수행해 주어야 합니다. 우선 대회의 데이터 탭에서 데이터셋을 다운로드 받으세요. 프로젝트 최상단에 `resources` 디렉토리를 생성한 후, 데이터셋의 압축을 풀어주세요.

```bash
$ tree
.
├── config
│   └── ...
├── resources
│   ├── dirty_mnist_2nd
│   │  ├── 00000.png
│   │  ├── 00001.png
│   │  ├── 00002.png
│   │  ├── 00003.png
│   │  └── ...
│   ├── mnist_data
│   │  ├── submission.csv
│   │  ├── test.csv
│   │  └── train.csv
│   ├── dirty_mnist_2nd_answer.csv
│   ├── sample_submission.csv
│   └── test_dirty_mnist_2nd
│       ├── 50000.png
│       ├── 50001.png
│       ├── 50002.png
│       ├── 50003.png
│       └── ...
└── src
    └── ...
```
이후 `resources/mnist_data/test.csv`를 분할해 주겠습니다. 이로부터 추가적인 학습 이미지와 새로운 validation 이미지를 생성하도록 합니다.

```python
import pandas as pd

test = pd.read_csv("resources/mnist_data/test.csv")

test.iloc[:, :-3000].to_csv("resources/mnist_data/test-split-1.csv", index=False)
test.iloc[:, -3000:].to_csv("resources/mnist_data/test-split-2.csv", index=False)
```

```bash
$ python utils/synthesize_images.py --mnist-data resources/mnist_data/test-split-1.csv \
                                    --output-dir dirty_mnist_2nd \
                                    --output-csv dirty_mnist_2nd_answer.csv \
                                    --num-images 350000
$ python utils/synthesize_images.py --mnist-data resources/mnist_data/test-split-2.csv \
                                    --output-dir val_dirty_mnist_2nd \
                                    --output-csv val_dirty_mnist_2nd_answer.csv \
                                    --num-images 5000
```
최상단에 생성된 `dirty_mnist_2nd`, `dirty_mnist_2nd_answer.csv`를 `resources`에 위치한 기존 학습 데이터와 병합해 주세요. `val_dirty_mnist_2nd`와 `val_dirty_mnist_2nd_answer.csv` 파일은 단순히 옮기면 됩니다.

추가적인 학습 데이터와 validation 데이터가 생성되었다면, config 파일을 수정해 주세요. 다음은 `config/tf_efficientnetv2_m_in21k.yaml`의 내용입니다.

```yaml
data:
  train:
    filenames: resources/dirty_mnist_2nd/*.png
    labels: resources/dirty_mnist_2nd_answer.csv
  validation:
    filenames: resources/val_dirty_mnist_2nd/*.png
    labels: resources/val_dirty_mnist_2nd_answer.csv
  image_size: 384

model:
  model_name: tf_efficientnetv2_m_in21k
  pretrained: true
  in_chans: 1
  num_classes: 26
  drop_rate: 0.5
  drop_path_rate: 0.2

optim:
  opt: fusedadam
  lr: 1e-3
  betas: [0.9, 0.999]
  eps: 1e-6
  weight_decay: 1e-5
  filter_bias_and_bn: false

train:
  epochs: 30
  batch_size: 512
  accumulate_grad_batches: 1
  gradient_clip_val: 0.0
  gradient_checkpointing: true
  validation_interval: 0.2
  log_every_n_steps: 10
```
여기서 실험에 사용할 여러 하이퍼파라미터를 수정할 수 있습니다. [아래의 benchmark table](#5-experiments-and-performance-benchmark)을 재현할 때 관련된 값을 수정해 주세요. 혹은 후술하겠지만 인자를 통해 변경할 수도 있습니다.

config 파일을 수정한 후 모델을 학습해 주세요. [benchmark table](#5-experiments-and-performance-benchmark)을 참고하여 재현에 필요한 여러 설정으로 학습하면 됩니다.
```bash
$ python src/train.py config/tf_efficientnetv2_m_in21k.yaml
```
만약 하이퍼파라미터를 수정하고자 한다면, config 파일을 변경하는 것 대신 다음처럼 실행할 수도 있습니다.
```bash
$ python src/train.py config/tf_efficientnetv2_m_in21k.yaml train.batch_size=512 \
                                                            train.gradient_checkpointing=true \
                                                            optim.lr=5e-3
```

이후 학습된 모델이 저장되었는지 확인하세요. `*.pt` 파일을 통해 다음과 같이 실행하여 이미지를 예측하세요.

```bash
$ python src/predict.py tf_efficientnetv2_m_in21k-XXXXX.pt \
                        resources/val_dirty_mnist_2nd \
                        --use-tta \
                        --return-probs 
```
만약 이미지가 256 해상도로 학습되었다면 `--image-size 256`과 같이 명시해 주어야 합니다.
이후 생성된 파일은 각 모델의 예측에 대한 확률값을 포함하고 있습니다. 이를 통해 다음과 같이 앙상블을 진행해 주면 됩니다. 본 팀의 최고 성능을 도출한 앙상블 가중치는 다음과 같습니다.

| name | public lb | weights |
|:-:|:-:|:-:|
| **d269b1** | **0.8917** | **1.5** |
| 26527b | 0.8908 | 1.0 |
| f43ccc | 0.8896 | 1.0 |
| 0545cd | 0.8886 | 1.0 |
| d6f15a | 0.8884 | 1.0 |

앙상블한 public lb 점수는 0.9002615385입니다. 각 모델별 이름과 하이퍼파라미터 조합은 [다음 문단](#5-experiments-and-performance-benchmark)을 참고하세요.

## 5. Experiments and performance benchmark

다음은 다양한 하이퍼파라미터 조합에 대한 성능 실험 결과입니다. 이를 참고하여 앙상블을 진행했습니다.

dataset v1은 35만 장 대신 15만 장의 합성 이미지로 구성된 데이터셋입니다. v1과 v2의 validation set이 서로 다름을 유의하세요.

| name | model | batch size | grad ckpt | epochs | augmented epochs | image size | dataset | val/v1 | val/v2 | val/v2 (tta) | public lb (tta) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 084a1d | tf_efficientnetv2_m_in21k | 32 | false | 10 | 40 | 384 | v1 | 0.8278 | 0.8306 | 0.8365 | 0.8844 |
| 0545cd | tf_efficientnetv2_m_in21k | 32 | false | 30 | 120 | 384 | v1 | 0.8344 | 0.8371 | 0.8418 | 0.8886 |
| d6240c | tf_efficientnetv2_m_in21k | 32 | false | 10 | 40 | 256 | v1 | 0.8224 | 0.8255 | 0.8332 | 0.8773 |
| f43ccc | tf_efficientnetv2_m_in21k | 64 | false | 30 | 240 | 384 | v2 | - | 0.8581 | 0.8629 | 0.8896 |
| e2082d | tf_efficientnetv2_m_in21k | 32 | false | 30 | 240 | 384 | v2 | - | 0.8505 | 0.8546 | 0.8823 |
| d6f15a | tf_efficientnetv2_l_in21k | 32 | false | 30 | 240 | 384 | v2 | - | 0.8524 | 0.8580 | 0.8884 |
| eb6ac8 | tf_efficientnetv2_s_in21k | 128 | false | 30 | 240 | 384 | v2 | - | 0.8602 | 0.8653 | 0.8848 |
| 29b10b | tf_efficientnetv2_s_in21k | 128 | false | 30 | 240 | 384 | v2 | - | 0.8581 | 0.8638 | 0.8872 |
| 26527b | tf_efficientnetv2_m_in21k | 256 | true | 30 | 240 | 384 | v2 | - | 0.8646 | 0.8699 | 0.8908 |
| d269b1 | tf_efficientnetv2_m_in21k | 512 | true | 30 | 240 | 384 | v2 | - | 0.8651 | 0.8700 | 0.8917 |
| 10c041 | tf_efficientnetv2_m_in21k | 512 | true | 30 | 240 | 384 | v2 | - | 0.8646 | 0.8707 | 0.8865 |
| e3a6e3 | tf_efficientnetv2_m_in21k | 512 | true | 30 | 240 | 384 | v2 | - | 0.8607 | 0.8675 | 0.8872 |