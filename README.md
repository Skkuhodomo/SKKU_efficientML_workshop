# On-Device ResNet 모델 경량화 및 가속 실습 자료

이 저장소는 ResNet18 모델을 양자화(Quantization)와 희소화(Sparsity) 기법을 적용하여, 효율적으로 경량화하고 가속화하기 위한 실습 코드 및 관련 유틸리티를 제공합니다!

## 📁 프로젝트 파일 구성

* `resnet_sequential.py`: ResNet 모델의 각 레이어에 대해 순차적으로 압축(양자화 및 희소화)을 적용하는 핵심 로직을 포함합니다.
* `Quantizer.py`: 모델의 가중치를 양자화하는 기능을 담당하는 클래스입니다. 비트 단위 양자화를 위한 파라미터 검색 및 적용을 수행합니다.
* `CombinedCompressor.py`: 주어진 레이어에 대해 입력(`Hessian`) 정보를 수집하고, 이를 기반으로 프루닝(Pruning) 및 양자화(Quantization)를 효율적으로 수행하는 주요 압축 클래스입니다.
* `utils.py`: 모델 레이어를 찾는 `find_layers_resnet` 함수, 시드 설정(`set_seed`), 장치(`device`) 설정, CIFAR 전용 ResNet18 정의, 모델의 정확도 측정(`get_acc`) 등 유틸리티 함수들을 포함합니다.
* `requirements.txt`: 이 프로젝트를 실행하는 데 필요한 모든 Python 라이브러리 목록이 포함되어 있습니다. [cite_start](`pip install -r requirements.txt` 명령어로 설치 가능) [cite: 1]
