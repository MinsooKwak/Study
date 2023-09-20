### 필요 라이브러리 설치

|1|2|3|4|5|
|:---:|:---:|:---:|:---:|:---:|
|numpy|pandas|openAI gym|matplotlib|jupyter|

- 설치 코드
  ```
  # conda 가상환경에서 다음 설치
  pip install numpy pandas gym jupyter
  ```

- pytorch (https://pytorch.org/)
  - 타 버전 사용시 위 링크 참조
    ```
    # gpu 사용하지 않을 때
    conda install pytorh cpuonly -c pytorch
    ```

- install 확인
  ```
  >>> python
  >>> import numpy
  >>> import pandas
  >>> import gym
  >>> import matplotlib.pyplot as plt
  ```
