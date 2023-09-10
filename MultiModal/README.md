# Multi Modal
"modality"란 무언가가 표현되거나 인식되는 방식으로, 어떻게 해석하고 인식하는지를 제공한다. </br></br>


### 인간의 원시 센서 (원시 데이터)
- 내부의 여러 data source를 fuse하는 등의 과정이 필요

- 원시 모달(ex. image, test, audio) => **추상 모달**을 구성 </br></br>


### McGurk Effect 
: 하나의 모달이 다른 모달에 우세한 경우에 대한 실험과 결과 (Multimodal, 2011)

- 음성과 영상을 두고 영상은 동일하나 소리가 다른 경우를 실험
- 보는 것이 듣는 것에 우선한다는 결과

### Core Challenge
**1. Representation learning** : Fuse해서 새로운 Representation 찾는 것 </br>
  ```
  [표현 결합하는 방법]

  1) Join

  2) Fussion

  3) Coordination :

      - 따로 representation하고 유사 특징을 공유하는 공간에서 coordination

      - 각 representation이 도움되는 경우 조정해 함께 작업

  4) Fission (분열) :

        - 통합된 representation이 아닌 여러개의 결합 representation을 만드는 것

        - 두 모달리티의 결합 feature를 추출해 문제 해결시 유용한 방식
  ```
        
**2. alignment**
   - 하나의 요소가(ex.프레임) 둘 이상의 text로 representation 될 때
   - 각 모달리티가 어떻게 representation 될 수 있는지 알고 link 만드는 것이 중요
   - 잘못된 alignment 가질 수 있어서 인간의 개입 필요할 때가 있음

  ```
  [alignment의 주요 2가지 접근 방식]

  1) explicit alignment (명시적 alignment) : 어떤 frame이 어떤 text 갖고 있는지 명시적으로 알 때

  2) implicit alignment (암시적 alignment) : 모달리티 내부 alignment labeling
  ```

**3. Translation** : 한 모달리티 -> 다른 모달리티

**4. Fusion** : 멀티 모달에서 데이터(정보)를 결합해 문제를 해결하고 예측 수행하는 표현 방법

**5. co-learning** </br>
  : Poor modality와 Rich modality가 있을 것인데, Rich modality 통해 poor modality 돕거나 모델링함
