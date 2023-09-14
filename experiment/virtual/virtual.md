### Virtual Character

- 지시문 생성 : 구체적 지시문 생성 (Chatgpt, prompt)

- 이미지 변환 : 텍스트 -> 이미지 ([Dream.ai](https://dream.ai/) 활용)
  - 음성
  - 입모양 동작
  - 제한 : 20회

- 이미지에 대사와 스타일 지정하여 영상 생성 ([D-ID](https://d-id.com/))
  - 제한 : 계정당 20회 생성


</br>

**1. 프롬프트 작성 - 이미지 기획**

  - chatGPT
  
  - 프롬프트 엔지니어링


```
질문) 내 특징을 바탕으로 한국인 버츄얼 아바타를 만들어볼거야.

특징: 20대, 여자, 데이터 연구원, 조용한, 대담한, 목표 지향적, ENTJ, 온화한 성격

아래 요소들을 상상해서 만들어줘.
스타일과 성별, 얼굴형, 헤어스타일, 피부, 눈, 코, 입, 체격, 패션
```
```
180자 이내로, 한줄로 쉼표로 구분해서 만들어줘
```

**2. dreamai 프로프트에 작성 (기타 추가 요소 반영)**
   - 타입 지정, create
   - edit with text
   - finalize => 이미지 저장

  
**3. 이미지로 말하는 동영상 생성**
   - d-id 좌측 create video
   - add > 이미지 파일 등록
   - 우측 > script 작성 / language 설정 / voice 설정
   - Generate

**4. 참고 사항 (프롬프트)**
   - Eleven Labs, MidJourney, Bing AI
     
