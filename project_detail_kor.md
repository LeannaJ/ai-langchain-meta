# Meta Project - 멀티플랫폼 트렌드 분석 시스템

## 📋 프로젝트 개요

**Meta Project**는 YouTube, Reddit, TikTok, Twitter(X), Google Trends 등 다양한 소셜 미디어 플랫폼에서 트렌딩 콘텐츠를 수집하고 분석하여 통합적인 트렌드 스코어를 계산하는 종합적인 시스템입니다. LangChain과 LDA 토픽 모델링을 활용하여 다중 플랫폼 데이터를 처리하고, AI 기반 분류 및 가중치 시스템을 통해 정확한 트렌드 분석을 제공합니다.

## 🎯 프로젝트 목표

1. **다중 플랫폼 데이터 통합**: 5개 주요 소셜 미디어 플랫폼의 트렌딩 데이터를 수집
2. **AI 기반 토픽 분석**: LDA 토픽 모델링과 LLM을 활용한 의미적 토픽 그룹화
3. **크로스 플랫폼 트렌드 스코어링**: 플랫폼별 가중치와 크로스 플랫폼 보너스를 적용한 통합 스코어 계산
4. **자동화된 워크플로우**: GitHub Actions를 통한 정기적인 데이터 수집 및 분석 자동화
5. **미디어 타입별 최적화**: 비디오, 텍스트, 이미지 콘텐츠에 따른 차별화된 분석

## 🏗️ 시스템 아키텍처

### 전체 워크플로우 구조

```
[데이터 수집] → [전처리] → [토픽 모델링] → [통합] → [정규화] → [그룹화] → [분류] → [스코어링] → [출력]
```

### 주요 컴포넌트

1. **데이터 수집 에이전트들**
   - YouTube Agent: YouTube Trending API
   - Reddit Agent: Reddit API (r/all/hot)
   - TikTok Agent: Playwright 기반 웹 스크래핑
   - Twitter Agent: Trends24.in 스크래핑
   - Google Trends Agent: BigQuery 기반 데이터 수집

2. **LangChain 워크플로우**
   - LDA 토픽 모델링
   - LLM 기반 토픽 라벨링
   - 유사 토픽 클러스터링
   - 카테고리 분류

3. **스코어링 시스템**
   - 플랫폼별 가중치 적용
   - 크로스 플랫폼 보너스 계산
   - 미디어 타입별 최적화

## 📊 데이터 수집 시스템

### 1. YouTube Agent (`youtube_agent.py`)

**기능:**
- YouTube Data API v3를 활용한 트렌딩 비디오 수집
- 8개 지역(US, IN, GB, CA, DE, JP, BR, AU)에서 데이터 수집
- 각 비디오당 상세 메트릭 계산

**수집 데이터:**
- 기본 정보: 제목, 채널명, 업로드 시간, 카테고리
- 참여도 지표: 조회수, 좋아요 수, 댓글 수
- 파생 지표: 시간당 조회수, 좋아요 대비 조회수 비율
- 채널 정보: 구독자 수, 채널 생성일

**메트릭 계산:**
```python
View Velocity = View Count / Video Age (Hours)
Like Velocity = Like Count / Video Age (Hours)
Comment Velocity = Comment Count / Video Age (Hours)
Like-to-View Ratio = (Like Count / View Count) * 100
```

### 2. Reddit Agent (`reddit_agent.py`)

**기능:**
- Reddit API OAuth 인증을 통한 r/all/hot 게시물 수집
- 1,000개 게시물 수집 목표
- 페이지네이션을 통한 효율적인 데이터 수집

**수집 데이터:**
- 게시물 기본 정보: 제목, 내용, 작성자, 서브레딧
- 참여도 지표: 업보트, 댓글 수, 조회수
- 메타데이터: 생성 시간, 스코어, URL

### 3. TikTok Agent (`TikTok Script with LLM support.py`)

**기능:**
- Gemini AI를 활용한 트렌딩 해시태그 생성
- Playwright를 통한 웹 스크래핑
- 해시태그별 인기 비디오 URL 수집

**AI 기반 해시태그 생성:**
```python
PROMPT_TEMPLATE = """
You are a social media trend analyst. Output 3–5 trending TikTok hashtags as JSON.
Format: {"hashtags": [{"tag": "hashtag1", "popularity_score": 0-100}]}
"""
```

### 4. Twitter Agent (`twitter_agent.py`)

**기능:**
- Trends24.in 웹사이트 스크래핑
- Playwright를 통한 동적 콘텐츠 수집
- 다국가 트렌드 데이터 수집

**수집 데이터:**
- 트렌드 키워드
- 트윗 수
- 트렌드 지속 시간

### 5. Google Trends Agent (`trend_bot.py`)

**기능:**
- Google BigQuery를 통한 Google Trends 데이터 수집
- 최근 7일간의 데이터 분석
- 상승 트렌드와 인기 트렌드 분리 수집

**수집 데이터:**
- 상승 트렌드: term, dma_hits, coverage_ratio, median_gain, spread_intensity_score
- 인기 트렌드: term, dma_hits, avg_rank, total_score, coverage_ratio

## 🔄 LangChain 워크플로우 (`final_langchain_workflow.py`)

### Step 1: 데이터 로딩
```python
def fetch_youtube_popular():
    latest = get_latest_csv("Scraped_Data/youtube_trending_analysis_*.csv")
    return pd.read_csv(latest)
```

### Step 2: 데이터 전처리
- 플랫폼별 데이터 타입 정규화
- 결측값 처리
- 메트릭 정규화

### Step 3: LDA 토픽 모델링

**전처리:**
```python
def preprocess_text(df, text_cols):
    # NLTK 기반 텍스트 전처리
    # - 소문자 변환, 특수문자 제거
    # - 불용어 제거, 어간 추출
    # - 의미 없는 단어 필터링
```

**토픽 모델링:**
```python
def topic_modeling(texts, num_topics=200):
    # Gensim LDA 모델 훈련
    # - 200개 토픽 추출
    # - 각 문서별 토픽 할당
    # - 토픽별 키워드 추출
```

**LLM 기반 라벨링:**
```python
def label_topics_with_llm(topic_keywords_str, topic_counts):
    # OpenAI GPT를 활용한 자연스러운 토픽 라벨 생성
    # - 키워드 기반 의미적 라벨링
    # - 일관성 있는 네이밍
```

### Step 4: 플랫폼별 메트릭 추출
```python
def extract_platform_metrics(df, platform):
    # 플랫폼별 표준화된 메트릭 추출
    # - YouTube: topic, video_count, total_engagement
    # - Reddit: topic, doc_count, total_engagement
    # - TikTok: hashtag, views
    # - Twitter: trend, duration, tweet_count
    # - Google Trends: term, median_gain
```

### Step 5: MinMax 정규화
```python
def apply_minmax_scaling(df):
    # 플랫폼별 0-1 정규화
    # frequency_norm, engagement_norm 계산
```

### Step 6: 유사 토픽 클러스터링

**1단계: 유사도 기반 클러스터링**
```python
def calculate_keyword_similarity(keywords1, keywords2):
    # Jaccard 유사도 + 문자열 유사도 결합
    # 임계값 0.6 이상 시 그룹화
```

**2단계: LLM 기반 의미적 그룹화**
```python
def llm_enhanced_grouping(initial_groups, topics_list):
    # OpenAI GPT를 활용한 의미적 토픽 그룹화
    # - 초기 그룹 검토 및 개선
    # - 의미적으로 유사한 토픽 병합
```

### Step 7: 카테고리 분류
```python
def classify_topic_category(topic):
    # 8개 카테고리로 분류:
    # - Beauty & Fashion
    # - Technology & Innovation
    # - Lifestyle & Health
    # - News & Politics
    # - Sports & Fitness
    # - Education & Learning
    # - Business & Finance
    # - Entertainment & Media
```

### Step 8: 트렌드 스코어 계산

**플랫폼 가중치 시스템:**
```python
weights = {
    'equal': {'Reddit': 0.2, 'X/Twitter': 0.2, 'YouTube': 0.2, 'TikTok': 0.2, 'Google Trends': 0.2},
    'video': {'YouTube': 0.5, 'TikTok': 0.4, 'Reddit': 0.05, 'X/Twitter': 0.05, 'Google Trends': 0.0},
    'text': {'Reddit': 0.4, 'X/Twitter': 0.4, 'YouTube': 0.1, 'TikTok': 0.1, 'Google Trends': 0.0},
    'image': {'TikTok': 0.4, 'Reddit': 0.3, 'X/Twitter': 0.3, 'YouTube': 0.0, 'Google Trends': 0.0}
}
```

**스코어 계산 공식:**
```python
# 플랫폼별 스코어: S_p = w_p * (α·F_p + β·E_p)
# 크로스 플랫폼 보너스: λ * (platform_count - 1)
# 최종 스코어: Final Score = S_p + Cross-platform Bonus
```

## 🤖 AI/ML 기술 스택

### 자연어 처리 (NLP)
- **NLTK**: 텍스트 전처리, 불용어 제거, 어간 추출
- **Gensim**: LDA 토픽 모델링, 문서-단어 매트릭스 생성
- **LangChain**: LLM 체인 구성 및 관리

### 머신러닝
- **LDA (Latent Dirichlet Allocation)**: 토픽 모델링
- **Jaccard 유사도**: 키워드 유사도 계산
- **SequenceMatcher**: 문자열 유사도 계산
- **MinMax 정규화**: 메트릭 표준화

### AI 모델
- **OpenAI GPT**: 토픽 라벨링, 의미적 그룹화, 카테고리 분류
- **Google Gemini**: TikTok 해시태그 생성

## 📈 수학적 모델링

### 1. 토픽 모델링 (LDA)
```
P(topic|document) = P(document|topic) * P(topic) / P(document)
```

### 2. 유사도 계산
```
Jaccard Similarity = |A ∩ B| / |A ∪ B|
String Similarity = SequenceMatcher(a, b).ratio()
Combined Similarity = 0.6 * Jaccard + 0.4 * String
```

### 3. 트렌드 스코어링
```
S_p = w_p * (α·F_p + β·E_p)
Cross-platform Bonus = λ * (platform_count - 1)
Final Score = S_p + Cross-platform Bonus
```

**매개변수:**
- α = 0.5 (빈도 가중치)
- β = 0.5 (참여도 가중치)
- λ = 0.1 (크로스 플랫폼 보너스 계수)

## 🔧 기술적 구현 세부사항

### 데이터 파이프라인
1. **GitHub Actions 워크플로우**: 각 플랫폼별 자동 데이터 수집
2. **CSV 아티팩트 관리**: 최신 데이터 파일 자동 다운로드
3. **에러 핸들링**: 플랫폼별 장애 격리 및 재시도 로직
4. **데이터 검증**: 수집된 데이터의 품질 및 완성도 검증

### 성능 최적화
- **배치 처리**: LLM API 호출 최적화 (배치 크기: 10)
- **캐싱**: 채널 정보 및 토픽 모델 캐싱
- **병렬 처리**: 플랫폼별 독립적 데이터 처리
- **메모리 효율성**: 대용량 데이터셋 처리 최적화

### 확장성
- **모듈화**: 플랫폼별 독립적 에이전트 구조
- **설정 기반**: 환경 변수를 통한 유연한 설정
- **API 추상화**: 플랫폼별 API 차이점 추상화

## 📊 결과 및 성과

### 출력 파일 구조
1. **detailed_data_with_grouping_{timestamp}.csv**
   - 원본 데이터 + 그룹화 정보
   - 토픽별 클러스터링 결과
   - 플랫폼별 메트릭

2. **consolidated_scores_w_crossbonus_{media_type}_{timestamp}.csv**
   - 최종 트렌드 스코어
   - 미디어 타입별 가중치 적용
   - 크로스 플랫폼 보너스 포함

### 데이터 규모 (2025-07-30 기준)
- **총 레코드 수**: 1,012개 토픽
- **플랫폼별 데이터**: YouTube, Reddit, TikTok, Twitter, Google Trends
- **카테고리 분포**: 8개 주요 카테고리
- **크로스 플랫폼 토픽**: 다중 플랫폼에서 발견된 토픽들

### 성능 지표
- **처리 시간**: 약 10-15분 (전체 워크플로우)
- **정확도**: LLM 기반 분류로 높은 의미적 정확도
- **확장성**: 새로운 플랫폼 추가 용이

## 💪 강점 (Strengths)

### 1. 종합적인 데이터 수집
- **다중 플랫폼**: 5개 주요 소셜 미디어 플랫폼 통합
- **실시간성**: GitHub Actions를 통한 정기적 업데이트
- **다양성**: 비디오, 텍스트, 이미지 등 다양한 콘텐츠 타입

### 2. 고급 AI/ML 기술
- **LDA 토픽 모델링**: 의미적 토픽 추출
- **LLM 통합**: 자연스러운 토픽 라벨링 및 분류
- **유사도 분석**: 다차원적 유사도 계산

### 3. 정교한 스코어링 시스템
- **플랫폼별 가중치**: 미디어 타입에 따른 차별화
- **크로스 플랫폼 보너스**: 다중 플랫폼 트렌드 강화
- **정규화**: 플랫폼간 공정한 비교

### 4. 자동화 및 확장성
- **CI/CD 파이프라인**: GitHub Actions 자동화
- **모듈화**: 플랫폼별 독립적 개발 및 유지보수
- **설정 기반**: 환경 변수를 통한 유연한 설정

## ⚠️ 약점 (Weaknesses)

### 1. API 의존성
- **API 제한**: YouTube, Reddit API 할당량 제한
- **비용**: OpenAI API 사용료 발생
- **안정성**: 외부 API 장애 시 전체 시스템 영향

### 2. 데이터 품질 이슈
- **스크래핑 불안정성**: TikTok, Twitter 웹 스크래핑의 취약성
- **데이터 불균형**: 플랫폼별 데이터 양 차이
- **노이즈**: 스팸, 봇 활동으로 인한 데이터 오염

### 3. 성능 한계
- **처리 시간**: LLM API 호출로 인한 지연
- **메모리 사용량**: 대용량 데이터 처리 시 메모리 부족
- **확장성**: 단일 서버 환경의 처리 한계

### 4. 정확도 문제
- **언어 제한**: 영어 중심의 분석
- **문화적 맥락**: 지역별 문화적 차이 미반영
- **시계열 분석 부족**: 트렌드 변화 패턴 분석 미흡

## 🚀 Next Steps (향후 발전 방향)

### 1. 기술적 개선

#### AI/ML 고도화
- **BERT 기반 임베딩**: 더 정확한 의미적 유사도 계산
- **시계열 분석**: 트렌드 변화 패턴 및 예측 모델
- **감정 분석**: 트렌드의 감정적 톤 분석
- **멀티모달 분석**: 이미지, 비디오 콘텐츠 분석

#### 인프라 개선
- **클라우드 마이그레이션**: AWS/GCP 기반 확장 가능한 인프라
- **데이터베이스 도입**: PostgreSQL/MongoDB를 통한 데이터 관리
- **캐싱 시스템**: Redis를 통한 성능 최적화
- **모니터링**: Prometheus/Grafana 기반 시스템 모니터링

### 2. 기능 확장

#### 플랫폼 확장
- **Instagram**: 인스타그램 트렌드 수집
- **LinkedIn**: 비즈니스 트렌드 분석
- **Twitch**: 게임 스트리밍 트렌드
- **Discord**: 커뮤니티 트렌드

#### 분석 기능
- **실시간 대시보드**: Streamlit/FastAPI 기반 웹 인터페이스
- **알림 시스템**: 중요 트렌드 발생 시 알림
- **API 서비스**: 외부 서비스 연동을 위한 REST API
- **데이터 시각화**: Plotly/D3.js 기반 인터랙티브 차트

### 3. 비즈니스 적용

#### 마케팅 도구
- **콘텐츠 제작 가이드**: 트렌드 기반 콘텐츠 제안
- **타겟팅 분석**: 타겟 오디언스 트렌드 분석
- **경쟁사 모니터링**: 경쟁사 콘텐츠 트렌드 추적

#### 연구 도구
- **학술 연구**: 소셜 미디어 트렌드 연구 지원
- **정책 분석**: 공공 정책 관련 트렌드 분석
- **문화 연구**: 문화적 현상 및 트렌드 연구

### 4. 국제화 및 현지화

#### 다국어 지원
- **언어별 분석**: 한국어, 일본어, 중국어 등 다국어 지원
- **문화적 맥락**: 지역별 문화적 특성 반영
- **현지 플랫폼**: 네이버, 카카오톡 등 현지 플랫폼 통합

#### 지역별 최적화
- **시간대별 분석**: 지역별 시간대 고려
- **이벤트 기반 분석**: 지역별 주요 이벤트 연관 분석
- **트렌드 예측**: 지역별 트렌드 예측 모델

## 📚 기술 문서 및 참고 자료

### 주요 라이브러리
- **LangChain**: LLM 워크플로우 관리
- **Gensim**: 토픽 모델링
- **NLTK**: 자연어 처리
- **Pandas**: 데이터 처리
- **Playwright**: 웹 스크래핑
- **OpenAI**: GPT 모델 활용
- **Google BigQuery**: 대용량 데이터 분석

### API 문서
- **YouTube Data API v3**: https://developers.google.com/youtube/v3
- **Reddit API**: https://www.reddit.com/dev/api/
- **OpenAI API**: https://platform.openai.com/docs
- **Google Trends BigQuery**: https://console.cloud.google.com/bigquery

### 연구 논문
- **LDA 토픽 모델링**: Blei, D. M., et al. (2003). Latent dirichlet allocation
- **크로스 플랫폼 분석**: 다양한 소셜 미디어 플랫폼 통합 분석 방법론
- **트렌드 예측**: 시계열 데이터 기반 트렌드 예측 모델

## 🏆 프로젝트 성과 요약

Meta Project는 다중 소셜 미디어 플랫폼의 트렌드 데이터를 통합 분석하는 혁신적인 시스템을 구축했습니다. LDA 토픽 모델링과 LLM을 결합한 고급 AI 기술을 활용하여, 단순한 키워드 기반 분석을 넘어서는 의미적 트렌드 분석을 제공합니다.

### 핵심 성과
1. **5개 플랫폼 통합**: YouTube, Reddit, TikTok, Twitter, Google Trends
2. **AI 기반 토픽 분석**: 200개 토픽 추출 및 의미적 그룹화
3. **정교한 스코어링**: 플랫폼별 가중치 및 크로스 플랫폼 보너스
4. **완전 자동화**: GitHub Actions 기반 24/7 데이터 수집 및 분석
5. **확장 가능한 아키텍처**: 새로운 플랫폼 및 기능 추가 용이

이 프로젝트는 소셜 미디어 트렌드 분석의 새로운 패러다임을 제시하며, 마케팅, 연구, 정책 분석 등 다양한 분야에서 활용 가능한 강력한 도구로 발전할 잠재력을 가지고 있습니다.
