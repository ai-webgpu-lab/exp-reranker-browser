# Results

## 1. 실험 요약
- 저장소: exp-reranker-browser
- 커밋 해시: 194dc02
- 실험 일시: 2026-05-20T15:45:45.231Z -> 2026-05-20T15:45:46.751Z
- 담당자: ai-webgpu-lab
- 실험 유형: `ml`
- 상태: `success`

## 2. 질문
- 고정 candidate set에서 reranker latency와 top-k quality를 단일 보고 흐름으로 남길 수 있는가
- backend/fallback metadata와 scoring p50/p95가 같은 raw result에 남는가
- 실제 reranker runtime 연결 전 deterministic fixture로 비교 프로토콜을 고정할 수 있는가

## 3. 실행 환경
### 브라우저
- 이름: Chrome
- 버전: 147.0.7727.15

### 운영체제
- OS: Linux
- 버전: unknown

### 디바이스
- 장치명: Linux x86_64
- device class: `desktop-high`
- CPU: 16 threads
- 메모리: 32 GB
- 전원 상태: `unknown`

### GPU / 실행 모드
- adapter: navigator.gpu available
- backend: `webgpu`
- fallback triggered: `false`
- worker mode: `main`
- cache state: `warm`
- required features: ["shader-f16"]
- limits snapshot: {}

## 4. 워크로드 정의
- 시나리오 이름: Browser Reranker
- 입력 프로필: 6-candidates-15-query-tokens
- 데이터 크기: candidateCount=6; queryTokens=15; bestRelevantRank=1; backend=webgpu; automation=playwright-chromium, candidateCount=6; queryTokens=15; bestRelevantRank=1; backend=webgpu; realAdapter=fallback(adapter.loadModel is not a function); automation=playwright-chromium
- dataset: reranker-fixture-v1
- model_id 또는 renderer: browser-reranker-fixture-v1
- 양자화/정밀도: -
- resolution: -
- context_tokens: -
- output_tokens: -

## 5. 측정 지표
### 공통
- time_to_interactive_ms: 139.5 ~ 883.2 ms
- init_ms: 18.4 ~ 20 ms
- success_rate: 1
- peak_memory_note: 32 GB reported by browser
- error_type: -

### Embeddings / ML
- docs_per_sec: 300 ~ 326.09
- queries_per_sec: 50 ~ 54.35
- p95_ms: 0.1 ms
- recall_at_10: 1
- index_build_ms: 18.4 ~ 20 ms
- backends: webgpu
- fallback states: false

## 6. 결과 표
| Run | Scenario | Backend | Cache | Mean | P95 | Notes |
|---|---|---:|---:|---:|---:|---|
| 1 | Browser Reranker | webgpu | warm | 300 | 0.1 | queries/s=50, recall/top-k=1, metric=docs/s |
| 2 | Browser Reranker | webgpu | warm | 326.09 | 0.1 | queries/s=54.35, recall/top-k=1, metric=docs/s |

## 7. 관찰
- browser reranker baseline은 backend=webgpu, fallback_triggered=false로 기록됐다.
- reranker summary는 candidates/sec=300, p95_ms=0.1, top-k hit=1였다.
- playwright-chromium로 수집된 automation baseline이며 headless=true, browser=Chromium 147.0.7727.15.
- 실제 runtime/model/renderer 교체 전 deterministic harness 결과이므로, 절대 성능보다 보고 경로와 재현성 확인에 우선 의미가 있다.

## 8. Real Adapter vs Deterministic
- adapter: real=reranker-xenova-bge-reranker-base-300, deterministic=deterministic-mock
- adapter_run: real=connected, deterministic=deterministic
- success_rate: real=1, deterministic=1

## 9. 결론
- browser reranker readiness가 candidate scoring latency와 top-k quality를 raw JSON과 RESULTS.md 양쪽에 남기게 됐다.
- 다음 단계는 deterministic scorer를 실제 reranker runtime으로 교체하되 candidate set과 output ranking contract를 유지하는 것이다.
- 이후 `bench-reranker-latency`와 `bench-rag-endtoend`의 입력 baseline으로 재사용할 수 있다.

## 10. 첨부
- 스크린샷: ./reports/screenshots/01-browser-reranker.png, ./reports/screenshots/10-browser-reranker-real-reranker.png
- 로그 파일: ./reports/logs/01-browser-reranker.log, ./reports/logs/10-browser-reranker-real-reranker.log
- raw json: ./reports/raw/01-browser-reranker.json, ./reports/raw/10-browser-reranker-real-reranker.json
- 배포 URL: https://ai-webgpu-lab.github.io/exp-reranker-browser/
- 관련 이슈/PR: -
