# ta-screener (private)

개인용 암호화폐 TA 스크리너. DYOR.net 스타일의 기능을 본인 코드로 독립 구현한 것.

## 구조

- `scanner/scan.py` — 15분 스캔 스크립트 (Python)
- `data/screener.json` — 스캔 결과 (Actions가 15분마다 갱신)
- `index.html` — 정적 대시보드 (Vercel 배포용)
- `.github/workflows/scan.yml` — 15분 cron 스캔 + 결과 커밋

## 스캐너

데이터: `https://data-api.binance.vision` 공개 API (키 불필요)

- `/api/v3/exchangeInfo` → USDT 현물·TRADING 심볼 (레버리지 토큰 제외)
- 시간봉 `15m / 1h / 4h / 1d / 3d / 1w` × 심볼당 250 캔들

지표 (심볼×시간봉): RSI(14), MACD(12,26,9), ADX(14), Bollinger(20,2),
Supertrend(10,3), StochRSI(14), OBV, 거래량 비율 — `pandas`+`numpy`로 직접 구현.

종합 점수 (-100~+100):

| 지표 | 가중치 | 신호 정규화 |
|---|---|---|
| RSI | 0.20 | (50 - rsi)/50 |
| MACD | 0.20 | hist / 최근50 최대\|hist\| |
| ADX 방향 | 0.15 | sign(+DI−-DI) × min(adx,50)/50 |
| Bollinger %B | 0.15 | (0.5 − %B) × 2 |
| Supertrend | 0.15 | 종가 ≥ 선 → +1, else −1 |
| StochRSI | 0.10 | (50 − %K)/50 |
| OBV 기울기 | 0.05 | 10봉 기울기 정규화 |

MTF 가중: 15m 0.10 / 1h 0.15 / 4h 0.20 / 1d 0.25 / 3d 0.15 / 1w 0.15

실행:

```bash
pip install -r scanner/requirements.txt
python scanner/scan.py   # data/screener.json 생성
```

심볼 하나가 실패해도 전체 스캔은 계속된다. 실행 시간·API 호출량은 로그와 JSON에 기록.

## 대시보드

`index.html`을 정적 호스팅(Vercel 등)에 올리면 `data/screener.json`을 읽어
정렬·필터·검색 가능한 스캐너 테이블을 렌더링한다.

## 주의

- 점수는 참고용 스크리닝 점수이며 매매 권유가 아니다.
- Phase 2 (미구현): 차트 패턴 탐지, 자동 추세선, 다이버전스, 펌프앤덤프 탐지, Telegram 알림.
