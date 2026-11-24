"""
Mini NPU 시뮬레이터 (MAC 연산) - main.py 뼈대

이 파일은 "빈 칸 채우기" 형태의 뼈대입니다.
각 함수 안의 주석을 읽고, 직접 코드를 채워 넣으세요.
(지금은 모든 함수가 `pass`라서 실행해도 아무 일도 일어나지 않습니다.)

[추천 개발 순서] - 아래 순서대로 하나씩 만들고, 만들 때마다 직접 실행해서 확인하세요!
  1단계완: mac()                       -> 핵심 계산. 3x3 예시로 5.0, 1.0이 나오는지 확인
  2단계완: judge()                     -> 동점(epsilon) 처리
  3단계완: parse_row(), read_matrix()  -> 모드 1 입력
  4단계완: measure_ms()                -> 시간 측정
  5단계완: run_mode1()                 -> 모드 1 완성 #todo 5) 3x3 성능 분석을 출력한다. (benchmark([3]) 와 print_perf_table() 재사용)
  6단계완: normalize_label(), extract_size(), load_json(), load_filters()
  7단계완: validate_case(), run_case() -> 모드 2의 핵심
  8단계완: benchmark(), print_perf_table(), summarize(), run_mode2()
  9단계: select_mode(), main()       -> 전체 연결
  10단계(선택): 보너스 함수들

[공부할 때 도움이 되는 곳]
  - 파이썬 공식 문서 (docs.python.org/ko/3) : 내장 함수, json, time 모듈
  - 위키독스 "점프 투 파이썬" 같은 입문서 : 반복문, 함수, 예외 처리, 딕셔너리
  - 각 함수 주석의 [검색 키워드]를 그대로 검색창에 넣어 보세요.

[주의] numpy, pandas 같은 외부 라이브러리는 사용 금지입니다.
       표준 라이브러리(json, time, dataclasses, typing)만 사용합니다.
"""

import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# 자료형 이름표 (타입 별칭)
# ---------------------------------------------------------------------------
# 타입 별칭은 "긴 자료형에 짧은 이름을 붙인 것"입니다. 실행에는 영향이 없고,
# 사람이 코드를 읽을 때 "이 함수가 어떤 값을 주고받는지" 알기 쉽게 해 줍니다.
# [검색 키워드] python typing List Dict Tuple 타입 힌트, python type alias

Matrix = List[List[float]]          # N x N 2차원 배열. 예: [[0,1,0],[1,1,1],[0,1,0]]
Flat = List[float]                  # 보너스 ①: 길이 N*N 의 1차원 배열
Label = str                         # 표준 라벨: "Cross" 또는 "X"
Filters = Dict[int, Dict[str, Matrix]]
# Filters 예시: {5: {"Cross": [[...]], "X": [[...]]}, 13: {...}, 25: {...}}
#   - 바깥 키 5, 13, 25 는 크기(N)
#   - 안쪽 키 "Cross", "X" 는 표준 라벨 (JSON 의 "cross"/"x" 가 아님!)

# ---------------------------------------------------------------------------
# 상수 (프로그램 전체에서 쓰는 고정값)
# ---------------------------------------------------------------------------
# 숫자나 문자열을 코드 여기저기에 직접 쓰면 오타가 나기 쉽습니다.
# 한 곳에 이름을 붙여 두고 그 이름을 쓰세요.

EPS = 1e-9               # 동점 판정 허용오차 (|a-b| < EPS 이면 같은 점수로 취급)
REPEAT = 10              # 성능 측정 반복 횟수 (최소 10회)
DATA_PATH = "data.json"  # main.py 와 같은 폴더에 둔다

LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"


# ---------------------------------------------------------------------------
# 결과 클래스
# ---------------------------------------------------------------------------
@dataclass
class CaseResult:
    """data.json 의 패턴 1건을 판정한 결과를 담는 '상자'.

    dict 대신 클래스를 쓰면 result.verdict 처럼 점(.)으로 꺼낼 수 있고,
    필드 이름 오타를 빨리 발견할 수 있습니다.

    [검색 키워드] python dataclass 사용법
    """
    key: str                 # 케이스 이름. 예: "size_13_1"
    n: int                   # 패턴 크기. 예: 13 (크기를 알아내지 못했다면 0)
    cross_score: float       # Cross 필터와의 MAC 점수
    x_score: float           # X 필터와의 MAC 점수
    verdict: str             # "Cross" / "X" / "UNDECIDED"
    expected: str            # 정규화된 정답 라벨 ("Cross" 또는 "X")
    passed: bool             # verdict == expected 이면 True
    reason: str = ""         # FAIL 이유 (PASS 이면 빈 문자열)


# ===========================================================================
# 1. 핵심 계산
# ===========================================================================
def mac(pattern: Matrix, filt: Matrix) -> float:
    """패턴과 필터를 같은 위치끼리 곱해서 모두 더한 값(MAC 점수)을 돌려준다.

    예) 십자가 패턴과 십자가 필터(3x3)의 점수는 5.0 이어야 한다.
        십자가 패턴과 X 필터의 점수는 1.0 이어야 한다.
    """


    total=0.0
    for i in range(len(pattern)):
        for j in range(len(pattern[i])):
            total = total + pattern[i][j]*filt[i][j]
    return total

    # TODO 1) 점수를 담을 변수를 0.0 으로 만든다.   예: total = 0.0
    # TODO 2) 행(i)을 하나씩 돌고, 그 안에서 열(j)을 하나씩 도는 "이중 for문"을 쓴다.
    #         pattern[i][j] 와 filt[i][j] 는 "같은 위치"의 값이다.
    # TODO 3) 두 값을 곱해서 total 에 계속 더한다.  (Multiply -> Accumulate)
    # TODO 4) 반복이 끝나면 total 을 return 한다.
    #
    # 주의: 요구사항은 "반복문으로 직접 구현"을 요구합니다. numpy 는 물론 금지이고,
    #       sum(), zip() 등으로 한 줄에 끝내는 방식은 '직접 구현'으로 인정되는지 애매하니
    #       for문(곱하기 + 더하기)으로 풀어서 쓰세요.
    #
    # [검색 키워드] python 이중 for문, python 2차원 리스트 인덱싱, python 변수에 값 누적하기



def judge(score_a: float, score_b: float,
          label_a: Label = "A", label_b: Label = "B",
          eps: float = EPS) -> str:
    """두 점수를 비교해 더 높은 쪽의 라벨을 돌려준다. 거의 같으면 "UNDECIDED".

    모드 1 에서는 judge(a, b)                  -> "A" / "B" / "UNDECIDED"
    모드 2 에서는 judge(c, x, "Cross", "X")    -> "Cross" / "X" / "UNDECIDED"
    """
    score_diff = abs(score_a - score_b)
    if score_diff < eps:
        return LABEL_UNDECIDED
    elif score_a > score_b:
        return label_a
    else:
        return label_b


    # TODO 1) abs(score_a - score_b) < eps 이면 LABEL_UNDECIDED 를 return 한다.
    #         (부동소수점은 0.1 + 0.2 == 0.3 이 False 인 것처럼 아주 작은 오차가 생기므로
    #          "==" 로 비교하면 안 되고, 허용오차(epsilon)로 비교해야 한다.)
    # TODO 2) 그렇지 않으면 score_a > score_b 일 때 label_a, 아니면 label_b 를 return 한다.
    #
    # [검색 키워드] python 부동소수점 오차 0.1+0.2, python float 비교 epsilon, python abs 함수



# ===========================================================================
# 2. 모드 1: 사용자 입력 (3x3)
# ===========================================================================
def parse_row(line: str, n: int) -> List[float]:
    """공백으로 구분된 한 줄 문자열을 숫자 리스트로 바꾼다.

    예) parse_row("0 1 0", 3) -> [0.0, 1.0, 0.0]
    숫자가 아니거나 개수가 n 개가 아니면 ValueError 를 일으킨다.
    """

    rows = line.split()
    if  not rows:
        raise ValueError(f"입력 형식 오류: 숫자를 0개 입력했습니다. {n}개를 입력해 주세요.")
    elif len(rows) < n:
        raise ValueError(f"입력 형식 오류: 숫자를 {len(rows)}개만 입력했습니다. {n}개를 입력해 주세요.")  
    elif len(rows) > n:
        raise ValueError(f"입력 형식 오류: 숫자를 {len(rows)}개 입력했습니다. {n}개만 입력해 주세요.")
    else:
        try:
            return [float(x) for x in rows]
        except ValueError:
            raise ValueError(f"입력 형식 오류: 숫자가 아닌 값이 있습니다. 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요.")
    # 오류 종류를 나눠서, 사용자가 무엇을 고쳐야 하는지 알려 주는 한국어 메시지를 담아 ValueError 를 일으킵니다.
    #   1) 숫자가 n 개보다 적을 때 : "입력 형식 오류: 숫자를 2개만 입력했습니다. 3개를 입력해 주세요."
    #   2) 숫자가 n 개보다 많을 때 : "입력 형식 오류: 숫자를 4개 입력했습니다. 3개만 입력해 주세요."
    #   3) 숫자가 아닌 값이 있을 때 : "입력 형식 오류: 숫자가 아닌 값이 있습니다. 각 줄에 3개의 숫자를
    #                                  공백으로 구분해 입력하세요."
    #   (문구 끝에 문서 예시 문구를 덧붙이면 요구사항의 예시와도 맞습니다. 3 은 n 값으로 바꿔 쓰세요.)
    # TODO 1) line.split() 으로 공백 기준으로 쪼갠다. (결과는 문자열 리스트)
    # TODO 2) 조각의 개수를 len() 으로 센다.
    #         n 보다 적으면 위 1) 메시지로, n 보다 많으면 위 2) 메시지로 `raise ValueError(메시지)` 한다.
    #         (비어 있는 입력은 조각이 0개이므로 "적게 입력" 경우에 함께 들어간다)
    # TODO 3) 각 조각을 float() 로 숫자로 바꾼다.
    #         "abc" 같은 값은 float() 가 ValueError 를 일으키는데, 기본 메시지가 영어라서
    #         try/except ValueError 로 잡은 뒤 위 3) 한국어 메시지로 다시 raise ValueError(...) 한다.
    # TODO 4) 숫자 리스트를 return 한다.
    #
    # [검색 키워드] python split 문자열 나누기, python len 개수 세기, python float 변환 ValueError,
    #               python raise 예외 발생시키기, python except as e 예외 메시지 출력


def read_matrix(title: str, n: int = 3) -> Matrix:
    """콘솔에서 n 줄(= n 개의 행)을 한 줄씩 입력받아 n x n 행렬을 돌려준다. 틀린 행은 그 행만 다시 입력받는다.

    title 은 안내 문구에 쓴다. 예) "필터 A", "필터 B", "패턴"
    """

    matrix = []
    for r in range(n):                 # 행 번호 0 .. n-1
        while True:                    # 이 행이 맞게 입력될 때까지 반복
            line = input(f"{title} {r+1}행 입력 (숫자 {n}개, 공백 구분): ")
            try:
                row = parse_row(line,n)              # parse_row 를 호출해 숫자 리스트를 얻는다
                break                  # 성공하면 while 만 빠져나온다
            except ValueError as e:
                print(e)                 # 메시지를 출력한다 (같은 행을 다시 묻게 됨)
        matrix.append(row)             # 맞게 입력된 행을 추가
    return matrix
    # 입력 방식: 한 줄이 '1개 행'입니다. 행마다 번호를 붙여 안내하고, 틀린 행만 다시 입력받습니다.
    #   예) 필터 A 1행 입력 (숫자 3개, 공백 구분): 0 1 0
    #       필터 A 2행 입력 (숫자 3개, 공백 구분): 1 1 1
    #       필터 A 3행 입력 (숫자 3개, 공백 구분): 0 1 0
    # TODO 1) 결과를 담을 빈 리스트 matrix 를 만든다.
    # TODO 2) 행 번호 r 을 0 부터 n-1 까지 도는 for문을 쓴다. (몇 번째 행인지 알려 주기 위해)
    # TODO 3) for문 안에서 "while True:" 로 무한 반복을 만든다.
    #         (이 행이 올바르게 입력될 때까지만 다시 묻기 위해)
    # TODO 4) while 안에서 input() 의 안내 문구에 title 과 행 번호(r + 1)를 넣어 한 줄을 받는다.
    #         예) f"{title} {r + 1}행 입력 (숫자 {n}개, 공백 구분): "
    # TODO 5) try/except ValueError 로 parse_row(line, n) 을 호출한다.
    #         - 성공하면 얻은 행을 matrix 에 append 하고, break 로 while 만 빠져나온다.
    #         - ValueError 가 나면 `except ValueError as e:` 로 받아 print(e) 로 메시지를 출력하고,
    #           같은 행을 다시 입력받는다. (메시지는 parse_row 가 상황에 맞게 담아 준다:
    #           적게 입력 / 많이 입력 / 숫자 아님. while 의 처음으로 돌아가려면 continue 를 쓴다)
    # TODO 6) for문이 끝나면 n 개의 행이 모인 것이다. matrix 를 return 한다.
    #
    # (생각해 볼 것) 이 구조에서는 행 수가 항상 n 개로 고정되고, 열 수/숫자 오류는 행마다 검사됩니다.
    #                여러 줄을 한꺼번에 붙여넣어 3줄을 넘기면 남은 줄이 다음 input() 으로 넘어가는
    #                한계가 있으니, README 에 "한 줄씩 입력"이라고 안내해 두세요.
    #
    # [검색 키워드] python for range 반복, python 중첩 반복문 while 안의 for,
    #               python break 와 continue 차이, python f-string


def run_mode1() -> None:
    """모드 1 전체 흐름: 입력 -> 저장 확인 -> MAC 연산 -> 판정 -> 성능 분석(3x3)."""
    filter_a=read_matrix("필터 A")
    filter_b=read_matrix("필터 B")
    print("필터 A, 필터 B 저장 완료\n")
    pattern_a=read_matrix("패턴")
    time_a = measure_ms(mac, pattern_a, filter_a)
    time_b = measure_ms(mac, pattern_a, filter_b)
    score_a = mac(pattern_a, filter_a)
    score_b = mac(pattern_a, filter_b)
    verdict = judge(score_a, score_b)

    print(f"필터 A 점수: {score_a:.6f}  필터 B 점수: {score_b:.6f} ")
    print(f"필터 A 진행시간: {time_a:.6f} ms, 필터 B 진행시간: {time_b:.6f} ms")
    if verdict=="UNDECIDED":
        print("판정불가 |A-B| < 1e-9")
    elif verdict=="A": 
        print(f"판정: A")
    else:
        print(f"판정: B")
    rows = benchmark([3])
    print_perf_table(rows)

    # TODO 6) 3x3 성능 분석을 출력한다. (benchmark([3]) 와 print_perf_table() 재사용)      

    # 요구사항의 실행 흐름: 필터 A, B 입력 -> 저장 확인 -> 패턴 입력 -> MAC 연산 -> 판정 -> 성능 분석
    # TODO 1) read_matrix("필터 A"), read_matrix("필터 B") 로 필터 두 개를 먼저 입력받는다.
    # TODO 2) 필터가 저장되었다는 확인 문구를 출력한다. (예: "필터 A, B 저장 완료")
    #         그다음에 read_matrix("패턴") 으로 패턴을 입력받는다. (순서를 지키는 것이 중요!)
    # TODO 3) measure_ms(...) 로 mac(패턴, 필터A), mac(패턴, 필터B) 의 평균 시간(ms)을 잰다.
    #         점수 자체는 mac() 을 한 번 더 호출해서 얻는다. (시간 측정과 점수 계산을 분리)
    # TODO 4) judge(점수A, 점수B) 로 판정을 얻는다.
    # TODO 5) 결과를 출력한다. 점수 A/B, 연산 시간(평균/10회), 판정.
    #         판정이 "UNDECIDED" 이면 "판정 불가 (|A-B| < 1e-9)" 처럼 보기 좋게 출력한다.
    # TODO 6) 3x3 성능 분석을 출력한다. (benchmark([3]) 와 print_perf_table() 재사용)
    #
    # [검색 키워드] python f-string 출력 포맷, python 소수점 자리수 출력 :.3f



# ===========================================================================
# 3. 모드 2: data.json 분석
# ===========================================================================
def normalize_label(raw: str) -> Label:
    """여러 모양의 라벨을 표준 라벨("Cross" 또는 "X")로 통일한다.

    "+" -> "Cross", "cross" -> "Cross", "x" -> "X", "X" -> "X"
    알 수 없는 값이면 ValueError 를 일으킨다.
    """
    raw=raw.strip().lower()
    if raw in ("+", "cross"):
        return LABEL_CROSS
    elif raw=="x":
        return LABEL_X
    else:
        raise ValueError(f"알 수 없는 라벨: {raw}")
          
    # 왜 필요할까? data.json 의 expected 는 "+" / "x" 이고, filters 의 키는 "cross" / "x" 입니다.
    # 서로 모양이 달라서 그대로 비교하면 같은 뜻인데도 FAIL 이 나옵니다.
    # TODO 1) raw 의 앞뒤 공백을 지우고 소문자로 바꾼다. (strip(), lower())
    # TODO 2) "+" 또는 "cross" 이면 LABEL_CROSS, "x" 이면 LABEL_X 를 return 한다.
    # TODO 3) 그 외에는 raise ValueError(f"알 수 없는 라벨: {raw}") 를 한다.
    #
    # [검색 키워드] python 문자열 strip lower, python in 연산자 (a in ("+", "cross")), python if elif else



def extract_size(key: str) -> int:
    """패턴 키에서 크기 N 을 뽑는다.

    예) extract_size("size_13_1") -> 13
    형식이 틀리면 ValueError 를 일으킨다.
    """
    key_parts = key.split("_")
    if len(key_parts) not in (2,3) or key_parts[0] != "size":
        raise ValueError(f"키 형식 오류: {key}")
    try:
        size = int(key_parts[1])
        if size <= 0:
            raise ValueError
        return size
    except ValueError:
        raise ValueError(f"크기 형식 오류: {key}")
    
    # TODO 1) key.split("_") 로 쪼개면 ["size", "13", "1"] 이 된다.
    # TODO 2) 조각이 3개이고 첫 조각이 "size" 인지 확인한다. 아니면 ValueError.
    # TODO 3) 두 번째 조각을 int() 로 바꿔 return 한다.
    #
    # [검색 키워드] python split 구분자, python int 변환, python 문자열 슬라이싱



def load_json(path: str) -> dict:
    """JSON 파일을 읽어 dict 로 돌려준다."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"해당 JSON 파일이 없거나 손상되었습니다.: {e}")
        return {}  # 빈 딕셔너리를 반환하거나, 필요에 따라 다른 처리를 할 수 있습니다.
    

    # TODO 1) with open(path, "r", encoding="utf-8") as f: 로 파일을 연다.
    # TODO 2) json.load(f) 의 결과를 return 한다.
    # TODO 3) (생각해 볼 것) 파일이 없거나 JSON 이 깨져 있으면?
    #         요구사항에는 규칙이 없으니 직접 정한다. 예) 안내 문구를 출력하고 모드 선택으로 돌아가기.
    #         try/except FileNotFoundError, json.JSONDecodeError 를 사용해 보자.
    #
    # [검색 키워드] python json.load 파일 읽기, python with open 사용법,
    #               python FileNotFoundError 처리



def load_filters(data: dict) -> Filters:
    """data["filters"] 를 읽어 {크기: {"Cross": 행렬, "X": 행렬}} 모양으로 만든다."""
    try:
        filters = {}
        for size_key, inner_dict in data["filters"].items():
            n = int(extract_size(size_key))  # "size_5" -> 5
            filters[n] = {} # 빈 딕셔너리 초기화 filters={n:{}}
            for label, matrix in inner_dict.items():
                normalized_label = normalize_label(label)  # "cross" -> "Cross", "x" -> "X"
                filters[n][normalized_label] = matrix
            print(f"✓ {size_key} 필터 로드 완료 (Cross, X)")
        return filters
    except (KeyError, ValueError, IndexError) as e:
        print(f"필터 로드 중 오류 발생: {e}")
        return {}  # 빈 딕셔너리를 반환하거나, 필요에 따라 다른 처리를 할 수 있습니다.
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        return {}  # 빈 딕셔너리를 반환하거나, 필요에 따라 다른 처리를 할 수 있습니다.
    
    # TODO 1) data["filters"] 는 {"size_5": {"cross": ..., "x": ...}, "size_13": ..., ...} 모양이다.
    # TODO 2) 바깥 키("size_5")는 extract 하듯 "_" 로 쪼개 숫자 5 를 얻는다.
    # TODO 3) 안쪽 키("cross", "x")는 normalize_label() 로 표준 라벨로 바꿔 저장한다.
    # TODO 4) 로드할 때마다 "✓ size_5 필터 로드 완료 (Cross, X)" 처럼 출력한다.
    # TODO 5) 완성된 딕셔너리를 return 한다.
    #
    # [검색 키워드] python 딕셔너리 items() 반복, python 딕셔너리에 값 넣기 d[key] = value,
    #               python 중첩 딕셔너리



def validate_case(
    n: int,                         # n: int (기대하는 패턴/필터의 한 변 크기, 예: 5, 13)
    case: dict,                     # case: dict (패턴 1건 데이터, 예: {"input": [[...]], "expected": "+"})
    filters: Filters                # filters: Dict[int, Dict[str, List[List[float]]]] (전체 필터 딕셔너리)
) -> Tuple[bool, str]:              # 반환값: Tuple[bool, str] ((통과 여부, 실패 사유 문자열))

    """패턴 1건이 판정 가능한 올바른 데이터인지 검사한다.

    반환: (True, "") 이면 정상, (False, "이유") 이면 문제 있음.
    프로그램을 멈추지 않고 '이 케이스만 FAIL' 로 처리하기 위해, 오류를 예외 대신 값으로 돌려준다.
    """

    # filters_n: Optional[Dict[str, List[List[float]]]] (크기 n에 해당하는 {"Cross": 행렬, "X": 행렬} 또는 None)
    filters_n = filters.get(n)
    if not filters_n:
        return False, f"크기 {n}에 해당하는 필터가 없습니다."

    if "input" not in case or "expected" not in case:
        return False, "필수 키(input 또는 expected) 누락"

    # pattern: List[List[float]] (입력된 2차원 행렬 데이터)
    pattern = case["input"]

    
    # 1. 입력 패턴 행 개수 확인
    if len(pattern) !=n:
         return False, f"패턴의 행의 개수가 다릅니다 (기대 {n}, 입력 {len(pattern)})"
    for i in range(len(pattern)):
        if len(pattern[i]) != n:
            return False, f"패턴의 행 {i+1}의 열 개수가 다릅니다 (기대 {n}, 입력 {len(pattern[i])})"

    # 3. 필터("Cross", "X")의 크기 확인
    # label: str ("Cross" 또는 "X" 문자열)
    for label in (LABEL_CROSS, LABEL_X):
        if label not in filters_n:
            return False, f"{n}x{n} {label} 필터가 없습니다."
        
        filt = filters_n[label]
        
        # 필터 행 개수 확인
        if len(filt) != n:
            return False, f"{label} 필터의 행 개수가 다릅니다 (기대 {n}, 입력 {len(filt)})"
        
        # 필터 각 행의 열 개수 확인
        for i in range(len(filt)):
            if len(filt[i]) != n:
                return False, f"{label} 필터의 {i + 1}행 열 개수가 다릅니다 (기대 {n}, 입력 {len(filt[i])})"

    # 4. expected 라벨 정규화 가능 여부 확인
    try:
        normalize_label(case["expected"])
    except ValueError as e:         # e: ValueError (라벨 변환 실패 시 발생한 예외 객체)
        return False, f"expected 라벨 오류: {e}"

    return True, ""




    # 아래 항목을 하나씩 검사하고, 문제가 있으면 바로 (False, "이유") 를 return 한다.
    # TODO 1) filters 에 크기 n 이 있는가?                      (n in filters)
    # TODO 2) case 에 "input" 과 "expected" 키가 모두 있는가?   ("input" in case)
    # TODO 3) case["input"] 이 n x n 인가?  (행 개수 == n, 각 행의 길이 == n)
    # TODO 4) filters[n] 의 두 필터도 n x n 인가?               (스키마/크기 불일치 방지)
    # TODO 5) expected 가 normalize_label() 로 변환 가능한가?   (try/except ValueError)
    # TODO 6) 전부 통과하면 (True, "") 를 return 한다.
    #
    # 이유 문구 예: "필터 크기 불일치 (기대 13x13, 입력 12x12)", "expected 라벨 알 수 없음: 'o'"
    #
    # [검색 키워드] python 튜플 반환 여러 값 return, python len 리스트 길이,
    #               python 딕셔너리 키 존재 확인 in



def run_case(key: str, case: dict, filters: Filters) -> CaseResult:
    """패턴 1건을 처음부터 끝까지 처리해 CaseResult 로 돌려준다. (절대 예외로 멈추면 안 됨)"""
    
    try:
        N = extract_size(key)
    except ValueError:
        return CaseResult(key=key, n=0, cross_score=0.0, x_score=0.0,verdict=LABEL_UNDECIDED,
                          expected="", passed=False, reason="키 형식 오류")
    
    try:
        valid, reason = validate_case(N, case, filters)
        if not valid:
            return CaseResult(key=key, n=N, cross_score=0.0, x_score=0.0,
                              verdict=LABEL_UNDECIDED, expected="", passed=False,
                              reason=reason
        )
        
        cross_score = mac(case["input"], filters[N][LABEL_CROSS])
        x_score = mac(case["input"], filters[N][LABEL_X])
        verdict = judge(cross_score, x_score, LABEL_CROSS, LABEL_X)
        expected = normalize_label(case["expected"])
        passed = (verdict == expected)
        if not passed:
            if verdict == LABEL_UNDECIDED:
                reason = "동점(UNDECIDED) 규칙"
            else:
                reason = f"판정 {verdict}, expected {expected} 불일치"
        
        return CaseResult(key=key, n=N, cross_score=cross_score, x_score=x_score,
                          verdict=verdict, expected=expected, passed=passed,
                          reason=reason if not passed else ""
        )
    except Exception as e:
        return CaseResult(
            key=key, n=N, cross_score=0.0, x_score=0.0,
            verdict=LABEL_UNDECIDED, expected="", passed=False,
            reason=f"실행 오류: {e}"
    )


    # TODO 1) extract_size(key) 로 N 을 얻는다. 실패(ValueError)하면
    #         n=0, passed=False, reason="키 형식 오류" 인 CaseResult 를 만들어 return 한다.
    # TODO 2) validate_case() 를 호출한다. (False, 이유) 이면 이유를 담은 FAIL 결과를 return 한다.
    # TODO 3) filters[n]["Cross"], filters[n]["X"] 를 꺼내 mac() 을 두 번 호출한다.
    # TODO 4) judge(cross점수, x점수, LABEL_CROSS, LABEL_X) 로 판정을 얻는다.
    # TODO 5) expected = normalize_label(case["expected"])
    # TODO 6) passed = (판정 == expected).
    #         판정이 "UNDECIDED" 이면 expected 와 절대 같지 않으므로 자동으로 FAIL 이 된다.
    # TODO 7) FAIL 이면 reason 을 채운다. 예) 판정이 UNDECIDED 이면 "동점(UNDECIDED) 규칙",
    #         아니면 "판정 X, expected Cross 불일치".
    #         (README 의 '실패 원인 분석'에 쓸 재료가 됩니다)
    # TODO 8) CaseResult(...) 를 만들어 return 한다.
    #
    # [검색 키워드] python dataclass 객체 생성, python 조건 표현식 a if 조건 else b



def run_mode2() -> None:
    """모드 2 전체 흐름: 필터 로드 -> 패턴 판정 -> 성능 분석 -> 결과 요약."""
    try:
        data = load_json(DATA_PATH)
        filters = load_filters(data)
        results: List[CaseResult] = []
        for key, case in data.get("patterns", {}).items():
            result = run_case(key, case, filters)
            results.append(result)
            print(f"--- {key} ---")
            print(f"Cross 점수: {result.cross_score:.6f}  X 점수: {result.x_score:.6f}  "
                  f"판정: {result.verdict} | expected: {result.expected} | "
                  f"{'PASS' if result.passed else 'FAIL (' + result.reason + ')'}")
        rows = benchmark([3, 5, 13, 25])
        print_perf_table(rows)
        summarize(results)
    except Exception as e:
        print(f"모드 2 실행 중 오류 발생: {e}")
        # 필요에 따라 추가적인 오류 처리나 로깅을 할 수 있습니다.   

    # TODO 1) load_json(DATA_PATH) 로 데이터를 읽는다.
    # TODO 2) load_filters(data) 로 필터를 만든다.
    # TODO 3) results: List[CaseResult] = [] 빈 리스트를 만든다.
    # TODO 4) data["patterns"] 의 (key, case) 를 하나씩 돌며 run_case() 를 호출하고
    #         결과를 results 에 append 한다. 케이스마다 아래 형식으로 출력한다.
    #           --- size_5_1 ---
    #           Cross 점수: ...  X 점수: ...  판정: ... | expected: ... | PASS (또는 FAIL (사유))
    # TODO 5) benchmark([3, 5, 13, 25]) 로 성능을 재고 print_perf_table() 로 출력한다.
    # TODO 6) summarize(results) 로 총 테스트 / 통과 / 실패 / 실패 목록을 출력한다.
    #
    # [검색 키워드] python 딕셔너리 items 반복, python 리스트 append



# ===========================================================================
# 4. 성능 분석 / 결과 요약
# ===========================================================================
def measure_ms(func: Callable, *args, repeat: int = REPEAT) -> float:
    """func(*args) 를 repeat 번 실행하고 1회 평균 시간을 ms(밀리초)로 돌려준다."""
    t0 = time.perf_counter()
    for _ in range(repeat):
        func(*args)
    t1 = time.perf_counter() 
    ms = (t1 - t0) * 1000 / repeat
    return ms
    # TODO 1) 시작 시각을 time.perf_counter() 로 잰다.
    # TODO 2) for 문으로 func(*args) 를 repeat 번 호출한다.
    # TODO 3) 끝 시각을 다시 잰다.
    # TODO 4) (끝 - 시작) 은 '초' 단위다. 1000 을 곱해 ms 로 바꾸고, repeat 으로 나눠 평균을 만든다.
    # TODO 5) 평균(ms)을 return 한다.
    #
    # 주의: input() 이나 print(), 파일 읽기 같은 I/O 는 시간 측정 구간 '안'에 넣지 않는다.
    #       (연산 함수 호출 구간만 재야 N 에 따른 시간 변화를 정확히 볼 수 있다)
    #
    # [검색 키워드] python time.perf_counter 시간 측정, python *args 가변 인자,
    #               python 함수를 인자로 전달


def benchmark(sizes: List[int]) -> List[Tuple[int, float, int]]:
    """크기별 MAC 평균 시간을 재서 [(N, 평균 ms, N*N), ...] 로 돌려준다."""

    rows = []
    for N in sizes:
        # 임시 데이터: N x N 행렬을 1.0 으로 채운다. (보너스 ② make_cross(N), make_x(N) 을 쓰면 더 좋다)
        pattern = make_cross(N)
        filt = make_x(N)
        avg_ms = measure_ms(mac, pattern, filt)
        rows.append((N, avg_ms, N * N))
    return rows


    # TODO 1) rows 빈 리스트를 만든다.
    # TODO 2) sizes 의 각 N 에 대해 N x N 테스트 데이터를 만든다.
    #         보너스 ② make_cross(N), make_x(N) 을 만들었다면 그것을 쓰면 되고,
    #         아직이라면 [[1.0] * N for _ in range(N)] 같은 임시 데이터를 써도 된다.
    # TODO 3) measure_ms(mac, 패턴, 필터) 로 평균 시간(ms)을 얻는다.
    # TODO 4) rows.append((N, 평균_ms, N * N)) 로 모은다.
    # TODO 5) rows 를 return 한다.
    #
    # [검색 키워드] python 리스트 컴프리헨션 2차원 리스트 만들기, python 튜플 만들기



def print_perf_table(rows: List[Tuple[int, float, int]]) -> None:
    """benchmark() 의 결과를 표로 출력한다. 열: 크기(N×N) / 평균 시간(ms) / 연산 횟수(N²)."""
    print("\n=== 성능 분석 ===")
    print(f"{'크기(N×N)':<8}{'평균 시간(ms)':>15}{'연산 횟수(N²)':>15}")
    for n, ms, ops in rows:
        print(f"{n}×{n}".ljust(8) + f"{ms:.6f}".rjust(15) + f"{ops}".rjust(15))
      
    # TODO 1) 머리글을 출력한다.   크기   평균 시간(ms)   연산 횟수
    # TODO 2) rows 를 한 줄씩 돌며 "3×3     0.010        9" 처럼 열을 맞춰 출력한다.
    #         f-string 의 정렬/자리수 기능을 쓰면 보기 좋게 맞출 수 있다.
    #         예) f"{n}×{n}".ljust(8), f"{ms:.3f}".rjust(10)
    # TODO 3) (생각해 볼 것) N 이 3 -> 25 로 커질 때 시간이 몇 배가 되었는지, N² 비율과 비교해 보자.
    #         README 의 O(N²) 근거가 된다.
    #
    # [검색 키워드] python f-string 정렬 ljust rjust, python 문자열 포맷 소수점



def summarize(results: List[CaseResult]) -> None:
    """총 테스트 수 / 통과 수 / 실패 수와 실패 케이스 목록을 출력한다."""
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    failed = total - passed
    print(f"\n=== 결과 요약 ===")
    print(f"총 테스트: {total}개  통과: {passed}개  실패: {failed}개")
    if failed > 0:
        print("실패 케이스:")
        for result in results:
            if not result.passed:
                print(f"- {result.key}: {result.reason}")
    print("상세 원인 분석은 README.md 의 '결과 리포트' 참고")

    # TODO 1) total = len(results)
    # TODO 2) passed = passed 가 True 인 개수. (for문으로 세거나 sum(...) 사용)
    # TODO 3) failed = total - passed
    # TODO 4) "총 테스트: ..개  통과: ..개  실패: ..개" 를 출력한다.
    # TODO 5) 실패가 있으면 "실패 케이스:" 아래에 "- size_13_1: <reason>" 을 한 줄씩 출력한다.
    # TODO 6) 마지막에 "상세 원인 분석은 README.md 의 '결과 리포트' 참고" 안내를 출력한다.
    #
    # [검색 키워드] python sum 조건 개수 세기, python for 문 리스트 순회



# ===========================================================================
# 5. 보너스 (선택)
# ===========================================================================
def make_x(n: int) -> Matrix:
    """[보너스 ②] n x n 크기의 X 패턴을 만든다. (두 대각선이 1.0, 나머지는 0.0)"""    
    
    #matrix[n-1][n-1]=[]
    matrix = [[0.0] * n for _ in range(n)] 
    for i in range(n):
        for j in range(n):
            if i==j:
                matrix[i][j]=1.0
            elif i+j==n-1:
                matrix[i][j]=1.0
            else:
                matrix[i][j]=0.0
    return matrix    
    # TODO 1) 0.0 으로 채운 n x n 행렬을 만든다.
    # TODO 2) i == j (왼쪽 위 -> 오른쪽 아래 대각선) 이거나
    #         i + j == n - 1 (오른쪽 위 -> 왼쪽 아래 대각선) 이면 1.0 으로 바꾼다.
    # TODO 3) 행렬을 return 한다.
    #



def make_cross(n: int) -> Matrix:
    """[보너스 ②] n x n 크기의 십자가(Cross) 패턴을 만든다. (가운데 행과 가운데 열이 1.0, 나머지는 0.0)"""
    matrix = [[0.0] * n for _ in range(n)] 
    for i in range(n):
        for j in range(n):
            if (n%2==0) and (i==(n/2) or i==(n/2-1)): 
                matrix[i][j]=1.0
            elif n%2!=0 and i==(n//2):
                matrix[i][j]=1.0
            elif (n%2==0) and (j==(n/2) or j==(n/2-1)): 
                matrix[i][j]=1.0
            elif n%2!=0 and j==(n//2):
                matrix[i][j]=1.0
            else:
                matrix[i][j]=0.0  
    return matrix     

                
    # [검색 키워드] python 행렬 대각선 인덱스 i==j, python 반대 대각선 i+j==n-1
    #     # TODO 1) n x n 을 0.0 으로 채운 행렬을 만든다.
    #         주의: [[0.0] * n] * n 은 모든 행이 같은 리스트를 가리키는 함정이 있다!
    #         [[0.0] * n for _ in range(n)] 처럼 만들어야 한다.
    # TODO 2) 가운데 인덱스 mid = n // 2 를 구한다.  (n 이 짝수일 때의 규칙은 직접 정한다)
    # TODO 3) 모든 (i, j) 에 대해 i == mid 이거나 j == mid 이면 1.0 으로 바꾼다.
    # TODO 4) 행렬을 return 한다.
    #
    # [검색 키워드] python 2차원 리스트 얕은 복사 함정, python // 몫 연산자



def flatten(m: Matrix) -> Flat:
    """[보너스 ①] N x N 2차원 배열을 길이 N*N 의 1차원 리스트로 펼친다."""
    flat_list = []
    #[flat_list.extend(row) for row in m]
    for row in m:
        flat_list.extend(row)
    return flat_list


    
    # TODO 1) 행을 위에서부터 차례로 읽어 한 줄로 이어 붙인다.
    #         (r 행 c 열의 값은 결과 리스트의 r * N + c 번째 위치에 놓인다)
    # TODO 2) 1차원 리스트를 return 한다.
    #
    # [검색 키워드] python 2차원 리스트 1차원으로 펼치기, python 리스트 extend
    pass


def mac_flat(pattern: Flat, filt: Flat) -> float:
    """[보너스 ①] 1차원으로 펼친 패턴과 필터의 MAC 점수를 돌려준다. (이중 for문 대신 for문 하나)"""
    score = 0.0
    for i in range(len(pattern)):
        score += pattern[i] * filt[i]
    return score

    # TODO 1) for i in range(len(pattern)): 한 번만 돌며 pattern[i] * filt[i] 를 누적한다.
    # TODO 2) mac() 과 같은 점수가 나와야 한다. (다르면 flatten 이나 계산에 버그가 있다는 뜻)
    #
    # [검색 키워드] python range len 인덱스 반복



def compare_2d_1d(sizes: List[int]) -> List[Tuple[int, float, float]]:
    """[보너스 ①] 같은 입력으로 2D 방식과 1D 방식의 평균 시간을 비교해 [(N, 2D ms, 1D ms), ...] 로 돌려준다."""
    rows = []
    for N in sizes:
        pattern_2d = make_cross(N)
        filt_2d = make_x(N)
        pattern_1d = flatten(pattern_2d)
        filt_1d = flatten(filt_2d)

        avg_ms_2d = measure_ms(mac, pattern_2d, filt_2d)
        avg_ms_1d = measure_ms(mac_flat, pattern_1d, filt_1d)

        mac_score_2d = mac(pattern_2d, filt_2d)
        mac_score_1d = mac_flat(pattern_1d, filt_1d)

        rows.append((N, avg_ms_2d, avg_ms_1d))

        if abs(mac_score_2d - mac_score_1d) > 1e-6:
            print(f"경고: N={N}에서 2D와 1D 점수가 다릅니다. 2D={mac_score_2d:.6f}, 1D={mac_score_1d:.6f}")
        else:
            print(f"N={N}에서 2D와 1D 점수 일치. 2D={mac_score_2d:.6f}, 1D={mac_score_1d:.6f}")
        if avg_ms_1d != 0:       
            print(f"N={N} 2D={avg_ms_2d:.6f}ms, 1D={avg_ms_1d:.6f}ms 2D/1D = {avg_ms_2d/avg_ms_1d:.2f}배")

    return rows


    # TODO 1) 각 N 에 대해 패턴/필터를 '한 번만' 만든다. (두 방식에 같은 데이터를 쓰기 위해)
    # TODO 2) flatten 은 시간 측정 '밖'에서 미리 해 둔다. (변환 시간이 섞이지 않게)
    # TODO 3) measure_ms(mac, ...) 와 measure_ms(mac_flat, ...) 를 같은 반복 횟수로 잰다.
    # TODO 4) (N, 2D_ms, 1D_ms) 를 모아 return 한다.
    # TODO 5) 점수가 서로 같은지도 한 번 확인해 보자. 결과 표를 README 에 붙이면 좋다.
    #
    # (생각해 볼 것) 파이썬에서는 1D 가 항상 빠를까? 직접 재 보고 결과를 그대로 적자.
    #
    # [검색 키워드] python 성능 비교 측정, python 캐시 지역성 (참고용)
    pass


# ===========================================================================
# 6. 프로그램 시작점
# ===========================================================================
def select_mode() -> int:
    """모드를 입력받아 1 또는 2 를 돌려준다. 잘못 입력하면 다시 묻는다."""
    print("=== Mini NPU Simulator ===")
    print("[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. 2d vs 1d 성능 비교 (보너스 ①)")
    print("0. 종료")
    while True:
        choice = input("선택: ").strip()
        if choice in ("1", "2", "3", "0"):
            return int(choice)
        else:
            print("잘못된 선택입니다. 1 또는 2를 입력하세요.")
    


    # TODO 1) "=== Mini NPU Simulator ===" 와 [모드 선택] 안내를 출력한다.
    #         1. 사용자 입력 (3x3)   2. data.json 분석
    # TODO 2) while True 로 input("선택: ") 을 받는다.
    # TODO 3) "1" 또는 "2" 이면 int 로 바꿔 return, 아니면 안내 후 다시 묻는다.
    #
    # [검색 키워드] python while 입력 검증, python input strip
    pass


def main() -> None:
    """프로그램 진입점: 모드를 고르고 해당 모드를 실행한다."""
    mode = select_mode()
    if mode == 1:
        run_mode1()
    elif mode == 2:
        run_mode2()
    elif mode == 3:
        sizes = [3, 5, 13, 25]
        rows = compare_2d_1d(sizes)
    elif mode == 0:
        print("프로그램을 종료합니다.")
        return
    else:
        print("알 수 없는 모드입니다. 프로그램을 종료합니다.")
        return

    # TODO 1) select_mode() 로 모드를 얻는다.
    # TODO 2) 1 이면 run_mode1(), 2 이면 run_mode2() 를 호출한다.
    # TODO 3) (선택) 보너스 ② 를 모드 1 에 연결하려면 select_mode 에 메뉴를 추가하거나,
    #         read_matrix 대신 make_cross / make_x 를 쓰는 선택지를 넣어 보자.



# 이 파일을 직접 실행했을 때만 main() 이 호출된다. (다른 파일에서 import 할 때는 실행되지 않음)
# [검색 키워드] python if __name__ == "__main__" 의미
if __name__ == "__main__":
    main()
