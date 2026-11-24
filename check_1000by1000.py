"""check_1000by1000.py - 1000×1000 처리 확인용 스크립트 (모드 3 확장판)

main.py 의 함수(make_cross, make_x, mac, mac_flat, flatten, judge, measure_ms)를
그대로 가져다 써서 1000×1000 데이터를 만들고, 아래 항목을 한 번에 점검한다.

  1) 데이터 생성 시간
  2) Cross vs X 판정 (judge)         -> main.py 의 MAC 결과
  3) 2D vs 1D 비교 (모드 3 와 같은 방식, main.compare_2d_1d)
  4) 최적화 후보 비교 (README '1000×1000 처리' 섹션의 근거)
        2D mac / 1D mac_flat / array('d') / sum(map(mul)) / 희소 필터
     -> 모든 방식이 기존 mac() 과 점수가 1e-6 이내로 같은지 확인
  5) 메모리: list[list[float]] vs 1D list vs array('d')
  6) JSON 파싱 시간 (메모리 안에서 dumps/loads, 파일 입출력 제외)

실행:  python check_1000by1000.py          (N=1000)
       python check_1000by1000.py 500      (다른 크기로 시험)

주의: 시간/메모리 값은 PC 마다 다르다. 본인 PC 에서 나온 값을 README 에 적을 것.
"""

import json
import operator
import os
import sys
import time
import tracemalloc
from array import array

# main.py 가 같은 폴더에 있다. (어디서 실행해도 import 되도록 경로를 추가)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as npu  # main.py 는 __name__ == "__main__" 보호가 있어 import 해도 실행되지 않는다.

TOLERANCE = 1e-6  # 방식 간 점수 일치 허용오차 (README 의 검증 기준과 동일)


# ---------------------------------------------------------------------------
# 최적화 후보 (check 전용. main.py 의 mac() 은 과제 요구대로 직접 반복문 유지)
# ---------------------------------------------------------------------------
def mac_array(pattern: array, filt: array) -> float:
    """array('d') 로 저장된 1D 데이터의 MAC. C 수준 루프(sum(map(mul)))를 쓴다."""
    return sum(map(operator.mul, pattern, filt))


def mac_sum_map(pattern: list, filt: list) -> float:
    """1D list 에 sum(map(mul)) 만 적용한 MAC."""
    return sum(map(operator.mul, pattern, filt))


def to_sparse(filt_flat: list) -> list:
    """필터에서 0 이 아닌 칸만 (인덱스, 값) 으로 모은다. Cross/X 는 약 2N-1 개."""
    return [(i, v) for i, v in enumerate(filt_flat) if v != 0.0]


def mac_sparse(pattern_flat: list, sparse_filt: list) -> float:
    """희소 필터 MAC: 필터의 비영 칸만 곱한다. O(N²) -> O(N)."""
    score = 0.0
    for i, v in sparse_filt:
        score += pattern_flat[i] * v
    return score


# ---------------------------------------------------------------------------
# 보조: 크기 계산, 출력
# ---------------------------------------------------------------------------
def mb(nbytes: float) -> str:
    return f"{nbytes / 1024 / 1024:.1f} MB"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_same(name: str, score: float, base: float) -> bool:
    ok = abs(score - base) <= TOLERANCE
    print(f"  {name:<22} 점수 {score:>14.6f}  기준과 {'일치' if ok else '불일치(!)'}")
    return ok


# ---------------------------------------------------------------------------
# 점검 본체
# ---------------------------------------------------------------------------
def run_check(n: int) -> bool:
    all_ok = True
    print(f"=== {n}×{n} 처리 확인 (연산 횟수 N² = {n * n:,}, 반복 {npu.REPEAT}회 평균) ===")

    # 1) 데이터 생성 ------------------------------------------------------
    section("1) 데이터 생성")
    t0 = time.perf_counter()
    pattern = npu.make_cross(n)
    filt = npu.make_x(n)
    t1 = time.perf_counter()
    print(f"  make_cross + make_x : {(t1 - t0) * 1000:.1f} ms")

    # 2) 판정 -------------------------------------------------------------
    section("2) Cross vs X 판정 (패턴 = Cross)")
    cross_filter = npu.make_cross(n)
    x_filter = filt
    score_c = npu.mac(pattern, cross_filter)
    score_x = npu.mac(pattern, x_filter)
    result = npu.judge(score_c, score_x, npu.LABEL_CROSS, npu.LABEL_X)
    print(f"  Cross 필터 점수 {score_c:.6f} / X 필터 점수 {score_x:.6f} -> 판정: {result}")
    if result != npu.LABEL_CROSS:
        print("  경고: Cross 패턴이 Cross 로 판정되지 않았습니다.")
        all_ok = False

    # 3) 모드 3 와 같은 2D vs 1D 비교 ----------------------------------------
    section("3) 2D vs 1D 비교 (모드 3 와 동일: main.compare_2d_1d)")
    rows = npu.compare_2d_1d([n])
    if not rows:
        all_ok = False

    # 4) 최적화 후보 비교 -----------------------------------------------------
    section("4) 최적화 후보 비교 (README 우선순위 근거)")
    p_flat = npu.flatten(pattern)
    f_flat = npu.flatten(filt)
    p_arr = array("d", p_flat)
    f_arr = array("d", f_flat)
    sparse = to_sparse(f_flat)

    base = npu.mac(pattern, filt)
    candidates = [
        ("2D mac (기준)", npu.mac, (pattern, filt)),
        ("1D mac_flat", npu.mac_flat, (p_flat, f_flat)),
        ("1D sum(map(mul))", mac_sum_map, (p_flat, f_flat)),
        ("array('d') sum(map)", mac_array, (p_arr, f_arr)),
        (f"희소 필터({len(sparse):,}칸)", mac_sparse, (p_flat, sparse)),
    ]
    base_ms = None
    print(f"  {'방식':<22}{'평균(ms)':>12}{'기준 대비':>12}")
    for name, func, args in candidates:
        ms = npu.measure_ms(func, *args)
        if base_ms is None:
            base_ms = ms
        speed = f"{base_ms / ms:.1f}배" if ms > 0 else "-"
        print(f"  {name:<22}{ms:>12.4f}{speed:>12}")
    print("  -- 점수 일치 검사 (허용오차 1e-6) --")
    for name, func, args in candidates:
        all_ok &= check_same(name, func(*args), base)

    # 5) 메모리 -------------------------------------------------------------
    section("5) 메모리 (행렬 1개 기준, tracemalloc 으로 실측)")
    # 주의: make_cross 로 만든 행렬은 0.0 / 1.0 이 같은 float 객체를 가리켜 작게 나온다.
    #       data.json 처럼 파일에서 읽으면 값마다 새 float 객체가 생기므로 그 경우를 따로 잰다.
    text_one = json.dumps(pattern)
    tracemalloc.start()
    parsed = json.loads(text_one)
    _, peak_parsed = tracemalloc.get_traced_memory()
    cur_parsed, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del parsed

    tracemalloc.start()
    arr = array("d", p_flat)
    cur_arr, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del arr

    tracemalloc.start()
    built = npu.make_cross(n)
    cur_built, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del built

    print(f"  JSON 텍스트(행렬 1개)                   : {mb(len(text_one.encode('utf-8')))}")
    print(f"  json.loads 로 읽은 list[list[float]]    : {mb(cur_parsed)}  (파싱 중 피크 {mb(peak_parsed)})")
    print(f"  make_cross 로 만든 list[list[float]]    : {mb(cur_built)}  (0.0/1.0 객체를 공유해 작음)")
    print(f"  array('d')                              : {mb(cur_arr)}")
    print(f"  모드 2 처럼 행렬 4개를 json 으로 올리면 약 {mb(cur_parsed * 4)}  (행렬 1개 실측 x 4, 추정)")
    print("  ※ tracemalloc 은 코드를 느리게 하므로 이 구간의 시간은 쓰지 않는다.")

    # 6) JSON -------------------------------------------------------------
    section("6) JSON 파싱 시간 (메모리 안에서만 측정, 파일 입출력 제외)")
    text = json.dumps({"filter": filt, "pattern": pattern})
    t0 = time.perf_counter()
    back = json.loads(text)
    t1 = time.perf_counter()
    print(f"  행렬 2개 JSON 텍스트 크기: {mb(len(text.encode('utf-8')))}")
    print(f"  json.loads 시간         : {(t1 - t0) * 1000:.1f} ms")
    if back["filter"] != filt or back["pattern"] != pattern:
        print("  경고: JSON 왕복 결과가 원본과 다릅니다.")
        all_ok = False

    # 결과 ------------------------------------------------------------------
    print("\n=== 최종 결과 ===")
    print("모든 점검 통과" if all_ok else "일부 점검 실패 (위 경고 확인)")
    return all_ok


def main() -> None:
    n = 1000
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
            if n < 3:
                raise ValueError
        except ValueError:
            print("크기는 3 이상의 정수로 입력하세요. 예) python check_1000by1000.py 1000")
            return
    run_check(n)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단되었습니다.")
