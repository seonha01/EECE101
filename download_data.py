"""
CIFAR-100 데이터 받기 (약 170MB)

    python download_data.py

- 진행률 / 속도 / 남은 시간을 보여 줍니다. 서버가 느려서 30분 넘게 걸릴 수 있어요.
- 중간에 끊기거나 Ctrl+C로 멈춰도 괜찮습니다. **다시 실행하면 받던 곳부터 이어서** 받습니다.
- 다 받으면 파일이 온전한지(MD5) 확인하고 압축을 풉니다.

조교에게 data 폴더를 받았다면 이 스크립트는 필요 없습니다 (이미 있으면 바로 끝남).
"""
import os
import sys
import time
import hashlib
import tarfile
import urllib.request

URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
SIZE = 169001437
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TAR = os.path.join(ROOT, "cifar-100-python.tar.gz")
DONE = os.path.join(ROOT, "cifar-100-python", "test")      # 압축이 풀린 결과 파일 중 하나


def md5_ok(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == MD5


def show(done, speed):
    left = (SIZE - done) / speed if speed > 0 else 0
    bar = "#" * int(30 * done / SIZE)
    sys.stdout.write(f"\r  [{bar:<30}] {done / SIZE:6.1%}  {done / 1e6:5.0f}/{SIZE / 1e6:.0f} MB"
                     f"  {speed / 1e6:4.1f} MB/s  남은 시간 {left / 60:4.0f}분 ")
    sys.stdout.flush()


def download_once():
    # 이미 받은 만큼은 건너뛰고 나머지만 요청 (Range 헤더) -> 이어받기
    have = os.path.getsize(TAR) if os.path.exists(TAR) else 0
    if have >= SIZE:
        return
    req = urllib.request.Request(URL, headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(req, timeout=60) as r, open(TAR, "ab" if have else "wb") as f:
        if have and r.status != 206:            # 서버가 이어받기를 거부하면 처음부터
            f.seek(0); f.truncate(); have = 0
        t0, got = time.time(), 0
        for chunk in iter(lambda: r.read(1 << 16), b""):
            f.write(chunk)
            got += len(chunk)
            show(have + got, got / max(time.time() - t0, 1e-3))
    print()


def main():
    if os.path.exists(DONE):
        print("CIFAR-100이 이미 준비되어 있습니다:", ROOT)
        return
    os.makedirs(ROOT, exist_ok=True)
    print("CIFAR-100 다운로드 (끊겨도 다시 실행하면 이어서 받습니다)")
    for attempt in range(1, 31):
        try:
            download_once()
            break
        except KeyboardInterrupt:
            print("\n멈췄습니다. 다시 실행하면 이어서 받습니다.")
            sys.exit(1)
        except Exception as e:
            print(f"\n  연결이 끊겼습니다 ({type(e).__name__}). 5초 후 이어서 받기... ({attempt}/30)")
            time.sleep(5)
    print("  파일 확인 중 (MD5)...")
    if not md5_ok(TAR):
        os.remove(TAR)
        sys.exit("파일이 깨져 있어서 지웠습니다. 다시 실행해 주세요.")
    print("  압축 푸는 중...")
    with tarfile.open(TAR) as t:
        t.extractall(ROOT)
    print("완료:", os.path.join(ROOT, "cifar-100-python"))


if __name__ == "__main__":
    main()
