import os
import time

timeLeft = 10
while True:
    while timeLeft != 0:
        print("Re-upload in {timeLeft}s")
        timeLeft -= 1
        time.sleep(1)

    os.system('./scripts/BaiduPCS-Go-Linux/BaiduPCS-Go upload ./log /MLArtifacts')