import re
import html
import time
from pathlib import Path
from urllib.parse import unquote

import requests


PAGE_URL = "https://www.bilibili.com/opus/451041166030261533?from=search&spm_id_from=333.337.0.0"

OUT_DIR = Path("bili_opus_images")
OUT_DIR.mkdir(exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Referer": "https://www.bilibili.com/",
}

resp = requests.get(PAGE_URL, headers=headers, timeout=20)
resp.raise_for_status()

text = html.unescape(resp.text)
text = text.replace("\\/", "/")

# 这里不要只匹配 bfs/article，要匹配整个 bfs 路径
urls = re.findall(
    r"(?:https?:)?//i\d\.hdslb\.com/bfs/[^\"'<>\s)]+",
    text
)

cleaned_urls = []
seen = set()

for url in urls:
    if url.startswith("//"):
        url = "https:" + url

    url = unquote(url)

    # 去掉 @1192w、@672w_378h 这种缩略图参数
    url = re.sub(r"@.*$", "", url)

    # 去掉可能残留的符号
    url = url.rstrip(",.;]})")

    if url not in seen:
        seen.add(url)
        cleaned_urls.append(url)

print(f"找到 {len(cleaned_urls)} 张图片")

for idx, url in enumerate(cleaned_urls, start=1):
    suffix = Path(url.split("?")[0]).suffix
    if not suffix:
        suffix = ".jpg"

    filename = OUT_DIR / f"{idx:03d}{suffix}"

    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": headers["User-Agent"],
                "Referer": PAGE_URL,
            },
            timeout=30,
        )

        if r.status_code == 200 and r.content:
            filename.write_bytes(r.content)
            print(f"[OK] {filename}")
        else:
            print(f"[失败] {r.status_code} {url}")

    except Exception as e:
        print(f"[错误] {url}")
        print(e)

    time.sleep(0.3)

print("下载完成")