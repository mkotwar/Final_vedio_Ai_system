import os
import sys
import json
import urllib.request
import urllib.parse
from uuid import uuid4
import time

base_url = "http://127.0.0.1:8000"
video_path = r"c:\Mukul K\vinfo1\video-search-engine\data\videos\32ac5bc9-91ea-4cfe-8d82-a383f6d608c4.mp4"

print("Uploading video...")
import mimetypes

with open(video_path, 'rb') as f:
    video_data = f.read()

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = bytearray()
body.extend(f'--{boundary}\r\n'.encode('utf-8'))
body.extend(f'Content-Disposition: form-data; name="file"; filename="vid.mp4"\r\n'.encode('utf-8'))
body.extend(b'Content-Type: video/mp4\r\n\r\n')
body.extend(video_data)
body.extend(f'\r\n--{boundary}--\r\n'.encode('utf-8'))

req = urllib.request.Request(f"{base_url}/videos/upload", data=body)
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
try:
    resp = urllib.request.urlopen(req)
    res_data = json.loads(resp.read().decode('utf-8'))
    video_id = res_data['video_id']
    print(f"Video uploaded: {video_id}")
    
    print("Extracting frames...")
    req2 = urllib.request.Request(f"{base_url}/frames/extract", data=json.dumps({"video_id": video_id}).encode('utf-8'))
    req2.add_header('Content-Type', 'application/json')
    start = time.time()
    resp2 = urllib.request.urlopen(req2)
    res_data2 = json.loads(resp2.read().decode('utf-8'))
    print(f"Extraction took {time.time() - start:.2f}s")
    print(json.dumps(res_data2, indent=2))
except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, 'read'):
        print(e.read().decode('utf-8'))
