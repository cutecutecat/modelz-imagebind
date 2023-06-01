from typing import List, Literal, TypedDict
import httpx
import msgpack

with open("dog_audio.wav", "rb") as f:
    dog_audio_bytes = f.read()

with open("dog_image.jpg", "rb") as f:
    dog_image_bytes = f.read()

with open("dog_video.mp4", "rb") as f:
    dog_video_bytes = f.read()

# Change to real endpoint URL and token
REMOTE_URL = "https://xxx.modelz.tech/"
TOKEN = "mzi-ttt"


class Input(TypedDict):
    input: List[bytes | str]
    model: Literal[
        "imagebind-image",
        "imagebind-audio",
        "imagebind-video",
        "imagebind-text",
    ]


class Embedding(TypedDict):
    embedding: List[float]
    index: int
    object: Literal["embedding"]


class Output(TypedDict):
    data: List[Embedding]
    model: Literal[
        "imagebind-image",
        "imagebind-audio",
        "imagebind-video",
        "imagebind-text",
    ]
    object: Literal["list"]


cases: List[Input] = [
    {"model": "imagebind-text", "input": ["A dog", "doggery", "puppy"]},
    {"model": "imagebind-image", "input": [dog_image_bytes]},
    {"model": "imagebind-audio", "input": [dog_audio_bytes]},
    {"model": "imagebind-video", "input": [dog_video_bytes]},
]

for case in cases:
    print("case for task {}:".format(case["model"]))
    prediction = httpx.post(
        f"{REMOTE_URL}/inference",
        headers={"X-API-Key": TOKEN},
        data=msgpack.packb(case),
        timeout=httpx.Timeout(timeout=60.0),
    )
    if prediction.status_code == 200:
        output: Output = msgpack.unpackb(prediction.content)
        embeddings = output["data"]
        for emb in embeddings:
            print(
                f"{emb['index']}-{str(emb['embedding'])[:50]}...{str(emb['embedding'])[-50:]}"
            )
    else:
        print(prediction.status_code)
        print(prediction.content)
