# Object Detect

You can track the latest updates by visiting the project's github address：[Object Detect Repository](https://github.com/ciaoyizhen/object_detect)

Requirement:

```
python >= 3.10
```

## Goal
Use huggingface to implement a variety of tasks, and you can replace the model at any time without modifying the code.

## Train Step:
```
1. python -m venv .env
2. source .env/bin/activate
3. pip install -r requirements.txt
4. modify yaml config
5. torchrun main.py (yaml_path) or python main.py
```

## Eval Step:
```
python demo/inference_demo.py
```

## WebUI
```
1. python demo/web_demo.py
2. open link with your browser
```

> Note: during training, only the model file is saved, for the image pre-processing, it is not saved, you need to manually put the pre-processing configuration file into the model file to be used

## FAQ
1. open too many file
```
ulimit -n xxx  # increase open file
```
2. How to download a model to train
```
1. open this (https://github.com/ciaoyizhen/image_classification)
2. choose and download a model
3. modify yaml
```


## Support the Author

If you find this project helpful and would like to support the author, you can make a donation using WeChat Pay or Alipay by scanning the QR codes below.

**WeChat Pay:**

<img src="assets/WeChat%20Pay.jpg" alt="WeChat Pay QR Code" width="300"/>

**Alipay:**

<img src="assets/Alipay.jpg" alt="WeChat Pay QR Code" width="300"/>


Thank you for your support!
