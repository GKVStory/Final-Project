from transformers import pipeline

# 输入的是训练好的模型，输出的是翻译结果

translator = pipeline("translation", model="./my_opus")

text = "这是一个测试语句，如果您可以阅读本句话，说明模型已经成功训练"

print(translator(text)[0]['translation_text'])