from transformers import pipeline

# 输入的是训练好的模型，输出的是翻译结果

translator = pipeline("translation", model="./my_opus")

f = open("./text.txt")

text = f.read()

print(translator(text)[0]['translation_text'])