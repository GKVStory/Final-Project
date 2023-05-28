from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AdamWeightDecay, TFAutoModelForSeq2SeqLM

# 输入的是一个预训练模型和数据集，输出的是调整好参数的中间模型和分割完成后的两个数据集

raw_dataset = load_dataset("csv", data_files="./result.csv")
print("Done loading raw dataset...")

splited_dataset = raw_dataset["train"].train_test_split(test_size=0.2)
print("Done spliting dataset...")

modelname = "./opus"

model = TFAutoModelForSeq2SeqLM.from_pretrained(modelname)

tokenizer = AutoTokenizer.from_pretrained(modelname)

def preprocess(dataset):
    source = [i for i in dataset["target"]]
    target = [j for j in dataset["source"]]
    model_input = tokenizer(source, text_target=target, max_length=256, truncation=True)
    return model_input

tokenized_dataset = splited_dataset.map(preprocess, batched=True)

tokenized_dataset["train"].save_to_disk("./train_set")

tokenized_dataset["test"].save_to_disk("./test_set")

model.save_pretrained("./mid_opus")

tokenizer.save_pretrained("./mid_opus")


