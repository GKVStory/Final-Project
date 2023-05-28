import keras
from datasets import load_from_disk

from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AdamWeightDecay, TFAutoModelForSeq2SeqLM

# 输入的是中间模型和两个数据集，输出的训练后的模型

modelname = "./mid_opus"

model = TFAutoModelForSeq2SeqLM.from_pretrained(modelname)

tokenizer = AutoTokenizer.from_pretrained(modelname)

optimizer = AdamWeightDecay(
    learning_rate=2e-5, 
    weight_decay_rate=0.01
)

model = TFAutoModelForSeq2SeqLM.from_pretrained(modelname)
print("Done loading pre-trained model...")

model.compile(optimizer=optimizer)
print("Done setting parameters for model...")

train_dataset = load_from_disk("./train_set")

test_dataset = load_from_disk("./test_set")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=modelname, return_tensors="tf")

tf_train_set = model.prepare_tf_dataset(
    train_dataset,
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    test_dataset,
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

earlystop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

my_callback = [earlystop_callback]

model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=10, callbacks=my_callback)

print("Done training model...")

tokenizer.save_pretrained("./my_opus")

model.save_pretrained("./my_opus")

print("Done saving model...")

