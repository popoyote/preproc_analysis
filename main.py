from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Создаем токенизатор
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# Обучаем токенизатор на текстах
tokenizer.train(files=["text.txt"], trainer=trainer)

# Токенизируем текст
output = tokenizer.encode("Привет, мир!")
print(output.tokens)  # ['Привет', ',', 'мир', '!']