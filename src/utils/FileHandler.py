import pandas as pd
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine


class FileHandler:
    def __init__(self, filename):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        self.filename = filename
        self.df = pd.read_excel(self.filename)
        file_path = 'data/numbers.txt'
        numbers = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                numbers.append(line.strip())
        names = self.df['Name'].tolist()
        self.name_embeddings = [self.get_embedding(name) for name in names]
        self.numbers_embeddings = [self.get_embedding(number) for number in numbers]


    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()  # Use squeeze() here
    

    def update_marks(self, name, new_mark):
        name_embedding = self.get_embedding(name)
        name_similarities = [1 - cosine(name_embedding, embedding) for embedding in self.name_embeddings]
        print(name_similarities)
        max_name_similarity = max(name_similarities)
        if max_name_similarity < 0.65:
            raise ValueError("Name not found")
        name_index = name_similarities.index(max_name_similarity)

        number_embedding = self.get_embedding(str(new_mark))
        number_similarities = [1 - cosine(number_embedding, embedding) for embedding in self.numbers_embeddings]
        print(number_similarities)
        max_similarity = max(number_similarities)
        if max_similarity < 0.6:
            raise ValueError("Number not found")
        mark_index = number_similarities.index(max_similarity)
        new_mark = mark_index
        self.df.loc[name_index, "Marks"] = new_mark

    def write_file(self):
        self.df.to_excel(self.filename, index=False)


if __name__ == "__main__":
    handler = FileHandler("data/test.xlsx")
    handler.update_marks("أحمد محمد", "عشرا")
    handler.write_file()


