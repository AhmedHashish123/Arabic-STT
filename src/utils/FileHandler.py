import pandas as pd


class FileHandler:
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_excel(self.filename)

    def update_marks(self, name, new_mark):
        self.df.loc[self.df["Name"] == name, "Marks"] = new_mark

    def write_file(self):
        self.df.to_excel(self.filename, index=False)


if __name__ == "__main__":
    handler = FileHandler("data/test.xls")
    handler.update_marks("Serag Mohema", 10)
    handler.write_file()
