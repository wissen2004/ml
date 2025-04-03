import pandas as pd

def count_comments(file_path):
    try:
        data = pd.read_csv(file_path)

        print("Первые 15 записей из файла:")
        print(data.head(15))

        data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')
        data = data.dropna(subset=['Rating'])

        rating_counts = data['Rating'].value_counts().sort_index()
        total_comments = data['Review'].dropna().count()

        print("\nКоличество отзывов для каждого рейтинга:")
        for rating, count in rating_counts.items():
            print(f"Рейтинг {rating}: {count} отзывов")

        print(f"\nОбщее количество отзывов: {total_comments}")
    except FileNotFoundError:
        print("Файл не найден.")
    except KeyError:
        print("В CSV-файле нет необходимого столбца..")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    file_path = "./data/data.csv"
    count_comments(file_path)