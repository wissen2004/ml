import pandas as pd


def filter_csv(file_path, output_path):
    try:
        # Загрузка данных из CSV файла
        data = pd.read_csv(file_path)

        # Преобразование рейтинга в числовой формат (на случай некорректных данных)
        data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')

        # Удаление строк, где рейтинг равен 3, 4 или 5
        filtered_data = data[~data['Rating'].isin([3, 4, 5])]

        # Сохранение отфильтрованных данных в новый файл
        filtered_data.to_csv(output_path, index=False)
        print(f"Фильтрованный файл сохранен: {output_path}")
    except FileNotFoundError:
        print("Файл не найден. Проверьте путь к файлу.")
    except KeyError:
        print("В CSV-файле нет необходимого столбца 'Rating'. Убедитесь, что данные корректны.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    file_path = "C:/Users/ASUS/Downloads/data.csv"  # Путь к исходному файлу
    output_path = "C:/Users/ASUS/Downloads/filtered_data.csv"  # Путь для сохранения результата"
    filter_csv(file_path, output_path)
