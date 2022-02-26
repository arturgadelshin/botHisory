
# Функция парсинга исторического файла и создания отдельных корпусов
def parse_history():
    with open('history_date/history_text.txt', 'r', encoding='utf-8', ) as file:
        # read lines for file
        lines = file.readlines()

    for i, line in enumerate(lines):
        with open(f'newcorpus/{i}.txt', 'w', encoding='utf-8') as new_file:
            new_file.write(line)
    return 'Parsing - OK'
