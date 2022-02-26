import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import time
import natasha as nt
from parse_history_txt import parse_history
from aiogram import Bot, Dispatcher, executor, types
import nest_asyncio
import logging
from datetime import datetime

# add filemode="w" to overwrite
logging.basicConfig(filename="log.txt", level=logging.INFO)


follow_user = {
    'id': 0,
    'first_name': '',
    'last_name': '',
    'username': '',
    'datetime': '',
    'message': '',
    'answer': '',
}


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nest_asyncio.apply()

print(parse_history())


# Загружает предобученные эмбеддинги
segmenter = nt.Segmenter()
morph_vocab = nt.MorphVocab()
emb = nt.NewsEmbedding()
morph_tagger = nt.NewsMorphTagger(emb)
ner_tagger = nt.NewsNERTagger(emb)

#syntax_parser = nt.NewsSyntaxParser(emb)


# Функция нормализации текста
def Normalize(text):

    # Убираем знаки пунктуации из текста
    word_token = str(re.sub("[^\w]", " ", text))
    # Преобразуем очищенный текст в объект Doc и


    doc = nt.Doc(word_token)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tag_ner(ner_tagger)


    # Приводим каждое слово к его изначальной форме
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    resDict = {_.text: _.lemma for _ in doc.tokens}


    # Возвращаем результат в виде списка
    return [resDict[i] for i in resDict]


# Функция ответа на запрос
def Response(user_response):
    start = time.time()
    user_response = user_response.lower()
    robo_response = ''  # Будущий ответ нашего бота
    sent_tokens.append(user_response)  # Временно добавим запрос пользователя в наш корпус.
    TfidfVec = TfidfVectorizer(tokenizer=Normalize)  # Вызовем векторизатор TF-IDF
    tfidf = TfidfVec.fit_transform(sent_tokens)

    # Создадим вектора
    vals = cosine_similarity(tfidf[-1], tfidf)  # Через метод косинусного сходства найдем предложение с наилучшим результатом
    idx = vals.argsort()[0][-2]  # Запомним индексы этого предложения # 0 индекс
    flat = vals.flatten()  # Сглаживаем полученное косинусное сходство
    flat.sort()

    req_tfidf = flat[-2]
    sent_tokens.remove(user_response)
    end = time.time()
    if (req_tfidf == 0):  # Если сглаженное значение будет равно 0, то ответ не был найден
        robo_response = robo_response + "😞 Извините, я не нашел ответа...\n😐 Как вариант попробуйте уточнить вопрос\n⌛ Время обработки запроса составило {} секунд".format(round(end-start,3))
        answer_bot = 'None'
        return robo_response, answer_bot
    else:
        answer_bot = sent_tokens[idx]
        robo_response = "➡ "+robo_response + sent_tokens[idx] + "\n⌛ Время обработки запроса составило {} секунд".format(round(end-start,3))
        return robo_response, answer_bot




# Бот


# Задание корпуса
reader = PlaintextCorpusReader('newcorpus/', r'.*\.txt')
data = reader.raw(reader.fileids())
sent_tokens = nltk.sent_tokenize(data)


welcome_input = ["привет", "ку", "прив", "добрый день", "доброго времени суток", "здравствуйте",
                 "приветствую"]  # Список привествий
goodbye_input = ["пока", "стоп", "выход", "конец", "до свидания"]  # Список прощаний

# Открытие файла и чтение содержимого
with open('token.txt', 'r') as file_token:
  token = file_token.read()


bot = Bot(token=token)  # Инициализация бота
dp = Dispatcher(bot)  # Определение диспетчера


@dp.message_handler(commands=['start'])  # Хэндлер для функции start
async def hi_func(message: types.Message):
    await message.answer("🤗 Привет!\n Я бот-историк!")
    await message.answer("🖋 Напиши мне историческую дату, исторический факт или историческую личность, а я найду "
                         "подробную информацию... ")
    await message.answer("🖋 Хочешь узнать кто автор проекта? Жми /author")


@dp.message_handler(commands=['author'])  # Хэндлер для функции start
async def hi_func(message: types.Message):
    await message.answer("Автор проекта:\n Гадельшин Артур \nwww.arturgadelshin.ru")


@dp.message_handler()  # Хэндлер для функции считывания
async def search_func(message: types.message):
    follow_user.update({
        'id': message.from_user.id,
        'first_name': message.from_user.first_name,
        'last_name': message.from_user.last_name,
        'username': message.from_user.username,
        'message': message.text.lower(),
        'datetime': datetime.now(),
    })

    if (message.text).lower() in welcome_input:
        await message.answer('🤗 Привет!')
    elif (message.text).lower() in goodbye_input:
        await message.answer('🤗 Пока! \nБуду ждать вас!')
    else:
        await message.answer('🧐 Дайте подумать...')
        res, answer = Response(message.text)
        follow_user.update({'answer': answer})
        await message.answer(res)
    logging.info(follow_user)



# Запуск бота
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
