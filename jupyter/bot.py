
import telebot
import test

token = '5067381102:AAF0izad3WhkZBGyd2occcXo26_iSTBycs4'

bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет я машинное обучения пиши "точность" и я покажу свою точность')

@bot.message_handler(content_types='text')
def send_message(message):
    if message.text:
        if message.text == 'точность':
            bot.send_message(message.chat.id, f'Точность {round(test.model.score(test.x_test, test.y_test)*100,4)}, процентов')
        else:
            bot.send_message(message.chat.id, 'Вы ввели неправильное слово введите "точность"')


bot.infinity_polling()