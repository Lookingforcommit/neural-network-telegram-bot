import numpy as np
import requests
import telebot
import torch
from PIL import Image
import os
import configs
import time

model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=False)
import torchvision

weights_path = configs.path_to_weights

bot = telebot.TeleBot(configs.token)
model.classifier = torch.nn.Linear(1024, 2)
for param in model.parameters():
    param.requires_grad = False
model.load_state_dict(torch.load(weights_path, map_location="cpu"))


class MessageHandler(object):
    def __init__(self, message):
        self.message = message
        self.user_id = self.message.chat.id
        self.available_commands = ["start", "help", "commands"]
        self.message_id = None
        self.file_path = None

    @staticmethod
    def image_preprocessing(image_path):
        image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                           std=[0.229, 0.224, 0.225])])
        image = Image.open(image_path).convert("RGB")
        image.load()
        image = image.resize((256, 256))
        image = np.array(image, dtype=np.float32)
        image = np.array(image / 255.0, dtype=np.float32)
        image_tensor = image_transform(image)
        image_batch = image_tensor.unsqueeze(0)
        return image_batch

    def help(self):
        text = "Для получения информации по вашему рентгенографическому снимку, отправьте его в одном из " \
               "поддерживаемых вашим Telegram-клиентом форматов; для получения списка команд напишите /commands "
        bot.send_message(self.user_id, text)

    def start(self):
        text = "Привет! Я - бот, созданный для определения COVID-19 по рентгенографическим снимкам. Напишите '/help' " \
               "для дополнительной информации"
        bot.send_message(self.user_id, text)

    def file_uploading(self):
        self.message_id = self.message.photo[-1].file_id
        file_info = bot.get_file(self.message_id)
        file_path = file_info.file_path
        download_link = "https://api.telegram.org/file/bot" + configs.token + "/" + file_path
        file = requests.get(download_link)
        file_path_list = list(file_path)
        file_extension = "." + "".join((file_path_list[len(file_path_list) - 3:]))
        local_file_path = configs.work_directory + str(self.message.date) + file_extension
        out = open(local_file_path, "wb")
        out.write(file.content)
        out.close()
        self.file_path = local_file_path

    def classification(self):
        image_batch = MessageHandler.image_preprocessing(self.file_path)
        prediction = model(image_batch)
        os.remove(self.file_path)
        if prediction[0][0].item() >= prediction[0][1].item():
            text = "Нейросеть обнаружила у вас коронавирус. Мы настоятельно рекомендуем вам ограничить социальные " \
                   "контакты и дождаться подтверждения или опровержения диагноза от вашего лечащего врача "
        elif prediction[0][1].item() > prediction[0][0].item():
            text = "Нейросеть не обнаружила у вас признаков коронавируса, но мы всё же рекомендуем вам на время " \
                   "ограничить социальные контакты до получения результатов от вашего лечащего врача"
        else:
            text = "Undefined error"
        bot.send_message(self.user_id, text)

    def undefined_command(self):
        text = "Бип-бип неизвестная команда бип-бип"
        bot.send_message(self.user_id, text)

    def commands(self):
        text = "Список доступных комманд: " + str(self.available_commands)
        bot.send_message(self.user_id, text)


@bot.message_handler(commands=["start", "help", "commands"])  # Commands handler
def get_commands(message):
    if message:
        message_handler = MessageHandler(message)
        if message.text == "/start":
            message_handler.start()
        elif message.text == "/help":
            message_handler.help()
        elif message.text == "/commands":
            message_handler.commands()
        else:
            message_handler.undefined_command()


@bot.message_handler(content_types=["photo"])  # Image handler
def get_images(message):
    if message:
        message_handler = MessageHandler(message)
        message_handler.file_uploading()
        message_handler.classification()


@bot.message_handler(content_types=["text"])  # Text handler
def get_commands(message):
    if message:
        message_handler = MessageHandler(message)
        if message.text not in message_handler.available_commands:
            message_handler.undefined_command()


try:
    bot.polling(none_stop=True, interval=5)

except Exception as e:
    time.sleep(15)
