#              Imports
#         -----------------
import os
import shutil
from moviepy.editor import AudioFileClip
from numpy import array, frombuffer, int16
import audiosegment
import pyaudio
import wave
#         -----------------

# Text to Speech
#---------------------------------------------------
# Ваши личные данные от Yandex.Cloud
#  |   Иструкцию по их получению вы найдёте здесь: https://habr.com/ru/post/681566/
# \=/  Замечание: чтобы воспользоваться сервисом SpeechKit от Yandex вам необходимо привязать свою карту к аккаунту.
#  |              НО Yandex дарит 4000 рублей на ваши первые 30 дней использования его возможностей
import yandex_api_param

from speechkit import Session, SpeechSynthesis
#---------------------------------------------------

class Audio():
    def __init__(self, temp_path, path_video):
        self.temp_path = temp_path
        self.frames_offset = 0
        self.soundtracks_list = []
        self.base_soundtrack = AudioFileClip(path_video)
        self.base_soundtrack_duration = self.base_soundtrack.duration
        #print(f'Dur: {self.base_soundtrack_duration}')

    def create_sundtack(self, cut_indexes, seg_list, fps, name_voice):
        self.soundtracks_list = []
        offset_list = []
        for i in range(0, cut_indexes.shape[0] - 1):
            # -----------------------------------------------------------
            time_start = array([(((cut_indexes[i] + self.frames_offset) / fps) / 60) / 60,
                                   (((cut_indexes[i] + self.frames_offset) / fps) / 60) % 60,
                                   ((cut_indexes[i] + self.frames_offset) / fps) % 60,
                                   ((cut_indexes[i] + self.frames_offset) / fps) % 1], dtype='float32')

            # Если из-за погрешности разделения на кадры мы выходим за рамки видео
            if ((cut_indexes[i + 1] + self.frames_offset + 1) / fps) > self.base_soundtrack_duration:
                time_end = array([(self.base_soundtrack_duration / 60) / 60,
                                  (self.base_soundtrack_duration / 60) % 60,
                                  self.base_soundtrack_duration % 60,
                                  self.base_soundtrack_duration % 1], dtype='float32')

            else:
                time_end = array([(((cut_indexes[i + 1] + self.frames_offset) / fps) / 60) / 60,
                                  (((cut_indexes[i + 1] + self.frames_offset) / fps) / 60) % 60,
                                  ((cut_indexes[i + 1] + self.frames_offset) / fps) % 60,
                                  ((cut_indexes[i + 1] + self.frames_offset) / fps) % 1], dtype='float32')

            print(f'{int(time_start[0])}:{int(time_start[1])}:{int(time_start[2])}.{str(time_start[3])[2:5]} - {int(time_end[0])}:{int(time_end[1])}:{int(time_end[2])}.{str(time_end[3])[2:5]}')
            soundtrack = self.base_soundtrack.subclip(f'{int(time_start[0])}:{int(time_start[1])}:{int(time_start[2])}.{str(time_start[3])[2:5]}',
                                                      f'{int(time_end[0])}:{int(time_end[1])}:{int(time_end[2])}.{str(time_end[3])[2:5]}')

            # Озвучка классов голосом
            # -----------------------------------------------------------
            string_text = ''

            if seg_list == []:
                string_text = ''
            # Если модель генерации не подключена
            elif isinstance(seg_list[0],dict):
                for text in [f'{count} {class_}, ' for class_, count in zip(seg_list[i].keys(), seg_list[i].values())]:
                    string_text += text

            # Если модель генерации субтитров передала в seg_list все свои субтитры по порядку
            elif isinstance(seg_list[0],str):
                string_text = seg_list[i]

            if len(string_text) > 0:
                print(f'Text: {string_text}')
                voice, voice_offset = self.text2speech(string_text, name_voice, fps, '','output.wav')
                self.soundtracks_list.append(voice)
                offset_list.append(voice_offset)
            else:
                offset_list.append(0)
            self.soundtracks_list.append(soundtrack)

        return offset_list

    @staticmethod
    def text2speech(text, voice, fps, save_dir, name_file):
        session = Session.from_yandex_passport_oauth_token(yandex_api_param.oauth_token, yandex_api_param.catalog_id)

        synthesizeAudio = SpeechSynthesis(session)

        # `.synthesize_stream()` возвращает объект типа `io.BytesIO()` с аудиофайлом
        sample_rate = 16000
        audio_data = synthesizeAudio.synthesize_stream(
            text=text,
            voice=voice, format='lpcm', sampleRateHertz=sample_rate
        )

        # Данные озвучки нельзя получить с частотой 44,1кГц, поэтому мы воспользуемся некоторыми преобразованиями...
        num_channels = 1
        new_sample_rate = 44100
        chunk_size = 4000

        np_audio = frombuffer(audio_data, dtype=int16)
        np_audio = audiosegment.from_numpy_array(np_audio, sample_rate).resample(sample_rate_Hz=new_sample_rate,
                                                                                 sample_width=2,
                                                                                 channels=1).to_numpy_array()
        frames_length = int((np_audio.shape[0] / new_sample_rate) * fps)
        # Создаст wav файл
        if len(save_dir) > 0:
            path = f'{save_dir}/{name_file}.wav'
        else:
            path = f'{name_file}.wav'

        with wave.open(path, 'wb') as wf:
            p = pyaudio.PyAudio()
            wf.setnchannels(num_channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(new_sample_rate)
            for i in range(0, len(np_audio), chunk_size):
                wf.writeframes(np_audio[i:i + chunk_size])
            p.terminate()
        return AudioFileClip(path), frames_length