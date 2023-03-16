#              Imports
#         -----------------
from Cutter import Cutter
from Yolov7 import Yolov7
from Video import Video
from Audio import Audio
from Utils import Utils

import os
import time
from shutil import rmtree
import json
import pprint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from moviepy.editor import concatenate_audioclips, AudioFileClip, VideoFileClip
from moviepy.video import io
#         -----------------

class SSFV():
    def __init__(self, path_model, batch_size, parts_count=-1, resize_koef=1):
        """

        :param temp_path: Путь к создаваемой папке для хранения промежуточной информации
        :param path_video: Путь к исходному видео файлу
        :param path_model: Путь к модели нарезающей видео на сцены
        :param batch_size: Количество кадров для обработки за одну итерацию(чем больше, тем больше ОЗУ займёт программа,
                                                                            но положительно влияет на скорость)

        :param parts_count: Количество батчей, которое обработает программа для данного видео
                                                                                    (выполнит обработку не всего видео)
                            -1 - если хотите обработать весь исходный видеофайл
        :param fps: Количество кадров в секунду которое имеет исходное видео
        :param resize_koef: Коэффициент сжатия исходного видео
                            1 - если хотите минимально сжать исходное видео (к сожалению пока выходное видео может быть
                            максимум размера: 540 x 1280)
        """
        self.path_model = path_model
        self.batch_size = batch_size
        self.parts_count = parts_count

        self.resize_koef = resize_koef
        self.H = 540 // self.resize_koef
        self.W = 1280 // self.resize_koef


    def add_semantic_voice_subtitles(self, temp_path, path_video, fps, clear_temp_dir = True):
        self.fps = fps
        self.path_video = path_video
        self.temp_path = temp_path
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
        else:
            rmtree(self.temp_path)
            os.makedirs(self.temp_path)

        video = Video(temp_path=self.temp_path, fps=self.fps, width=self.W, height=self.H)
        cutter = Cutter(path=self.path_model)
        my_utils = Utils()
        audio = Audio(temp_path=self.temp_path,path_video=self.path_video)
        yolo = Yolov7(width=self.W)
        iteration = 0
        process_start = time.time()
        folder_count = 0
        for batch_frames, size in Video.generator_particular_frames_from_video(
                video_path=self.path_video,
                W=self.W,
                H=self.H,
                part_size=self.batch_size,
                part_count=self.parts_count):

            # Если размер прочитанного фрагмента видео меньше батча - указать, что идёт последняя итерация
            # Для этого рассчитаем количество частей как текущая "итерация + 1"
            if size != self.batch_size:
                self.parts_count = iteration + 1
            #          --------------------------
            # Разделение на сцены
            predict = cutter.predict(batch_frames, self.resize_koef)
            my_utils.cutter_correcter(predict)  # Убрать соседние разделения на кадры, если такие есть

            # Добавление в батч начала сцены прошлого батча
            batch_frames, cut_indexes = my_utils.batch_correcting(batch_frames, predict, iteration, self.parts_count)
            print(f'Размер батча: {batch_frames.shape}')
            print(f'Индексы переходов: {cut_indexes}')

            if ((not os.path.exists(self.temp_path + f'/{folder_count}')) and (cut_indexes.shape[0] >= 2)):
                os.makedirs(self.temp_path + f'/{folder_count}')
                folder_count += 1

            # cut_indexes - массив с инексами кадров из текущего батча для которых нейросеть считает,
            # что следующий кадр - начало новой сцены
            #          --------------------------
            # Сегментация
            seg_list = [] # список предсказаний для всех сцен батча
            for i in range(cut_indexes.shape[0] - 1):
                images_for_obj_det = my_utils.dynamic_cutter(cut_indexes[i], cut_indexes[i + 1])

                # Препроцессинг датасета
                dataset = yolo.get_dataset_from_frames(batch_frames, images_for_obj_det, 2)
                # Предсказание от YOLOv7
                obj_det_predict = yolo.predict(dataset,view_img=False)
                # Постпроцессинг результатов в словарь
                detected_dict = my_utils.predict2dict(obj_det_predict)
                print(detected_dict)
                seg_list.append(detected_dict)              # Вернуть для создания озвучки

            """
                     --------------------------
            Небольшие пояснения, как взять данные для входа в нейросеть

            for i in range(0, cut_indexes.shape[0] - 1): <-- Итерация по сценам
                * 
                for index in range(cut_indexes[i] + 1, cut_indexes[i + 1]): <-- Итерация по кадрам сцены
                    ...

            Вы можете взять целый четырёхмерный массив кадров так:

                        batch_frames[ cut_indexes[i] + 1: cut_indexes[i+1] , 0]

            В месте, где стоит "*"
                     --------------------------
            """
            #          --------------------------
            # Создать нейронку по генерации текста (субтитров) и записать в seg_list
            # Потенциальный вход: frames               ,    cut_indexes,        seg_list
            #                     "кадры исходного видео"  "границы сцен"   "список подсказок из YOLOv7"
            # Выход: seg_list - с результатами генерации для каждой сцены
            #          --------------------------

            # Озвучка сцен
            offset_list = audio.create_sundtack(cut_indexes=cut_indexes, seg_list=seg_list, fps=self.fps, name_voice='zahar')
            print(offset_list)
            #          --------------------------
            # Сохраниение части видеофайла
            video.save_batch(batch_frames, cut_indexes, offset_list)

            if len(cut_indexes) > 1:
                # Соединение батча звуковой дорожки в аудио файл
                batch_soundtrack = concatenate_audioclips(audio.soundtracks_list)
                batch_soundtrack.write_audiofile(f'{self.temp_path}/{folder_count-1}/2.mp3', buffersize=1_000, verbose=True, fps=44100)

                audio.frames_offset += cut_indexes[-1] + 1
            else:
                print('Весь батч переносится...')
            iteration += 1

            # Если были прочитаны все кадры исходного видео - закончить обработку
            if size != self.batch_size:
                break
        # Закончить запись видео
        video.out.release()
        video.print_frames_info()
        # Сбор полного аудио из частей (батчей)
        soundtracks_list = []
        for folder in range(len(os.listdir(f'{self.temp_path}')) - 1):
            soundtracks_list.append(AudioFileClip(f'{self.temp_path}/{folder}/2.mp3'))

        batch_soundtrack = concatenate_audioclips(soundtracks_list)
        batch_soundtrack.write_audiofile(f'{self.temp_path}/2.mp3', buffersize=5_000, verbose=True, fps=44100)

        # Сведение видео с аудио дорожкой
        try:
            io.ffmpeg_tools.ffmpeg_merge_video_audio((self.temp_path + f'/1.mp4'), (self.temp_path + f'/2.mp3'),
                                                     ('result.mp4'),
                                                     vcodec='copy', acodec='copy', ffmpeg_output=False, logger='bar')
        except:
            print('ERROR: Итоговое видео не сохранилось. Проверьте, что вы сохраняете все промежуточные файлы')

        print(f'{time.time() - process_start}sec')
        if (os.path.exists(self.temp_path) and clear_temp_dir):
            rmtree(self.temp_path)

#=====================================================================================================================

    def video2gifs(self, path_video, base_path, dir_name=''):
        self.path_video = path_video

        video = VideoFileClip(self.path_video)
        fps = int(video.fps)

        if dir_name == '':
            start_index = len(self.path_video) - self.path_video[::-1].find('/')
            dir_name = self.path_video[start_index:-4]

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        else:
            rmtree(dir_name)
            os.makedirs(dir_name)

        cutter = Cutter(path=self.path_model)
        my_utils = Utils()
        iteration = 0
        folder_count = 0
        frames_offset = 0
        process_time = time.time()
        for batch_frames, size in Video.generator_particular_frames_from_video(
                video_path=self.path_video,
                W=self.W,
                H=self.H,
                part_size=self.batch_size,
                part_count=self.parts_count):
            print(batch_frames.shape)
            # Если размер прочитанного фрагмента видео меньше батча - указать, что идёт последняя итерация
            # Для этого рассчитаем количество частей как текущая "итерация + 1"
            if size != self.batch_size:
                self.parts_count = iteration + 1
            #          --------------------------
            # Разделение на сцены
            predict = cutter.predict(batch_frames, self.resize_koef)
            my_utils.cutter_correcter(predict)  # Убрать соседние разделения на кадры, если такие есть

            # Добавление в батч начала сцены прошлого батча
            batch_frames, cut_indexes = my_utils.batch_correcting(batch_frames, predict, iteration, self.parts_count)
            print(f'Размер батча: {batch_frames.shape}')
            print(f'Индексы переходов: {cut_indexes}')
            for i in range(0, cut_indexes.shape[0] - 1):
                if cut_indexes[i+1] - cut_indexes[i] < 48:
                    continue
                # -----------------------------------------------------------
                time_start = np.array([(((cut_indexes[i] + frames_offset+1) / fps) / 60) / 60,
                                    (((cut_indexes[i] + frames_offset+1) / fps) / 60) % 60,
                                    ((cut_indexes[i] + frames_offset+1) / fps) % 60,
                                    ((cut_indexes[i] + frames_offset+1) / fps) % 1], dtype='float32')


                time_end = np.array([(((cut_indexes[i + 1] + frames_offset-1) / fps) / 60) / 60,
                                  (((cut_indexes[i + 1] + frames_offset-1) / fps) / 60) % 60,
                                  ((cut_indexes[i + 1] + frames_offset-1) / fps) % 60,
                                  ((cut_indexes[i + 1] + frames_offset-1) / fps) % 1], dtype='float32')

                print(f'{int(time_start[0])}:{int(time_start[1])}:{int(time_start[2])}.{str(time_start[3])[2:5]} - {int(time_end[0])}:{int(time_end[1])}:{int(time_end[2])}.{str(time_end[3])[2:5]}')
                gif_video = video.subclip(f'{int(time_start[0])}:{int(time_start[1])}:{int(time_start[2])}.{str(time_start[3])[2:5]}',
                                                      f'{int(time_end[0])}:{int(time_end[1])}:{int(time_end[2])}.{str(time_end[3])[2:5]}').resize(0.5)
                gif_video.write_gif(f'{base_path}/{dir_name}/{folder_count}.gif')
                folder_count+=1

            frames_offset += cut_indexes[i + 1] + 1
            iteration += 1
            # Если были прочитаны все кадры исходного видео - закончить обработку
            if size != self.batch_size:
                break
        print(time.time()-process_time,'sec')

    def video2cuts(self, path_video, save_dir):
        self.path_video = path_video

        start_index = len(self.path_video) - self.path_video[::-1].find('/')
        dir_name = self.path_video[start_index:-4]

        cutter = Cutter(path=self.path_model)
        my_utils = Utils()
        iteration = 0
        frames_offset = 0
        all_cut_indexes = [0]
        process_time = time.time()
        for batch_frames, size in Video.generator_particular_frames_from_video(
                video_path=self.path_video,
                W=self.W,
                H=self.H,
                part_size=self.batch_size,
                part_count=self.parts_count):
            # Если размер прочитанного фрагмента видео меньше батча - указать, что идёт последняя итерация
            # Для этого рассчитаем количество частей как текущая "итерация + 1"
            if size != self.batch_size:
                self.parts_count = iteration + 1
            #          --------------------------
            # Разделение на сцены
            predict = cutter.predict(batch_frames, self.resize_koef)
            my_utils.cutter_correcter(predict)  # Убрать соседние разделения на кадры, если такие есть

            # Добавление в батч начала сцены прошлого батча
            batch_frames, cut_indexes = my_utils.batch_correcting(batch_frames, predict, iteration, self.parts_count)

            if cut_indexes.shape[0] > 1:
                all_cut_indexes.extend((cut_indexes[1:]+frames_offset).tolist() )
                print(len(all_cut_indexes))
                frames_offset += cut_indexes[-1] + 1
            else:
                print(f'Добавлено {size + 1} вместо {cut_indexes[-1] + 1}')
                frames_offset += size + 1
            iteration += 1
            # Если были прочитаны все кадры исходного видео - закончить обработку
            if size != self.batch_size:
                break
        print(f'Количество склеек: {len(all_cut_indexes)}')
        print(f'Склейки: {all_cut_indexes}')

        cut_pairs = {}
        write_index = 0
        for i in range(len(all_cut_indexes)-1):
            if all_cut_indexes[i + 1] - all_cut_indexes[i] > 48:
                cut_pairs[write_index] = [all_cut_indexes[i],all_cut_indexes[i + 1]]
                write_index += 1
        pprint.pprint(cut_pairs)
        with open(f'{save_dir}/{dir_name}.json','w') as file:
            json.dump(cut_pairs,file)
        print(time.time()-process_time,'sec')

    def create_gifs(self, path_video, base_dir):
        self.path_video = path_video

        start_index = len(self.path_video) - self.path_video[::-1].find('/')
        video_name = self.path_video[start_index:-4]

        if not os.path.exists(f'{base_dir}/{video_name}'):
            os.makedirs(f'{base_dir}/{video_name}')
        else:
            rmtree(f'{base_dir}/{video_name}')
            os.makedirs(f'{base_dir}/{video_name}')

        video = VideoFileClip(self.path_video)
        fps = int(video.fps)

        with open(f'{base_dir}/{video_name}.json','r') as file:
            cut_pairs = json.load(file)
        frames_offset = 0
        folder_count = 0
        process_time = time.time()

        for i in range(len(cut_pairs)):
            start = cut_pairs[str(i)][0]
            end = cut_pairs[str(i)][1]

            if end - start < 48:
                continue
            # -----------------------------------------------------------
            time_start = np.array([(((start + frames_offset + 1) / fps) / 60) / 60,
                                   (((start + frames_offset + 1) / fps) / 60) % 60,
                                   ((start + frames_offset + 1) / fps) % 60,
                                   ((start + frames_offset + 1) / fps) % 1], dtype='float32')

            time_end = np.array([(((end + frames_offset - 1) / fps) / 60) / 60,
                                 (((end + frames_offset - 1) / fps) / 60) % 60,
                                 ((end + frames_offset - 1) / fps) % 60,
                                 ((end + frames_offset - 1) / fps) % 1], dtype='float32')

            print(
                f'{int(time_start[0])}:{int(time_start[1])}:{int(time_start[2])}.{str(time_start[3])[2:5]} - {int(time_end[0])}:{int(time_end[1])}:{int(time_end[2])}.{str(time_end[3])[2:5]}')
            gif_video = video.subclip(
                f'{int(time_start[0])}:{int(time_start[1])}:{int(time_start[2])}.{str(time_start[3])[2:5]}',
                f'{int(time_end[0])}:{int(time_end[1])}:{int(time_end[2])}.{str(time_end[3])[2:5]}').resize(0.5)
            gif_video.write_gif(f'{base_dir}/Острые_козырьки/{folder_count}.gif')
            folder_count += 1
        print(time.time() - process_time, 'sec')

    def voicer(self, labels_path, fps, save_dir):
        with open(labels_path,'r') as labels:
            text_labels = json.load(labels)
        for i in range(len(text_labels)):
            label = text_labels[str(i)]
            Audio.text2speech(text=label,
                              voice='zahar',
                              fps=fps,
                              save_dir=save_dir,
                              name_file=i)


if __name__ == '__main__':
    system = SSFV(path_model='C:/Users/Aleshka5/my_projects/model_film_cut_v9_93',
                batch_size=1500,
                parts_count=30,
                resize_koef=10
                )
    # system.add_semantic_voice_subtitles(temp_path='C:/Users/Aleshka5/Desktop/Git_repos/Semantic-subtitles-for-videos/temp',
    #            path_video='C:/Users/Aleshka5/my_projects/content/legenda_17.mp4',
    #            fps=24,
    #            clear_temp_dir=False)

    # system.video2gifs(path_video='C:/Users/Aleshka5/my_projects/content/legenda_17.mp4',
    #                   base_path = 'C:/Users/Aleshka5/Desktop/Git_repos/Semantic-subtitles-for-videos',
    #                   dir_name='')

    # system.video2cuts(path_video='C:/Users/Aleshka5/Desktop/Git_repos/Semantic-subtitles-for-videos/database/Острые_козырьки.mp4',
    #                   save_dir = 'C:/Users/Aleshka5/Desktop/Git_repos/Semantic-subtitles-for-videos/database')

    # system.create_gifs(path_video='C:/Users/Aleshka5/Desktop/Git_repos/Semantic-subtitles-for-videos/database/Острые_козырьки.mp4',
    #                    base_dir = 'C:/Users/Aleshka5/Desktop/Git_repos/Semantic-subtitles-for-videos/database')

    system.voicer(labels_path = 'C:/Users/Aleshka5/Desktop/Git_repos/Semantic-subtitles-for-videos/database/Острые_козырьки_labels.json',
                  fps = 25,
                  save_dir = 'C:/Users/Aleshka5/Desktop/Git_repos/Semantic-subtitles-for-videos/database/Острые_козырьки')