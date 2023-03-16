# Semantic-subtitles-for-videos
Файлы для базы можете скачать <a href="https://drive.google.com/file/d/1L8EGMgR7-SndFaOet4SgN3BmjvCeCN8g/view?usp=sharing">здесь</a>.
# Как пользоваться "data base create".
Скрипт написан в основном для создания и заполнения БД, но в нём уже есть две нужные функции: 
1) для получения превью изображений.<br>
2) для получения мета информации для просмотра видео.<br>
![image](https://user-images.githubusercontent.com/78702396/225712476-0e6b966f-3a78-4e06-8997-c583a0cc5a9a.png)
1. Пример вывода для первой функции:
![image](https://user-images.githubusercontent.com/78702396/225713615-b2812950-a1b9-4744-8180-12f9fbebf33c.png)
2. Пример вывода для второй функции: (cut_frame - кадр склейки видео, указан в кадрах. Чтобы получить время нужно умножить на fps)
![image](https://user-images.githubusercontent.com/78702396/225713438-14cb75d0-8266-4912-a946-562f8a7453f7.png)

В Python эти таблицы примут вид: [(столбец 1,столбец 2,столбец 3,...),(столбец 1,столбец 2,столбец 3,...)], где кортежи - строки <br>
-----------file_path----------------cut_frame---------------text_subtitle------------------------------voice_file_path---------voice_dur--fps<br>
[('/database/Острые_козырьки.mp4', 0, 'черный экран и появляется белая надпись', '/database/Острые_козырьки/0.wav', 70, 25),<br>
('/database/Острые_козырьки.mp4', 110, 'черный экран и появляется белая надпись', '/database/Острые_козырьки/1.wav', 70, 25), ... ]<br>

## Важно, что все пути к файлам считаются от папки database.
До неё должен вести путь base_path:
![image](https://user-images.githubusercontent.com/78702396/225715776-2c77c3f4-6b4f-479c-8c8d-8afef648d5b8.png)
