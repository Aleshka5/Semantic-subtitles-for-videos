import json
import psycopg2
from moviepy.editor import AudioFileClip
HOST = 'containers-us-west-197.railway.app'
USER = 'postgres'
PASSWORD = 'KyQKKbDzEFmd698DGe4h'
DB_NAME = 'railway'
PORT = '6258'
#ct - create table
# Parent tables
def ct_videos(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            drop table if exists videos cascade;        
            ''')
        cursor.execute(
        '''
        create table videos (
        id serial PRIMARY KEY,
        file_path text,
        preview_path text,
        name varchar(100),
        fps integer
        );
        '''
        )
        # print(cursor.fetchone())

def ct_gifs(connection):
    with connection.cursor() as cursor:
        cursor.execute(
        '''
        drop table if exists gifs cascade;        
        ''')
        cursor.execute(
        '''
        create table gifs (
        id serial PRIMARY KEY,        
        file_path text );
        '''
        )

def ct_voice_subtitles(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            drop table if exists voice_subtitles cascade;        
            ''')
        cursor.execute(
        '''            
        create table voice_subtitles (
        id serial PRIMARY KEY,
        id_cut_frame integer,
        text_subtitle text,
        voice_duration integer,
        voice_file_path text,        
        FOREIGN KEY (id_cut_frame) REFERENCES cut_scenes (id)
        on delete CASCADE
        on update RESTRICT        
        );
        '''
        )

# Child tables
def ct_cut_scenes(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            drop table if exists cut_scenes cascade;        
            ''')
        cursor.execute(
        '''            
        create table cut_scenes (
        id serial PRIMARY KEY,
        id_video integer,
        cut_frame integer,        
        id_gif integer,
        FOREIGN KEY (id_gif) REFERENCES gifs (id)
        on delete SET NULL
        on update RESTRICT,
        FOREIGN KEY (id_video) REFERENCES videos (id)
        on delete CASCADE
        on update RESTRICT
        );
        '''
        )


# Put dawn something to data base
def ct_func_add_video(connection):
    with connection.cursor() as cursor:
        cursor.execute(
        '''            
        create or replace function add_video (path_video text, 
                                              path_preview text,
                                              video_name varchar(100),
                                              fps integer,
                                              out cur_id integer) 
        language plpgsql
        as         
        $$        
        begin
        insert into videos (file_path,preview_path,name,fps) values (path_video,path_preview,video_name,fps);
        cur_id = id from videos where videos.name = video_name limit 1;        
        end;
        $$;
        '''
        )

def ct_proc_related_raws(connection):
    with connection.cursor() as cursor:
        cursor.execute(
        '''
        create or replace procedure add_cut_gif_label (video_id integer,
                                                       cut_frame_input integer,
                                                       gif_path text,
                                                       voice_path text,
                                                       text_label text,
                                                       duration_voice integer) 
        language plpgsql
        as         
        $$        
        declare
        gif_id int;
        cut_id int;
        begin
        insert into gifs (file_path) values (gif_path);
        gif_id = max(id) from gifs;
        insert into cut_scenes (id_video,cut_frame,id_gif) values (video_id,cut_frame_input,gif_id);
        cut_id = max(id) from cut_scenes; 
        insert into voice_subtitles (id_cut_frame,text_subtitle,voice_duration,voice_file_path) values (cut_id,text_label,duration_voice,voice_path);        
        end;
        $$;
        '''
        )

def ct_func_watch(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            '''
            create or replace function watch(video_name varchar(100))
            returns table(file_path text,cut_frame integer,text_subtitle text,voice_file_path text,voice_duration integer)                                     
            language sql
            as
            $$
            select file_path,cut_frame,text_subtitle,voice_file_path,voice_duration from 
            cut_scenes as c join videos as v
                on c.id_video = v.id
            join voice_subtitles as o
                on o.id_cut_frame = c.id
            where name = video_name
            order by cut_frame;
            $$;
            '''
        )
        res = cursor.fetchall()

# Take something from data base
def ct_view_all_previews(connection):
    with connection.cursor() as cursor:
        cursor.execute(
        '''            
        create or replace view all_previews as 
        select preview_path from videos;
        '''
        )

def add_raw_label(connection,video_id,cut_frame_input,gif_path,voice_path,text_label,duration_voice):
    with connection.cursor() as cursor:
        cursor.execute(
        f'''
        call add_cut_gif_label({video_id},{cut_frame_input},{gif_path},{voice_path},{text_label},{duration_voice});
        '''
        )
def add_video(connection,video_path,preview_path, video_name, fps):
    with connection.cursor() as cursor:
        cursor.execute(
        f'''            
        select * from add_video({video_path},{preview_path}, {video_name}, {fps});
        '''
        )
        res=cursor.fetchone()[0]
    return res

#    Needed views

def get_watch(connection,video_name):
    with connection.cursor() as cursor:
        cursor.execute(
            f'''                                     
            select * from watch({video_name})
            '''
        )
        res = cursor.fetchall()
    return res

def get_all_previews(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            '''                         
            select * from all_previews;
            '''
        )
        res = cursor.fetchall()
    return res

def sfromat(string):
    return "'"+string+"'"

if __name__ == '__main__':
    create = False
    fill = False
    take_all = True
    try:
        connection = psycopg2.connect(
            host=HOST,
            user=USER,
            password=PASSWORD,
            database=DB_NAME,
            port=PORT
        )
        connection.autocommit = True
# Create
        fps = 25
        base_path = 'C:/Users/Aleshka5/Desktop/Git_repos/Semantic-subtitles-for-videos'
        if create:
            ct_videos(connection)
            ct_gifs(connection)
            ct_cut_scenes(connection)
            ct_voice_subtitles(connection)
            last_id_video = 0
            last_id_cut = 0
            last_id_gif = 0
            last_id_label = 0
            ct_view_all_previews(connection)
            ct_func_add_video(connection)
            ct_proc_related_raws(connection)
# Fill DB
        if fill:
            with open(base_path+'/database/Острые_козырьки.json','r') as cut_file:
                cut_pairs = json.load(cut_file)
            with open(base_path+'/database/Острые_козырьки_labels.json','r') as labels_file:
                labels = json.load(labels_file)
            video_id = add_video(connection,sfromat('/database/Острые_козырьки.mp4'),sfromat('/database/Острые_козырьки_preview.JPG'), sfromat('Острые козырьки'), fps)
            print(f'Индекс добавленного видео: {video_id}')
            print('Alright')
            for i in range(len(cut_pairs)):
                gif_path = f'/database/Острые_козырьки/{i}.gif'
                voice_path = f'/database/Острые_козырьки/{i}.wav'
                text_label = labels[str(i)]
                cut_frame_input = cut_pairs[str(i)][0]
                duration_voice = int(AudioFileClip(base_path+f'/database/Острые_козырьки/{i}.wav').duration * fps)
                add_raw_label(connection,video_id,cut_frame_input,sfromat(gif_path),sfromat(voice_path),sfromat(text_label),duration_voice)
# Get something
        # Используется для пользовательских запросов
        if take_all:
            print(get_all_previews(connection)) # Взять все превью для каталога видео
            print(get_watch(connection,video_name = sfromat('Острые козырьки'))) # Взять информацию о конкретном видео для просмотра
    except Exception as _ex:
        print('Error while working with PostgreSQL', _ex)
    finally:
        if connection:
            connection.close()
            print('Connection closed')