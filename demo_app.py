import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import bar_chart_race as bcr
import streamlit as st
import ffmpeg
import rpy2.robjects as ro
from math import pi
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

with st.echo(code_location="below"):

    st.title('''
    Spotify trends
    ''')

    st.write('''
    Добрый день, коллега. Сегодня мы будем работать с базой данных Spotify, которая лежит на kaggle. 
    В рамках этого дэшборда мы познакомимся с самим датасетом и попытаемся сделать какие-нибудь выводы о развитии музыки.
    Было бы здорово, если ты бы сейчас открыл свой любимый музыкальный сервис, включил наушники и понаслаждался треками,
    которые будут упоминаться в этом небольшом исследовании)
    ''')

    st.write('''
    Датасет слишком большой, поэтому я приложил в файл свой файл zip с датасетами. Нужно его вложить в одну папку с demo_app.py
    Если хероку не сработает, то можно ввести streamlit run demo_app.py в терминал этого файла, открытый в PyCharm.
    ''')

    st.write('''
    Для начала я проведу небольшую "чистку" данных. А именно уберу лайвы от музыкантов, чтобы нам было чуть-чуть удобнее
    и ничего не могло сильно испортить наши данные.
    ''')

    spotify_track_data = pd.read_csv("tracks.csv")
    spotify_track_data.head()
    engine = create_engine('sqlite://', echo=False)
    spotify_track_data.to_sql('tracks', con=engine)

    engine.execute('''
                    select count (id)
                    from tracks
                    ''').fetchall()

    engine.execute('''
                    select count (id) 
                    from tracks 
                    where name like '%(Live%'
                    ''').fetchall()

    engine.execute('''
                    delete 
                    from tracks
                    where name like '%(Live'
                    ''')

    rows = engine.execute('''
                    select * 
                    from tracks
                    ''').fetchall()
    spotify_track_data = pd.DataFrame(list(rows))
    spotify_track_data.columns = ['index','id', 'name', 'popularity',
                                 'duration_ms', 'explicit', 'artists',
                                 'id_artists', 'release_date', 'danceability',
                                 'energy', 'key', 'loudness',
                                 'mode', 'speechiness', 'acousticness',
                                 'instrumentalness', 'liveness', 'valence',
                                 'tempo', 'time_signature']

    spotify_track_data.artists = spotify_track_data.artists.replace('['']', np.nan)
    spotify_track_data.release_date = pd.to_datetime(spotify_track_data.release_date)
    spotify_track_data['year'] = (pd.to_datetime(spotify_track_data.release_date)).dt.year.apply(pd.to_numeric)
    spotify_track_data['month'] = (pd.to_datetime(spotify_track_data.release_date)).dt.month
    st.write(spotify_track_data.head())

    st.write('''
    Вот так выглядит наш датасет. Есть несколько небольших вопросов: что такое valence, tempo и key. Немного посидев в гугле
    можно понять, что valence- это "радость/позитивность" песни, tempo= beats per minute (удары в минуту), а key - 
    нота, которая чаще всего используется в песне.
    ''')

    st.write('''
    Хочется увидеть, какое распределение имеют эти данные. На графиках ниже представлены всевозможные распределения для
    разных параметров, которые есть в таблице. 
    ''')

    bins = np.arange(0, 1000000, 10000)
    fig1 = plt.figure(figsize=[15, 7])
    sns.histplot(data = spotify_track_data, x = 'duration_ms', bins = bins)
    plt.title('Duration Distribution')
    plt.xlabel('duration')
    plt.ylabel('Number of songs')
    st.pyplot(fig1)

    st.write('''
    Видно, что где-то около 200000 милисекунд у нас и набирается максимум. 200000 милисекунд = 3 минуты 20 секунд. Вполне
    естественный результат.
    ''')

    binsize = 1
    bins = np.arange(0, spotify_track_data['popularity'].max()+binsize, binsize)
    fig2 = plt.figure(figsize=[15, 7])
    sns.histplot(data = spotify_track_data, x = 'popularity', bins = bins)
    plt.title('Popularity Distribution')
    plt.xlabel('Popularity')
    plt.ylabel('Number of songs')
    st.pyplot(fig2)

    st.write('''
    Огромная часть треков имеет популярность 0, но это что-то просто близкое к 0 на самом деле и измеряется в современной популярности. 
    В списке "непопулярных" достаточно много треков Моцарта(Mozart: Symphony No. 35 in D Major, K. 385 "Haffner": I. Allegro con spirito), 
    Баха (Leipziger Choräle: Schmücke dich, o liebe Seele, BWV 654) и Луи Армстронга (Wild Man Blues). 
    ''')

    binsize = 0.01
    bins = np.arange(0, spotify_track_data['danceability'].max()+binsize, binsize)
    fig3 = plt.figure(figsize=[15, 7])
    sns.histplot(data = spotify_track_data, x = 'danceability', bins = bins)
    plt.title('Danceability Distribution')
    plt.xlabel('Danceability')
    plt.ylabel('Number of songs')
    st.pyplot(fig3)

    st.write('''
    Очень интуитивный результат. Люди в основном слушают музыку, чтобы потанцевать.
    ''')

    binsize = 0.02
    bins = np.arange(0, spotify_track_data['acousticness'].max()+binsize, binsize)
    fig4 = plt.figure(figsize=[15, 7])
    sns.histplot(data = spotify_track_data, x = 'acousticness', bins = bins)
    plt.title('Acousticness Distribution')
    plt.xlabel('Acousticness')
    plt.ylabel('Number of songs')
    st.pyplot(fig4)

    st.write('''
    Тоже достаточно очевидно. В современности музыку выпускают чересчур часто (по сравнению с прошлым). Нынешние технологии
    позваляют отказывать от приобритения акустических инструментов в пользу электронного звучания
    ''')

    binsize = 0.02
    bins = np.arange(0, spotify_track_data['energy'].max()+binsize, binsize)
    fig5 = plt.figure(figsize=[15, 7])
    sns.histplot(data = spotify_track_data, x = 'energy', bins = bins)
    plt.title('Energy Distribution')
    plt.xlabel('Energy')
    plt.ylabel('Number of songs')
    st.pyplot(fig5)

    st.write('''
    Немного странный результат, ведь кажется, что из-за вышесказанного музыка будет более энергичной, но нет, распределение
    имеет максимум где-то в 0,5-0,6. В эту категорию попадает огромное количество треков Битлс (Anna (Go To Him) - Remastered 2009),
    Señorita от Шона Мендеса и Wow. от Пост Малона. Треки, действительно, не самые энергичные.
    ''')

    binsize = 2
    bins = np.arange(0, spotify_track_data['tempo'].max()+binsize, binsize)
    fig6 = plt.figure(figsize=[15, 7])
    sns.histplot(data = spotify_track_data, x = 'tempo', bins = bins, kde = True)
    plt.title('Tempo Distribution')
    plt.xlabel('Tempo in BPM')
    plt.ylabel('Number of songs')
    st.pyplot(fig6)

    st.write('''
    Темп обладает очень интуитивным распределением: очень мало треков ниже 80 бпм, основная масса от 90 до 140 (танцевальная), больше уже
    слишком тяжело воспринимать на слух. В категорию с 0 бпм попадают так называемые Pause Track в альбомах джаз исполнителей. 
    Возможно, это нужно было, чтобы домотать что-то, а может просто так вставлена тишина. Треком с максимальным бпм является:
    ('誰來愛我', "['楊燦明']"). Я его послушал и не понял, откуда там бпм под 250, скорее всего какой-то баг спотифая.
    ''')

    spotify_year_data = pd.read_csv('data_by_year_o.csv')
    spotify_year_data = spotify_year_data.set_index('year')
    spotify_year_data = spotify_year_data.drop('duration_ms' , axis = 1)
    spotify_year_data = spotify_year_data.drop('loudness' , axis = 1)
    spotify_year_data = spotify_year_data.drop('tempo' , axis = 1)
    spotify_year_data = spotify_year_data.drop('popularity' , axis = 1)
    spotify_year_data = spotify_year_data.drop('key' , axis = 1)
    spotify_year_data = spotify_year_data.drop('mode' , axis = 1)
    spotify_year_data.head()

    st.write('''
    Как я уже показывал на примерах, музыка прошлого менее популярна, чем музыка современная. Давайте посмотрим, как 
    развивались показатели, на которые мы смотрели только что, со временем. (Здесь должен быть анимированный график, но 
    он почему-то не работает в стримлите. Написаны комментарии по этому поводу в комментриях к этому участку кода.
    ''')
    #Почему-то этот график не работает в стримлите. Если вы вставите его в ноутбук, сделаете pip intsll ffmpeg-python
    #то там все заработает (работает очень долго). Да, я установил ffmepg-python в PyCharm, но все равно не работает(
    #st.pyplot(bcr.bar_chart_race(df = spotify_year_data,
    #                  title = 'yr',
    #                  cmap = 'prism',
    #                  fixed_order = True,
    #                  steps_per_period = 30))

    st.write('''
    Теперь давайте переведем key и mode в привычные нам обозначения. И посмотрим, какие ноты самые популярные.
    ''')

    key_mode = pd.DataFrame()
    semi = ['C', 'Csharp', 'D', 'Dsharp', 'E', 'F', 'Fsharp', 'G', 'Gsharp', 'A', 'Asharp', 'H', 'C']
    minmaj = {0.0:'min', 1.0:'maj'}
    key_mode['mode_str'] = spotify_track_data['mode'].replace(minmaj)
    key_mode['key_str'] =  spotify_track_data['key'].apply(lambda x: semi[x])
    key_mode['key_mode'] = key_mode['key_str'] + '_' + key_mode['mode_str']
    fig7 = plt.figure(figsize=(15,7.5))
    sns.histplot(key_mode['key_mode'].sort_values(), kde = True)
    plt.xticks(rotation=90)
    plt.title('Distribution of Keys', size=15)
    st.pyplot(fig7)

    st.write('''
    Видно, что в целом музыка построена на трех нотах, при этом все они мажорные. Разговоры про три аккорда от русских
    исполнителей типа Loqimean или Скриптонита оказались правдой) 
    ''')

    st.write('''
    Но нас, как истинных потребителей интересует популярность. От чего она зависит? Давайте рассмотрим корелляции 
    ''')

    fig8 = plt.figure(figsize=(15, 7))
    ax2 = sns.regplot(data=spotify_track_data.sample(1000), x='loudness', y='popularity')
    ax2.set_title('Correlation between loudness and popularity')
    ax2.set_xlabel('loudness')
    st.pyplot(fig8)

    st.write('''
    Достаточно интересный результат. Вообще неочевидно, что громкость музыки как-то влияет на ее популярность, но это так, как мы
    видим из графика. Коррелляция очевидно положительна.
    ''')

    fig9 = plt.figure(figsize=(14.70, 8.27))
    ax2 = sns.regplot(data=spotify_track_data.sample(1000), x='energy', y='popularity')
    ax2.set_title('Correlation between energy and popularity')
    ax2.set_xlabel('energy')
    st.pyplot(fig9)

    st.write('''
    Зависимость также прослеживается. Энергичная музыка становится более популярной. Достаточно логично.
    ''')

    fig10 = plt.figure(figsize=(15, 7))
    ax2 = sns.regplot(data=spotify_track_data.sample(2000), x='danceability', y='popularity')
    ax2.set_title('Correlation between danceability and popularity')
    ax2.set_xlabel('danceability')
    st.pyplot(fig10)

    st.write('''
    Зависимость также присутствует, но она меньше. Все же не всегда людям нравится музыка, под которую можно танцевать в 
    клубах. Звучит обнадеживающе)
    ''')

    fig11 = plt.figure(figsize=(15, 7))
    ax2 = sns.regplot(data=spotify_track_data.sample(2000), x='valence', y='popularity')
    ax2.set_title('Correlation between valence and popularity')
    ax2.set_xlabel('valence')
    st.pyplot(fig11)

    st.write('''
    Зависимость не прослеживается. Кажется, что она примерно 0, то есть людям одновременно нравятся и веселые песни и 
    "на погрустить". 
    ''')

    st.write('''
    Но, вдруг, это все эти нулевые выборсы, которых очень много. Вдруг именно треки с большой популярностью тянут нас вниз,
    не давая сделать никаких выводов о музыке. Давайте тогда взглянем на top-100 треков сл спотифая прямо сейчас. Как они, кажется
    , устроены. В основном, они должны быть танцевальными (спасибо тиктоку), не сильно лиричными и с малым количеством инструментала. 
    Но конечно же, там и есть треки, которые популярны именно благодаря акустике и музыкальности (не обязательно танцевальной).
    ''')

    df_top_100 = spotify_track_data.sort_values(by=['popularity'], ascending=False)[0:100]
    st.write(df_top_100.head())

    fig12 = plt.figure(figsize=(15, 7))
    ax2 = sns.regplot(data=df_top_100, x='danceability', y='popularity')
    ax2.set_title('Correlation between danceability and popularity')
    ax2.set_xlabel('danceability')
    st.pyplot(fig12)

    fig13 = plt.figure(figsize=(15, 7))
    ax2 = sns.regplot(data=df_top_100, x='energy', y='popularity')
    ax2.set_title('Correlation between energy and popularity')
    ax2.set_xlabel('energy')
    st.pyplot(fig13)

    fig14 = plt.figure(figsize=(15, 7))
    ax2 = sns.regplot(data=df_top_100, x='acousticness', y='popularity')
    ax2.set_title('Correlation between acousticness and popularity')
    ax2.set_xlabel('acousticness')
    st.pyplot(fig14)

    fig15 = plt.figure(figsize=(15, 7))
    ax2 = sns.regplot(data=df_top_100, x='speechiness', y='popularity');
    ax2.set_title('Correlation between speechiness and popularity');
    ax2.set_xlabel('speechiness')
    st.pyplot(fig15)

    st.write('''
    Практически никаких выводов снова сделать нельзя. Причем мы видим, что на самом деле топ-100 очень разнообразен, 
    нельзя сказать, что все треки танцевальные, нельзя сказать, что все они лишены акустики, и нельзя сказать, что среди 
    них нет медлячков.
    ''')

    fig16 = plt.figure(figsize=(15, 7))
    ax2 = sns.regplot(data=df_top_100, x='year', y='popularity');
    ax2.set_title('Correlation between year and popularity');
    ax2.set_xlabel('year')
    st.pyplot(fig16)

    st.write('''
    Вот отсюда уже можно сделать какой-то вывод. Видно, что вкусы людей очень быстро меняются и адаптируются к новейшим трекам. 
    Публика хочет нового и нового.
    ''')

    st.write('''
    Теперь давайте посмотрим на то, как менялись самые популярные треки по годам. Будем брать топ-50 песен. Самую популярную, 
    медиану и последнюю. Видно, как снижается instrumentallness и acousticness по годам. (Извините, но грузится очень долго)
    ''')

    x = int(st.number_input(label='Пожалуйста, введите год, который вас интересует. С 1922. Не забудьте нажать Enter после ввода)'))

    r = ro.r
    r['source']('test.R')
    filter_year_function_r = ro.globalenv['filter_year']
    get_three_function_r = ro.globalenv['get_three']
    with localconverter(ro.default_converter + pandas2ri.converter):
        spotify_track_data_r = ro.conversion.py2rpy(spotify_track_data)
    df_result_r = filter_year_function_r(spotify_track_data_r, x)
    df_fin_res = get_three_function_r(df_result_r)
    ro.r.assign("my_df", df_fin_res)
    ro.r("save(my_df, file='{}')".format('3songs.r'))
    with localconverter(ro.default_converter + pandas2ri.converter):
        pd_from_r_df = ro.conversion.rpy2py(df_result_r)
        pd_from_r_df_1 = ro.conversion.rpy2py(df_fin_res)

    pd_from_r_df = pd_from_r_df.reset_index()
    pd_from_r_df_1 = pd_from_r_df_1.reset_index()

    pd_for_plot = pd_from_r_df_1.drop(['level_0', 'index', 'id', 'popularity',
                                       'duration_ms', 'explicit', 'artists', 'id_artists',
                                       'release_date', 'key', 'mode', 'year', 'month', 'tempo',
                                       'loudness', 'time_signature'], axis=1)

    categories = ['danceability', 'energy', 'acousticness',
                  'instrumentalness', 'valence', 'speechiness', 'liveness']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    fig17 = plt.figure(figsize=(15,7))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1], ["0.2", "0.4", "0.6", '0.8', '1'], color="grey", size=7)
    plt.ylim(0, 1.2)
    values = pd_for_plot.loc[0].drop('name').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=pd_for_plot.name[0])
    ax.fill(angles, values, 'b', alpha=0.1)
    values = pd_for_plot.loc[1].drop('name').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=pd_for_plot.name[1])
    ax.fill(angles, values, 'r', alpha=0.1)
    values = pd_for_plot.loc[2].drop('name').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=pd_for_plot.name[2])
    ax.fill(angles, values, 'r', alpha=0.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    st.pyplot(fig17)