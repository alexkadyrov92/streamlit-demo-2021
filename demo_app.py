import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from matplotlib import pyplot as plt

df = pd.read_csv("https://gist.githubusercontent.com/alexkadyrov92/3ae5358d4547cb6a6f3ab0e79c346cf2/raw/9e7a8397e3f0d055a0ef59d214d282f1c1b65cad/exoplanet.eu_catalog.csv", )
df1 = df.drop(['mass_error_min', 'mass_error_max', 'mass_sini_error_min', 'mass_sini_error_max', 'radius_error_min', 'radius_error_max','orbital_period_error_min', 'orbital_period_error_max', 'semi_major_axis_error_min', 'semi_major_axis_error_max','eccentricity_error_min', 'eccentricity_error_max','inclination_error_min', 'inclination_error_max','omega_error_min', 'omega_error_max', 'tperi_error_min', 'tperi_error_max', 'tconj_error_min', 'tconj_error_max', 'tzero_tr_error_min', 'tzero_tr_error_max', 'tzero_tr_sec_error_min', 'tzero_tr_sec_error_max', 'lambda_angle_error_min', 'lambda_angle_error_max', 'impact_parameter_error_min', 'impact_parameter_error_max', 'tzero_vr_error_min', 'tzero_vr_error_max', 'k_error_min', 'k_error_max', 'temp_calculated_error_min', 'temp_calculated_error_max', 'geometric_albedo_error_min', 'geometric_albedo_error_max', 'star_distance_error_min', 'star_distance_error_max', 'star_metallicity_error_min', 'star_metallicity_error_max', 'star_mass_error_min', 'star_mass_error_max', 'star_radius_error_min', 'star_radius_error_max','star_age_error_min', 'star_age_error_max', 'star_teff_error_min', 'star_teff_error_max',], axis = 1)
st.title("Exoplanets Dashboard")
"Этот дэшборд вдохновлен курсом 'Введение в астрофизику', который читают на факультете Совместного Бакалавриата ВШЭ и РЭШ. В качестве главного источника данных был взят сайт exoplanets.eu. Зрителю предлагается посмотреть на визуализацию некоторых данных с этого датасэта. В основном будут рассмотрены методы открытия экзопланет, а также экзопланеты, которые могут (однажды) стать новым домом для человечества."
df = df1.groupby('detection_type')['# name'].nunique()
explode = (0, 0, 0, 0, 0.2, 0, 0, 0)
fig1, ax1 = plt.subplots()
labels = ["Astrometry", "Default", "Imaging", "Microlensing", "Primary Transit", "Radial Velocity",
          "TTV", "Timing"]
ax1.pie(df, explode=explode, counterclock=False,
        shadow=True, startangle=120)
plt.legend(labels)
plt.title("Доли способов открытия экзопланет за все время")
ax1.axis('equal')
"Сначала давайте рассмотрим методы обнаружения экзопланет. Это не так просто, так как в обычный телескоп их очень сложно увидеть из-за того, что звезды своим светом блокируют свет планет."
"Всего существует 8 способов обнаружить экзопланету (тут я не очень уверен, так как неясно, что за метод 'Default', но в датасете он присутствует), про каждый из них можно отдельно почитать, к примеру, на Википедии. На первой круговой диаграмме видно, какое количество ото всех экзопланет было открыто тем или иным способом"
st.pyplot(fig1)
"Видно, что метод Primary Transit сильно доминирует, им открыто более 3000 экзопланет, но засчет чего? Почему засекать малейшие измнения в движении звезд или улавливать небольшие световые искривления сложнее, чем периодически мониторить свет звезд?"
"В поиске ответа на этот вопрос нам поможет вторая диаграмма, которая наглядно показывает то, какой метод был наиболее популярен в определенное время."
datedect = pd.DataFrame(index=df1["discovered"].sort_values().dropna().unique(),
                        columns=df1["detection_type"].unique(), dtype=int)
datedect.fillna(0, inplace=True)
for i in range (0, len(df1)):
    for j in df1["detection_type"].unique():
        if df1.iloc[i]["detection_type"] == j and np.isnan(df1.iloc[i]["discovered"]) == False:
            datedect.loc[df1.iloc[i]["discovered"]][j] += 1
datedect["sum"] = datedect.sum(axis=1)
datedect2 = pd.DataFrame(index=df1["discovered"].sort_values().dropna().unique(),
                         columns=df1["detection_type"].unique(), dtype=float)
datedect2.fillna(0, inplace=True)
for i in range (0, len(datedect)):
    for j in df1["detection_type"].unique():
        datedect2.iloc[i][j] = datedect.iloc[i][j]/datedect.iloc[i]["sum"]*100
fig2 = plt.figure()
plt.stackplot(datedect2.index, datedect2.values.T, labels=df1["detection_type"].unique())
plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Доли способов открытия экзопланет по годам")
plt.xlabel("Год")
plt.ylabel("Процент открытых экзопланет")
st.pyplot(fig2)
"В определенный момент произошел бум Транзитного метода, а именно годах в 2005-2010. Ищем какую-нибудь информацию по этому поводу в гугле. Получаем ответ на наш вопрос: запуски космических миссий CoRoT и Kepler в 2006 и 2009 годах соответственно. Эти телескопы были запущены как раз для поиска экзопланет транзитным методом."
"Второе, что очень интересует в экзопланетах - наличие жизни на них. Пока что на вопрос о наличии ответить трудно, но можно ответить на другой. Может ли в теории планета поддерживать жизнь?"
"На данный момент существует не так много критериев, которые могли бы показать, поддерживает ли планета жизнь или нет. Я избрал те, которые точно присутствуют в нашем датасете. А именно, массу планеты, массу звезды, к которой принадлежит эта планета, 'металличность' и температура этой звезды, наличие на планете химических элементов, необходимых для поддерживания белковой жизни."
"Хотелось бы узнать, какой из критериев наиболее общо обхватывает возможность наличия жизни на экзопланете."
crit = {"Mass of the planet" : "mass > 0.85/317.8",
        "Star Temperature" : "4000<star_teff<7000",
        "Star Metallicity" : "star_metallicity>0.01",
        "C" : "molecules.str.contains('C', na=False)",
        "O" : "molecules.str.contains('O', na=False)",
        "H" : "molecules.str.contains('H', na=False)",
        "N" : "molecules.str.contains('N', na=False)"}
labels1 = list(["Mass of the planet", "Star Temperature", "Star Metallicity", 'C', "O", "H", "N"])
new_df = pd.DataFrame(index=["Habitable Exoplanets"],
                      columns=labels1,
                      dtype=int)
d=0
for i in new_df.columns:
    new_df.loc["Habitable Exoplanets"][i] = len(df1.query(crit.get(i)))
    d = d + len(df1.query(crit.get(i)))
fig3, ax2 = plt.subplots()
ax2.bar(x="Mass of the planet", height=new_df.loc["Habitable Exoplanets"]["Mass of the planet"], color=["red"])
ax2.bar(x="Star Temperature", height=new_df.loc["Habitable Exoplanets"]["Star Temperature"], color=["green"])
ax2.bar(x="Star Metallicity", height=new_df.loc["Habitable Exoplanets"]["Star Metallicity"], color=["blue"])
ax2.bar(x="C", height=new_df.loc["Habitable Exoplanets"]["C"], color=["orange"])
ax2.bar(x="O", height=new_df.loc["Habitable Exoplanets"]["O"], color=["cyan"])
ax2.bar(x="H", height=new_df.loc["Habitable Exoplanets"]["H"], color=["black"])
ax2.bar(x="N", height=new_df.loc["Habitable Exoplanets"]["N"], color=["purple"])
ax2.set_ylabel("Количество обнаруженных экзопланет")
ax2.xaxis.set_visible(False)
plt.title("Количество экзопланет, удоволетворяющих криетриям жизнеспособности")
plt.legend(labels1, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig3)
"Теперь хотелось бы посмотреть, а сколько планет подходит подо все критерии, сколько только под 3? под 4? И так далее."
selection = st.multiselect("Select habitability parameters", ["Mass of the planet", "Star Temperature",
                                                              "Star Metallicity", 'C', "O", "H", "N"])
helpdf = df1
for i in selection:
    helpdf = helpdf.query(crit.get(i))
scatter_df = helpdf.dropna(subset = ["star_distance", "radius"])
scatter_df["xaxis"] = pd.Series(1, index=scatter_df.index)
scatter_df["star_distance_l.y"] = scatter_df["star_distance"]*3.26156
scatter_df["r_earth"] = scatter_df["radius"]*11.2
earth = {"# name" : "Earth", "xaxis" : 1, "star_name" : "Sun", "star_distance_l.y" : 0, "r_earth" : 1, "detection_type":
         "Known"}
scatter_df = scatter_df.append(earth, ignore_index=True)
st.write(helpdf)
f"Существует только {len(helpdf)} планет, которые удоволетворяют количеству выбранных критериев жизнеобеспечения."
"После того, как мы поняли, много их или мало, можно задать вопрос, который интересует каждого: 'А сколько же лететь до этой планеты? Попадет ли человечесто при моей жизни на планету с другими живыми существами?' Для ответа на этот вопрос можно посмотреть на график ниже. По оси Х указаны световые года (грубо говоря: если завтра человечество изобретет двигатель, позволяющий летать со скоростью света, через сколько лет мы доберемся до данной планеты). Также зрителю предлагается сравнить размеры планеты с Землей (точкой в самом начале). Лучше всего расширить этот график на максимум."
fig4 = px.scatter(data_frame=scatter_df, x="star_distance_l.y", y="xaxis", size="r_earth",
                  labels=dict(x="Дистанция до планеты (в световых годах)", color="Метод открытия"),
                  hover_name="# name", color="detection_type")
fig4.update_xaxes(title_text = "Дистанция до планеты (в световых годах)")
fig4.update_yaxes(title_text = " ")
fig4.update_layout(title_text = "Жизнеспособные экзопланеты")
fig4.update_coloraxes(colorbar_title_text = "Метод открытия")
st.plotly_chart(fig4)
