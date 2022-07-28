#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

# USER BASED RECOMMENDER PROJE ADIMLARI
# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması

# Kullanıcıların davranış benzerlikleri üzerinden film önerileri yapılacak.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 100)

# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
movie = pd.read_csv('datasets/movie.csv')
movie.head()
movie.shape

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
rating = pd.read_csv('datasets/rating.csv')
rating.head()
rating.shape
rating["userId"].nunique()


# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.
df = movie.merge(rating, how="left", on="movieId")
df.head(50)
df.shape


# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız.
# Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.

# df.groupby("title")["title"].count()
comment_counts = pd.DataFrame(df["title"].value_counts())

comment_counts

# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz
rare_movies = comment_counts[comment_counts["title"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape


# Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz. Yani user-movie df matrisi oluşturulması adımı.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()


# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie.csv')
    rating = pd.read_csv('datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


user_movie_df = create_user_movie_df()

user_movie_df.head()

#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.
random_user = 108170

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Kaç tane film oyladı belirlediğimiz user
len(movies_watched)

#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
# Ve yeni bir df oluşturuyoruz.

user_movie_count = movies_watched_df.T.count()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.

# user ile benzer filmlerin en az %60 ını izlesin ki onlar üzerinden bir öneri yapayım.
# En az 1 kez izleyen de var bu benim userım ile aynı filmleri izleyebilir yorumu yapamıyorum.
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

# %60 üzeri benzer filmlere oy veren user sayısı nedir?
len(users_same_movies)

#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. Belirlemiş olduğumuz userın ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer kullanıcıları (Top Users) bulacağız.

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı
# şekilde movies_watched_df dataframe’ini filtreleyiniz.

# Belirlemiş olduğumuz user ve diğer userların bir araya getirilmesi
# Burada artık belirlemiş olduğumuz user ve bu usera en çok benzeyen kullanıcılar yer almakta.
# isin: yapısı şunları içeriyorsa getir işlemi yapmaktadır.
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.

# İtem based recommender da sutünlarda filmler vardı. Peki burada ne olması gerekiyor?
# Sutünlarda userlar olması gerekiyor. Bunun için final_df'in transpozu alınarak userlar sutüne alınmıştır.
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df[corr_df["user_id_1"] == random_user]

# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)

# Belirlediğimiz user ile en yüksek korelasyona sahip olan userlar kimler?
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz.

# Şu anda bizim usera en benzer userları bulduk. Peki ne önereceğim? Userları mı önereceğim?
# Bu kullanıcıların id bilgileri ile hangi filme kaç puan verdiklerine gidiyor olacağım.
# Bu tabloda çoklamalar var neden? Bir kullanıcı birden çok filme puan vermiş olabilir.
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

# random_userın buradan çıkarılması işlemi
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].nunique()
top_users_ratings.head()

#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Problem ne? Öneri yapmak istediğimizde göz önünde bulundurmamız gereken iki metrik var.
# 1) Korelasyon
# 2) Rating: filmlere vermiş oldukları puanlar

# Problem 1
# Bir filme bizim belirlemiş olduğumuz userla benzer alışkanlığa
# sahip birisi yüksek puanlama birisi düşük puanlama vermiş olabilir.

# Problem 2
# Bizim belirlemiş olduğumuz user ile benzer korelasyona sahip userlar incelenildğinde yüksek korelasyonlu olup
# 4.5 rating veren ve düşük korelasyonlu olup 4.5 rating verenler var hangisini dikkate alacağım?


# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
# Öyle bir skor yapmalıyız ki hem ratingin etkisi hemde korelasyonun etkisi göz önünde bulundurulsun.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Adım 2: Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.

# Problem ne? Filmler çokluyor çünkü userlar aynı filmleri izlemiş olabilirler.
# Bunu engellemek adına filmlere göre group by alıp weighted_ratingin mean'i alınarak bu durumda çözülür.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.

# weighted_rating'i 3.5'ten büyük olanları getirelim: Bunlar artık önerilecek olan filmler.
recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)


# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

# 0    Mystery Science Theater 3000: The Movie (1996)
# 1                               Natural, The (1984)
# 2                             Super Troopers (2001)
# 3                         Christmas Story, A (1983)
# 4                       Sonatine (Sonachine) (1993)

#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının izlediği filmlerden en son en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
# 5 öneri user-based 5 öneri item-based olacak şekilde 10 öneri yapınız.

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170
import pandas as pd
# Adım 1: movie,rating veri setlerini okutunuz.
movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')

# Traih değişkenine çevrilmesi işlemi
rating["timestamp"] = pd.to_datetime(rating["timestamp"])

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Adım 3: User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)


# Son iki adımı uygulayan fonksiyon
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)

# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık.
movies_from_item_based[1:6].index

# ITEM BASED
# ['My Science Project (1985)', 'Mediterraneo (1991)',
#        'Old Man and the Sea, The (1958)',
#        'National Lampoon's Senior Trip (1995)', 'Clockwatchers (1997)'],

# USER BASED
# 0    Mystery Science Theater 3000: The Movie (1996)
# 1                               Natural, The (1984)
# 2                             Super Troopers (2001)
# 3                         Christmas Story, A (1983)
# 4                       Sonatine (Sonachine) (1993)
