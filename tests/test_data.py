import numpy as np

from reelrank import data


def test_load_ml1m_format(tmp_path):
    (tmp_path / "ratings.dat").write_text("1::10::5::100\n1::20::3::200\n2::10::4::150\n")
    (tmp_path / "movies.dat").write_bytes("10::Am\xe9lie (2001)::Comedy|Romance\n20::Heat (1995)::Action\n"
                                          .encode("latin-1"))
    ratings, movies = data.load_ml1m(tmp_path)
    assert ratings.shape == (3, 4)
    assert movies.loc[0, "title"].startswith("Am")
    assert movies["year"].tolist() == [2001.0, 1995.0]
    assert movies.loc[0, "genres"] == ["Comedy", "Romance"]
    assert movies.loc[0, "tags"] == []


def test_load_ml32m_format_with_tags(tmp_path):
    (tmp_path / "ratings.csv").write_text("userId,movieId,rating,timestamp\n1,10,4.5,100\n2,20,2.0,200\n")
    (tmp_path / "movies.csv").write_text('movieId,title,genres\n10,"Heat (1995)",Action\n'
                                         '20,Untitled,(no genres listed)\n')
    (tmp_path / "tags.csv").write_text("userId,movieId,tag,timestamp\n" +
                                       "".join(f"{u},10,Heist,1\n" for u in range(3)) + "1,20,rare,1\n")
    ratings, movies = data.load_ml32m(tmp_path)
    assert ratings["rating"].tolist() == [4.5, 2.0]
    assert movies.loc[0, "tags"] == ["heist"]          # lower-cased, kept (3 uses)
    assert movies.loc[1, "tags"] == []                 # 1 use < min_tag_count
    assert movies.loc[1, "genres"] == [] and np.isnan(movies.loc[1, "year"])
