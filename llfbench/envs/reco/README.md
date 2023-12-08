# Movie Recommendation Environment

The environment relied on `justwatch` API, which was used to retrieve platform-specific movie information.
However, around September 2023, `justwatch` turned off their public unofficial API access.

The original code is still available at `movie_rec_justwatch.py`.

We now only rely on `omdb` API, which is used to retrieve movie information.
And the environment is under `movie_rec.py`.