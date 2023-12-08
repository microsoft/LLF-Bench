# Movie Recommendation Environment

We now only use `omdb` API, which is used to retrieve movie information.
The environment is under `movie_rec.py`.

In order to use this environment, please follow the instruction here to register and get your own user key:

https://www.omdbapi.com/apikey.aspx

Then, you can set the environment variable `OMDB_API_KEY` to your key:
```bash
export OMDB_API_KEY=your_key
```

An old version of the environment relied on `justwatch` API, which was used to retrieve platform-specific movie information.
However, around September 2023, `justwatch` turned off their public unofficial API access.

The original code is still available at `movie_rec_justwatch.py`.