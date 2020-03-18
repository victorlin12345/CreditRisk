def XY_from_df(df):
    features = list(df.columns)[1:]
    Y = df['risk'].values
    X = df[features].values
    return X, Y