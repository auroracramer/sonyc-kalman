# -*- coding: utf-8 -*-
from .filter import KalmanFilter
try:
    from .KalmanVariationalAutoencoder import KalmanVariationalAutoencoder
except:
    from KalmanVariationalAutoencoder import KalmanVariationalAutoencoder
