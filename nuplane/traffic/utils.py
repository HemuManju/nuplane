import math

# Geology constants
R = 6371000  # Radius of third rock from the sun, in metres
FT = 12 * 0.0254  # 1 FOOT = 12 INCHES
NAUTICAL_MILE = 1.852  # Nautical mile in meters 6076.118ft=1nm


def haversine(lat1, lat2, long1, long2):  # in radians.
    dlat, dlong = lat2 - lat1, long2 - long1
    return math.pow(math.sin(dlat / 2), 2) + math.cos(lat1) * math.cos(lat2) * math.pow(
        math.sin(dlong / 2), 2
    )


def distance(p1, p2):  # in degrees.
    lat1, lat2 = math.radians(p1['lat']), math.radians(p2['lat'])
    long1, long2 = math.radians(p1['lon']), math.radians(p2['lon'])
    a = haversine(lat1, lat2, long1, long2)
    return 2 * R * math.asin(math.sqrt(a))  # in m
