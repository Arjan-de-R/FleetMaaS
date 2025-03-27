import datetime
import random

def determine_treq(trip, date):
    '''Determine the departure time for a given trip'''

    year = date.get("year", 2025)
    month = date.get("month", 4)
    day = date.get("day", 1)

    hour = trip.exact_depart_hour
    minute = trip.exact_depart_minute
    seconds = random.randint(0, 59)

    # Create the datetime object
    dt = datetime.datetime(year, month, day, hour, minute, seconds)

    # Format as yyyy-mm-dd tt:tt:tt
    formatted_dt = dt.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_dt

def eucledian_distance(x1, y1, x2, y2):
    '''Calculate the Euclidean distance between two points'''
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5