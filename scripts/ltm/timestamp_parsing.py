"""Timestamp parsing utils for the Long Term Memory system."""

import math
from datetime import datetime


def get_time_difference_message(past_timestamp):
    """Converts a timestamp from the past to a human-readable 'X days ago' format."""
    datetime_format = "%Y-%m-%d %H:%M:%S"
    past = datetime.strptime(past_timestamp, datetime_format)
    now = datetime.utcnow()
    delta = now - past

    days = math.floor(delta.days)
    if days == 0:
        hours = math.floor(delta.seconds / 3600)
        if hours == 0:
            minutes = math.floor(delta.seconds / 60)
            message = "{} minutes ago".format(minutes)
        else:
            message = "{} hours ago".format(hours)
    elif days == 1:
        message = "1 day ago"
    else:
        message = "{} days ago".format(days)
    return message
